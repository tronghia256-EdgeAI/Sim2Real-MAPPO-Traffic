"""
Self-Organizing Traffic Light (SOTL) controller — SUMO baseline.

Reference
---------
Cools, S.-B., Gershenson, C., & D'Hooghe, B. (2013).
Self-organizing traffic lights: A realistic simulation.
In Advances in Applied Self-organizing Systems (pp. 45–55). Springer.

Algorithm
---------
Each intersection independently tracks two opposing lane groups (A and B).
At every step, for the currently green group and the currently red group:

    n_green  = total vehicles on the green-phase approach lanes
    n_red    = total vehicles on the red-phase approach lanes

Switch decision (applied per TLS per step):

    1. Force switch   if  steps_on_phase >= phi_max
    2. Demand switch  if  steps_on_phase >= phi_min  AND  n_red >= kappa

Otherwise hold the current phase.

Parameters (defaults calibrated to step_length=5 s, max_green=60 s)
-------------------------------------------------------------------
    phi_min  =  2 steps  (= 10 s minimum green, same as env min_green_time)
    phi_max  = 12 steps  (= 60 s maximum green, same as env max_green_time)
    kappa    =  5 vehicles  (switch threshold on red approach)

``n_red`` and ``n_green`` are derived from ``raw_vehicle_count`` in the
lane cache, which is populated by ObservationBuilder._lane_metrics() via
TraCI — the same metric used in MAPPO observation building.

Usage
-----
    python experiment/baselines/sotl.py                       # 10 episodes
    python experiment/baselines/sotl.py --episodes 5 --kappa 8 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.config import build_default_config, TrafficEnvConfig
from src.traffic_env.envs.multi_agent import MappoTrafficEnv

# ── defaults ──────────────────────────────────────────────────────────────────
SUMO_CFG = _ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"
DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 1080

# SOTL thresholds (literature defaults)
DEFAULT_PHI_MIN = 2    # env steps (= 10 s at step_length=5 s)
DEFAULT_PHI_MAX = 12   # env steps (= 60 s)
DEFAULT_KAPPA   = 5    # vehicles waiting on red approach before switch


# =============================================================================
# Policy
# =============================================================================

class SOTLPolicy:
    """
    Stateful Self-Organizing Traffic Light controller.

    Internal state per TLS:
        _current_action    — currently active signal group (0 = A, 1 = B)
        _steps_on_phase    — steps elapsed since last phase switch

    The policy requests a switch when the demand threshold is met and the
    minimum green window has closed.  The environment's own min-green
    enforcement acts as a safety net if ``phi_min < env.min_green_steps``.

    Parameters
    ----------
    lane_groups : {tls_id: (group_a_lanes, group_b_lanes)}
    phi_min     : minimum steps before a demand-triggered switch is allowed
    phi_max     : maximum steps before a forced switch occurs
    kappa       : vehicle count threshold on the red approach that triggers a switch
    """

    def __init__(
        self,
        lane_groups: Dict[str, Tuple[List[str], List[str]]],
        phi_min: int = DEFAULT_PHI_MIN,
        phi_max: int = DEFAULT_PHI_MAX,
        kappa: int   = DEFAULT_KAPPA,
    ) -> None:
        self.lane_groups = lane_groups
        self.phi_min = int(phi_min)
        self.phi_max = int(phi_max)
        self.kappa   = int(kappa)
        self._tls_ids = list(lane_groups.keys())

        # Per-TLS mutable state
        self._current_action: Dict[str, int] = {}
        self._steps_on_phase: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        """Reset all per-TLS state to phase 0, step 0."""
        self._current_action = {tid: 0 for tid in self._tls_ids}
        self._steps_on_phase = {tid: 0 for tid in self._tls_ids}

    # ------------------------------------------------------------------
    def select_actions(
        self,
        lane_cache: Dict[str, Dict[str, float]],
    ) -> Dict[str, int]:
        """
        Evaluate the SOTL rule for each TLS and return the action map.

        Parameters
        ----------
        lane_cache : output of ObservationBuilder.build_lane_cache()
            Must contain ``raw_vehicle_count`` for every monitored lane.
        """
        actions: Dict[str, int] = {}

        for tls_id in self._tls_ids:
            group_a, group_b = self.lane_groups[tls_id]
            current = self._current_action[tls_id]
            steps   = self._steps_on_phase[tls_id]

            # Identify green / red groups for this step
            green_lanes = group_a if current == 0 else group_b
            red_lanes   = group_b if current == 0 else group_a

            n_red = sum(
                float(lane_cache.get(lid, {}).get("raw_vehicle_count", 0.0))
                for lid in red_lanes
            )

            # SOTL switching logic
            if steps >= self.phi_max:
                # Force switch — maximum green exceeded
                want_switch = True
            elif steps >= self.phi_min and n_red >= self.kappa:
                # Demand-driven switch — competing approach is congested
                want_switch = True
            else:
                want_switch = False

            if want_switch:
                new_action = 1 - current   # toggle between 0 and 1
                self._current_action[tls_id] = new_action
                self._steps_on_phase[tls_id] = 0
            else:
                self._steps_on_phase[tls_id] = steps + 1

            actions[tls_id] = self._current_action[tls_id]

        return actions

    # ------------------------------------------------------------------
    @property
    def phase_summary(self) -> Dict[str, Tuple[int, int]]:
        """Return {tls_id: (current_action, steps_on_phase)} for diagnostics."""
        return {
            tid: (self._current_action[tid], self._steps_on_phase[tid])
            for tid in self._tls_ids
        }


# =============================================================================
# Episode runner
# =============================================================================

def _collect_network_metrics(
    info_dict: Dict,
    lane_cache: Dict[str, Dict[str, float]],
    tls_ids: List[str],
) -> Tuple[float, float, float, float]:
    """
    Extract one-step network-level scalars.

    Returns
    -------
    (halted_total, waiting_proxy, avg_speed_mps, throughput)
    """
    halted_total = sum(
        float(info_dict.get(tid, {}).get("queue_total_proxy", 0.0))
        for tid in tls_ids
    )
    waiting_proxy = float(np.mean([
        float(info_dict.get(tid, {}).get("avg_waiting_proxy", 0.0))
        for tid in tls_ids
    ]))
    throughput = float(np.mean([
        float(info_dict.get(tid, {}).get("current_passed", 0.0))
        for tid in tls_ids
    ]))
    speeds = [
        float(m.get("raw_speed", 0.0))
        for m in lane_cache.values()
        if m
    ]
    avg_speed_mps = float(np.mean(speeds)) if speeds else 0.0

    return halted_total, waiting_proxy, avg_speed_mps, throughput


def run_episode(
    cfg: TrafficEnvConfig,
    policy: SOTLPolicy,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    gui: bool = False,
) -> Dict[str, float]:
    """
    Run one evaluation episode with the SOTL policy.

    Returns
    -------
    dict with keys:
        total_waiting    – Σ avg_waiting_proxy over all steps
        mean_halted      – mean halted-vehicle count per step (network)
        mean_speed_mps   – mean avg lane speed (m/s) per step
        final_throughput – vehicles that exited at episode end
        episode_steps    – actual steps completed
    """
    env = MappoTrafficEnv(config=cfg, gui=gui, use_libsumo=not gui)
    obs, _ = env.reset(seed=seed)
    policy.reset()   # reset SOTL phase state for this episode

    halted_hist: List[float] = []
    waiting_hist: List[float] = []
    speed_hist: List[float] = []
    final_passed: float = 0.0

    for step in range(max_steps):
        # Build lane cache before action (current SUMO state)
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)

        # SOTL decision
        action_dict = policy.select_actions(lane_cache)

        obs, reward, terminated, truncated, info = env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())

        # Refresh cache after env step for accurate post-step metrics
        lane_cache_post = env.obs_builder.build_lane_cache(env.tls_ids)

        halted, waiting, speed, passed = _collect_network_metrics(
            info, lane_cache_post, env.tls_ids
        )
        halted_hist.append(halted)
        waiting_hist.append(waiting)
        speed_hist.append(speed)
        final_passed = passed

        if done:
            break

    env.close()

    return {
        "total_waiting":    float(np.sum(waiting_hist)),
        "mean_halted":      float(np.mean(halted_hist)) if halted_hist else 0.0,
        "mean_speed_mps":   float(np.mean(speed_hist)) if speed_hist else 0.0,
        "final_throughput": final_passed,
        "episode_steps":    len(halted_hist),
    }


# =============================================================================
# Multi-episode evaluation
# =============================================================================

def evaluate(
    n_episodes: int = DEFAULT_EPISODES,
    max_steps: int  = DEFAULT_MAX_STEPS,
    sumo_cfg: Path  = SUMO_CFG,
    base_seed: int  = 0,
    phi_min: int    = DEFAULT_PHI_MIN,
    phi_max: int    = DEFAULT_PHI_MAX,
    kappa: int      = DEFAULT_KAPPA,
    gui: bool       = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run ``n_episodes`` independent episodes, return mean ± std and 95 % CI.

    Seeds are base_seed + episode_index, ensuring reproducibility.
    """
    if not sumo_cfg.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumo_cfg}")

    cfg = build_default_config(
        sumo_cfg_path=str(sumo_cfg),
        gui=gui,
        max_steps=max_steps,
    )
    lane_groups = cfg.get_lane_groups()
    lane_groups = {tid: lane_groups[tid] for tid in cfg.tls_ids if tid in lane_groups}

    policy = SOTLPolicy(
        lane_groups=lane_groups,
        phi_min=phi_min,
        phi_max=phi_max,
        kappa=kappa,
    )

    keys = ["total_waiting", "mean_halted", "mean_speed_mps", "final_throughput"]
    records: Dict[str, List[float]] = {k: [] for k in keys}

    print(f"\n{'='*60}")
    print(f"  SOTL  — {n_episodes} episodes  "
          f"[phi_min={phi_min}, phi_max={phi_max}, kappa={kappa}]")
    print(f"{'='*60}")
    print(f"  SUMO cfg : {sumo_cfg.name}")
    print(f"  Base seed: {base_seed}")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        seed = base_seed + ep
        t0 = time.perf_counter()
        result = run_episode(cfg, policy, seed=seed, max_steps=max_steps, gui=gui)
        elapsed = time.perf_counter() - t0

        for k in keys:
            records[k].append(result[k])

        print(
            f"  ep {ep+1:>2}/{n_episodes} | seed={seed} | "
            f"steps={result['episode_steps']:>4} | "
            f"waiting={result['total_waiting']:>8.1f} | "
            f"halted={result['mean_halted']:>6.2f} | "
            f"speed={result['mean_speed_mps']:>5.2f} m/s | "
            f"passed={result['final_throughput']:>5.0f} | "
            f"wall={elapsed:.1f}s"
        )

    # ── Statistics ────────────────────────────────────────────────────────────
    n = n_episodes
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array(records[k])
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n)
        summary[k] = {
            "mean": mean, "std": std, "ci95": ci95,
            "min":  float(np.min(arr)), "max": float(np.max(arr)),
        }

    print(f"\n{'='*60}")
    print("  SUMMARY  (mean ± 95% CI)")
    print(f"{'='*60}")
    fmt = "  {:<22} {:>9.2f}  ±  {:>7.2f}   [{:.2f}, {:.2f}]"
    for k, s in summary.items():
        lo = s["mean"] - s["ci95"]
        hi = s["mean"] + s["ci95"]
        print(fmt.format(k, s["mean"], s["ci95"], lo, hi))
    print(f"{'='*60}\n")

    return summary


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOTL ATSC baseline — SUMO evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",  type=int, default=DEFAULT_EPISODES)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--phi-min",   type=int, default=DEFAULT_PHI_MIN,
                   help="Minimum env steps on green before switch allowed")
    p.add_argument("--phi-max",   type=int, default=DEFAULT_PHI_MAX,
                   help="Maximum env steps on green before forced switch")
    p.add_argument("--kappa",     type=int, default=DEFAULT_KAPPA,
                   help="Vehicle count threshold on red approach for switch")
    p.add_argument("--sumo-cfg",  type=str, default=str(SUMO_CFG))
    p.add_argument("--gui",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        sumo_cfg=Path(args.sumo_cfg),
        base_seed=args.seed,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        kappa=args.kappa,
        gui=args.gui,
    )
