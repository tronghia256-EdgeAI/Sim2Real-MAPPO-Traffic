"""
Max Pressure (MP) traffic signal controller — SUMO baseline.

Reference
---------
Varaiya, P. (2013). Max pressure control of a network of signalized
intersections. Transportation Research Part C, 36, 177–195.

Algorithm
---------
At every decision step, for each intersection independently:

    Pressure(group) = Σ effective_queue_norm(lane_i)  for lane_i in group

    action = 0  if  Pressure(group_A) >= Pressure(group_B)
    action = 1  otherwise

``effective_queue_norm`` is the ratio of halted vehicle length to lane
length (∈ [0, 1]), computed by ObservationBuilder._lane_metrics() via
TraCI exactly as used during MAPPO training.  Using the same metric
ensures a fair, apples-to-apples comparison.

The environment (MappoTrafficEnv) enforces its own minimum green time
constraint, so Max Pressure simply issues a pressure-optimal request
every step; the env silently holds the current phase when the min-green
window has not yet elapsed.

Usage
-----
    python experiment/baselines/max_pressure.py              # 10 episodes
    python experiment/baselines/max_pressure.py --episodes 5 --seed 0
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
DEFAULT_MAX_STEPS = 1080  # 1 episode = 5 400 s simulation time


# =============================================================================
# Policy
# =============================================================================

class MaxPressurePolicy:
    """
    Stateless Max Pressure controller.

    At each call to ``select_actions`` the policy queries the current
    lane cache and picks the signal group with higher total pressure.
    No internal state is needed — all memory lives in the SUMO simulation.

    Parameters
    ----------
    lane_groups : {tls_id: (group_a_lanes, group_b_lanes)}
        Opposing lane groups per intersection.  Must match the groups
        used to assemble the policy's observation vector (same convention
        as ``DEFAULT_MANUAL_LANE_GROUPS`` in config.py).
    """

    def __init__(
        self,
        lane_groups: Dict[str, Tuple[List[str], List[str]]],
    ) -> None:
        self.lane_groups = lane_groups

    def select_actions(
        self,
        lane_cache: Dict[str, Dict[str, float]],
    ) -> Dict[str, int]:
        """
        Return {tls_id: action} based on current lane pressures.

        action = 0 → group_A gets green (env maps this to green phase index 0)
        action = 1 → group_B gets green (env maps this to green phase index 1)

        Parameters
        ----------
        lane_cache : output of ObservationBuilder.build_lane_cache()
            Keys are lane IDs; values contain at least ``effective_queue_norm``.
        """
        actions: Dict[str, int] = {}
        for tls_id, (group_a, group_b) in self.lane_groups.items():
            pressure_a = sum(
                float(lane_cache.get(lid, {}).get("effective_queue_norm", 0.0))
                for lid in group_a
            )
            pressure_b = sum(
                float(lane_cache.get(lid, {}).get("effective_queue_norm", 0.0))
                for lid in group_b
            )
            actions[tls_id] = 0 if pressure_a >= pressure_b else 1
        return actions


# =============================================================================
# Episode runner
# =============================================================================

def _collect_network_metrics(
    info_dict: Dict,
    lane_cache: Dict[str, Dict[str, float]],
    tls_ids: List[str],
    speed_cap: float,
) -> Tuple[float, float, float, float]:
    """
    Extract one-step network-level scalars.

    Returns
    -------
    (halted_total, waiting_proxy, avg_speed_mps, throughput_delta)
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

    # Average raw speed across all monitored lanes (m/s)
    speeds = [
        float(m.get("raw_speed", 0.0))
        for m in lane_cache.values()
        if m
    ]
    avg_speed_mps = float(np.mean(speeds)) if speeds else 0.0

    return halted_total, waiting_proxy, avg_speed_mps, throughput


def run_episode(
    cfg: TrafficEnvConfig,
    policy: MaxPressurePolicy,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    gui: bool = False,
) -> Dict[str, float]:
    """
    Run one evaluation episode and return aggregate metrics.

    Returns
    -------
    dict with keys:
        total_waiting    – Σ avg_waiting_proxy over all steps × tls
        mean_halted      – mean halted-vehicle count per step (network)
        mean_speed_mps   – mean avg lane speed (m/s) per step
        final_throughput – total vehicles that exited the network
        episode_steps    – actual steps completed
    """
    env = MappoTrafficEnv(config=cfg, gui=gui, use_libsumo=not gui)
    obs, _ = env.reset(seed=seed)

    halted_hist: List[float] = []
    waiting_hist: List[float] = []
    speed_hist: List[float] = []
    final_passed: float = 0.0

    for step in range(max_steps):
        # Build lane cache for pressure computation
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)

        # Max Pressure decision
        action_dict = policy.select_actions(lane_cache)

        obs, reward, terminated, truncated, info = env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())

        # Refresh cache after step (SUMO has advanced)
        lane_cache_post = env.obs_builder.build_lane_cache(env.tls_ids)

        halted, waiting, speed, passed = _collect_network_metrics(
            info, lane_cache_post, env.tls_ids,
            speed_cap=cfg.observation.speed_cap,
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
    max_steps: int = DEFAULT_MAX_STEPS,
    sumo_cfg: Path = SUMO_CFG,
    base_seed: int = 0,
    gui: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run ``n_episodes`` independent episodes, return mean ± std and 95 % CI.

    Seeds are deterministic: episode i uses seed ``base_seed + i``,
    ensuring reproducibility and independence across episodes.
    """
    if not sumo_cfg.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumo_cfg}")

    cfg = build_default_config(
        sumo_cfg_path=str(sumo_cfg),
        gui=gui,
        max_steps=max_steps,
    )
    lane_groups = cfg.get_lane_groups()
    # Filter to only TLS IDs the env will manage
    lane_groups = {tid: lane_groups[tid] for tid in cfg.tls_ids if tid in lane_groups}
    policy = MaxPressurePolicy(lane_groups=lane_groups)

    keys = ["total_waiting", "mean_halted", "mean_speed_mps", "final_throughput"]
    records: Dict[str, List[float]] = {k: [] for k in keys}

    print(f"\n{'='*55}")
    print(f"  MAX PRESSURE  — {n_episodes} evaluation episodes")
    print(f"{'='*55}")
    print(f"  SUMO cfg : {sumo_cfg.name}")
    print(f"  Base seed: {base_seed}")
    print(f"{'='*55}\n")

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
    # 95 % CI: mean ± 1.96 * (std / sqrt(n))
    n = n_episodes
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array(records[k])
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n)
        summary[k] = {"mean": mean, "std": std, "ci95": ci95,
                      "min": float(np.min(arr)), "max": float(np.max(arr))}

    print(f"\n{'='*55}")
    print("  SUMMARY  (mean ± 95% CI)")
    print(f"{'='*55}")
    fmt = "  {:<22} {:>9.2f}  ±  {:>7.2f}   [{:.2f}, {:.2f}]"
    for k, s in summary.items():
        lo = s["mean"] - s["ci95"]
        hi = s["mean"] + s["ci95"]
        print(fmt.format(k, s["mean"], s["ci95"], lo, hi))
    print(f"{'='*55}\n")

    return summary


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Max Pressure ATSC baseline — SUMO evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",  type=int,   default=DEFAULT_EPISODES)
    p.add_argument("--max-steps", type=int,   default=DEFAULT_MAX_STEPS)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--sumo-cfg",  type=str,   default=str(SUMO_CFG))
    p.add_argument("--gui",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        sumo_cfg=Path(args.sumo_cfg),
        base_seed=args.seed,
        gui=args.gui,
    )
