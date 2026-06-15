"""
Unified comparative evaluation: MAPPO vs Max Pressure vs SOTL.

Runs each controller for ``--episodes`` independent episodes (default 10)
using deterministic seeds [base_seed, base_seed+1, …].  Reports
mean ± 95% CI for:

    - total_waiting    : Σ avg_waiting_proxy over all steps
    - mean_halted      : mean halted-vehicle count per step (network-wide)
    - mean_speed_mps   : mean avg lane speed (m/s) per step
    - final_throughput : vehicles that exited the network at episode end

DEPRECATED — proxy metrics, not paper-grade. Use experiment/runners/eval_compare.py.

Usage
-----
    python experiment/runners/_deprecated/evaluate.py
    python experiment/runners/_deprecated/evaluate.py --episodes 5 --seed 42
    python experiment/runners/_deprecated/evaluate.py --no-mappo   # skip slow policy load
    python experiment/runners/_deprecated/evaluate.py --gui        # launch SUMO GUI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project root on sys.path (file is 3 levels deep: experiment/runners/_deprecated) ──
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.config import build_default_config, TrafficEnvConfig
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from src.core.policy_loader import PolicyLoader

from experiment.baselines.max_pressure import (
    MaxPressurePolicy,
    _collect_network_metrics as _mp_metrics,
)
from experiment.baselines.sotl import (
    SOTLPolicy,
    _collect_network_metrics as _sotl_metrics,
)

# ── paths ─────────────────────────────────────────────────────────────────────
SUMO_CFG   = _ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"
MODEL_PATH = _ROOT / "models" / "mappo" / "20260418_215140" / "best_model.pt"

DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 1080

METRIC_KEYS = ["total_waiting", "mean_halted", "mean_speed_mps", "final_throughput"]


# =============================================================================
# MAPPO episode runner
# =============================================================================

def _run_mappo_episode(
    cfg: TrafficEnvConfig,
    policy: PolicyLoader,
    seed: Optional[int],
    max_steps: int,
    gui: bool,
) -> Dict[str, float]:
    env = MappoTrafficEnv(config=cfg, gui=gui, use_libsumo=not gui)
    obs, _ = env.reset(seed=seed)

    halted_hist: List[float] = []
    waiting_hist: List[float] = []
    speed_hist: List[float] = []
    final_passed: float = 0.0

    for _ in range(max_steps):
        action_dict = policy.predict(
            {tid: np.asarray(obs[tid], dtype=np.float32) for tid in cfg.tls_ids},
            deterministic=True,
        )
        obs, _, terminated, truncated, info = env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())

        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
        halted, waiting, speed, passed = _mp_metrics(
            info, lane_cache, env.tls_ids, speed_cap=cfg.observation.speed_cap
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


def _run_mp_episode(
    cfg: TrafficEnvConfig,
    policy: MaxPressurePolicy,
    seed: Optional[int],
    max_steps: int,
    gui: bool,
) -> Dict[str, float]:
    env = MappoTrafficEnv(config=cfg, gui=gui, use_libsumo=not gui)
    obs, _ = env.reset(seed=seed)

    halted_hist: List[float] = []
    waiting_hist: List[float] = []
    speed_hist: List[float] = []
    final_passed: float = 0.0

    for _ in range(max_steps):
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
        action_dict = policy.select_actions(lane_cache)
        obs, _, terminated, truncated, info = env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())

        lane_cache_post = env.obs_builder.build_lane_cache(env.tls_ids)
        halted, waiting, speed, passed = _mp_metrics(
            info, lane_cache_post, env.tls_ids, speed_cap=cfg.observation.speed_cap
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


def _run_sotl_episode(
    cfg: TrafficEnvConfig,
    policy: SOTLPolicy,
    seed: Optional[int],
    max_steps: int,
    gui: bool,
) -> Dict[str, float]:
    env = MappoTrafficEnv(config=cfg, gui=gui, use_libsumo=not gui)
    obs, _ = env.reset(seed=seed)
    policy.reset()

    halted_hist: List[float] = []
    waiting_hist: List[float] = []
    speed_hist: List[float] = []
    final_passed: float = 0.0

    for _ in range(max_steps):
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
        action_dict = policy.select_actions(lane_cache)
        obs, _, terminated, truncated, info = env.step(action_dict)
        done = any(terminated.values()) or any(truncated.values())

        lane_cache_post = env.obs_builder.build_lane_cache(env.tls_ids)
        halted, waiting, speed, passed = _sotl_metrics(
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
# Statistics helpers
# =============================================================================

def _compute_stats(records: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    n = len(next(iter(records.values())))
    summary: Dict[str, Dict[str, float]] = {}
    for k, vals in records.items():
        arr = np.array(vals)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n)
        summary[k] = {
            "mean": mean, "std": std, "ci95": ci95,
            "min":  float(np.min(arr)), "max": float(np.max(arr)),
        }
    return summary


def _run_controller(
    name: str,
    run_fn,
    cfg: TrafficEnvConfig,
    policy,
    n_episodes: int,
    max_steps: int,
    base_seed: int,
    gui: bool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
    """Run ``n_episodes`` and return (summary_stats, raw_records)."""
    records: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {name}  —  {n_episodes} episodes")
    print(bar)

    for ep in range(n_episodes):
        seed = base_seed + ep
        t0 = time.perf_counter()
        result = run_fn(cfg, policy, seed, max_steps, gui)
        elapsed = time.perf_counter() - t0

        for k in METRIC_KEYS:
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

    summary = _compute_stats(records)
    return summary, records


# =============================================================================
# Comparison table printer
# =============================================================================

def _print_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    n_episodes: int,
) -> None:
    controllers = list(results.keys())
    col_w = 22

    metric_labels = {
        "total_waiting":    "Total waiting (s)",
        "mean_halted":      "Mean halted (veh)",
        "mean_speed_mps":   "Mean speed (m/s)",
        "final_throughput": "Throughput (veh)",
    }

    sep = "=" * (32 + col_w * len(controllers))
    print(f"\n\n{sep}")
    print(f"  COMPARATIVE RESULTS  (n={n_episodes}, mean ± 95% CI)")
    print(sep)

    # Header
    header = f"  {'Metric':<30}"
    for c in controllers:
        header += f"  {c:^{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for key, label in metric_labels.items():
        row = f"  {label:<30}"
        for c in controllers:
            s = results[c][key]
            cell = f"{s['mean']:.2f} ± {s['ci95']:.2f}"
            row += f"  {cell:^{col_w}}"
        print(row)

    print(sep)

    # Improvement vs. the WORST controller per metric (lower is better for waiting/halted)
    if len(controllers) >= 2 and "MAPPO" in results:
        print("\n  MAPPO improvement over each baseline (primary metrics):\n")
        for key, label in metric_labels.items():
            mappo_mean = results["MAPPO"][key]["mean"]
            direction  = -1 if key in ("total_waiting", "mean_halted") else 1
            for c in controllers:
                if c == "MAPPO":
                    continue
                base_mean = results[c][key]["mean"]
                if base_mean != 0:
                    rel = direction * (mappo_mean - base_mean) / abs(base_mean) * 100
                    sign = "+" if rel >= 0 else ""
                    print(f"    {label:<30}  vs {c:<14}  {sign}{rel:.1f}%")
        print()

    print(sep + "\n")


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate(
    n_episodes: int = DEFAULT_EPISODES,
    max_steps: int  = DEFAULT_MAX_STEPS,
    sumo_cfg: Path  = SUMO_CFG,
    model_path: Path = MODEL_PATH,
    base_seed: int  = 0,
    run_mappo: bool = True,
    run_mp: bool    = True,
    run_sotl: bool  = True,
    phi_min: int    = 2,
    phi_max: int    = 12,
    kappa: int      = 5,
    gui: bool       = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:

    if not sumo_cfg.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumo_cfg}")

    cfg = build_default_config(
        sumo_cfg_path=str(sumo_cfg),
        gui=gui,
        max_steps=max_steps,
    )

    lane_groups_all = cfg.get_lane_groups()
    lane_groups = {tid: lane_groups_all[tid] for tid in cfg.tls_ids if tid in lane_groups_all}

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_records: Dict[str, Dict[str, List[float]]] = {}

    # ── MAPPO ─────────────────────────────────────────────────────────────────
    if run_mappo:
        if not model_path.exists():
            print(f"[WARN] Model checkpoint not found: {model_path}  — skipping MAPPO.")
        else:
            mappo_policy = PolicyLoader(
                checkpoint_path=model_path,
                obs_dim=cfg.local_obs_dim,
                action_dim=2,
            ).load()
            summary, records = _run_controller(
                "MAPPO", _run_mappo_episode, cfg, mappo_policy,
                n_episodes, max_steps, base_seed, gui,
            )
            all_results["MAPPO"] = summary
            all_records["MAPPO"] = records

    # ── Max Pressure ──────────────────────────────────────────────────────────
    if run_mp:
        mp_policy = MaxPressurePolicy(lane_groups=lane_groups)
        summary, records = _run_controller(
            "Max Pressure", _run_mp_episode, cfg, mp_policy,
            n_episodes, max_steps, base_seed, gui,
        )
        all_results["Max Pressure"] = summary
        all_records["Max Pressure"] = records

    # ── SOTL ──────────────────────────────────────────────────────────────────
    if run_sotl:
        sotl_policy = SOTLPolicy(
            lane_groups=lane_groups,
            phi_min=phi_min,
            phi_max=phi_max,
            kappa=kappa,
        )
        summary, records = _run_controller(
            "SOTL", _run_sotl_episode, cfg, sotl_policy,
            n_episodes, max_steps, base_seed, gui,
        )
        all_results["SOTL"] = summary
        all_records["SOTL"] = records

    _print_comparison(all_results, n_episodes)
    return all_results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified evaluation: MAPPO vs Max Pressure vs SOTL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",   type=int,  default=DEFAULT_EPISODES)
    p.add_argument("--max-steps",  type=int,  default=DEFAULT_MAX_STEPS)
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--sumo-cfg",   type=str,  default=str(SUMO_CFG))
    p.add_argument("--model",      type=str,  default=str(MODEL_PATH),
                   help="Path to MAPPO checkpoint (.pt)")
    p.add_argument("--no-mappo",   action="store_true", help="Skip MAPPO evaluation")
    p.add_argument("--no-mp",      action="store_true", help="Skip Max Pressure evaluation")
    p.add_argument("--no-sotl",    action="store_true", help="Skip SOTL evaluation")
    p.add_argument("--phi-min",    type=int,  default=2)
    p.add_argument("--phi-max",    type=int,  default=12)
    p.add_argument("--kappa",      type=int,  default=5)
    p.add_argument("--gui",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    print(
        "\n[DEPRECATED] evaluate.py reports PROXY metrics (queue-based 'waiting', "
        "per-step 'throughput') and is NOT paper-grade.\n"
        "             For tripinfo-based, multi-seed, statistically-tested tables "
        "use:  experiment/runners/eval_compare.py\n"
    )
    args = _parse_args()
    evaluate(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        sumo_cfg=Path(args.sumo_cfg),
        model_path=Path(args.model),
        base_seed=args.seed,
        run_mappo=not args.no_mappo,
        run_mp=not args.no_mp,
        run_sotl=not args.no_sotl,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        kappa=args.kappa,
        gui=args.gui,
    )
