from __future__ import annotations

"""
eval_compare.py
===============
Unified, tripinfo-based, multi-seed comparison harness — the single source of
the paper's main result tables (TABLE-4 / TABLE-5).

Why this exists
---------------
The previous comparison paths (experiment/runners/_deprecated/evaluate.py,
_deprecated/compare_traffic_metrics.py) reported PROXY metrics (mean-queue × step_length as
"waiting", a broken per-step getArrivedNumber as "throughput") and defaulted to
the legacy checkpoint. This harness instead:

  * drives EVERY method through SUMO with --tripinfo-output and reads the SAME
    rigorous metrics for all of them (mean travel time, mean waiting, time loss,
    P95 waiting, served trips) via experiment.common.tripinfo;
  * runs the env-path methods (MAPPO / IPPO / privileged / max-pressure / SOTL /
    Webster / fixed-time) through MappoTrafficEnv so step semantics, min/max-green
    and yellow handling are identical, and the SUMO-native actuated baseline
    through its own program — same scenario, same seeds;
  * aggregates over N evaluation seeds (mean ± 95% CI) and runs the paper's
    statistical protocol vs a reference method (Mann-Whitney U, Holm-Bonferroni
    across the method family, Cohen's d).

All learned-policy metadata (obs_mode, tls roster) is resolved from the
checkpoint's sibling run_config.json so a privileged (38-dim) checkpoint is
evaluated with the matching observation space automatically.

Usage (from project root)
-------------------------
    # scaled network, full method set, 5 seeds
    python experiment/runners/eval_compare.py \
        --network n3_grid \
        --checkpoint models/mappo/<run_id>/best_model.pt \
        --methods mappo webster actuated maxpressure sotl fixed \
        --seeds 42 123 456 789 1337

    # add IPPO and the privileged upper bound (separate checkpoints)
    python experiment/runners/eval_compare.py --network n2_corridor \
        --checkpoint           models/mappo/<mappo_run>/best_model.pt \
        --ippo-checkpoint      models/mappo/<ippo_run>/best_model.pt \
        --privileged-checkpoint models/mappo/<priv_run>/best_model.pt \
        --methods mappo ippo privileged webster actuated maxpressure sotl fixed

Outputs (results/paper1_mappo/eval_tables/<network>_<ts>/):
    summary.json   full per-method per-seed values + aggregates + stats
    table.csv      one row per (method, metric): mean, ci95, p_adj, cohens_d
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.config import (
    DEFAULT_MANUAL_LANE_GROUPS,
    build_default_config,
    load_lane_groups_json,
)
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from src.core.policy_loader import PolicyLoader
from experiment.common.tripinfo import parse_tripinfo_means
from experiment.baselines.max_pressure import MaxPressurePolicy
from experiment.baselines.sotl import SOTLPolicy
from experiment.baselines.webster import WebsterPolicy, measure_lane_flows, DEFAULT_PCU
from experiment.baselines import actuated as actuated_mod

# tripinfo metrics, with optimisation direction (lower-is-better except served).
# CO2/fuel are post-hoc (never in the reward); NaN unless --emissions is set.
METRIC_KEYS: Tuple[str, ...] = (
    "mean_travel_time_s",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "p95_waiting_time_s",
    "n_trips",
    "total_co2_mg",
    "total_fuel_mg",
)
LOWER_IS_BETTER: Dict[str, bool] = {
    "mean_travel_time_s": True,
    "mean_waiting_time_s": True,
    "mean_time_loss_s": True,
    "p95_waiting_time_s": True,
    "n_trips": False,  # served trips: higher is better
    "total_co2_mg": True,
    "total_fuel_mg": True,
}
METRIC_LABELS: Dict[str, str] = {
    "mean_travel_time_s": "Travel time (s)",
    "mean_waiting_time_s": "Waiting (s)",
    "mean_time_loss_s": "Time loss (s)",
    "p95_waiting_time_s": "P95 waiting (s)",
    "n_trips": "Served trips",
    "total_co2_mg": "CO2 (mg/ep)",
    "total_fuel_mg": "Fuel (mg/ep)",
}

# SUMO flag enabling the emissions device for every vehicle (post-hoc CO2/fuel)
EMISSIONS_FLAGS = ["--device.emissions.probability", "1"]

LEARNED_METHODS = {"mappo", "ippo", "privileged"}


# ---------------------------------------------------------------------------
# Scenario resolution
# ---------------------------------------------------------------------------

class Scenario:
    """Resolved file paths + lane-group topology for one evaluation scenario."""

    def __init__(
        self,
        sumo_cfg: Path,
        net_path: Path,
        vtypes_path: Path,
        tls_ids: Tuple[str, ...],
        lane_groups: Dict[str, Tuple[List[str], List[str]]],
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.net_path = net_path
        self.vtypes_path = vtypes_path
        self.tls_ids = tls_ids
        self.lane_groups = lane_groups


def resolve_scenario(args: argparse.Namespace) -> Scenario:
    if args.network:
        base = _ROOT / "sumo_configs" / "networks" / args.network
        cfg_name = "sumo_config_eval.sumocfg" if args.split == "eval" else "sumo_config.sumocfg"
        sumo_cfg = base / cfg_name
        groups_path = base / "lane_groups.json"
        net_path = base / "intersections.net.xml"
        vtypes_path = base / "vtypes.add.xml"
        if not sumo_cfg.exists():
            raise SystemExit(f"[scenario] missing {sumo_cfg}")
        tls_ids, lane_groups = load_lane_groups_json(str(groups_path))
        return Scenario(sumo_cfg, net_path, vtypes_path, tls_ids, lane_groups)

    if not args.sumo_cfg:
        raise SystemExit("either --network or --sumo-cfg is required")
    sumo_cfg = Path(args.sumo_cfg)
    net_path = sumo_cfg.parent / "intersections.net.xml"
    vtypes_path = sumo_cfg.parent / "vtypes.add.xml"
    if args.lane_groups:
        tls_ids, lane_groups = load_lane_groups_json(str(args.lane_groups))
    else:
        lane_groups = {t: (list(a), list(b)) for t, (a, b) in DEFAULT_MANUAL_LANE_GROUPS.items()}
        tls_ids = tuple(lane_groups)
    return Scenario(sumo_cfg, net_path, vtypes_path, tls_ids, lane_groups)


# ---------------------------------------------------------------------------
# Generic env-driven episode (tripinfo)
# ---------------------------------------------------------------------------

ActionFn = Callable[[MappoTrafficEnv, Dict[str, np.ndarray], int, float], Dict[str, int]]


def _build_env_cfg(scenario: Scenario, args: argparse.Namespace, obs_mode: str,
                   obs_overrides: Optional[Dict[str, Any]] = None):
    kw: Dict[str, Any] = dict(obs_overrides or {})
    return build_default_config(
        sumo_cfg_path=str(scenario.sumo_cfg),
        gui=False,
        tls_ids=scenario.tls_ids,
        manual_lane_groups=scenario.lane_groups,
        step_length=args.step_length,
        max_steps=args.max_steps,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        obs_mode=obs_mode,
        **kw,
    )


def run_env_episode(
    env_cfg,
    make_action: ActionFn,
    seed: int,
    max_steps: int,
    tripinfo_path: Path,
    on_reset: Optional[Callable[[], None]] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Drive MappoTrafficEnv with an arbitrary action function; return tripinfo means."""
    env = MappoTrafficEnv(
        config=env_cfg,
        gui=False,
        use_libsumo=False,  # traci required for tripinfo flush + parity with baselines
        extra_sumo_args=[
            "--tripinfo-output", str(tripinfo_path),
        ] + list(extra_args or []),
    )
    obs_dict, _ = env.reset(seed=seed)
    if on_reset is not None:
        on_reset()
    step_len = float(env_cfg.sim.step_length)
    try:
        for step in range(max_steps):
            action = make_action(env, obs_dict, step, step_len)
            obs_dict, _, terminated, truncated, _ = env.step(action)
            if any(terminated.values()) or any(truncated.values()):
                break
    finally:
        env.close()  # flushes tripinfo
    return parse_tripinfo_means(tripinfo_path)


# ---------------------------------------------------------------------------
# Per-method action providers
# ---------------------------------------------------------------------------

def _load_run_config(checkpoint: Path) -> dict:
    cfg_path = checkpoint.parent / "run_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _obs_mode_for_checkpoint(checkpoint: Path) -> str:
    rc = _load_run_config(checkpoint)
    mode = rc.get("env_cfg", {}).get("observation", {}).get("obs_mode")
    return mode if mode in ("proxy", "privileged") else "proxy"


def _obs_overrides_for_checkpoint(checkpoint: Path) -> Dict[str, Any]:
    """Rebuild the EXACT obs space a checkpoint was trained with (incl. VI-F
    ablations) from its run_config.json, so the actor's input dim matches.
    """
    rc = _load_run_config(checkpoint)
    obs = rc.get("env_cfg", {}).get("observation", {})
    out: Dict[str, Any] = {"lane_truncated": bool(obs.get("lane_truncated", False))}
    if obs.get("lane_feature_names"):
        out["lane_feature_names"] = tuple(obs["lane_feature_names"])
    if obs.get("tls_feature_names"):
        out["tls_feature_names"] = tuple(obs["tls_feature_names"])
    return out


def resolve_checkpoints(value) -> List[Path]:
    """Resolve --checkpoint into a concrete list of best_model.pt files.

    Accepts a single .pt file, a directory (globbed for **/best_model.pt — the
    per-seed campaign layout <campaign>/models/<job>/<run_id>/best_model.pt), a
    glob pattern string, or a list of any of these. Order-preserving + dedup.
    """
    if value is None:
        return []
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    found: List[Path] = []
    for it in items:
        p = Path(it)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.glob("**/best_model.pt")))
        else:  # treat as a glob pattern
            found.extend(sorted(Path().glob(str(it))))
    seen, out = set(), []
    for p in found:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def _mean_metrics(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average tripinfo metrics over a learned checkpoint's eval seeds -> 1 sample."""
    out: Dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = np.array([float(m.get(key, np.nan)) for m in metric_dicts], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = float(np.mean(vals)) if vals.size else float("nan")
    return out


def eval_learned(
    method: str,
    checkpoints: Sequence[Path],
    scenario: Scenario,
    args: argparse.Namespace,
    seeds: Sequence[int],
    tmp: Path,
) -> List[Dict[str, float]]:
    """Evaluate every per-seed checkpoint; sample = one checkpoint's mean over
    the eval route-seeds, so n_samples == number of training seeds (the policy
    variance source). With a single checkpoint this degrades to a point estimate.
    """
    samples: List[Dict[str, float]] = []
    for ci, ckpt in enumerate(checkpoints):
        obs_mode = _obs_mode_for_checkpoint(ckpt)
        env_cfg = _build_env_cfg(scenario, args, obs_mode,
                                 obs_overrides=_obs_overrides_for_checkpoint(ckpt))
        obs_dim = env_cfg.local_obs_dim
        policy = PolicyLoader(
            checkpoint_path=ckpt, obs_dim=obs_dim, action_dim=2,
        ).load()

        def make_action(env, obs_dict, step, step_len, _p=policy):
            return _p.predict(
                {t: np.asarray(obs_dict[t], dtype=np.float32) for t in env.tls_ids},
                deterministic=True,
            )

        emi = EMISSIONS_FLAGS if getattr(args, "emissions", False) else []
        seed_metrics: List[Dict[str, float]] = []
        for seed in seeds:
            tri = tmp / f"{method}_c{ci}_{seed}.xml"
            m = run_env_episode(env_cfg, make_action, seed, args.max_steps, tri, extra_args=emi)
            seed_metrics.append(m)
            _log_seed(f"{method}[{ci}]", seed, m)
        samples.append(_mean_metrics(seed_metrics))
    return samples


def eval_maxpressure(scenario, args, seeds, tmp) -> List[Dict[str, float]]:
    env_cfg = _build_env_cfg(scenario, args, obs_mode="proxy")
    policy = MaxPressurePolicy(lane_groups=scenario.lane_groups)

    def make_action(env, obs_dict, step, step_len):
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
        return policy.select_actions(lane_cache)

    return _run_seeds("maxpressure", env_cfg, make_action, args, seeds, tmp)


def eval_sotl(scenario, args, seeds, tmp) -> List[Dict[str, float]]:
    env_cfg = _build_env_cfg(scenario, args, obs_mode="proxy")
    policy = SOTLPolicy(
        lane_groups=scenario.lane_groups,
        phi_min=args.phi_min, phi_max=args.phi_max, kappa=args.kappa,
    )

    def make_action(env, obs_dict, step, step_len):
        lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
        return policy.select_actions(lane_cache)

    return _run_seeds("sotl", env_cfg, make_action, args, seeds, tmp,
                      on_reset=policy.reset)


def eval_fixed(scenario, args, seeds, tmp) -> List[Dict[str, float]]:
    env_cfg = _build_env_cfg(scenario, args, obs_mode="proxy")
    green_steps = max(1, round(args.fixed_green_s / env_cfg.sim.step_length))

    def make_action(env, obs_dict, step, step_len):
        phase = (step // green_steps) % 2
        return {t: phase for t in env.tls_ids}

    return _run_seeds("fixed", env_cfg, make_action, args, seeds, tmp)


def eval_webster(scenario, args, seeds, tmp) -> List[Dict[str, float]]:
    """Per seed: measure historical flows (static program) then drive WebsterPolicy."""
    env_cfg = _build_env_cfg(scenario, args, obs_mode="proxy")
    all_lanes = [l for ga, gb in scenario.lane_groups.values() for l in list(ga) + list(gb)]
    measure_seconds = args.max_steps * env_cfg.sim.step_length

    results = []
    for seed in seeds:
        flows = measure_lane_flows(
            scenario.sumo_cfg, all_lanes, measure_seconds, seed, DEFAULT_PCU,
        )
        policy = WebsterPolicy(
            lane_groups=scenario.lane_groups, lane_flows_pcu_h=flows,
            sat_flow_pcu_h=args.sat_flow,
        )

        def make_action(env, obs_dict, step, step_len, _p=policy):
            return _p.select_actions(sim_time_s=step * step_len)

        emi = EMISSIONS_FLAGS if getattr(args, "emissions", False) else []
        tri = tmp / f"webster_{seed}.xml"
        m = run_env_episode(env_cfg, make_action, seed, args.max_steps, tri, extra_args=emi)
        results.append(m)
        _log_seed("webster", seed, m)
    return results


def eval_actuated(scenario, args, seeds, tmp) -> List[Dict[str, float]]:
    """SUMO-native gap-based actuated; runs its own program (not via the env)."""
    add_path = scenario.net_path.parent / "tls_actuated.add.xml"
    tls_ids = actuated_mod.write_actuated_add(scenario.net_path, add_path)
    duration = args.max_steps * args.step_length
    emi = EMISSIONS_FLAGS if getattr(args, "emissions", False) else None

    results = []
    for seed in seeds:
        tri = tmp / f"actuated_{seed}.xml"
        actuated_mod.run_actuated(
            scenario.sumo_cfg, add_path, scenario.vtypes_path,
            tls_ids, seed, duration, tri, extra_args=emi,
        )
        m = parse_tripinfo_means(tri)
        results.append(m)
        _log_seed("actuated", seed, m)
    return results


def _run_seeds(method, env_cfg, make_action, args, seeds, tmp, on_reset=None):
    emi = EMISSIONS_FLAGS if getattr(args, "emissions", False) else []
    results = []
    for seed in seeds:
        tri = tmp / f"{method}_{seed}.xml"
        m = run_env_episode(env_cfg, make_action, seed, args.max_steps, tri,
                            on_reset=on_reset, extra_args=emi)
        results.append(m)
        _log_seed(method, seed, m)
    return results


def _log_seed(method: str, seed: int, m: Dict[str, float]) -> None:
    print(
        f"  {method:<12} seed={seed:<5} | travel={m['mean_travel_time_s']:7.1f}s "
        f"wait={m['mean_waiting_time_s']:6.1f}s p95={m['p95_waiting_time_s']:6.1f}s "
        f"trips={m['n_trips']:.0f}"
    )


# ---------------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------------

def aggregate(per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    n = len(per_seed)
    for key in METRIC_KEYS:
        vals = np.array([float(r.get(key, np.nan)) for r in per_seed], dtype=float)
        finite = vals[np.isfinite(vals)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(finite.size) if finite.size > 1 else 0.0
        out[key] = {"mean": mean, "std": std, "ci95": ci95,
                    "n": int(finite.size), "values": vals.tolist()}
    return out


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp2 = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    sp = float(np.sqrt(sp2))
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def mann_whitney_p(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 1 or b.size < 1:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values across a family of comparisons."""
    valid = {k: v for k, v in pvals.items() if np.isfinite(v)}
    if not valid:
        return {k: float("nan") for k in pvals}
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min((m - i) * p, 1.0))
        adj[name] = running
    for k in pvals:
        adj.setdefault(k, float("nan"))
    return adj


def pct_improvement(method_mean: float, ref_mean: float, lower_is_better: bool) -> float:
    if not np.isfinite(method_mean) or not np.isfinite(ref_mean) or ref_mean == 0:
        return float("nan")
    direction = -1.0 if lower_is_better else 1.0
    return direction * (method_mean - ref_mean) / abs(ref_mean) * 100.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    aggregates: Dict[str, Dict[str, Dict[str, float]]],
    reference: str,
) -> dict:
    """aggregates: {method: {metric: {mean, ci95, values, ...}}}."""
    methods = list(aggregates)
    stats: Dict[str, Dict[str, dict]] = {}

    for metric in METRIC_KEYS:
        lib = LOWER_IS_BETTER[metric]
        ref_vals = aggregates.get(reference, {}).get(metric, {}).get("values", [])
        raw_p: Dict[str, float] = {}
        per_method: Dict[str, dict] = {}
        for method in methods:
            if method == reference:
                continue
            mv = aggregates[method][metric]["values"]
            p = mann_whitney_p(ref_vals, mv)
            raw_p[method] = p
            per_method[method] = {
                "p_raw": p,
                "cohens_d": cohens_d(ref_vals, mv),
                "pct_improvement_ref_vs_method": pct_improvement(
                    aggregates[reference][metric]["mean"],
                    aggregates[method][metric]["mean"], lib,
                ),
            }
        adj = holm_bonferroni(raw_p)
        for method in per_method:
            per_method[method]["p_holm"] = adj.get(method, float("nan"))
        stats[metric] = per_method

    return {"reference": reference, "stats": stats}


def print_table(aggregates, report) -> None:
    methods = list(aggregates)
    ref = report["reference"]
    col_w = 20
    print("\n" + "=" * (24 + col_w * len(methods)))
    print(f"  COMPARISON (mean ± 95% CI; reference = {ref}; † p_holm<0.05 vs {ref})")
    print("=" * (24 + col_w * len(methods)))
    header = f"  {'Metric':<20}"
    for mth in methods:
        header += f"{mth:>{col_w}}"
    print(header)
    print("-" * len(header))
    for metric in METRIC_KEYS:
        row = f"  {METRIC_LABELS[metric]:<20}"
        for mth in methods:
            a = aggregates[mth][metric]
            cell = f"{a['mean']:.1f}±{a['ci95']:.1f}"
            if mth != ref:
                p = report["stats"][metric].get(mth, {}).get("p_holm", float("nan"))
                if np.isfinite(p) and p < 0.05:
                    cell += "†"
            row += f"{cell:>{col_w}}"
        print(row)
    print("=" * (24 + col_w * len(methods)) + "\n")


def write_outputs(out_dir: Path, scenario_name: str, aggregates, report, args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": scenario_name,
        "split": args.split,
        "seeds": list(args.seeds),
        "max_steps": args.max_steps,
        "min_green_time": args.min_green_time,
        "max_green_time": args.max_green_time,
        "reference": report["reference"],
        "aggregates": aggregates,
        "stats": report["stats"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import csv
    ref = report["reference"]
    with open(out_dir / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "ci95", "n_seeds",
                    "p_holm_vs_ref", "cohens_d", "pct_improvement_vs_ref"])
        for method in aggregates:
            for metric in METRIC_KEYS:
                a = aggregates[method][metric]
                st = report["stats"][metric].get(method, {}) if method != ref else {}
                w.writerow([
                    method, metric,
                    f"{a['mean']:.4f}", f"{a['ci95']:.4f}", a["n"],
                    f"{st.get('p_holm', float('nan')):.4g}" if method != ref else "",
                    f"{st.get('cohens_d', float('nan')):.4g}" if method != ref else "",
                    f"{st.get('pct_improvement_ref_vs_method', float('nan')):.2f}" if method != ref else "",
                ])
    print(f"[out] wrote {out_dir/'summary.json'} and {out_dir/'table.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    scenario = resolve_scenario(args)
    seeds = list(args.seeds)
    scenario_name = args.network or Path(args.sumo_cfg).parent.name

    print(f"\nScenario: {scenario_name} ({args.split}) | {len(scenario.tls_ids)} TLS | "
          f"{len(seeds)} seeds | methods: {', '.join(args.methods)}")

    checkpoints = {
        "mappo": resolve_checkpoints(args.checkpoint),
        "ippo": resolve_checkpoints(args.ippo_checkpoint),
        "privileged": resolve_checkpoints(args.privileged_checkpoint),
    }

    tmp = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.gettempdir()) / "eval_compare"
    tmp.mkdir(parents=True, exist_ok=True)

    aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in args.methods:
        t0 = time.time()
        print(f"\n--- {method} ---")
        try:
            if method in LEARNED_METHODS:
                ckpts = checkpoints.get(method, [])
                if not ckpts:
                    print(f"  [skip] {method}: no checkpoint(s) found")
                    continue
                print(f"  {len(ckpts)} checkpoint(s) x {len(seeds)} eval seeds")
                per_seed = eval_learned(method, ckpts, scenario, args, seeds, tmp)
            elif method == "maxpressure":
                per_seed = eval_maxpressure(scenario, args, seeds, tmp)
            elif method == "sotl":
                per_seed = eval_sotl(scenario, args, seeds, tmp)
            elif method == "webster":
                per_seed = eval_webster(scenario, args, seeds, tmp)
            elif method == "actuated":
                per_seed = eval_actuated(scenario, args, seeds, tmp)
            elif method == "fixed":
                per_seed = eval_fixed(scenario, args, seeds, tmp)
            else:
                print(f"  [skip] unknown method {method!r}")
                continue
        except Exception as exc:  # one method failing must not kill the campaign
            print(f"  [error] {method} failed: {type(exc).__name__}: {exc}")
            continue
        aggregates[method] = aggregate(per_seed)
        print(f"  ({time.time()-t0:.1f}s)")

    if not aggregates:
        raise SystemExit("[eval] no method produced results")

    reference = args.reference if args.reference in aggregates else next(iter(aggregates))
    report = build_report(aggregates, reference)

    print_table(aggregates, report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "results" / "paper1_mappo" / "eval_tables" / f"{scenario_name}_{ts}"
    write_outputs(out_dir, scenario_name, aggregates, report, args)
    return {"aggregates": aggregates, "report": report, "out_dir": str(out_dir)}


def _force_utf8_stdout() -> None:
    # Windows consoles default to a locale codepage (e.g. cp1258) that lacks
    # '->', the dagger, etc.; printing them crashes when stdout is redirected.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    _force_utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", choices=["n1", "n2_corridor", "n3_grid"], default=None)
    p.add_argument("--split", choices=["train", "eval"], default="eval")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="explicit cfg (alternative to --network)")
    p.add_argument("--lane-groups", type=str, default=None,
                   help="lane_groups.json (with --sumo-cfg); else DEFAULT_MANUAL_LANE_GROUPS")
    p.add_argument("--methods", nargs="+",
                   default=["mappo", "webster", "actuated", "maxpressure", "sotl", "fixed"],
                   help="subset of: mappo ippo privileged webster actuated maxpressure sotl fixed")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    p.add_argument("--reference", type=str, default="mappo",
                   help="method to test others against (Mann-Whitney/Holm/Cohen)")

    # each accepts a .pt file, a directory (globbed for **/best_model.pt — the
    # per-seed campaign layout), or a glob pattern; multiple checkpoints become
    # per-training-seed samples for the CI / significance tests.
    p.add_argument("--checkpoint", nargs="+", default=None,
                   help="MAPPO checkpoint(s): file / dir / glob (per-seed → CI)")
    p.add_argument("--ippo-checkpoint", nargs="+", default=None)
    p.add_argument("--privileged-checkpoint", nargs="+", default=None)

    p.add_argument("--emissions", action="store_true",
                   help="enable SUMO emissions device -> post-hoc CO2/fuel per episode (TABLE-10)")
    p.add_argument("--max-steps", type=int, default=1080)
    p.add_argument("--step-length", type=int, default=5,
                   help="seconds per decision step (for the SUMO-native actuated duration)")
    p.add_argument("--min-green-time", type=int, default=15)
    p.add_argument("--max-green-time", type=int, default=60)

    # baseline hyperparameters
    p.add_argument("--fixed-green-s", type=float, default=30.0)
    p.add_argument("--phi-min", type=int, default=2)
    p.add_argument("--phi-max", type=int, default=12)
    p.add_argument("--kappa", type=int, default=5)
    p.add_argument("--sat-flow", type=float, default=1800.0)

    p.add_argument("--tmp-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
