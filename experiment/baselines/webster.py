"""
Webster Fixed-Time controller — SUMO baseline (replaces the 30 s equal-split strawman).

Reference
---------
Webster, F.V. (1958). Traffic Signal Settings. Road Research Technical Paper 39.

    C = (1.5 * L + 5) / (1 - Y)

    L : total lost time per cycle  = n_phases * (yellow + startup_lost)
    Y : sum over phases of the CRITICAL lane flow ratio  y_i = q_i / s_i
        q_i : highest PCU flow among the phase's lanes (PCU/h)
        s_i : saturation flow (PCU/h/lane)

Green splits:  g_i = (y_i / Y) * (C - L), floored at the env's min green.

Vietnamese mixed traffic is handled in PCU (passenger car units):
moto 0.3, car 1.0, truck 2.0, bus 2.5 (override via --pcu).

Two-pass protocol
-----------------
1. MEASUREMENT: run the scenario under the net's built-in static program with
   no external control, counting unique vehicles (PCU-weighted) per lane.
   This is the "historical average arrival rate" input to Webster.
2. EVALUATION: drive MappoTrafficEnv with the fitted WebsterPolicy — the SAME
   env path used by MAPPO / MaxPressure / SOTL, so step semantics, min-green
   and yellow handling are identical across methods. Tripinfo metrics are
   written and summarized.

Usage (from project root):
    python experiment/baselines/webster.py \
        --network n3_grid --seed 42
    python experiment/baselines/webster.py \
        --sumo-cfg sumo_configs/networks/n2_corridor/sumo_config_eval.sumocfg \
        --lane-groups sumo_configs/networks/n2_corridor/lane_groups.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.config import build_default_config, load_lane_groups_json
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from experiment.common.tripinfo import parse_tripinfo_means

DEFAULT_PCU = {"moto": 0.3, "car": 1.0, "truck": 2.0, "bus": 2.5}

# parity with BaseSumoEnv default flags so measurement == evaluation dynamics
ENV_PARITY_FLAGS = [
    "--time-to-teleport", "160",
    "--waiting-time-memory", "1000",
    "--no-warnings",
    "--no-step-log",
]


# =============================================================================
# Webster policy
# =============================================================================

@dataclass
class WebsterTiming:
    cycle: float
    green_a: float
    green_b: float
    y_a: float
    y_b: float


@dataclass
class WebsterPolicy:
    """Fixed-time controller with per-junction Webster-optimal cycle/splits.

    Parameters
    ----------
    lane_groups : {tls_id: (group_a_lanes, group_b_lanes)}
        Phase groups (group_a = action 0 / phase 0; group_b = action 1).
    lane_flows_pcu_h : {lane_id: flow in PCU/h}
        Measured historical average arrival rates per lane.
    """

    lane_groups: Mapping[str, Tuple[List[str], List[str]]]
    lane_flows_pcu_h: Mapping[str, float]
    sat_flow_pcu_h: float = 1800.0
    yellow: float = 3.0
    startup_lost: float = 2.0
    min_cycle: float = 40.0
    max_cycle: float = 120.0
    min_green: float = 10.0
    max_y: float = 0.85          # Webster diverges as Y -> 1; clamp when oversaturated
    timings: Dict[str, WebsterTiming] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for tls_id, (group_a, group_b) in self.lane_groups.items():
            self.timings[tls_id] = self._fit_one(group_a, group_b)

    def _critical_ratio(self, lanes: List[str]) -> float:
        flows = [float(self.lane_flows_pcu_h.get(l, 0.0)) for l in lanes]
        q_crit = max(flows) if flows else 0.0
        return q_crit / max(self.sat_flow_pcu_h, 1e-9)

    def _fit_one(self, group_a: List[str], group_b: List[str]) -> WebsterTiming:
        y_a = self._critical_ratio(group_a)
        y_b = self._critical_ratio(group_b)
        # keep a usable split even with zero measured flow
        y_a = max(y_a, 1e-3)
        y_b = max(y_b, 1e-3)

        big_y = y_a + y_b
        if big_y > self.max_y:                      # oversaturated: rescale, keep ratio
            scale = self.max_y / big_y
            y_a, y_b, big_y = y_a * scale, y_b * scale, self.max_y

        lost = 2.0 * (self.yellow + self.startup_lost)          # L, 2 phases
        cycle = (1.5 * lost + 5.0) / max(1.0 - big_y, 1e-9)     # Webster optimum
        cycle = float(np.clip(cycle, self.min_cycle, self.max_cycle))

        effective = cycle - lost
        g_a = max(self.min_green, (y_a / big_y) * effective)
        g_b = max(self.min_green, (y_b / big_y) * effective)
        cycle_actual = g_a + g_b + 2.0 * self.yellow
        return WebsterTiming(cycle=cycle_actual, green_a=g_a, green_b=g_b, y_a=y_a, y_b=y_b)

    def select_actions(self, sim_time_s: float) -> Dict[str, int]:
        """Return {tls_id: 0|1} from each junction's fixed schedule.

        Timeline per cycle: [green_a][yellow][green_b][yellow]. The env
        inserts its own yellow on a switch; the schedule's yellow slots
        simply hold the boundary, so requested greens are fully served.
        """
        actions: Dict[str, int] = {}
        for tls_id, t in self.timings.items():
            pos = sim_time_s % t.cycle
            actions[tls_id] = 0 if pos < t.green_a + self.yellow else 1
        return actions

    def describe(self) -> str:
        lines = [f"{'TLS':>6}  {'cycle':>6}  {'gA':>6}  {'gB':>6}  {'yA':>5}  {'yB':>5}"]
        for tls_id, t in sorted(self.timings.items()):
            lines.append(
                f"{tls_id:>6}  {t.cycle:6.1f}  {t.green_a:6.1f}  {t.green_b:6.1f}"
                f"  {t.y_a:5.3f}  {t.y_b:5.3f}"
            )
        return "\n".join(lines)


# =============================================================================
# Pass 1 — flow measurement (historical average arrival rates)
# =============================================================================

def measure_lane_flows(
    sumo_cfg: Path,
    lane_ids: List[str],
    duration_s: int,
    seed: int,
    pcu: Mapping[str, float],
) -> Dict[str, float]:
    """Run SUMO under its built-in static program; count unique PCU per lane.

    Returns {lane_id: PCU/h}.
    """
    import traci

    cmd = ["sumo", "-c", str(sumo_cfg), "--seed", str(seed), "--start"] + ENV_PARITY_FLAGS
    traci.start(cmd, label="webster_measure")
    conn = traci.getConnection("webster_measure")

    seen_per_lane: Dict[str, set] = {l: set() for l in lane_ids}
    pcu_per_lane: Dict[str, float] = {l: 0.0 for l in lane_ids}
    type_cache: Dict[str, float] = {}

    def pcu_of(veh_id: str) -> float:
        if veh_id not in type_cache:
            try:
                type_id = conn.vehicle.getTypeID(veh_id).lower()
            except Exception:
                type_id = "car"
            weight = 1.0
            for key, w in pcu.items():
                if type_id.startswith(key):
                    weight = w
                    break
            type_cache[veh_id] = weight
        return type_cache[veh_id]

    try:
        for _ in range(int(duration_s)):
            conn.simulationStep()
            for lane in lane_ids:
                try:
                    for veh_id in conn.lane.getLastStepVehicleIDs(lane):
                        if veh_id not in seen_per_lane[lane]:
                            seen_per_lane[lane].add(veh_id)
                            pcu_per_lane[lane] += pcu_of(veh_id)
                except Exception:
                    pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    hours = duration_s / 3600.0
    return {lane: count / max(hours, 1e-9) for lane, count in pcu_per_lane.items()}


# =============================================================================
# Pass 2 — evaluation through the shared env
# =============================================================================

def evaluate_policy(
    sumo_cfg: Path,
    tls_ids: Tuple[str, ...],
    lane_groups: Dict[str, Tuple[List[str], List[str]]],
    policy: WebsterPolicy,
    seed: int,
    max_steps: int,
    tripinfo_path: Path,
) -> Dict[str, float]:
    env_cfg = build_default_config(
        sumo_cfg_path=str(sumo_cfg),
        gui=False,
        tls_ids=tls_ids,
        manual_lane_groups=lane_groups,
        max_steps=max_steps,
    )
    env = MappoTrafficEnv(
        config=env_cfg,
        gui=False,
        use_libsumo=False,
        extra_sumo_args=[
            "--tripinfo-output", str(tripinfo_path),
            "--tripinfo-output.write-unfinished",
        ],
    )
    env.reset(seed=seed)

    step_len = float(env_cfg.sim.step_length)
    total_reward = 0.0
    for step in range(max_steps):
        actions = policy.select_actions(sim_time_s=step * step_len)
        _, reward_dict, terminated, truncated, _ = env.step(actions)
        total_reward += float(np.mean(list(reward_dict.values()))) if reward_dict else 0.0
        if any(terminated.values()) or any(truncated.values()):
            break
    env.close()  # flushes tripinfo

    metrics = parse_tripinfo_means(tripinfo_path)
    metrics["mean_step_reward"] = total_reward / max_steps
    return metrics


# =============================================================================
# CLI
# =============================================================================

def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.network:
        base = _ROOT / "sumo_configs" / "networks" / args.network
        cfg = base / ("sumo_config_eval.sumocfg" if args.split == "eval" else "sumo_config.sumocfg")
        groups = base / "lane_groups.json"
        return cfg, groups
    if not args.sumo_cfg or not args.lane_groups:
        raise SystemExit("either --network or both --sumo-cfg and --lane-groups are required")
    return Path(args.sumo_cfg), Path(args.lane_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=["n1", "n2_corridor", "n3_grid"], default=None)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--sumo-cfg", type=str, default=None)
    parser.add_argument("--lane-groups", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=1080)
    parser.add_argument("--measure-seconds", type=int, default=5400,
                        help="measurement-pass duration (default: full episode)")
    parser.add_argument("--sat-flow", type=float, default=1800.0, help="PCU/h/lane")
    parser.add_argument("--pcu", type=str, default=None,
                        help='JSON override, e.g. \'{"moto":0.3,"car":1.0}\'')
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    sumo_cfg, groups_path = resolve_paths(args)
    tls_ids, lane_groups = load_lane_groups_json(str(groups_path))
    all_lanes = [l for ga, gb in lane_groups.values() for l in list(ga) + list(gb)]
    pcu = dict(DEFAULT_PCU, **(json.loads(args.pcu) if args.pcu else {}))

    print(f"[1/3] measuring lane flows: {sumo_cfg.name}, {args.measure_seconds}s, "
          f"seed={args.seed} ({len(all_lanes)} lanes)")
    flows = measure_lane_flows(sumo_cfg, all_lanes, args.measure_seconds, args.seed, pcu)

    policy = WebsterPolicy(
        lane_groups=lane_groups, lane_flows_pcu_h=flows, sat_flow_pcu_h=args.sat_flow,
    )
    print("[2/3] Webster timings (per junction, no offsets):")
    print(policy.describe())

    tripinfo_path = Path(tempfile.gettempdir()) / f"webster_tripinfo_{args.seed}.xml"
    print(f"[3/3] evaluation pass via MappoTrafficEnv ({args.episode_steps} steps)")
    metrics = evaluate_policy(
        sumo_cfg, tls_ids, lane_groups, policy,
        seed=args.seed, max_steps=args.episode_steps, tripinfo_path=tripinfo_path,
    )

    print("\n=== Webster Fixed-Time results ===")
    for key, value in metrics.items():
        print(f"  {key:>22}: {value:.2f}" if isinstance(value, float) else f"  {key:>22}: {value}")

    if args.out_json:
        payload = {
            "method": "webster",
            "sumo_cfg": str(sumo_cfg),
            "seed": args.seed,
            "sat_flow_pcu_h": args.sat_flow,
            "pcu": pcu,
            "timings": {t: vars(w) for t, w in policy.timings.items()},
            "metrics": metrics,
        }
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
