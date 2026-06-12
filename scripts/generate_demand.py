"""
generate_demand.py
==================
Time-varying (trapezoidal) demand generation for N1 / N2 / N3 networks.

Replaces uniform randomTrips demand with a within-episode peak profile so a
single 1080-step episode (x 5 s = 5400 s) covers off-peak -> rush hour ->
medium -> cool-down. Departures are sampled from an inhomogeneous Poisson
process (per-second draws) over the piecewise-linear rate profile:

    t (s)      0 ....... 900 ...... 1500 ...... 3000 ...... 3600 ...... 4800 ...... 5400
    rate     LOW         ramp        PEAK        ramp        MEDIUM      taper -> LOW
    veh/s    0.40    0.40->1.67      1.67    1.67->0.83      0.83     0.83->0.40

Rates are NETWORK-WIDE for the N1 reference (6 entry edges). For larger
networks the profile is scaled by  n_entries / 6  by default (--scale auto)
so the per-approach load — what each agent actually faces — stays comparable
across N1/N2/N3.

Train/eval consistency (fixes the historical drift): every vehicle gets
    type="mixed_traffic" departLane="random" departPos="random" departSpeed="random"
in BOTH training and evaluation files; only the RNG seed differs
(eval seed = train seed + 1000 by default -> held-out departure pattern + ODs).

Routing: trips are routed with duarouter (resolves the Vietnamese
vTypeDistribution into concrete per-vehicle types, deterministic via --seed).
If duarouter is unavailable the raw trips file is written as the .rou.xml and
SUMO routes on insertion (slower; a warning is printed).

Run (after scripts/generate_networks.py):
    python scripts/generate_demand.py                       # all networks
    python scripts/generate_demand.py --network n3_grid --seed 42
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = ROOT / "sumo_configs" / "networks"

EPISODE_SECONDS = 5400          # 1080 env steps x 5 s
N1_REFERENCE_ENTRIES = 6        # rate spec is defined for N1's 6 entry edges

TRIP_ATTRS = {
    "type": "mixed_traffic",
    "departLane": "random",
    "departPos": "random",
    "departSpeed": "random",
}


# ---------------------------------------------------------------------------
# Demand profile
# ---------------------------------------------------------------------------

def trapezoid_profile(low: float, peak: float, medium: float) -> List[Tuple[float, float]]:
    """Piecewise-linear (time, rate) knots over one episode."""
    return [
        (0.0,    low),
        (900.0,  low),
        (1500.0, peak),
        (3000.0, peak),
        (3600.0, medium),
        (4800.0, medium),
        (5400.0, low),
    ]


def rate_at(t: float, knots: Sequence[Tuple[float, float]]) -> float:
    """Linear interpolation over profile knots; clamped at the ends."""
    if t <= knots[0][0]:
        return knots[0][1]
    for (t0, r0), (t1, r1) in zip(knots, knots[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return r1
            return r0 + (r1 - r0) * (t - t0) / (t1 - t0)
    return knots[-1][1]


def sample_departures(
    knots: Sequence[Tuple[float, float]],
    scale: float,
    horizon_s: int,
    rng: np.random.Generator,
) -> List[float]:
    """Inhomogeneous Poisson departures via per-second Poisson draws."""
    departs: List[float] = []
    for second in range(horizon_s):
        lam = rate_at(second + 0.5, knots) * scale
        n = int(rng.poisson(lam))
        if n > 0:
            departs.extend(second + rng.random(n))
    departs.sort()
    return departs


# ---------------------------------------------------------------------------
# Network topology: entry / exit edges
# ---------------------------------------------------------------------------

def find_entries_exits(net_path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (entries, exits) as lists of (edge_id, fringe_node_id).

    A fringe node has exactly one neighbour (a dead-end stub endpoint).
    Entry = edge leaving a fringe node; exit = edge arriving at one. This is
    robust to junction 'type' attributes across netgenerate / netedit nets.
    """
    root = ET.parse(net_path).getroot()
    neighbours: Dict[str, set] = {}
    edges: List[Tuple[str, str, str]] = []
    for edge in root.iter("edge"):
        if edge.get("function") == "internal":
            continue
        eid, n_from, n_to = edge.get("id"), edge.get("from"), edge.get("to")
        if not eid or not n_from or not n_to:
            continue
        edges.append((eid, n_from, n_to))
        neighbours.setdefault(n_from, set()).add(n_to)
        neighbours.setdefault(n_to, set()).add(n_from)

    fringe = {n for n, nbrs in neighbours.items() if len(nbrs) == 1}
    entries = [(eid, n_from) for eid, n_from, _ in edges if n_from in fringe]
    exits = [(eid, n_to) for eid, _, n_to in edges if n_to in fringe]
    if not entries or not exits:
        raise RuntimeError(f"{net_path}: found {len(entries)} entries / {len(exits)} exits")
    return sorted(entries), sorted(exits)


# ---------------------------------------------------------------------------
# Trip + route file generation
# ---------------------------------------------------------------------------

def write_trips(
    out_path: Path,
    departs: Sequence[float],
    entries: Sequence[Tuple[str, str]],
    exits: Sequence[Tuple[str, str]],
    rng: np.random.Generator,
    id_prefix: str,
) -> int:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    attrs = " ".join(f'{k}="{v}"' for k, v in TRIP_ATTRS.items())
    n_written = 0
    for i, depart in enumerate(departs):
        entry_edge, entry_node = entries[int(rng.integers(len(entries)))]
        # destination on a different fringe node (no stub U-turns)
        candidates = [e for e in exits if e[1] != entry_node]
        exit_edge, _ = candidates[int(rng.integers(len(candidates)))]
        lines.append(
            f'    <trip id="{id_prefix}{i}" depart="{depart:.2f}" '
            f'from="{entry_edge}" to="{exit_edge}" {attrs}/>'
        )
        n_written += 1
    lines.append("</routes>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_written


def route_with_duarouter(
    net_path: Path, trips_path: Path, vtypes_path: Path, out_path: Path, seed: int
) -> bool:
    if shutil.which("duarouter") is None:
        return False
    # vtypes are redirected to a scratch file and discarded: the canonical
    # definitions stay in vtypes.add.xml (loaded by the sumocfg); embedding
    # them in the route file would duplicate ids and abort SUMO.
    vtype_scratch = out_path.with_suffix(".vtypes.scratch.xml")
    cmd = [
        "duarouter",
        "-n", str(net_path),
        "-r", str(trips_path),
        "--additional-files", str(vtypes_path),
        "-o", str(out_path),
        "--vtype-output", str(vtype_scratch),
        "--seed", str(seed),
        "--ignore-errors",
        "--repair",
        "--no-warnings",
        "--no-step-log",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] duarouter failed ({result.returncode}): {result.stderr[:300]}")
        return False
    # drop scratch + .alt files duarouter writes next to the output
    alt = out_path.with_suffix(out_path.suffix + ".alt.xml")
    legacy_alt = Path(str(out_path).replace(".rou.xml", ".rou.alt.xml"))
    for p in (alt, legacy_alt, vtype_scratch):
        if p.exists():
            p.unlink()
    return True


def generate_for_network(
    name: str,
    seed: int,
    rates: Tuple[float, float, float],
    scale_arg: str,
) -> None:
    net_dir = NETWORKS_DIR / name
    net_path = net_dir / "intersections.net.xml"
    vtypes_path = net_dir / "vtypes.add.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"{net_path} — run scripts/generate_networks.py first")

    entries, exits = find_entries_exits(net_path)
    scale = len(entries) / N1_REFERENCE_ENTRIES if scale_arg == "auto" else float(scale_arg)

    low, medium, peak = rates
    knots = trapezoid_profile(low=low, peak=peak, medium=medium)

    print(f"\n=== {name} ===")
    print(f"  entries={len(entries)} exits={len(exits)} scale={scale:.2f} "
          f"(network rates: low={low*scale:.2f} med={medium*scale:.2f} peak={peak*scale:.2f} veh/s)")

    for split, split_seed in (("train", seed), ("eval", seed + 1000)):
        rng = np.random.default_rng(split_seed)
        departs = sample_departures(knots, scale, EPISODE_SECONDS, rng)
        trips_path = net_dir / f"demand_{split}.trips.xml"
        rou_path = net_dir / f"demand_{split}.rou.xml"
        n = write_trips(trips_path, departs, entries, exits, rng, id_prefix=f"{split}_")

        if route_with_duarouter(net_path, trips_path, vtypes_path, rou_path, split_seed):
            routed = "duarouter"
        else:
            shutil.copyfile(trips_path, rou_path)
            routed = "trips-as-routes (SUMO online routing — install duarouter for speed)"
            print(f"  [WARN] {split}: falling back to {routed}")

        peak_count = sum(1 for d in departs if 1500 <= d < 3000)
        print(f"  [OK] {split}: {n} vehicles (peak window 1500-3000s: {peak_count}, "
              f"~{peak_count/1500.0:.2f} veh/s), seed={split_seed}, routing={routed}")
        print(f"       -> {rou_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=["n1", "n2_corridor", "n3_grid"], default=None,
                        help="generate for one network only (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="train seed; eval uses seed+1000 (default 42)")
    parser.add_argument("--low-rate", type=float, default=0.40,
                        help="off-peak rate, veh/s at N1 reference scale")
    parser.add_argument("--medium-rate", type=float, default=0.83,
                        help="medium rate, veh/s at N1 reference scale")
    parser.add_argument("--peak-rate", type=float, default=1.67,
                        help="rush-hour rate, veh/s at N1 reference scale")
    parser.add_argument("--scale", default="auto",
                        help="'auto' = n_entries/6 (per-approach load parity with N1), "
                             "or an explicit float")
    args = parser.parse_args()

    targets = [args.network] if args.network else ["n1", "n2_corridor", "n3_grid"]
    for name in targets:
        generate_for_network(
            name,
            seed=args.seed,
            rates=(args.low_rate, args.medium_rate, args.peak_rate),
            scale_arg=args.scale,
        )
    print("\nDone. sumo_config.sumocfg uses demand_train.rou.xml; "
          "sumo_config_eval.sumocfg uses demand_eval.rou.xml.")


if __name__ == "__main__":
    sys.exit(main())
