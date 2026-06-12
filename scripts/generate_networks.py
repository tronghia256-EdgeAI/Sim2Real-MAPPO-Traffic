"""
generate_networks.py
====================
Generate the scaled SUMO networks for Paper 1 (schema 1.1.0):

    N1  2-junction corridor   — copied from sumo_configs/training (self-contained)
    N2  1x5 signalized corridor (netgenerate grid 5x1 + attached side streets)
    N3  4x4 signalized grid     (netgenerate grid 4x4 + attached side streets)

For every network this script writes, under sumo_configs/networks/<name>/:

    intersections.net.xml     the network
    vtypes.add.xml            Vietnamese mixed-traffic vehicle types (copied)
    lane_groups.json          {tls_id: [group_a_lanes, group_b_lanes]} derived
                              FROM THE TLS PROGRAM ITSELF (phase 0 vs phase 2),
                              so groups are correct by construction and pass
                              check_obs_match.py CHECK 6
    sumo_config.sumocfg       train config (routes: demand_train.rou.xml)
    sumo_config_eval.sumocfg  eval config  (routes: demand_eval.rou.xml)

Requirements: SUMO_HOME set, `netgenerate` on PATH (SUMO >= 1.16).

Run:
    python scripts/generate_networks.py            # all networks
    python scripts/generate_networks.py --only n3_grid
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = ROOT / "sumo_configs" / "networks"
N1_NET = ROOT / "sumo_configs" / "training" / "intersections.net.xml"
VTYPES_SRC = ROOT / "sumo_configs" / "training" / "vtypes.add.xml"

# expected signalized junction counts (hard validation)
EXPECTED_TLS = {"n2_corridor": 5, "n3_grid": 16}

NETGEN_SPECS: Dict[str, List[str]] = {
    "n2_corridor": [
        "--grid",
        "--grid.x-number", "5",
        "--grid.y-number", "1",
        "--grid.length", "200",
        "--grid.attach-length", "200",
    ],
    "n3_grid": [
        "--grid",
        "--grid.x-number", "4",
        "--grid.y-number", "4",
        "--grid.length", "200",
        "--grid.attach-length", "200",
    ],
}

COMMON_NETGEN_FLAGS = [
    "--default.lanenumber", "2",       # 2 lanes/approach, same as N1
    "--default.speed", "13.89",        # 50 km/h urban
    "--no-turnarounds", "true",
    "--default-junction-type", "traffic_light",
    "--tls.layout", "opposites",       # NS-green / yellow / EW-green / yellow
    "--tls.green.time", "30",
    "--tls.yellow.time", "3",
]

SUMOCFG_TEMPLATE = """<configuration>
    <input>
        <net-file value="intersections.net.xml"/>
        <route-files value="{route_file}"/>
        <additional-files value="vtypes.add.xml"/>
    </input>

    <processing>
        <time-to-teleport value="-1"/> <lateral-resolution value="0.4"/> </processing>
</configuration>
"""


def run_netgenerate(name: str, out_net: Path) -> None:
    cmd = ["netgenerate"] + NETGEN_SPECS[name] + COMMON_NETGEN_FLAGS + ["-o", str(out_net)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"netgenerate failed for {name}:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Lane-group derivation from the TLS program (authoritative)
# ---------------------------------------------------------------------------

def derive_lane_groups(net_path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    """Derive {tls: (group_a, group_b)} from each TLS program.

    group_a = lanes whose connections show green in the FIRST green phase,
    group_b = lanes green in the SECOND green phase. Assignment is unified at
    the EDGE level (all lanes of an approach edge go to the phase where the
    edge has the most protected 'G' signals), because the 1.1.0 observation
    aggregates per approach edge — an edge split across groups would be
    inconsistent with the obs layout.
    """
    root = ET.parse(net_path).getroot()

    # tls -> linkIndex -> (edge, lane_id)
    tls_links: Dict[str, Dict[int, Tuple[str, str]]] = {}
    for conn in root.iter("connection"):
        tls = conn.get("tl")
        if tls is None:
            continue
        link_idx = int(conn.get("linkIndex", "-1"))
        from_edge = conn.get("from", "")
        from_lane = conn.get("fromLane", "0")
        if link_idx < 0 or not from_edge or from_edge.startswith(":"):
            continue
        tls_links.setdefault(tls, {})[link_idx] = (from_edge, f"{from_edge}_{from_lane}")

    groups: Dict[str, Tuple[List[str], List[str]]] = {}
    for logic in root.iter("tlLogic"):
        tls_id = logic.get("id", "")
        phases = [p.get("state", "") for p in logic.iter("phase")]
        green_phase_idx = [i for i, s in enumerate(phases)
                           if any(c in "Gg" for c in s) and "y" not in s.lower()]
        if len(green_phase_idx) < 2:
            raise RuntimeError(
                f"{net_path.name}: TLS {tls_id} has {len(green_phase_idx)} green "
                f"phases (need >= 2). Program: {phases}"
            )
        pa, pb = green_phase_idx[0], green_phase_idx[1]
        links = tls_links.get(tls_id, {})

        # per-edge protected-green votes in each candidate phase
        edge_votes: Dict[str, List[int]] = {}   # edge -> [G-count in pa, in pb, any-green pa, any-green pb]
        edge_lanes: Dict[str, List[str]] = {}
        for idx, (edge, lane_id) in sorted(links.items()):
            sa = phases[pa][idx] if idx < len(phases[pa]) else "r"
            sb = phases[pb][idx] if idx < len(phases[pb]) else "r"
            v = edge_votes.setdefault(edge, [0, 0, 0, 0])
            v[0] += sa == "G"
            v[1] += sb == "G"
            v[2] += sa in "Gg"
            v[3] += sb in "Gg"
            if lane_id not in edge_lanes.setdefault(edge, []):
                edge_lanes[edge].append(lane_id)

        group_a: List[str] = []
        group_b: List[str] = []
        for edge, v in edge_votes.items():
            if (v[0], v[2]) >= (v[1], v[3]):
                group_a.extend(edge_lanes[edge])
            else:
                group_b.extend(edge_lanes[edge])

        if not group_a or not group_b:
            raise RuntimeError(
                f"{net_path.name}: TLS {tls_id} produced an empty phase group "
                f"(a={group_a}, b={group_b}) — program layout not 'opposites'?"
            )
        groups[tls_id] = (sorted(group_a), sorted(group_b))
    return groups


def validate_groups(net_path: Path, groups: Dict[str, Tuple[List[str], List[str]]]) -> None:
    """Replicate check_obs_match CHECK 6 locally: fail fast at generation time."""
    root = ET.parse(net_path).getroot()
    edge_to = {
        e.get("id"): e.get("to")
        for e in root.iter("edge")
        if e.get("function") != "internal" and e.get("id") and e.get("to")
    }
    owner: Dict[str, str] = {}
    for tls_id, (ga, gb) in groups.items():
        assert ga and gb, f"{tls_id}: empty group"
        assert not set(ga) & set(gb), f"{tls_id}: groups overlap"
        for lane in ga + gb:
            edge = lane.rsplit("_", 1)[0]
            assert edge_to.get(edge) == tls_id, (
                f"{tls_id}: lane {lane} approaches {edge_to.get(edge)}, not {tls_id}"
            )
            assert owner.setdefault(lane, tls_id) == tls_id, (
                f"lane {lane} claimed by {owner[lane]} and {tls_id}"
            )


# ---------------------------------------------------------------------------
# Per-network assembly
# ---------------------------------------------------------------------------

def write_network_dir(name: str, net_src: Path | None) -> None:
    out_dir = NETWORKS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_net = out_dir / "intersections.net.xml"

    if net_src is not None:                      # n1: copy existing net
        shutil.copyfile(net_src, out_net)
    else:                                        # n2/n3: generate
        run_netgenerate(name, out_net)

    shutil.copyfile(VTYPES_SRC, out_dir / "vtypes.add.xml")

    groups = derive_lane_groups(out_net)
    expected = EXPECTED_TLS.get(name)
    if expected is not None and len(groups) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} signalized junctions, got {len(groups)} "
            f"({sorted(groups)})"
        )
    validate_groups(out_net, groups)

    payload = {
        "schema_version": "1.1.0",
        "network": name,
        "tls_ids": sorted(groups),
        "lane_groups": {t: [list(a), list(b)] for t, (a, b) in groups.items()},
        "note": (
            "group_a = lanes green in the first green phase of the TLS program; "
            "group_b = second green phase. Derived by scripts/generate_networks.py; "
            "validated against the net topology (CHECK 6 invariants)."
        ),
    }
    with open(out_dir / "lane_groups.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    (out_dir / "sumo_config.sumocfg").write_text(
        SUMOCFG_TEMPLATE.format(route_file="demand_train.rou.xml"), encoding="utf-8"
    )
    (out_dir / "sumo_config_eval.sumocfg").write_text(
        SUMOCFG_TEMPLATE.format(route_file="demand_eval.rou.xml"), encoding="utf-8"
    )

    n_lanes = sum(len(a) + len(b) for a, b in groups.values())
    print(f"  [OK] {name}: {len(groups)} TLS, {n_lanes} grouped lanes -> {out_dir.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["n1", "n2_corridor", "n3_grid"], default=None)
    args = parser.parse_args()

    targets = [args.only] if args.only else ["n1", "n2_corridor", "n3_grid"]
    for name in targets:
        print(f"\n=== {name} ===")
        write_network_dir(name, net_src=N1_NET if name == "n1" else None)

    print("\nDone. Next: python scripts/generate_demand.py")


if __name__ == "__main__":
    sys.exit(main())
