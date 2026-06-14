from __future__ import annotations

"""
generate_ood_scenarios.py  (paper VI-D)
=======================================
Out-of-distribution evaluation scenarios for the scaled networks (n2_corridor,
n3_grid). Policies are TRAINED on the within-episode trapezoid demand
(generate_demand.py); here we build HELD-OUT distributions to probe
generalization:

  1. Uniform-rate demand at three levels (held-out vs the trapezoid shape):
        low    0.40 veh/s   medium 0.83 veh/s   high 1.67 veh/s   (N1 scale)
     + asymmetric: medium rate with origins skewed to half the fringe (directional).
        -> demand_ood_<profile>.rou.xml + sumo_config_ood_<profile>.sumocfg

  2. Motorcycle-share sweep {50, 73, 90}% (the formal sensitivity defense of the
     central 73% mix; 73 = the training value, regenerated for symmetry). Other
     classes keep their original proportions, rescaled to fill 1 - moto.
        -> vtypes_moto<pct>.add.xml + sumo_config_moto<pct>.sumocfg
        (these reuse the held-out demand_eval.rou.xml; only the vehicle mix changes)

Rates scale by n_entries/6 like generate_demand (per-approach load parity).
Routing uses duarouter when available, else trips-as-routes (SUMO online routing).

Run (after generate_networks.py + generate_demand.py):
    python scripts/generate_ood_scenarios.py                       # n2 + n3
    python scripts/generate_ood_scenarios.py --network n3_grid

Evaluate (per scenario), e.g.:
    python experiment/runners/eval_compare.py \
        --sumo-cfg sumo_configs/networks/n3_grid/sumo_config_ood_high.sumocfg \
        --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
        --checkpoint <campaign>/models/n3_grid_mappo_proxy_seed* \
        --methods mappo webster actuated maxpressure sotl fixed
    # moto sweep (learned + env-path baselines; actuated uses the cfg's mix):
    python experiment/runners/eval_compare.py \
        --sumo-cfg sumo_configs/networks/n3_grid/sumo_config_moto90.sumocfg \
        --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
        --checkpoint <...> --methods mappo webster maxpressure sotl fixed
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_demand import (  # reuse the proven helpers
    EPISODE_SECONDS, N1_REFERENCE_ENTRIES,
    find_entries_exits, sample_departures, write_trips, route_with_duarouter,
)

NETWORKS_DIR = ROOT / "sumo_configs" / "networks"

# (profile -> (rate veh/s at N1 scale, asymmetric?))
OOD_PROFILES = {
    "low": (0.40, False),
    "medium": (0.83, False),
    "high": (1.67, False),
    "asymmetric": (0.83, True),
}
# deterministic per-profile seed offset (NOT hash() — that is per-process random)
_PROFILE_SEED_OFFSET = {"low": 11, "medium": 22, "high": 33, "asymmetric": 44}
MOTO_PCTS = (50, 73, 90)
BASE_OTHER = {"car": 0.17, "truck": 0.07, "bus": 0.03}  # sum 0.27 at 73% moto


def _write_cfg(cfg_path: Path, route_file: str, vtypes_file: str) -> None:
    cfg_path.write_text(
        "<configuration>\n"
        "    <input>\n"
        '        <net-file value="intersections.net.xml"/>\n'
        f'        <route-files value="{route_file}"/>\n'
        f'        <additional-files value="{vtypes_file}"/>\n'
        "    </input>\n"
        "    <processing>\n"
        '        <time-to-teleport value="-1"/> <lateral-resolution value="0.4"/>\n'
        "    </processing>\n"
        "</configuration>\n",
        encoding="utf-8",
    )


def _const_knots(rate: float) -> List[Tuple[float, float]]:
    return [(0.0, rate), (float(EPISODE_SECONDS), rate)]


def gen_ood_demand(name: str, seed: int, scale_arg: str) -> None:
    net_dir = NETWORKS_DIR / name
    net_path = net_dir / "intersections.net.xml"
    vtypes_path = net_dir / "vtypes.add.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"{net_path} — run generate_networks.py first")

    entries, exits = find_entries_exits(net_path)
    scale = len(entries) / N1_REFERENCE_ENTRIES if scale_arg == "auto" else float(scale_arg)
    print(f"\n=== {name} OOD demand (entries={len(entries)} scale={scale:.2f}) ===")

    for profile, (rate, asym) in OOD_PROFILES.items():
        # eval-distinct seed per profile (held-out from training's seed+1000)
        prof_seed = seed + 2000 + _PROFILE_SEED_OFFSET[profile]
        rng = np.random.default_rng(prof_seed)
        departs = sample_departures(_const_knots(rate), scale, EPISODE_SECONDS, rng)

        # asymmetric: skew origins to half the fringe by repeating them (weighted pick)
        origins: Sequence = entries
        if asym:
            half = max(1, len(entries) // 2)
            origins = list(entries[:half]) * 3 + list(entries[half:])

        trips = net_dir / f"demand_ood_{profile}.trips.xml"
        rou = net_dir / f"demand_ood_{profile}.rou.xml"
        n = write_trips(trips, departs, origins, exits, rng, id_prefix=f"ood_{profile}_")
        if not route_with_duarouter(net_path, trips, vtypes_path, rou, prof_seed):
            shutil.copyfile(trips, rou)
            print(f"  [WARN] {profile}: duarouter unavailable — trips-as-routes")
        _write_cfg(net_dir / f"sumo_config_ood_{profile}.sumocfg",
                   f"demand_ood_{profile}.rou.xml", "vtypes.add.xml")
        print(f"  [OK] ood_{profile}: {n} veh (rate {rate*scale:.2f} veh/s{' ASYM' if asym else ''})")


def gen_moto_sweep(name: str) -> None:
    net_dir = NETWORKS_DIR / name
    base_vtypes = net_dir / "vtypes.add.xml"
    if not base_vtypes.exists():
        raise FileNotFoundError(f"{base_vtypes} missing")
    eval_rou = net_dir / "demand_eval.rou.xml"
    route_ref = "demand_eval.rou.xml" if eval_rou.exists() else "demand_train.rou.xml"
    print(f"\n=== {name} motorcycle-share sweep (demand={route_ref}) ===")

    other_sum = sum(BASE_OTHER.values())
    for pct in MOTO_PCTS:
        moto = pct / 100.0
        rest = max(1.0 - moto, 0.0)
        scale = rest / other_sum if other_sum > 0 else 0.0
        probs = {"moto": moto, **{k: v * scale for k, v in BASE_OTHER.items()}}

        tree = ET.parse(base_vtypes)
        for vt in tree.getroot().iter("vType"):
            vid = vt.get("id")
            if vid in probs:
                vt.set("probability", f"{probs[vid]:.4f}")
        out_vtypes = net_dir / f"vtypes_moto{pct}.add.xml"
        tree.write(out_vtypes, encoding="utf-8", xml_declaration=False)
        _write_cfg(net_dir / f"sumo_config_moto{pct}.sumocfg",
                   route_ref, f"vtypes_moto{pct}.add.xml")
        print(f"  [OK] moto{pct}: probs=" +
              " ".join(f"{k}={v:.3f}" for k, v in probs.items()))


def gen_lanebased_cfgs(name: str) -> None:
    """Write lane-based (sublane OFF) twins of the train/eval cfgs for the
    sublane-vs-lane cross-evaluation (VI-F / C4): a policy trained under
    lane-based dynamics is tested on the sublane scenario and vice versa.
    Identical to the originals but with the <lateral-resolution> element removed.
    """
    import re
    net_dir = NETWORKS_DIR / name
    print(f"\n=== {name} lane-based cfgs (sublane OFF, for cross-eval) ===")
    for base in ("sumo_config.sumocfg", "sumo_config_eval.sumocfg"):
        src = net_dir / base
        if not src.exists():
            print(f"  [skip] {base} not found")
            continue
        txt = src.read_text(encoding="utf-8")
        # drop the lateral-resolution element (disables the sublane model)
        txt = re.sub(r"<lateral-resolution[^/]*/>", "", txt)
        out = net_dir / base.replace(".sumocfg", "_lanebased.sumocfg")
        out.write_text(txt, encoding="utf-8")
        print(f"  [OK] {out.name}")


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", choices=["n2_corridor", "n3_grid"], default=None,
                   help="default: both scaled networks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", default="auto")
    p.add_argument("--skip-ood", action="store_true")
    p.add_argument("--skip-moto", action="store_true")
    p.add_argument("--skip-lanebased", action="store_true")
    args = p.parse_args()

    targets = [args.network] if args.network else ["n2_corridor", "n3_grid"]
    for name in targets:
        if not args.skip_ood:
            gen_ood_demand(name, args.seed, args.scale)
        if not args.skip_moto:
            gen_moto_sweep(name)
        if not args.skip_lanebased:
            gen_lanebased_cfgs(name)
    print("\nDone. Evaluate each generated sumo_config_*.sumocfg with eval_compare "
          "(--sumo-cfg ... --lane-groups ...). See docstring for examples.")


if __name__ == "__main__":
    main()
