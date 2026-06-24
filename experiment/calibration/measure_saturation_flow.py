"""Saturation-flow calibration for the mixed motorcycle-dominant network.

Reproduces the per-lane saturation flow reported in Section V (Experimental
Setup) of Paper 1. Uses the standard queue-discharge method:

  * a single signalized approach, one lane at the paper's urban lane width
    (3.2 m), with SUMO's sublane model (lateral resolution 0.4 m);
  * the moto73 vehicle mix (73% motorcycle / 17% car / 7% truck / 3% bus),
    identical to ``sumo_configs/networks/*/vtypes_moto73.add.xml``;
  * a long red builds a deep standing queue that fills the approach; on green
    the bulk discharge of that queue is measured (skipping the first few
    vehicles = startup lost time, restricted to the deeply-saturated window
    where the halting count stays high).

Reported metrics, per lane:
  * raw discharge in veh/h (high under motorcycle filtering — several
    two-wheelers discharge abreast within one lane);
  * PCU-normalized discharge in PCU/h (moto PCU 0.30, Webster convention),
    which is the saturation flow quoted in the paper (~1800 PCU/h/lane) and
    matches the ``sat_flow_pcu_h`` default of ``experiment/baselines/webster.py``.

Usage
-----
    python experiment/calibration/measure_saturation_flow.py
    python experiment/calibration/measure_saturation_flow.py --seeds 42 123 456

Requires SUMO (netconvert + sumo) on PATH with SUMO_HOME set.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# moto73 vehicle mix — kept in sync with sumo_configs/networks/*/vtypes_moto73.add.xml
VTYPE_DISTRIBUTION = """  <vTypeDistribution id="mixed_traffic">
    <vType id="moto"  vClass="motorcycle" length="2.0" width="0.8" accel="4.5" decel="6.0" sigma="0.7" maxSpeed="15" speedDev="0.4" probability="0.7300" latAlignment="arbitrary" minGapLat="0.12" maxSpeedLat="1.5" minGap="1.0" impatience="0.7" jmIgnoreKeepClear="1.0"/>
    <vType id="car"   vClass="passenger"  length="4.8" width="1.8" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="20" speedDev="0.15" probability="0.1700" latAlignment="center" minGapLat="0.5" minGap="2.0"/>
    <vType id="truck" vClass="truck"      length="7.5" width="2.5" accel="1.5" decel="3.5" sigma="0.4" maxSpeed="16" speedDev="0.1" probability="0.0700" latAlignment="center" minGap="2.5"/>
    <vType id="bus"   vClass="bus"        length="12.0" width="2.5" accel="1.2" decel="3.0" sigma="0.3" maxSpeed="14" speedDev="0.05" probability="0.0300" latAlignment="center" minGap="2.5"/>
  </vTypeDistribution>"""

PCU = {"motorcycle": 0.30, "passenger": 1.0, "truck": 2.0, "bus": 2.5}


def build_inputs(d: Path, lane_w: float, ab_len: float, bc_len: float, lat_res: float, step: float) -> Path:
    netconvert = str(Path(os.environ["SUMO_HOME"]) / "bin" / "netconvert")
    (d / "n.nod.xml").write_text(
        '<nodes>\n'
        '  <node id="A" x="0" y="0" type="priority"/>\n'
        f'  <node id="B" x="{ab_len}" y="0" type="traffic_light"/>\n'
        f'  <node id="C" x="{ab_len + bc_len}" y="0" type="priority"/>\n'
        '</nodes>\n', encoding="utf-8")
    (d / "n.edg.xml").write_text(
        '<edges>\n'
        f'  <edge id="AB" from="A" to="B" numLanes="1" width="{lane_w}" speed="15"/>\n'
        f'  <edge id="BC" from="B" to="C" numLanes="1" width="{lane_w}" speed="15"/>\n'
        '</edges>\n', encoding="utf-8")
    subprocess.run([netconvert, "-n", str(d / "n.nod.xml"), "-e", str(d / "n.edg.xml"),
                    "-o", str(d / "net.net.xml"), "--no-turnarounds", "--tls.guess", "false",
                    "--no-warnings"], check=True)
    (d / "r.rou.xml").write_text(
        '<routes>\n' + VTYPE_DISTRIBUTION + '\n'
        '  <route id="thru" edges="AB BC"/>\n'
        '  <flow id="f" type="mixed_traffic" route="thru" begin="0" end="200" '
        'vehsPerHour="30000" departLane="free" departSpeed="0" departPos="free"/>\n'
        '</routes>\n', encoding="utf-8")
    cfg = d / "s.sumocfg"
    cfg.write_text(
        '<configuration>\n'
        '  <input><net-file value="net.net.xml"/><route-files value="r.rou.xml"/></input>\n'
        '  <processing>\n'
        f'    <lateral-resolution value="{lat_res}"/>\n'
        f'    <step-length value="{step}"/>\n'
        '    <default.speeddev value="0"/>\n'
        '  </processing>\n'
        '</configuration>\n', encoding="utf-8")
    return cfg


def measure_one(cfg: Path, seed: int, red_end: float, sim_end: float,
                halt_hi: int, skip_first: int, moto_pcu: float) -> dict:
    import traci
    traci.start(["sumo", "-c", str(cfg), "--no-step-log", "--no-warnings", "--seed", str(seed)])
    tls = traci.trafficlight.getIDList()
    if not tls:
        traci.close()
        raise RuntimeError("no TLS created on the test approach")
    TLS = tls[0]
    nlinks = len(traci.trafficlight.getRedYellowGreenState(TLS))

    seen: set = set()
    crossings: list = []          # (time, vClass)
    deep_times: list = []         # times at which AB halting >= halt_hi
    t = 0.0
    while t < sim_end:
        traci.trafficlight.setRedYellowGreenState(TLS, ("r" if t < red_end else "G") * nlinks)
        traci.simulationStep()
        t = traci.simulation.getTime()
        if traci.edge.getLastStepHaltingNumber("AB") >= halt_hi:
            deep_times.append(t)
        for vid in traci.edge.getLastStepVehicleIDs("BC"):
            if vid not in seen:
                seen.add(vid)
                crossings.append((t, traci.vehicle.getVehicleClass(vid)))
    traci.close()

    post = [(tc, vc) for tc, vc in crossings if tc >= red_end]
    deep_after = [tt for tt in deep_times if tt >= red_end]
    if len(post) <= skip_first or not deep_after:
        raise RuntimeError(f"seed {seed}: queue too shallow (crossings={len(post)})")
    # saturated window: from after startup lost time to the end of the deeply-queued period
    t_start = post[skip_first - 1][0]
    t_end = deep_after[-1]
    win = [(tc, vc) for tc, vc in post if t_start <= tc <= t_end]
    dur = t_end - t_start
    if dur < 15 or len(win) < 10:
        raise RuntimeError(f"seed {seed}: thin window dur={dur:.1f}s n={len(win)}")

    pcu_map = dict(PCU, motorcycle=moto_pcu)
    sum_pcu = sum(pcu_map.get(vc, 1.0) for _, vc in win)
    moto_share = sum(1 for _, vc in win if vc == "motorcycle") / len(win)
    return {
        "seed": seed, "n": len(win), "dur": dur, "moto_share": moto_share,
        "veh_h": len(win) / dur * 3600.0, "pcu_h": sum_pcu / dur * 3600.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--lane-width", type=float, default=3.2)
    ap.add_argument("--lateral-res", type=float, default=0.4)
    ap.add_argument("--moto-pcu", type=float, default=0.30)
    ap.add_argument("--red-end", type=float, default=200.0)
    ap.add_argument("--sim-end", type=float, default=520.0)
    ap.add_argument("--halt-hi", type=int, default=8)
    ap.add_argument("--skip-first", type=int, default=5)
    args = ap.parse_args()

    if "SUMO_HOME" not in os.environ:
        print("ERROR: SUMO_HOME is not set."); return 1

    with tempfile.TemporaryDirectory(prefix="satflow_") as tmp:
        cfg = build_inputs(Path(tmp), args.lane_width, 250.0, 200.0, args.lateral_res, 0.5)
        rows = []
        for s in args.seeds:
            try:
                rows.append(measure_one(cfg, s, args.red_end, args.sim_end,
                                        args.halt_hi, args.skip_first, args.moto_pcu))
            except Exception as e:  # noqa: BLE001 — report and continue across seeds
                print(f"  [skip] {e}")

    if not rows:
        print("No valid measurements."); return 1

    ver = subprocess.check_output(["sumo", "--version"]).decode().split()[4]
    print("\n=============== SATURATION FLOW (queue-discharge) ===============")
    print(f"lane={args.lane_width} m | sublane res={args.lateral_res} m | mix=moto73 | "
          f"moto PCU={args.moto_pcu} | SUMO {ver}")
    print(f"{'seed':>6} {'win(s)':>7} {'moto%':>6} {'veh/h':>8} {'PCU/h':>8}")
    for r in rows:
        print(f"{r['seed']:>6} {r['dur']:>7.1f} {r['moto_share']*100:>5.1f}% "
              f"{r['veh_h']:>8.0f} {r['pcu_h']:>8.0f}")
    pcu = [r["pcu_h"] for r in rows]
    veh = [r["veh_h"] for r in rows]
    med_veh = statistics.median(veh)
    # Robust PCU estimate: the raw discharge is well-measured, but the per-window
    # PCU/h is sensitive to the realised motorcycle fraction in a short window.
    # Converting the median raw discharge by the *calibrated* mix-average PCU
    # (73% moto / 17% car / 7% truck / 3% bus) gives the stable saturation flow.
    avg_pcu_mix = 0.73 * args.moto_pcu + 0.17 * 1.0 + 0.07 * 2.0 + 0.03 * 2.5
    print("-" * 64)
    print(f"median raw discharge : {med_veh:.0f} veh/h/lane  (range {min(veh):.0f}-{max(veh):.0f})")
    print(f"per-window PCU       : median {statistics.median(pcu):.0f} PCU/h  "
          f"(range {min(pcu):.0f}-{max(pcu):.0f}; sample-sensitive via window moto%)")
    print(f"mix-based PCU (robust): {med_veh * avg_pcu_mix:.0f} PCU/h/lane "
          f"= {med_veh:.0f} veh/h x {avg_pcu_mix:.3f} PCU/veh")
    print("=> Section V reports raw ~2450 veh/h/lane and ~1500 PCU/h/lane at the 73% mix.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
