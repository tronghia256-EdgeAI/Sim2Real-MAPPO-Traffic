"""
run_pilot.py
============
W3-4 pilot campaign: 1-seed MAPPO v3 training on the largest network (N3 4x4
grid by default) to benchmark wall-clock cost and verify convergence machinery
before the multi-seed cluster campaign.

Pipeline:
  1. Preflight: lane_groups.json topology re-validation (CHECK 6 invariants).
  2. Training subprocess: experiment/runners/train_ppo.py
     (--algo mappo, 1 seed, time-varying demand_train.rou.xml).
     The trainer dual-saves run_config.json and writes bench.json with
     wall-clock + steps/s instrumentation.
  3. Verification: asserts run_config.json carries env_cfg.version == 1.1.0
     (the schema stamp separating valid data from legacy artifacts).
  4. Evaluation: greedy-policy episode on the HELD-OUT demand
     (sumo_config_eval.sumocfg) with --tripinfo-output; mean travel time and
     mean waiting time parsed from tripinfo.
  5. Reporting: results/paper1_mappo/pilot_<network>/<run_id>/
        pilot_summary.json   consolidated config + bench + eval metrics
        metrics.csv          one row per metric (spreadsheet-friendly)
     plus TensorBoard scalars (pilot_eval/*) when TB is importable.

Usage (from project root):
    python scripts/run_pilot.py                          # N3, 100k pilot steps
    python scripts/run_pilot.py --total-timesteps 20000  # quick mechanics check
    python scripts/run_pilot.py --network n2_corridor --algo ippo
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_ROOT = ROOT / "results" / "paper1_mappo"
REQUIRED_SCHEMA_VERSION = "1.2.0"   # 1.1.0 obs schema + 1.2.0 reward revision


# ---------------------------------------------------------------------------
# Step 1 — preflight
# ---------------------------------------------------------------------------

def preflight(net_dir: Path) -> dict:
    groups_path = net_dir / "lane_groups.json"
    net_path = net_dir / "intersections.net.xml"
    for p in (groups_path, net_path, net_dir / "demand_train.rou.xml",
              net_dir / "demand_eval.rou.xml"):
        if not p.exists():
            raise SystemExit(
                f"[preflight] missing {p} — run scripts/generate_networks.py "
                f"and scripts/generate_demand.py first"
            )

    import xml.etree.ElementTree as ET
    data = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = data["lane_groups"]

    edge_to = {
        e.get("id"): e.get("to")
        for e in ET.parse(net_path).getroot().iter("edge")
        if e.get("function") != "internal"
    }
    owner: dict = {}
    for tls_id, (ga, gb) in groups.items():
        assert ga and gb and not set(ga) & set(gb), f"[preflight] bad groups for {tls_id}"
        for lane in list(ga) + list(gb):
            edge = lane.rsplit("_", 1)[0]
            assert edge_to.get(edge) == tls_id, (
                f"[preflight] lane {lane} approaches {edge_to.get(edge)}, not {tls_id}"
            )
            assert owner.setdefault(lane, tls_id) == tls_id
    print(f"[preflight] OK - {len(groups)} TLS, lane groups topology-consistent")
    return data


# ---------------------------------------------------------------------------
# Step 2 — training subprocess
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace, net_dir: Path) -> Path:
    cmd = [
        sys.executable, str(ROOT / "experiment" / "runners" / "train_ppo.py"),
        "--mode", "train",
        "--algo", args.algo,
        "--sumo-cfg", str(net_dir / "sumo_config.sumocfg"),
        "--lane-groups", str(net_dir / "lane_groups.json"),
        "--seed", str(args.seed),
        "--total-timesteps", str(args.total_timesteps),
        "--rollout-horizon", str(args.rollout_horizon),
        "--min-green-time", str(args.min_green_time),
        "--max-green-time", str(args.max_green_time),
        "--ckpt-dir", "models/mappo",
        "--log-dir", "logs/rl",
    ]
    print(f"[train] $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    run_dir: Path | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"  | {line.rstrip()}")
        m = re.search(r"Training completed\. Run dir: (.+)", line)
        if m:
            run_dir = Path(m.group(1).strip())
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"[train] training subprocess failed (exit {proc.returncode})")
    if run_dir is None:
        raise SystemExit("[train] could not locate run dir in trainer output")
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    print(f"[train] done in {time.time()-t0:.1f}s -> {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Step 3 — schema-version verification
# ---------------------------------------------------------------------------

def verify_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "run_config.json"
    run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    version = run_cfg.get("env_cfg", {}).get("version")
    if version != REQUIRED_SCHEMA_VERSION:
        raise SystemExit(
            f"[verify] {cfg_path} carries env_cfg.version={version!r}, "
            f"required {REQUIRED_SCHEMA_VERSION!r} — refusing to record as paper data"
        )
    print(f"[verify] run_config.json stamped version {version} OK (dual-saved by trainer)")
    return run_cfg


# ---------------------------------------------------------------------------
# Step 4 — held-out evaluation with tripinfo
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    run_dir: Path, net_dir: Path, seed: int, max_steps: int,
    min_green_time: int = 15, max_green_time: int = 60,
) -> dict:
    import numpy as np
    import torch

    from experiment.common.tripinfo import parse_tripinfo_means
    from experiment.runners.train_ppo import Actor, RunningMeanStd
    from src.traffic_env.config import build_default_config, load_lane_groups_json
    from src.traffic_env.envs.multi_agent import MappoTrafficEnv

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    tls_ids, lane_groups = load_lane_groups_json(str(net_dir / "lane_groups.json"))
    # green times must match the trained config — max_green_time scales the
    # green_timer_norm obs feature, so a mismatch silently skews the policy input
    env_cfg = build_default_config(
        sumo_cfg_path=str(net_dir / "sumo_config_eval.sumocfg"),
        gui=False,
        tls_ids=tls_ids,
        manual_lane_groups=lane_groups,
        max_steps=max_steps,
        min_green_time=min_green_time,
        max_green_time=max_green_time,
    )
    tripinfo_path = Path(tempfile.gettempdir()) / f"pilot_tripinfo_{run_dir.name}.xml"
    env = MappoTrafficEnv(
        config=env_cfg, gui=False, use_libsumo=False,
        extra_sumo_args=[
            "--tripinfo-output", str(tripinfo_path),
            "--tripinfo-output.write-unfinished",
        ],
    )

    obs_dim = env_cfg.local_obs_dim
    actor = Actor(obs_dim, 2)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    obs_dict, _ = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    done = False
    with torch.no_grad():
        while not done and steps < max_steps:
            actions = {}
            for tls_id in tls_ids:
                obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
                dist = actor.distribution(torch.as_tensor(obs).unsqueeze(0))
                actions[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())
            obs_dict, reward, terminated, truncated, _ = env.step(actions)
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            steps += 1
            done = any(terminated.values()) or any(truncated.values())
    env.close()

    metrics = parse_tripinfo_means(tripinfo_path)
    metrics["eval_steps"] = steps
    metrics["mean_step_reward"] = total_reward / max(steps, 1)
    metrics["checkpoint"] = ckpt_path.name
    print(f"[eval] {steps} steps on held-out demand | "
          f"travel={metrics['mean_travel_time_s']:.1f}s "
          f"waiting={metrics['mean_waiting_time_s']:.1f}s "
          f"(n={metrics['n_trips']})")
    return metrics


# ---------------------------------------------------------------------------
# Step 5 — reporting
# ---------------------------------------------------------------------------

def write_report(
    args: argparse.Namespace, run_dir: Path, run_cfg: dict, eval_metrics: dict
) -> Path:
    bench = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
    out_dir = RESULTS_ROOT / f"pilot_{args.network}" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pilot": {
            "network": args.network,
            "algo": args.algo,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "min_green_time": args.min_green_time,
            "max_green_time": args.max_green_time,
            "schema_version": run_cfg["env_cfg"]["version"],
            "num_agents": len(run_cfg["tls_ids"]),
        },
        "bench": bench,
        "eval_heldout": eval_metrics,
        "run_dir": str(run_dir),
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flat = {
        "network": args.network, "algo": args.algo, "seed": args.seed,
        "min_green": args.min_green_time, "max_green": args.max_green_time,
        "steps": bench["global_steps"], "wall_clock_s": bench["elapsed_seconds"],
        "mean_sps": bench["mean_steps_per_second"],
        "proj_hours_2M": bench["projected_hours_2M_steps"],
        "proj_hours_2M_x5seeds": bench["projected_hours_2M_x5_seeds"],
        "mean_travel_time_s": eval_metrics["mean_travel_time_s"],
        "mean_waiting_time_s": eval_metrics["mean_waiting_time_s"],
        "p95_waiting_time_s": eval_metrics["p95_waiting_time_s"],
        "n_trips": eval_metrics["n_trips"],
    }
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    # optional TensorBoard scalars (skipped gracefully on broken TF installs)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(out_dir / "tb"))
        for key in ("mean_travel_time_s", "mean_waiting_time_s", "p95_waiting_time_s"):
            tb.add_scalar(f"pilot_eval/{key}", float(eval_metrics[key]), bench["global_steps"])
        tb.add_scalar("pilot_bench/mean_sps", float(bench["mean_steps_per_second"]),
                      bench["global_steps"])
        tb.close()
    except Exception as exc:
        print(f"[report] TensorBoard skipped ({type(exc).__name__})")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=["n1", "n2_corridor", "n3_grid"],
                        default="n3_grid")
    parser.add_argument("--algo", choices=["mappo", "ippo"], default="mappo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--eval-max-steps", type=int, default=1080)
    parser.add_argument("--min-green-time", type=int, default=15)
    parser.add_argument("--max-green-time", type=int, default=60)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="skip training; resume reporting from an existing "
                             "completed run dir (e.g. models/mappo/<run_id>)")
    args = parser.parse_args()

    net_dir = ROOT / "sumo_configs" / "networks" / args.network

    preflight(net_dir)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        print(f"[train] skipped — using existing run dir {run_dir}")
    else:
        run_dir = run_training(args, net_dir)
    run_cfg = verify_run_config(run_dir)
    eval_metrics = evaluate_checkpoint(run_dir, net_dir, seed=args.seed + 1000,
                                       max_steps=args.eval_max_steps,
                                       min_green_time=args.min_green_time,
                                       max_green_time=args.max_green_time)
    out_dir = write_report(args, run_dir, run_cfg, eval_metrics)

    bench = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 64)
    print("PILOT SUMMARY")
    print("=" * 64)
    print(f"  network / algo / seed : {args.network} / {args.algo} / {args.seed}")
    print(f"  steps trained         : {bench['global_steps']}")
    print(f"  wall-clock            : {bench['elapsed_seconds']:.0f}s "
          f"({bench['elapsed_seconds']/3600:.2f}h)")
    print(f"  mean SPS              : {bench['mean_steps_per_second']:.1f}")
    print(f"  projected 2M steps    : {bench['projected_hours_2M_steps']:.1f}h")
    print(f"  projected 2M x 5 seeds: {bench['projected_hours_2M_x5_seeds']:.1f}h")
    print(f"  held-out travel time  : {eval_metrics['mean_travel_time_s']:.1f}s")
    print(f"  held-out waiting time : {eval_metrics['mean_waiting_time_s']:.1f}s")
    print(f"  report                : {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
