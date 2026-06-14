"""
eval_noise_robustness.py
========================
Paper VI-E: sensing-noise robustness sweep for a trained vision-proxy policy.

Loads a PROXY checkpoint (the deployable arm) and re-evaluates it on held-out
demand under the calibrated sensing-noise model (src/traffic_env/components/
obs_noise.py), sweeping the noise envelope scale {0, 0.5, 1, 2} plus the two
structural failure modes (single-camera dropout, one-step pipeline delay).

The noise is injected on the RAW [0,1] per-agent observation BEFORE obs_rms
normalization — exactly where ε enters in deployment: the camera produces a
noisy proxy feature vector, which the policy then normalizes with its stored
training statistics. Training itself is noiseless (ε = 0); this is an eval-only
study and never touches the campaign checkpoints' weights.

⚠ The numbers are only paper-ready once NoiseConfig.is_calibrated is True, i.e.
class_flip_rate and occupancy_bias_sigma have been replaced with measured
detector statistics. The runner prints a loud warning otherwise but still runs
(useful for plumbing / scale-curve shape during development).

Usage (from project root, after a proxy checkpoint exists):
    python experiment/runners/eval_noise_robustness.py \
        --checkpoint results/paper1_mappo/<campaign>/models/n3_grid_mappo_proxy_seed42/<run>/best_model.pt \
        --network n3_grid --seeds 42 123 456 789 1337
    # -> results/paper1_mappo/noise_robustness/<ckpt_tag>/{summary.json, sweep.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_ROOT = ROOT / "results" / "paper1_mappo" / "noise_robustness"
DEFAULT_SCALES = (0.0, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Conditions to evaluate
# ---------------------------------------------------------------------------

def build_conditions(
    scales: Sequence[float],
    max_lanes: int,
    base_kwargs: Dict,
) -> List[Dict]:
    """One NoiseConfig-kwargs dict per evaluated condition.

    Scale sweep (Gaussian/feature noise scaled by the envelope) + two structural
    conditions evaluated at the nominal 1x envelope: camera dropout of approach 0
    and a one-step observation delay.
    """
    conditions: List[Dict] = []
    for s in scales:
        conditions.append({"label": f"scale_{s:g}", "scale": s, **base_kwargs})
    conditions.append({
        "label": "dropout_approach0", "scale": 1.0,
        "camera_dropout_approaches": (0,), **base_kwargs,
    })
    conditions.append({
        "label": "delay_1step", "scale": 1.0, "delay_steps": 1, **base_kwargs,
    })
    return conditions


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    *,
    actor,
    obs_rms,
    env_factory,
    tls_ids: Sequence[str],
    noise_cfg,
    seeds: Sequence[int],
    max_steps: int,
) -> Dict[str, float]:
    import numpy as np
    import torch

    from experiment.common.tripinfo import parse_tripinfo_means
    from src.traffic_env.components.obs_noise import SensingNoiseModel

    travel, waiting, p95, ntrips = [], [], [], []
    for seed in seeds:
        env, tripinfo_path = env_factory(seed)
        noise = SensingNoiseModel(
            config=noise_cfg,
            max_lanes_per_tls=env.config.max_lanes_per_tls,
            lane_feature_dim=env.config.lane_feature_dim,
            tls_feature_dim=env.config.observation.tls_feature_dim,
            seed=seed,
        )
        obs_dict, _ = env.reset(seed=seed)
        steps, done = 0, False
        with torch.no_grad():
            while not done and steps < max_steps:
                actions = {}
                for tls_id in tls_ids:
                    raw = np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1)
                    raw = noise.apply(raw, agent_id=tls_id)   # ε before normalization
                    obs = obs_rms.normalize(raw)
                    dist = actor.distribution(torch.as_tensor(obs).unsqueeze(0))
                    actions[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                steps += 1
                done = any(terminated.values()) or any(truncated.values())
        env.close()

        m = parse_tripinfo_means(tripinfo_path)
        travel.append(m["mean_travel_time_s"])
        waiting.append(m["mean_waiting_time_s"])
        p95.append(m["p95_waiting_time_s"])
        ntrips.append(m["n_trips"])

    return {
        "mean_travel_time_s": float(np.mean(travel)),
        "std_travel_time_s": float(np.std(travel)),
        "mean_waiting_time_s": float(np.mean(waiting)),
        "p95_waiting_time_s": float(np.mean(p95)),
        "n_trips": float(np.mean(ntrips)),
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import numpy as np
    import torch

    from experiment.runners.train_ppo import Actor, RunningMeanStd
    from src.traffic_env.components.obs_noise import NoiseConfig
    from src.traffic_env.config import build_default_config, load_lane_groups_json
    from src.traffic_env.envs.multi_agent import MappoTrafficEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--network", default="n3_grid",
                        choices=["n1", "n2_corridor", "n3_grid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--eval-max-steps", type=int, default=1080)
    parser.add_argument("--min-green-time", type=int, default=15)
    parser.add_argument("--max-green-time", type=int, default=60)
    args = parser.parse_args()

    net_dir = ROOT / "sumo_configs" / "networks" / args.network
    tls_ids, lane_groups = load_lane_groups_json(str(net_dir / "lane_groups.json"))

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # resolve obs_mode from the checkpoint's run_config; noise robustness is a
    # PROXY-arm study — refuse a privileged checkpoint rather than silently mixing
    run_cfg_path = ckpt_path.parent / "run_config.json"
    obs_mode = "proxy"
    if run_cfg_path.exists():
        obs_mode = (json.loads(run_cfg_path.read_text(encoding="utf-8"))
                    .get("env_cfg", {}).get("observation", {}).get("obs_mode", "proxy"))
    if obs_mode != "proxy":
        raise SystemExit(
            f"[noise] checkpoint obs_mode={obs_mode!r}; the VI-E robustness study "
            f"applies to the deployable PROXY arm only"
        )

    def env_factory(seed: int):
        env_cfg = build_default_config(
            sumo_cfg_path=str(net_dir / "sumo_config_eval.sumocfg"),
            gui=False, tls_ids=tls_ids, manual_lane_groups=lane_groups,
            max_steps=args.eval_max_steps, obs_mode="proxy",
            min_green_time=args.min_green_time, max_green_time=args.max_green_time,
        )
        tripinfo_path = Path(tempfile.gettempdir()) / f"noise_{ckpt_path.parent.name}_{seed}.xml"
        env = MappoTrafficEnv(
            config=env_cfg, gui=False, use_libsumo=False,
            extra_sumo_args=["--tripinfo-output", str(tripinfo_path),
                             "--tripinfo-output.write-unfinished"],
        )
        return env, tripinfo_path

    # build actor from the checkpoint's own obs dim
    probe_cfg = build_default_config(
        sumo_cfg_path="x.sumocfg", tls_ids=tls_ids,
        manual_lane_groups=lane_groups, obs_mode="proxy",
    )
    obs_dim = probe_cfg.local_obs_dim
    actor = Actor(obs_dim, 2)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    noise_cfg_path = ROOT / "configs" / "noise_config.json"
    if noise_cfg_path.exists():
        calib = NoiseConfig.from_json(noise_cfg_path)
        print(f"[noise] loaded calibrated config from {noise_cfg_path}")
    else:
        calib = NoiseConfig()
    if not calib.is_calibrated:
        print("⚠ [noise] NoiseConfig placeholders (class_flip_rate, "
              "occupancy_bias_sigma) are UNCALIBRATED — results are NOT paper-ready "
              "until set from measured detector statistics. Run "
              "experiment/runners/calibrate_noise.py to produce configs/noise_config.json.")

    conditions = build_conditions(args.scales, probe_cfg.max_lanes_per_tls, base_kwargs={
        "queue_calib_sigma": calib.queue_calib_sigma,
        "speed_sigma_mps": calib.speed_sigma_mps,
        "pressure_calib_sigma": calib.pressure_calib_sigma,
        "class_flip_rate": calib.class_flip_rate,
        "occupancy_bias_sigma": calib.occupancy_bias_sigma,
    })

    rows: List[Dict] = []
    for cond in conditions:
        label = cond.pop("label")
        noise_cfg = NoiseConfig(**cond)
        res = evaluate_condition(
            actor=actor, obs_rms=obs_rms, env_factory=env_factory,
            tls_ids=tls_ids, noise_cfg=noise_cfg, seeds=args.seeds,
            max_steps=args.eval_max_steps,
        )
        res["condition"] = label
        rows.append(res)
        print(f"[noise] {label:<20} travel={res['mean_travel_time_s']:.1f}s "
              f"wait={res['mean_waiting_time_s']:.1f}s p95={res['p95_waiting_time_s']:.1f}s")

    out_dir = RESULTS_ROOT / ckpt_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "network": args.network,
        "obs_mode": obs_mode,
        "calibrated": calib.is_calibrated,
        "seeds": args.seeds,
        "conditions": rows,
    }, indent=2), encoding="utf-8")
    with open(out_dir / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "mean_travel_time_s", "std_travel_time_s",
            "mean_waiting_time_s", "p95_waiting_time_s", "n_trips", "n_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"[noise] wrote {out_dir}")


if __name__ == "__main__":
    main()
