"""
reward_ablation.py
==================
Ablation study over MAPPO reward weight configurations.

Tests five configurations against the trained policy evaluated in the medium
SUMO scenario.  Results are saved to:
    results/paper1_mappo/ablation/reward_ablation_results.csv

Usage (from project root):
    python experiment/ablation/reward_ablation.py \\
        --checkpoint models/mappo/20260418_215140/best_model.pt \\
        --sumo-cfg   sumo_configs/evaluation/medium/sumo_config.sumocfg \\
        --episodes   5 \\
        --seeds      42 123 456
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.config import RewardConfig, build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from src.core.policy_loader import PolicyLoader


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    name: str
    description: str
    weights: Dict[str, float] = field(default_factory=dict)


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        name="full",
        description="Full reward (all components — baseline)",
        weights={
            "queue":             -1.0,
            "pressure":          -0.5,
            "throughput":        +1.0,
            "switch_penalty":    -0.1,
            "low_speed_penalty": -0.2,
            "waiting_time":      -0.3,
        },
    ),
    AblationConfig(
        name="no_pressure",
        description="Remove pressure term (ablate PRESSLIGHT signal)",
        weights={
            "queue":             -1.0,
            "pressure":           0.0,
            "throughput":        +1.0,
            "switch_penalty":    -0.1,
            "low_speed_penalty": -0.2,
            "waiting_time":      -0.3,
        },
    ),
    AblationConfig(
        name="no_throughput",
        description="Remove throughput term (ablate cooperative signal)",
        weights={
            "queue":             -1.0,
            "pressure":          -0.5,
            "throughput":         0.0,
            "switch_penalty":    -0.1,
            "low_speed_penalty": -0.2,
            "waiting_time":      -0.3,
        },
    ),
    AblationConfig(
        name="queue_only",
        description="Queue penalty only (minimal reward)",
        weights={
            "queue":             -1.0,
            "pressure":           0.0,
            "throughput":         0.0,
            "switch_penalty":     0.0,
            "low_speed_penalty":  0.0,
            "waiting_time":       0.0,
        },
    ),
    AblationConfig(
        name="vision_proxy",
        description="Vision-deployment weights (disable noisy speed signal)",
        weights={
            "queue":             -1.0,
            "pressure":          -0.5,
            "throughput":        +0.5,
            "switch_penalty":    -0.1,
            "low_speed_penalty":  0.0,
            "waiting_time":      -0.3,
        },
    ),
]


# ---------------------------------------------------------------------------
# Evaluation helpers (copied from train_ppo.py to avoid circular imports)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    def __init__(self, shape: tuple) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


@dataclass
class EpisodeResult:
    ablation_name: str
    seed: int
    total_reward: float
    episode_length: int
    throughput: float
    co2_total_mg: float


def _run_episode(
    env: MappoTrafficEnv,
    actor: torch.nn.Module,
    obs_rms: RunningMeanStd,
    tls_ids: List[str],
    seed: int,
    max_steps: int,
    device: torch.device,
) -> Tuple[float, int, float, float]:
    obs_dict, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    throughput = 0.0
    co2 = 0.0
    prev_passed = 0.0
    done = False

    while not done and steps < max_steps:
        actions: Dict[str, int] = {}
        for tls_id in tls_ids:
            obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
            actions[tls_id] = action

        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
        steps += 1

        for payload in next_info.values():
            co2 += float(payload.get("co2_mg_per_s", 0.0))

        first_info = next(iter(next_info.values()), {})
        cur_passed = float(first_info.get("current_passed", prev_passed))
        throughput += max(cur_passed - prev_passed, 0.0)
        prev_passed = cur_passed

        done = any(terminated.values()) or any(truncated.values())
        obs_dict = next_obs

    return total_reward, steps, throughput, co2


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds: List[int] = list(args.seeds)
    out_dir = _ROOT / "results" / "paper1_mappo" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "reward_ablation_results.csv"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    results: List[EpisodeResult] = []

    for ablation in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation.name}")
        print(f"  {ablation.description}")
        print(f"  weights: {ablation.weights}")
        print(f"{'='*60}")

        reward_cfg = RewardConfig(weights=ablation.weights)
        env_cfg = build_default_config(
            sumo_cfg_path=args.sumo_cfg,
            gui=False,
            max_steps=args.max_steps,
        )
        env_cfg = env_cfg._replace(reward_config=reward_cfg) if hasattr(env_cfg, "_replace") else env_cfg

        env = MappoTrafficEnv(
            config=env_cfg,
            gui=False,
            use_libsumo=True,
            no_step_log=True,
            print_warnings=False,
            add_default_flags=True,
        )

        try:
            obs_dict, _ = env.reset(seed=seeds[0])
            tls_ids = list(env_cfg.tls_ids)
            obs_dim = env_cfg.local_obs_dim
            action_dim = getattr(env, "action_dim", 2)

            from experiment.runners.train_ppo import Actor
            actor = Actor(obs_dim, action_dim).to(device)
            actor.load_state_dict(ckpt["actor_state_dict"])
            actor.eval()

            obs_rms = RunningMeanStd((obs_dim,))
            if "obs_rms" in ckpt:
                obs_rms.load_state_dict(ckpt["obs_rms"])

            for seed in seeds:
                reward, length, tp, co2 = _run_episode(
                    env, actor, obs_rms, tls_ids,
                    seed=seed, max_steps=args.max_steps, device=device,
                )
                res = EpisodeResult(
                    ablation_name=ablation.name,
                    seed=seed,
                    total_reward=reward,
                    episode_length=length,
                    throughput=tp,
                    co2_total_mg=co2,
                )
                results.append(res)
                print(f"  seed={seed:>4} | reward={reward:>8.3f} | len={length:>4} | tp={tp:>6.0f} | co2={co2:>8.1f}mg")
        finally:
            env.close()

    # --- Write CSV ---
    fieldnames = list(EpisodeResult.__dataclass_fields__.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # --- Print summary table ---
    print(f"\n{'Ablation':<20} {'Mean Reward':>12} {'Std':>7} {'Mean TP':>10} {'Mean CO2':>12}")
    print("-" * 65)
    for ablation in ABLATION_CONFIGS:
        subset = [r for r in results if r.ablation_name == ablation.name]
        rewards = [r.total_reward for r in subset]
        tps = [r.throughput for r in subset]
        co2s = [r.co2_total_mg for r in subset]
        print(
            f"{ablation.name:<20} {np.mean(rewards):>12.3f} {np.std(rewards):>7.3f} "
            f"{np.mean(tps):>10.0f} {np.mean(co2s):>12.1f}"
        )

    print(f"\nResults saved to: {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward component ablation study")
    parser.add_argument("--checkpoint", type=str,
                        default="models/mappo/20260418_215140/best_model.pt")
    parser.add_argument("--sumo-cfg", type=str,
                        default="sumo_configs/evaluation/medium/sumo_config.sumocfg")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
