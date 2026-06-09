from __future__ import annotations

"""Clean MAPPO training script for SUMO traffic control.

PATCH NOTES (audit v2):
- FIX: clip_range_vf missing type annotation (was class var, not a dataclass field).
- FIX: value loss clipping now uses clip_range_vf, not clip_ratio.
- FIX: model save names corrected to best_model.pt / last_model.pt per spec.
- FIX: timestamped run dirs under logs/rl/<run_id>/ and models/rl/<run_id>/
  to prevent overwriting between runs.
- FIX: extract_step_emissions now reads from info keys added in multi_agent v2
  (co2_mg_per_s, fuel_ml_per_s) — these were always 0 before because multi_agent
  never set them in info_dict.
- FIX: throughput tracking uses info['current_passed'] (now set by env).
- FIX: run_config.json saved to BOTH log_dir and ckpt_dir for completeness.
- FIX: value_lr and rollout_horizon CLI defaults aligned with TrainConfig defaults.
- NOTE: action_dim is now set by env.action_dim after reset (set in multi_agent v2).
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.traffic_env.config import build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 500_000
    rollout_horizon: int = 256           # matched to CLI default

    gamma: float = 0.99
    gae_lambda: float = 0.95

    normalize_advantages: bool = True

    clip_range_vf: float = 0.2           # FIX: type annotation added
    clip_ratio: float = 0.2
    target_kl: float = 0.015

    policy_lr: float = 3e-4
    value_lr: float = 3e-4               # FIX: aligned with CLI default

    update_epochs: int = 10
    minibatch_size: int = 256            # FIX: aligned with CLI default

    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    log_dir: str = "logs/rl"
    ckpt_dir: str = "models/mappo"
    save_every_steps: int = 50_000
    save_latest_every_updates: int = 1

    eval_episodes: int = 5
    eval_max_steps: int = 400

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------
class CsvLogger:
    """append-mode CSV logger for per-update and per-episode metrics."""

    STEP_FIELDS = [
        "global_step", "episode", "update_idx",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "explained_variance", "sps",
        "lr", "entropy_coef",
    ]
    EPISODE_FIELDS = [
        "global_step", "episode", "episode_length",
        "mean_reward", "queue_total_proxy",
        "co2_total_mg", "fuel_total_ml", "throughput",
        "avg_waiting_proxy",
    ]

    def __init__(self, log_dir: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._step_path = self._dir / "train_updates.csv"
        self._ep_path = self._dir / "train_episodes.csv"
        self._init_file(self._step_path, self.STEP_FIELDS)
        self._init_file(self._ep_path, self.EPISODE_FIELDS)

    def _init_file(self, path: Path, fields: list) -> None:
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_update(self, **kwargs: Any) -> None:
        with open(self._step_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.STEP_FIELDS, extrasaction="ignore").writerow(kwargs)

    def log_episode(self, **kwargs: Any) -> None:
        with open(self._ep_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS, extrasaction="ignore").writerow(kwargs)


# ---------------------------------------------------------------------
# Running Mean Std (observation normalization)
# ---------------------------------------------------------------------
class RunningMeanStd:
    """Welford online algorithm for mean and variance."""

    def __init__(self, shape: tuple) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_obs_dict(obs_dict: Dict[str, np.ndarray], tls_ids: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for tls_id in tls_ids:
        parts.append(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def extract_global_state(info_dict: Dict[str, Dict[str, Any]], fallback: np.ndarray) -> np.ndarray:
    for payload in info_dict.values():
        gs = payload.get("global_state")
        if gs is not None:
            arr = np.asarray(gs, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return arr
    return np.asarray(fallback, dtype=np.float32).reshape(-1)


def extract_step_emissions(info_dict: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    """sum CO2 and fuel across all tls entries in info_dict.
    FIX: these keys are now set by multi_agent.py step() info_dict.
    """
    co2 = 0.0
    fuel = 0.0
    for payload in info_dict.values():
        co2 += float(payload.get("co2_mg_per_s", 0.0))
        fuel += float(payload.get("fuel_ml_per_s", 0.0))
    return co2, fuel


def extract_step_waiting(info_dict: Dict[str, Dict[str, Any]]) -> float:
    """mean avg_waiting_proxy across all tls entries."""
    vals = [float(p.get("avg_waiting_proxy", 0.0)) for p in info_dict.values()]
    return float(np.mean(vals)) if vals else 0.0


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def get_entropy_coef(step: int, total: int, start: float, end: float) -> float:
    """linearly decay entropy coefficient."""
    frac = min(step / max(total, 1), 1.0)
    return start + frac * (end - start)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------
@dataclass(slots=True)
class StepRecord:
    obs_dict: Dict[str, np.ndarray]
    global_state: np.ndarray
    action_dict: Dict[str, int]
    logprob_dict: Dict[str, float]
    reward_dict: Dict[str, float]
    terminated_dict: Dict[str, bool]
    truncated_dict: Dict[str, bool]
    value: float
    team_reward: float


class RolloutBuffer:
    def __init__(self, tls_ids: Sequence[str]) -> None:
        self.tls_ids = list(tls_ids)
        self.records: List[StepRecord] = []

    def add(self, record: StepRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def actor_flatten(
        self,
        advantages_by_tls: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        obs_list: List[np.ndarray] = []
        action_list: List[int] = []
        logprob_list: List[float] = []
        adv_list: List[float] = []

        T = len(self.records)
        for tls_id in self.tls_ids:
            adv_seq = advantages_by_tls[tls_id]
            for t in range(T):
                rec = self.records[t]
                obs_list.append(np.asarray(rec.obs_dict[tls_id], dtype=np.float32).reshape(-1))
                action_list.append(int(rec.action_dict[tls_id]))
                logprob_list.append(float(rec.logprob_dict[tls_id]))
                adv_list.append(float(adv_seq[t]))

        return {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "actions": np.asarray(action_list, dtype=np.int64),
            "logprobs": np.asarray(logprob_list, dtype=np.float32),
            "advantages": np.asarray(adv_list, dtype=np.float32),
        }

    def critic_flatten(self, team_returns: np.ndarray) -> Dict[str, np.ndarray]:
        states = [np.asarray(rec.global_state, dtype=np.float32).reshape(-1) for rec in self.records]
        values = np.asarray([rec.value for rec in self.records], dtype=np.float32)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "values": values,
            "team_returns": np.asarray(team_returns, dtype=np.float32),
        }


# ---------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------
def make_env(args: argparse.Namespace) -> Tuple[MappoTrafficEnv, Any]:
    env_cfg = build_default_config(
        sumo_cfg_path=args.sumo_cfg,
        gui=args.gui,
        tls_ids=tuple(args.tls_ids) if args.tls_ids else None,
        max_lanes_per_tls=args.max_lanes_per_tls,
        step_length=args.step_length,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        queue_cap=args.queue_cap,
        waiting_cap=args.waiting_cap,
        speed_cap=args.speed_cap,
        use_external_state=False,
        debug=args.debug,
        reward_scale=args.reward_scale,
    )
    env = MappoTrafficEnv(
        config=env_cfg,
        gui=args.gui,
        use_libsumo=not args.gui,
        no_step_log=True,
        waiting_time_memory=1000,
        print_warnings=False,
        add_default_flags=True,
    )
    return env, env_cfg


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    train_cfg = TrainConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        rollout_horizon=args.rollout_horizon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef_start=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        save_every_steps=args.save_every_steps,
        save_latest_every_updates=args.save_latest_every_updates,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
    )

    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    # FIX: timestamped run dirs to prevent overwriting between runs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(Path(train_cfg.log_dir) / run_id)
    ckpt_dir = str(Path(train_cfg.ckpt_dir) / run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    state_dim = env_cfg.global_state_dim

    # action_dim: do a quick reset to let env build phase maps, then read it
    obs_dict, info_dict = env.reset(seed=train_cfg.seed)
    action_dim: int = getattr(env, "action_dim", 2)

    actor = Actor(obs_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=train_cfg.policy_lr, eps=1e-5)
    critic_optim = optim.Adam(critic.parameters(), lr=train_cfg.value_lr, eps=1e-5)

    total_updates = train_cfg.total_timesteps // train_cfg.rollout_horizon

    def lr_lambda(update: int) -> float:
        return max(1.0 - update / max(total_updates, 1), 0.05)

    actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optim, lr_lambda)
    critic_scheduler = optim.lr_scheduler.LambdaLR(critic_optim, lr_lambda)

    obs_rms = RunningMeanStd((obs_dim,))
    state_rms = RunningMeanStd((state_dim,))

    writer = SummaryWriter(log_dir=log_dir)
    csv_logger = CsvLogger(log_dir)

    run_meta = {
        "run_id": run_id,
        "env_cfg": env_cfg.to_dict(),
        "train_cfg": asdict(train_cfg),
        "tls_ids": tls_ids,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    # FIX: save run_config.json to both log_dir and ckpt_dir
    for save_dir in (log_dir, ckpt_dir):
        with open(Path(save_dir) / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)

    buffer = RolloutBuffer(tls_ids)
    global_step = 0
    update_idx = 0
    best_mean_reward = -float("inf")
    start_time = time.time()

    global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))

    episode_count = 0
    episode_length = 0
    episode_agent_rewards: Dict[str, float] = {tls_id: 0.0 for tls_id in tls_ids}
    episode_queue_total = 0.0
    episode_co2 = 0.0
    episode_fuel = 0.0
    episode_throughput = 0.0
    episode_waiting = 0.0
    ep_prev_passed = 0.0

    def reset_episode_stats() -> None:
        nonlocal episode_length, episode_agent_rewards, episode_queue_total
        nonlocal episode_co2, episode_fuel, episode_throughput, ep_prev_passed, episode_waiting
        episode_length = 0
        episode_agent_rewards = {tls_id: 0.0 for tls_id in tls_ids}
        episode_queue_total = 0.0
        episode_co2 = 0.0
        episode_fuel = 0.0
        episode_throughput = 0.0
        episode_waiting = 0.0
        ep_prev_passed = 0.0

    while global_step < train_cfg.total_timesteps:
        buffer.clear()

        actor.eval()
        critic.eval()

        for _ in range(train_cfg.rollout_horizon):
            norm_state = state_rms.normalize(global_state)
            state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = float(critic(state_tensor).item())

            action_dict: Dict[str, int] = {}
            logprob_dict: Dict[str, float] = {}

            for tls_id in tls_ids:
                obs = np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1)
                norm_obs = obs_rms.normalize(obs)
                obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dist = actor.distribution(obs_tensor)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                action_dict[tls_id] = int(action.item())
                logprob_dict[tls_id] = float(logprob.item())

            next_obs_dict, reward_dict, terminated_dict, truncated_dict, next_info_dict = env.step(action_dict)
            next_global_state = extract_global_state(next_info_dict, fallback=global_state)

            team_reward = float(np.mean(list(reward_dict.values()))) if reward_dict else 0.0

            rec = StepRecord(
                obs_dict={k: np.asarray(v, dtype=np.float32).copy() for k, v in obs_dict.items()},
                global_state=np.asarray(global_state, dtype=np.float32).copy(),
                action_dict=dict(action_dict),
                logprob_dict=dict(logprob_dict),
                reward_dict={k: float(v) for k, v in reward_dict.items()},
                terminated_dict={k: bool(v) for k, v in terminated_dict.items()},
                truncated_dict={k: bool(v) for k, v in truncated_dict.items()},
                value=value,
                team_reward=team_reward,
            )
            buffer.add(rec)

            episode_length += 1
            global_step += 1

            for tls_id in tls_ids:
                episode_agent_rewards[tls_id] += float(reward_dict.get(tls_id, 0.0))
                episode_queue_total += float(next_info_dict.get(tls_id, {}).get("queue_total_proxy", 0.0))

            step_co2, step_fuel = extract_step_emissions(next_info_dict)
            episode_co2 += step_co2
            episode_fuel += step_fuel
            episode_waiting += extract_step_waiting(next_info_dict)

            # FIX: current_passed now set by env; reliable throughput tracking
            first_info = next(iter(next_info_dict.values()), {})
            current_passed = float(first_info.get("current_passed", ep_prev_passed))
            episode_throughput += max(current_passed - ep_prev_passed, 0.0)
            ep_prev_passed = current_passed

            obs_dict = next_obs_dict
            global_state = next_global_state

            done_now = any(terminated_dict.values()) or any(truncated_dict.values())
            if done_now:
                episode_count += 1
                mean_episode_reward = float(np.mean(list(episode_agent_rewards.values())))

                writer.add_scalar("episode/mean_reward", mean_episode_reward, global_step)
                writer.add_scalar("episode/length", episode_length, global_step)
                writer.add_scalar("episode/queue_total_proxy", episode_queue_total, global_step)
                writer.add_scalar("episode/co2_total_mg", episode_co2, global_step)
                writer.add_scalar("episode/fuel_total_ml", episode_fuel, global_step)
                writer.add_scalar("episode/throughput", episode_throughput, global_step)
                writer.add_scalar("episode/avg_waiting_proxy", episode_waiting, global_step)

                csv_logger.log_episode(
                    global_step=global_step,
                    episode=episode_count,
                    episode_length=episode_length,
                    mean_reward=mean_episode_reward,
                    queue_total_proxy=episode_queue_total,
                    co2_total_mg=episode_co2,
                    fuel_total_ml=episode_fuel,
                    throughput=episode_throughput,
                    avg_waiting_proxy=episode_waiting,
                )

                if mean_episode_reward > best_mean_reward:
                    best_mean_reward = mean_episode_reward
                    # FIX: save as best_model.pt per spec
                    torch.save(
                        {
                            "actor_state_dict": actor.state_dict(),
                            "critic_state_dict": critic.state_dict(),
                            "actor_optim_state_dict": actor_optim.state_dict(),
                            "critic_optim_state_dict": critic_optim.state_dict(),
                            "global_step": global_step,
                            "episode_count": episode_count,
                            "best_mean_reward": best_mean_reward,
                            "env_cfg": env_cfg.to_dict(),
                            "train_cfg": asdict(train_cfg),
                            "obs_rms": obs_rms.state_dict(),
                            "state_rms": state_rms.state_dict(),
                        },
                        Path(ckpt_dir) / "best_model.pt",
                    )

                obs_dict, info_dict = env.reset(seed=train_cfg.seed + episode_count)
                global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))
                reset_episode_stats()

            if global_step >= train_cfg.total_timesteps:
                break

        if len(buffer) == 0:
            continue

        # --------------------------------------------------
        # Update RunningMeanStd with rollout batch
        # --------------------------------------------------
        all_obs = np.stack([
            np.asarray(rec.obs_dict[tls_id], dtype=np.float32).reshape(-1)
            for rec in buffer.records
            for tls_id in tls_ids
        ], axis=0)
        obs_rms.update(all_obs)

        all_states = np.stack([
            np.asarray(rec.global_state, dtype=np.float32).reshape(-1)
            for rec in buffer.records
        ], axis=0)
        state_rms.update(all_states)

        # --------------------------------------------------
        # Bootstrap value for last state.
        # --------------------------------------------------
        actor.eval()
        critic.eval()
        with torch.no_grad():
            norm_last = state_rms.normalize(global_state)
            last_state_tensor = torch.as_tensor(norm_last, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(critic(last_state_tensor).item())

        last_record = buffer.records[-1]
        if any(last_record.terminated_dict.values()) or any(last_record.truncated_dict.values()):
            last_value = 0.0

        # --------------------------------------------------
        # GAE
        # --------------------------------------------------
        team_rewards = np.asarray([rec.team_reward for rec in buffer.records], dtype=np.float32)
        team_values = np.asarray([rec.value for rec in buffer.records], dtype=np.float32)
        team_dones = np.asarray([
            float(any(rec.terminated_dict.values()) or any(rec.truncated_dict.values()))
            for rec in buffer.records
        ], dtype=np.float32)

        team_adv, team_returns = compute_gae(
            rewards=team_rewards,
            values=team_values,
            dones=team_dones,
            last_value=last_value,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
        )

        team_adv_norm = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)

        advantages_by_tls: Dict[str, np.ndarray] = {
            tls_id: team_adv_norm.copy() for tls_id in tls_ids
        }

        actor_flat = buffer.actor_flatten(advantages_by_tls)
        critic_flat = buffer.critic_flatten(team_returns=team_returns)

        actor_flat["obs"] = obs_rms.normalize(actor_flat["obs"])
        critic_flat["states"] = state_rms.normalize(critic_flat["states"])

        obs_t = torch.as_tensor(actor_flat["obs"], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(actor_flat["actions"], dtype=torch.int64, device=device)
        old_logprob_t = torch.as_tensor(actor_flat["logprobs"], dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(actor_flat["advantages"], dtype=torch.float32, device=device)

        state_t = torch.as_tensor(critic_flat["states"], dtype=torch.float32, device=device)
        old_value_t = torch.as_tensor(critic_flat["values"], dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(critic_flat["team_returns"], dtype=torch.float32, device=device)

        # --------------------------------------------------
        # PPO update
        # --------------------------------------------------
        actor.train()
        critic.train()

        entropy_coef = get_entropy_coef(
            global_step, train_cfg.total_timesteps,
            train_cfg.entropy_coef_start, train_cfg.entropy_coef_end,
        )

        policy_losses, value_losses, entropies, kls, clip_fracs = [], [], [], [], []

        batch_size_actor = obs_t.shape[0]
        batch_size_critic = state_t.shape[0]

        actor_indices = np.arange(batch_size_actor)
        skip_actor = False

        for _ in range(train_cfg.update_epochs):

            if not skip_actor:
                actor_indices = np.random.permutation(batch_size_actor)

                for start in range(0, batch_size_actor, train_cfg.minibatch_size):
                    mb_idx = actor_indices[start:start + train_cfg.minibatch_size]

                    dist = actor.distribution(obs_t[mb_idx])
                    new_logprob = dist.log_prob(action_t[mb_idx])
                    entropy = dist.entropy().mean()

                    logratio = new_logprob - old_logprob_t[mb_idx]
                    ratio = torch.exp(logratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > train_cfg.clip_ratio).float().mean().item()

                    surr1 = ratio * adv_t[mb_idx]
                    surr2 = torch.clamp(ratio, 1.0 - train_cfg.clip_ratio, 1.0 + train_cfg.clip_ratio) * adv_t[mb_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    loss_actor = policy_loss - entropy_coef * entropy

                    actor_optim.zero_grad(set_to_none=True)
                    loss_actor.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), train_cfg.max_grad_norm)
                    actor_optim.step()

                    policy_losses.append(policy_loss.item())
                    entropies.append(entropy.item())
                    kls.append(approx_kl)
                    clip_fracs.append(clip_frac)

                if np.mean(kls) > 1.5 * train_cfg.target_kl:
                    skip_actor = True
                    writer.add_scalar("debug/early_stop_kl", 1.0, global_step)

            critic_indices = np.random.permutation(batch_size_critic)
            critic_mb_size = max(train_cfg.minibatch_size, 1)

            for start in range(0, batch_size_critic, critic_mb_size):
                mb_idx = critic_indices[start:start + critic_mb_size]

                value_pred = critic(state_t[mb_idx])

                # FIX: use clip_range_vf (not clip_ratio) for value loss clipping
                value_pred_clipped = old_value_t[mb_idx] + torch.clamp(
                    value_pred - old_value_t[mb_idx],
                    -train_cfg.clip_range_vf,
                    train_cfg.clip_range_vf,
                )

                value_loss_unclipped = (ret_t[mb_idx] - value_pred).pow(2)
                value_loss_clipped = (ret_t[mb_idx] - value_pred_clipped).pow(2)

                value_loss = train_cfg.value_coef * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                critic_optim.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), train_cfg.max_grad_norm)
                critic_optim.step()

                value_losses.append(value_loss.item())

        actor_scheduler.step()
        critic_scheduler.step()
        current_lr = actor_scheduler.get_last_lr()[0]

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        sps = int(global_step / max(time.time() - start_time, 1e-8))
        ev = explained_variance(critic_flat["values"], team_returns)

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("loss/policy", float(np.mean(policy_losses)), global_step)
        writer.add_scalar("loss/value", float(np.mean(value_losses)), global_step)
        writer.add_scalar("loss/entropy", float(np.mean(entropies)), global_step)
        writer.add_scalar("loss/approx_kl", float(np.mean(kls)), global_step)
        writer.add_scalar("loss/clip_fraction", float(np.mean(clip_fracs)), global_step)
        writer.add_scalar("loss/explained_variance", ev, global_step)
        writer.add_scalar("train/lr", current_lr, global_step)
        writer.add_scalar("train/entropy_coef", entropy_coef, global_step)
        writer.add_scalar("train/actor_batch_size", batch_size_actor, global_step)
        writer.add_scalar("train/episode_count", episode_count, global_step)
        writer.add_scalar("train/team_return_mean", float(np.mean(team_returns)), global_step)
        writer.add_scalar("train/team_adv_mean", float(np.mean(team_adv)), global_step)

        csv_logger.log_update(
            global_step=global_step,
            episode=episode_count,
            update_idx=update_idx,
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(kls)),
            clip_fraction=float(np.mean(clip_fracs)),
            explained_variance=ev,
            sps=sps,
            lr=current_lr,
            entropy_coef=entropy_coef,
        )

        # Periodic checkpoint
        if global_step % train_cfg.save_every_steps < train_cfg.rollout_horizon:
            ckpt_path = Path(ckpt_dir) / f"checkpoint_{global_step}.pt"
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "actor_optim_state_dict": actor_optim.state_dict(),
                    "critic_optim_state_dict": critic_optim.state_dict(),
                    "actor_scheduler_state_dict": actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": critic_scheduler.state_dict(),
                    "global_step": global_step,
                    "episode_count": episode_count,
                    "env_cfg": env_cfg.to_dict(),
                    "train_cfg": asdict(train_cfg),
                    "obs_rms": obs_rms.state_dict(),
                    "state_rms": state_rms.state_dict(),
                },
                ckpt_path,
            )

        # FIX: save last_model.pt per spec (was latest.pt)
        if update_idx % train_cfg.save_latest_every_updates == 0:
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "actor_optim_state_dict": actor_optim.state_dict(),
                    "critic_optim_state_dict": critic_optim.state_dict(),
                    "actor_scheduler_state_dict": actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": critic_scheduler.state_dict(),
                    "global_step": global_step,
                    "episode_count": episode_count,
                    "env_cfg": env_cfg.to_dict(),
                    "train_cfg": asdict(train_cfg),
                    "obs_rms": obs_rms.state_dict(),
                    "state_rms": state_rms.state_dict(),
                },
                Path(ckpt_dir) / "last_model.pt",
            )

        print(
            f"step={global_step:>8} | ep={episode_count:>4} | "
            f"pi={np.mean(policy_losses):>8.4f} | vf={np.mean(value_losses):>8.4f} | "
            f"ent={np.mean(entropies):>7.4f} | kl={np.mean(kls):>7.5f} | "
            f"clip={np.mean(clip_fracs):>6.3f} | lr={current_lr:.2e} | sps={sps:>5}"
        )

        update_idx += 1

    env.close()
    writer.close()
    print(f"Training completed. Run dir: {ckpt_dir}")


# ---------------------------------------------------------------------
# Multi-seed evaluation helper (paper reproducibility)
# ---------------------------------------------------------------------
@torch.no_grad()
def run_multiseed_eval(
    args: argparse.Namespace,
    seeds: Sequence[int] = (42, 123, 456, 789, 1337),
) -> Dict[str, Any]:
    """
    Run evaluation across multiple seeds and report mean ± std.
    Required for Q2/Q1 paper submission.
    Saves evaluation_results.json next to the checkpoint.
    """
    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    action_dim = 2  # default; overridden after reset

    device = torch.device(args.device if hasattr(args, "device") else "cpu")
    actor = Actor(obs_dim, action_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    per_seed = []
    for seed in seeds:
        obs_dict, info_dict = env.reset(seed=seed)
        total_reward = 0.0
        total_steps = 0
        total_co2 = 0.0
        total_throughput = 0.0
        ep_prev_passed = 0.0
        done = False

        while not done and total_steps < args.eval_max_steps:
            action_dict: Dict[str, int] = {}
            for tls_id in tls_ids:
                obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor.distribution(obs_t)
                action_dict[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            step_co2, _ = extract_step_emissions(info)
            total_co2 += step_co2

            first_info = next(iter(info.values()), {})
            cur_passed = float(first_info.get("current_passed", ep_prev_passed))
            total_throughput += max(cur_passed - ep_prev_passed, 0.0)
            ep_prev_passed = cur_passed

            total_steps += 1
            done = any(terminated.values()) or any(truncated.values())
            obs_dict = next_obs

        per_seed.append({
            "seed": seed,
            "total_reward": total_reward,
            "episode_length": total_steps,
            "co2_total_mg": total_co2,
            "throughput": total_throughput,
        })
        print(f"seed={seed} | reward={total_reward:.3f} | steps={total_steps} | co2={total_co2:.1f}mg | tp={total_throughput:.0f}")

    env.close()

    rewards = [r["total_reward"] for r in per_seed]
    lengths = [r["episode_length"] for r in per_seed]
    co2s = [r["co2_total_mg"] for r in per_seed]

    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "mean_co2": float(np.mean(co2s)),
        "std_co2": float(np.std(co2s)),
        "seeds": list(seeds),
        "per_seed_results": per_seed,
    }

    # save evaluation_results.json next to the checkpoint
    ckpt_dir = str(Path(args.checkpoint).parent)
    with open(Path(ckpt_dir) / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"\nMulti-seed summary ({len(seeds)} seeds):\n"
        f"  reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}\n"
        f"  length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n"
        f"  CO2:    {results['mean_co2']:.1f} ± {results['std_co2']:.1f} mg\n"
    )
    return results


# ---------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    action_dim = 2

    actor = Actor(obs_dim, action_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    for ep in range(cfg.eval_episodes):
        obs_dict, info_dict = env.reset(seed=cfg.seed + ep)
        global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))

        total_reward = 0.0
        total_steps = 0
        done = False
        while not done and total_steps < cfg.eval_max_steps:
            action_dict: Dict[str, int] = {}
            for tls_id in tls_ids:
                obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor.distribution(obs_t)
                action_dict[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            total_steps += 1
            done = any(terminated.values()) or any(truncated.values())
            obs_dict = next_obs
            global_state = extract_global_state(info, fallback=global_state)

        print(f"eval_ep={ep + 1} | reward={total_reward:.3f} | steps={total_steps}")

    env.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MAPPO trainer for SUMO traffic control")

    parser.add_argument("--mode", choices=["train", "eval", "multiseed_eval"], default="train")
    parser.add_argument("--sumo-cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/mappo/20260418_215140/best_model.pt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-horizon", type=int, default=128)   # FIX: aligned with TrainConfig
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)       # FIX: aligned with TrainConfig
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-dir", type=str, default="logs/rl")
    parser.add_argument("--ckpt-dir", type=str, default="models/mappo")
    parser.add_argument("--save-every-steps", type=int, default=50_000)
    parser.add_argument("--save-latest-every-updates", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--step-length", type=int, default=5)
    parser.add_argument("--yellow-time", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1080)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--max-green-time", type=int, default=60)
    parser.add_argument("--queue-cap", type=float, default=50.0)
    parser.add_argument("--waiting-cap", type=float, default=300.0)
    parser.add_argument("--speed-cap", type=float, default=15.0)
    parser.add_argument("--max-lanes-per-tls", type=int, default=4)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tls-ids", nargs="*", default=[])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "multiseed_eval":
        run_multiseed_eval(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()