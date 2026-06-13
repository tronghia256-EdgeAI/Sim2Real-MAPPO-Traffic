from __future__ import annotations

"""
configuration layer for a mappo-ready traffic environment.

this module keeps all tunable parameters in one place so the env, reward,
observation, and vision adapters can share the same schema without hard-coding
magic numbers across files.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_TLS_IDS: Tuple[str, ...] = ("J0", "J2")

DEFAULT_MANUAL_LANE_GROUPS: Dict[str, Tuple[List[str], List[str]]] = {
    "J0": (
        ["-E5_0", "-E5_1", "-E6_0", "-E6_1"],
        ["-E0_0", "-E0_1", "-E1_0", "-E1_1"],
    ),
    # S3 FIX: J2 group B previously listed -E1_* (edge J2->J0, an approach to J0,
    # already owned by J0's group B). J2's true incoming corridor edge is E1
    # (J0->J2). The wrong assignment corrupted J2's pressure_norm observation,
    # its pressure reward partition, and the MaxPressure baseline at J2.
    "J2": (
        ["-E3_0", "-E3_1", "-E4_0", "-E4_1"],
        ["E1_0", "E1_1", "-E2_0", "-E2_1"],
    ),
}

DEFAULT_LANE_FEATURE_NAMES: Tuple[str, ...] = (
    "effective_queue_norm",
    "occupancy_norm",
    "avg_speed_norm",
    "motorbike_share",
    "heavy_vehicle_share",
)

# W5-9 campaign: extra simulator-exact per-approach features for the
# MAPPO-privileged upper-bound arm (paper VI-C "cost of deployability").
# Deliberately namespaced "privileged_" so the RewardCalculator (which reads
# e.g. "waiting_time_norm") never picks them up — the reward must stay
# identical across the privileged and proxy arms; only the obs differs.
PRIVILEGED_EXTRA_LANE_FEATURE_NAMES: Tuple[str, ...] = (
    "privileged_halt_count_norm",     # exact halted-vehicle count / queue_cap
    "privileged_waiting_norm",        # lane accumulated waiting time / waiting_cap
    "privileged_vehicle_count_norm",  # exact vehicle count / queue_cap
)

DEFAULT_TLS_FEATURE_NAMES: Tuple[str, ...] = (
    "phase_one_hot_0",
    "phase_one_hot_1",
    "phase_one_hot_2",
    "phase_one_hot_3",
    "green_timer_norm",
    "pressure_norm",
)

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "queue":            -1.0,   # nonlinear (mean_q^2); primary congestion signal
    "pressure":         -0.5,   # phase-aware imbalance (PRESSLIGHT); 0 if green_lanes absent
    "throughput":        1.0,   # cooperative cleared-vehicles; shared across agents
    "switch_penalty":   -0.1,   # phase-switching cost
    "low_speed_penalty":-0.2,   # mean speed below threshold; reduce/zero for vision deploy
    "waiting_time":     -0.3,   # accumulated delay; uses waiting_time_norm or queue proxy
}


@dataclass(slots=True)
class SimConfig:
    """simulation and rollout parameters."""

    sumo_cfg_path: str = ""
    gui: bool = False
    step_length: int = 5
    yellow_time: int = 3
    max_steps: int = 1080
    min_green_time: int = 15
    max_green_time: int = 60
    seed: Optional[int] = None
    debug: bool = False

    def validate(self) -> None:
        if not self.sumo_cfg_path:
            raise ValueError("sumo_cfg_path must not be empty")
        if self.step_length <= 0:
            raise ValueError("step_length must be > 0")
        if self.yellow_time < 0:
            raise ValueError("yellow_time must be >= 0")
        if self.yellow_time > self.step_length:
            raise ValueError("yellow_time must be <= step_length")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.min_green_time < 0 or self.max_green_time <= 0:
            raise ValueError("green time constraints must be positive")
        if self.min_green_time > self.max_green_time:
            raise ValueError("min_green_time cannot exceed max_green_time")


@dataclass(slots=True)
class MultiAgentConfig:
    """multi-agent topology for mappo."""

    tls_ids: Tuple[str, ...] = DEFAULT_TLS_IDS
    max_lanes_per_tls: int = 4
    shared_policy: bool = True
    action_mode: str = "discrete_phase"
    centralized_value: bool = True

    def validate(self) -> None:
        if not self.tls_ids:
            raise ValueError("tls_ids must contain at least one traffic light id")
        if self.max_lanes_per_tls <= 0:
            raise ValueError("max_lanes_per_tls must be > 0")
        if self.action_mode not in {"discrete_phase"}:
            raise ValueError("unsupported action_mode")

    @property
    def num_agents(self) -> int:
        return len(self.tls_ids)


@dataclass(slots=True)
class ObservationConfig:
    """observation schema for lane-level and tls-level features."""

    lane_feature_names: Tuple[str, ...] = DEFAULT_LANE_FEATURE_NAMES
    tls_feature_names: Tuple[str, ...] = DEFAULT_TLS_FEATURE_NAMES
    queue_cap: float = 50.0
    waiting_cap: float = 300.0
    speed_cap: float = 15.0
    use_external_state: bool = False
    # "proxy" (default): camera-computable features only — the deployable arm.
    # "privileged": proxy features + PRIVILEGED_EXTRA_LANE_FEATURE_NAMES from
    # exact simulator state — upper-bound arm, never deployable.
    obs_mode: str = "proxy"

    def validate(self) -> None:
        if not self.lane_feature_names:
            raise ValueError("lane_feature_names must not be empty")
        if not self.tls_feature_names:
            raise ValueError("tls_feature_names must not be empty")
        if self.obs_mode not in {"proxy", "privileged"}:
            raise ValueError("obs_mode must be 'proxy' or 'privileged'")
        if self.obs_mode == "privileged":
            missing = [n for n in PRIVILEGED_EXTRA_LANE_FEATURE_NAMES
                       if n not in self.lane_feature_names]
            if missing:
                raise ValueError(
                    f"obs_mode='privileged' requires privileged lane features "
                    f"in lane_feature_names; missing: {missing}"
                )
        if self.obs_mode == "proxy":
            leaked = [n for n in self.lane_feature_names
                      if n in PRIVILEGED_EXTRA_LANE_FEATURE_NAMES]
            if leaked:
                raise ValueError(
                    f"obs_mode='proxy' must not contain privileged features: {leaked}"
                )
        if self.queue_cap <= 0:
            raise ValueError("queue_cap must be > 0")
        if self.waiting_cap <= 0:
            raise ValueError("waiting_cap must be > 0")
        if self.speed_cap <= 0:
            raise ValueError("speed_cap must be > 0")

    @property
    def lane_feature_dim(self) -> int:
        return len(self.lane_feature_names)

    @property
    def tls_feature_dim(self) -> int:
        return len(self.tls_feature_names)

    @property
    def local_obs_dim(self) -> int:
        return self.lane_feature_dim + self.tls_feature_dim

    def local_obs_dim_for(self, max_lanes_per_tls: int) -> int:
        return max_lanes_per_tls * self.lane_feature_dim + self.tls_feature_dim

    def global_state_dim(self, num_agents: int, max_lanes_per_tls: int) -> int:
        return num_agents * self.local_obs_dim_for(max_lanes_per_tls)


@dataclass(slots=True)
class RewardConfig:
    """reward shaping and clipping settings."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    reward_scale: float = 1.0
    # 1.2.0 ranges with default weights: queue [-1,0], signed pressure
    # [-0.5,+0.5], local throughput [0,+1], switch [-0.1,0], low_speed
    # [-0.04,0], waiting [-0.3,0]  =>  raw in [-1.94, +1.50].
    reward_clip_low: float = -2.0
    reward_clip_high: float = 1.5
    throughput_norm_divisor: float = 20.0
    low_speed_threshold: float = 0.20
    jam_speed_threshold: float = 0.50
    jam_queue_threshold: float = 0.50

    def validate(self) -> None:
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be > 0")
        if self.reward_clip_low >= self.reward_clip_high:
            raise ValueError("reward_clip_low must be smaller than reward_clip_high")
        if self.throughput_norm_divisor <= 0:
            raise ValueError("throughput_norm_divisor must be > 0")
        if self.low_speed_threshold < 0:
            raise ValueError("low_speed_threshold must be non-negative")


@dataclass(slots=True)
class VisionBridgeConfig:
    """settings for the vision-to-state adapter."""

    enabled: bool = False
    buffer_size: int = 30
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    min_track_age: int = 2
    max_track_gap: int = 10
    camera_ids: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0, 1]")
        if self.min_track_age < 0:
            raise ValueError("min_track_age must be >= 0")
        if self.max_track_gap < 0:
            raise ValueError("max_track_gap must be >= 0")


@dataclass(slots=True)
class TrafficEnvConfig:
    """top-level configuration object used by the env factory."""

    sim: SimConfig = field(default_factory=SimConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    vision: VisionBridgeConfig = field(default_factory=VisionBridgeConfig)
    manual_lane_groups: Optional[Dict[str, Tuple[List[str], List[str]]]] = None
    # 1.1.0: per-approach obs aggregation (S1), signed pressure_norm (S2),
    # J2 lane-group topology fix (S3). Checkpoints trained before 1.1.0 are
    # incompatible with this observation schema.
    # 1.2.0: reward revision for the W5-9 campaign — queue penalty mean(q^2)
    # (anti-starvation), SIGNED pressure reward term, LOCAL per-agent
    # lane-exit throughput (replaces global arrival split). Obs schema is
    # unchanged from 1.1.0; reward semantics differ, so results across
    # versions must not be pooled.
    version: str = "1.2.0"

    def validate(self) -> None:
        self.sim.validate()
        self.multi_agent.validate()
        self.observation.validate()
        self.reward.validate()
        self.vision.validate()

    @property
    def tls_ids(self) -> Tuple[str, ...]:
        return self.multi_agent.tls_ids

    @property
    def max_lanes_per_tls(self) -> int:
        return self.multi_agent.max_lanes_per_tls

    @property
    def num_agents(self) -> int:
        return self.multi_agent.num_agents

    @property
    def local_obs_dim(self) -> int:
        return self.observation.local_obs_dim_for(self.max_lanes_per_tls)

    @property
    def global_state_dim(self) -> int:
        return self.observation.global_state_dim(
            num_agents=self.num_agents,
            max_lanes_per_tls=self.max_lanes_per_tls,
        )

    @property
    def observation_dim(self) -> int:
        return self.global_state_dim

    @property
    def lane_feature_dim(self) -> int:
        return self.observation.lane_feature_dim

    @property
    def tls_feature_dim(self) -> int:
        return self.observation.tls_feature_dim

    def get_lane_groups(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """return manual lane groups if provided, otherwise default known groups."""
        if self.manual_lane_groups is not None:
            return self.manual_lane_groups
        return DEFAULT_MANUAL_LANE_GROUPS

    def to_dict(self) -> Dict[str, Any]:
        """export a json-friendly snapshot for logging and experiment tracking."""
        return {
            "version": self.version,
            "sim": asdict(self.sim),
            "multi_agent": {
                **asdict(self.multi_agent),
                "num_agents": self.num_agents,
            },
            "observation": {
                **asdict(self.observation),
                "lane_feature_dim": self.lane_feature_dim,
                "tls_feature_dim": self.tls_feature_dim,
                "local_obs_dim": self.local_obs_dim,
                "global_state_dim": self.global_state_dim,
                "observation_dim": self.observation_dim,
            },
            "reward": asdict(self.reward),
            "vision": asdict(self.vision),
            "manual_lane_groups": self.get_lane_groups(),
        }

    def build_state_schema(self) -> Dict[str, Any]:
        """build a schema that can be shared with the vision adapter."""
        lane_features = [
            {"index": i, "name": name, "normalization": "[0,1]"}
            for i, name in enumerate(self.observation.lane_feature_names)
        ]
        tls_features = [
            {"index": i, "name": name, "normalization": "[0,1]"}
            for i, name in enumerate(self.observation.tls_feature_names)
        ]

        phase_lane_groups: Dict[str, Dict[str, List[str]]] = {}
        for tls_id, (group_a, group_b) in self.get_lane_groups().items():
            phase_lane_groups[tls_id] = {
                "phase_order_0": list(group_a),
                "phase_order_1": list(group_b),
            }

        return {
            "schema_name": "mappo_sumo_traffic_state",
            "version": self.version,
            "description": (
                "flat multi-agent observation schema for mappo training and vision-to-state alignment"
            ),
            "tls_ids": list(self.tls_ids),
            "num_agents": self.num_agents,
            "max_lanes_per_tls": self.max_lanes_per_tls,
            "observation_dim": self.observation_dim,
            "lane_feature_dim": self.lane_feature_dim,
            "tls_feature_dim": self.tls_feature_dim,
            "lane_order_rule": (
                "use traci.trafficlight.getControlledLanes(tls_id), remove duplicates while preserving order, "
                "and drop internal lanes that start with ':'"
            ),
            "phase_lane_groups": phase_lane_groups,
            "features": {
                "lane_features": lane_features,
                "tls_features": tls_features,
            },
            "notes": [
                "phase control is synchronized across agents by default",
                "the actor should consume local observations, while the critic should consume the global state",
                "if a step contains a phase switch, the env must advance through yellow time before continuing the remaining green interval",
                "use the same schema for both sumo observations and external vision state injection",
            ],
        }


def load_lane_groups_json(
    path: str,
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[List[str], List[str]]]]:
    """Load (tls_ids, manual_lane_groups) from a lane_groups.json.

    Files are produced by scripts/generate_networks.py for the scaled
    networks (n2_corridor, n3_grid, ...) and validated against the net
    topology by check_obs_match.py CHECK 6.
    """
    import json
    from pathlib import Path as _Path

    with open(_Path(path), encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("lane_groups", {})
    groups: Dict[str, Tuple[List[str], List[str]]] = {
        tls_id: (list(pair[0]), list(pair[1])) for tls_id, pair in raw.items()
    }
    if not groups:
        raise ValueError(f"no lane_groups found in {path}")
    tls_ids = tuple(data.get("tls_ids") or sorted(groups))
    return tls_ids, groups


def build_default_config(
    *,
    sumo_cfg_path: str,
    gui: bool = False,
    tls_ids: Optional[Sequence[str]] = None,
    max_lanes_per_tls: int = 4,
    step_length: int = 5,
    yellow_time: int = 3,
    max_steps: int = 1080,
    min_green_time: int = 15,
    max_green_time: int = 60,
    queue_cap: float = 50.0,
    waiting_cap: float = 300.0,
    speed_cap: float = 15.0,
    use_external_state: bool = False,
    obs_mode: str = "proxy",
    manual_lane_groups: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
    debug: bool = False,
    reward_scale: float = 1.0,
) -> TrafficEnvConfig:
    """helper constructor for scripts and experiments.

    ``obs_mode='privileged'`` appends PRIVILEGED_EXTRA_LANE_FEATURE_NAMES to the
    per-approach feature set (8 lane features instead of 5), enlarging the obs
    vector; the reward and TLS features are untouched so the privileged arm
    differs from the proxy arm in observation only.
    """
    if obs_mode == "privileged":
        lane_feature_names: Tuple[str, ...] = (
            DEFAULT_LANE_FEATURE_NAMES + PRIVILEGED_EXTRA_LANE_FEATURE_NAMES
        )
    elif obs_mode == "proxy":
        lane_feature_names = DEFAULT_LANE_FEATURE_NAMES
    else:
        raise ValueError(f"unknown obs_mode {obs_mode!r}; expected 'proxy' or 'privileged'")

    cfg = TrafficEnvConfig(
        sim=SimConfig(
            sumo_cfg_path=sumo_cfg_path,
            gui=gui,
            step_length=step_length,
            yellow_time=yellow_time,
            max_steps=max_steps,
            min_green_time=min_green_time,
            max_green_time=max_green_time,
            debug=debug,
        ),
        multi_agent=MultiAgentConfig(
            tls_ids=tuple(tls_ids) if tls_ids is not None else DEFAULT_TLS_IDS,
            max_lanes_per_tls=max_lanes_per_tls,
        ),
        observation=ObservationConfig(
            lane_feature_names=lane_feature_names,
            queue_cap=queue_cap,
            waiting_cap=waiting_cap,
            speed_cap=speed_cap,
            use_external_state=use_external_state,
            obs_mode=obs_mode,
        ),
        reward=RewardConfig(reward_scale=reward_scale),
        manual_lane_groups=manual_lane_groups,
    )
    cfg.validate()
    return cfg


__all__ = [
    "DEFAULT_LANE_FEATURE_NAMES",
    "DEFAULT_MANUAL_LANE_GROUPS",
    "DEFAULT_REWARD_WEIGHTS",
    "DEFAULT_TLS_FEATURE_NAMES",
    "DEFAULT_TLS_IDS",
    "PRIVILEGED_EXTRA_LANE_FEATURE_NAMES",
    "MultiAgentConfig",
    "ObservationConfig",
    "RewardConfig",
    "SimConfig",
    "TrafficEnvConfig",
    "VisionBridgeConfig",
    "build_default_config",
    "load_lane_groups_json",
]