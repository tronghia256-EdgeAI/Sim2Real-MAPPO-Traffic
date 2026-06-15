from __future__ import annotations

"""
obs_noise.py
============
Eval-time sensing-noise model for the vision-proxy robustness study
(paper VI-E). Instantiates the noise process ε of the observation projection
o_{i,t} = φ_i(s_t) + ε_{i,t} (Section III-B) on the assembled 26-dim per-agent
proxy observation, parameterized from the deployment perception stack rather
than from convenience distributions.

This module is used ONLY at evaluation time. Training observations are computed
noiselessly from SUMO (ε = 0); main-campaign policies never see this noise. The
"proxy+noise" train-time arm, if run, is a separate domain-randomization ablation
and is out of scope here.

Per-feature noise on the interleaved per-approach layout (proxy mode):
    approach a, lane-feature f  ->  index 5a + f,  a in {0..3}
      f=0 effective_queue_norm   : multiplicative N(0, queue_calib_sigma^2)
                                    (px-per-meter calibration error, +/-10%)
      f=1 occupancy_norm         : multiplicative N(0, occupancy_bias_sigma^2)
                                    (perspective-residual bias)   [PLACEHOLDER]
      f=2 avg_speed_norm         : additive N(0, sigma_v^2); full sigma below
                                    speed_low_regime, reduced above
                                    (ByteTrack displacement noise)
      f=3 motorbike_share        : additive N(0, class_flip_rate^2)
      f=4 heavy_vehicle_share    : additive N(0, class_flip_rate^2)
                                    (YOLO class confusion)         [PLACEHOLDER]
    TLS dims (20..25):
      20..23 phase_one_hot       : NOT perturbed — internal action->phase state
      24     green_timer_norm    : NOT perturbed — internal controller clock
      25     pressure_norm        : multiplicative N(0, pressure_calib_sigma^2)
                                    on its deviation from 0.5 (inherits queue
                                    calibration error in deployment)

Structural failure modes:
    camera dropout : zero all 5 lane features of a chosen approach slot
    one-step delay : policy receives the previous step's noisy observation

⚠ class_flip_rate and occupancy_bias_sigma are PLACEHOLDERS. They must be set
from measured detector statistics (YOLO confusion matrix + ROI geometry
evaluation) before the numbers enter the paper — see NoiseConfig.is_calibrated.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.traffic_env.config import DEFAULT_LANE_FEATURE_NAMES


# Feature name -> offset within a 5-feature approach block (proxy layout).
_LANE_FEATURE_OFFSET: Dict[str, int] = {
    name: idx for idx, name in enumerate(DEFAULT_LANE_FEATURE_NAMES)
}


@dataclass(slots=True)
class NoiseConfig:
    """Calibrated sensing-noise parameters (1x envelope at scale=1.0).

    Values with a measured basis are taken from docs/paper1_tsc/state.md; the two flagged
    PLACEHOLDER fields must be replaced from detector-stack evaluation before use
    in the paper. ``scale`` multiplies the whole envelope for the {0, 0.5, 1, 2}
    robustness sweep.
    """

    # measured-basis parameters
    queue_calib_sigma: float = 0.10        # +/-10% px-per-meter calibration (docs/paper1_tsc/state.md)
    speed_sigma_mps: float = 1.0           # ByteTrack +/-1 m/s at low speed (docs/paper1_tsc/state.md)
    speed_low_regime_mps: float = 3.0      # full sigma below this speed
    speed_high_regime_factor: float = 0.25 # sigma multiplier above the low regime
    pressure_calib_sigma: float = 0.10     # inherits queue calibration error
    speed_cap_mps: float = 15.0            # to convert speed sigma to normalized units

    # PLACEHOLDER parameters — set from measured detector statistics
    class_flip_rate: float = 0.05          # YOLO class-confusion share jitter  [PLACEHOLDER]
    occupancy_bias_sigma: float = 0.08     # perspective-residual bias           [PLACEHOLDER]

    # global envelope scale for the robustness sweep
    scale: float = 1.0

    # structural failure modes
    camera_dropout_approaches: Tuple[int, ...] = ()  # approach slots to zero (0..max_lanes-1)
    delay_steps: int = 0                              # 0 or 1

    # True once parameters come from a measured III-E calibration file
    # (configs/noise_config.json via from_json / calibrate_noise.py).
    calibrated: bool = False

    def validate(self) -> None:
        if self.scale < 0:
            raise ValueError("scale must be >= 0")
        if self.delay_steps not in (0, 1):
            raise ValueError("delay_steps must be 0 or 1")
        for name in ("queue_calib_sigma", "speed_sigma_mps", "pressure_calib_sigma",
                     "class_flip_rate", "occupancy_bias_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.speed_cap_mps <= 0:
            raise ValueError("speed_cap_mps must be > 0")

    @property
    def is_calibrated(self) -> bool:
        """True once parameters have a measured basis.

        Either the explicit ``calibrated`` flag is set (loaded from a III-E
        calibration file), or the two PLACEHOLDER fields no longer hold their
        default values. Robustness numbers must not enter the paper until this
        is True.
        """
        return bool(self.calibrated) or not (
            self.class_flip_rate == 0.05 and self.occupancy_bias_sigma == 0.08
        )

    def scaled(self, scale: float) -> "NoiseConfig":
        """Return a copy with a new envelope ``scale`` (for the sweep)."""
        new = NoiseConfig(**{f: getattr(self, f) for f in self.__slots__})  # type: ignore[attr-defined]
        new.scale = float(scale)
        return new

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "NoiseConfig":
        """Load a calibrated config written by calibrate_noise.py.

        Accepts either a flat dict of fields or a ``{"params": {...}}`` wrapper
        (the calibrate_noise.py output, which also carries provenance/error
        metadata that is ignored here). Loading a file implies ``calibrated``
        unless the file explicitly says otherwise.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        params = data.get("params", data)
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: Dict[str, Any] = {k: v for k, v in params.items() if k in field_names}
        if kwargs.get("camera_dropout_approaches") is not None:
            kwargs["camera_dropout_approaches"] = tuple(kwargs["camera_dropout_approaches"])
        cfg = cls(**kwargs)
        cfg.calibrated = bool(params.get("calibrated", True))
        cfg.validate()
        return cfg


@dataclass(slots=True)
class SensingNoiseModel:
    """Applies NoiseConfig to assembled per-agent proxy observations.

    Stateful only for the one-step delay (per-agent buffer). RNG is seeded for
    reproducibility; call :meth:`reset` at episode boundaries.
    """

    config: NoiseConfig
    lane_feature_dim: int = 5
    max_lanes_per_tls: int = 4
    tls_feature_dim: int = 6
    seed: int = 0
    lane_feature_names: Sequence[str] = DEFAULT_LANE_FEATURE_NAMES

    _rng: np.random.Generator = field(init=False)
    _delay_buf: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _offset: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self._rng = np.random.default_rng(self.seed)
        self._delay_buf = {}
        self._offset = {
            name: idx for idx, name in enumerate(self.lane_feature_names)
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the delay buffer and (optionally) reseed the RNG."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._delay_buf = {}

    # ------------------------------------------------------------------
    def apply(self, obs: np.ndarray, agent_id: str = "_default") -> np.ndarray:
        """Return a noisy copy of one agent's local observation.

        ``agent_id`` keys the one-step-delay buffer so multiple agents can share
        a single model instance without crosstalk.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        s = float(self.config.scale)
        if s <= 0.0:
            noisy = obs.copy()
        else:
            noisy = self._perturb(obs, s)

        # structural: camera dropout zeros a whole approach slot's lane features
        for a in self.config.camera_dropout_approaches:
            base = a * self.lane_feature_dim
            if 0 <= base and base + self.lane_feature_dim <= noisy.size:
                noisy[base: base + self.lane_feature_dim] = 0.0

        # structural: one-step pipeline-latency delay on the noisy stream
        if self.config.delay_steps == 1:
            prev = self._delay_buf.get(agent_id)
            self._delay_buf[agent_id] = noisy
            if prev is not None:
                return prev

        return noisy

    # ------------------------------------------------------------------
    def _perturb(self, obs: np.ndarray, s: float) -> np.ndarray:
        cfg = self.config
        out = obs.copy()
        n_lane_block = self.max_lanes_per_tls * self.lane_feature_dim

        q_off = self._offset.get("effective_queue_norm")
        occ_off = self._offset.get("occupancy_norm")
        spd_off = self._offset.get("avg_speed_norm")
        moto_off = self._offset.get("motorbike_share")
        heavy_off = self._offset.get("heavy_vehicle_share")

        speed_norm_sigma = cfg.speed_sigma_mps / cfg.speed_cap_mps
        low_regime_norm = cfg.speed_low_regime_mps / cfg.speed_cap_mps

        for a in range(self.max_lanes_per_tls):
            base = a * self.lane_feature_dim

            if q_off is not None:
                i = base + q_off
                out[i] = out[i] * (1.0 + self._rng.normal(0.0, cfg.queue_calib_sigma * s))

            if occ_off is not None:
                i = base + occ_off
                out[i] = out[i] * (1.0 + self._rng.normal(0.0, cfg.occupancy_bias_sigma * s))

            if spd_off is not None:
                i = base + spd_off
                # ByteTrack noise is worst in the low-speed regime
                sigma = speed_norm_sigma if out[i] < low_regime_norm \
                    else speed_norm_sigma * cfg.speed_high_regime_factor
                out[i] = out[i] + self._rng.normal(0.0, sigma * s)

            if moto_off is not None:
                i = base + moto_off
                out[i] = out[i] + self._rng.normal(0.0, cfg.class_flip_rate * s)

            if heavy_off is not None:
                i = base + heavy_off
                out[i] = out[i] + self._rng.normal(0.0, cfg.class_flip_rate * s)

        # pressure_norm (last TLS dim): perturb its signed deviation from 0.5,
        # which is what inherits the queue calibration error in deployment
        p_idx = n_lane_block + self.tls_feature_dim - 1
        if 0 <= p_idx < out.size:
            dev = out[p_idx] - 0.5
            out[p_idx] = 0.5 + dev * (1.0 + self._rng.normal(0.0, cfg.pressure_calib_sigma * s))

        # all features are bounded [0,1]; phase one-hot and timer are untouched
        np.clip(out[: n_lane_block], 0.0, 1.0, out=out[: n_lane_block])
        out[p_idx] = float(np.clip(out[p_idx], 0.0, 1.0))
        return out.astype(np.float32, copy=False)


__all__ = ["NoiseConfig", "SensingNoiseModel"]
