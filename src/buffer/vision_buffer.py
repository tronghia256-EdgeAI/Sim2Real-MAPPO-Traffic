from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from types import SimpleNamespace
from typing import Any, Deque, Dict, Optional, Union
import logging

import numpy as np
from src.adapters.vision_to_state import VisionStatePacket

logger = logging.getLogger(__name__)


@dataclass
class BufferedVisionState:
    obs: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    semantic_state: Dict[str, Any] = field(default_factory=dict)
    raw_packet: Any = None
    timestamp: float = 0.0
    buffer_size: int = 0


class VisionBuffer:
    """
    Temporal buffer for vision-derived PPO states with EMA smoothing.
    
    HIGH-5 FIX: EMA alpha is now loaded from config (state_config.json).
    No longer hardcoded to 0.6.
    
    Keeps the observation dimension unchanged and smooths frame-to-frame noise
    using exponential moving average (EMA).
    
    Parameters
    ----------
    maxlen : int
        Buffer window size (frames to keep).
    ema_alpha : float
        EMA smoothing factor ∈ [0, 1].
        0.0 = no smoothing (use latest)
        1.0 = no history (average equally)
        0.6 = standard (60% latest, 40% history)
    use_median : bool
        If True, use median smoothing instead of EMA.
    clip_obs : bool
        If True, clip observations to [0, 1] after smoothing.
    """
    def __init__(
        self,
        maxlen: int = 5,
        ema_alpha: float = 0.6,
        use_median: bool = False,
        clip_obs: bool = True,
    ) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in [0, 1]")

        self.maxlen = int(maxlen)
        self.ema_alpha = float(ema_alpha)
        self.use_median = bool(use_median)
        self.clip_obs = bool(clip_obs)

        self._buffer: Deque[Any] = deque(maxlen=self.maxlen)
        self._last_smoothed: Optional[np.ndarray] = None
        
        logger.info(
            "VisionBuffer initialized: maxlen=%d, ema_alpha=%.2f, use_median=%s, clip_obs=%s",
            self.maxlen, self.ema_alpha, self.use_median, self.clip_obs
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._last_smoothed = None

    def push(self, packet: VisionStatePacket) -> BufferedVisionState:
        if packet is None:
            raise ValueError("packet cannot be None")
        self._buffer.append(packet)
        return self._build_buffered(packet)

    def push_obs(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        semantic_state: Optional[Dict[str, Any]] = None,
        timestamp: float = 0.0,
    ) -> BufferedVisionState:
        packet = SimpleNamespace(
            obs=np.asarray(obs, dtype=np.float32),
            info=info or {},
            semantic_state=semantic_state or {},
            timestamp=float(timestamp),
        )
        return self.push(packet)  # type: ignore[arg-type]

    def get_latest(self) -> Optional[BufferedVisionState]:
        if not self._buffer:
            return None
        return self._build_buffered(self._buffer[-1])

    def ready(self, min_len: int = 1) -> bool:
        return len(self._buffer) >= max(1, int(min_len))

    def __len__(self) -> int:
        return len(self._buffer)

    def _build_buffered(self, packet: Any) -> BufferedVisionState:
        obs = self._smooth_obs()
        info = self._smooth_info()
        semantic = self._merge_semantics()

        return BufferedVisionState(
            obs=obs,
            info=info,
            semantic_state=semantic,
            raw_packet=packet,
            timestamp=float(getattr(packet, "timestamp", 0.0) or 0.0),
            buffer_size=len(self._buffer),
        )

    def _smooth_obs(self) -> np.ndarray:
        if not self._buffer:
            raise RuntimeError("Cannot smooth an empty buffer.")

        obs_list = [np.asarray(getattr(p, "obs"), dtype=np.float32).reshape(-1) for p in self._buffer]
        shapes = {arr.shape[0] for arr in obs_list}
        if len(shapes) != 1:
            raise RuntimeError("All buffered observations must have the same shape.")

        stack = np.stack(obs_list, axis=0)

        if self.use_median:
            smoothed = np.median(stack, axis=0).astype(np.float32)
        elif len(obs_list) == 1:
            smoothed = obs_list[0].copy()
        else:
            ema = obs_list[0].copy()
            alpha = self.ema_alpha
            for arr in obs_list[1:]:
                ema = alpha * arr + (1.0 - alpha) * ema
            smoothed = ema.astype(np.float32)

        if self.clip_obs:
            smoothed = np.clip(smoothed, 0.0, 1.0)

        self._last_smoothed = smoothed.copy()
        return smoothed

    def _smooth_info(self) -> Dict[str, Any]:
        infos = [getattr(p, "info", {}) or {} for p in self._buffer]
        accident_found = any(bool(info.get("accident_found", False)) for info in infos)
        accident_conf = max([float(info.get("accident_conf", 0.0)) for info in infos] + [0.0])
        return {
            "accident_found": accident_found,
            "accident_conf": accident_conf,
            "buffer_size": len(self._buffer),
        }

    def _merge_semantics(self) -> Dict[str, Any]:
        latest = self._buffer[-1]
        sem = dict(getattr(latest, "semantic_state", {}) or {})
        sem["buffer_size"] = len(self._buffer)
        sem["history_ts"] = [float(getattr(p, "timestamp", 0.0) or 0.0) for p in self._buffer]
        return sem

    @property
    def last_smoothed_obs(self) -> Optional[np.ndarray]:
        return None if self._last_smoothed is None else self._last_smoothed.copy()


def build_buffered_state(vision_buffer: VisionBuffer, packet: VisionStatePacket) -> BufferedVisionState:
    return vision_buffer.push(packet)