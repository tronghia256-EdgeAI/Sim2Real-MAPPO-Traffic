"""
camera_manager_v2.py
====================
Production-grade multi-camera manager.

Key improvements over v1
------------------------
- CameraHealthState enum: ACTIVE / STALE / DEAD
- Latest-frame strategy: deque(maxlen=1) — inference always gets the newest frame
- Timestamp validation: frames older than `stale_threshold_sec` are rejected
- Per-camera moving-average FPS (clamped to [0, FPS_CLAMP_MAX])
- Fail-safe: `get_valid_frames()` returns only ACTIVE cameras; caller knows
  when to fall back to the fixed-cycle traffic policy
- Thread-safe without unnecessary copies in the hot path
- Reconnection logic: camera worker retries after `reconnect_interval_sec`

Design
------
Each camera runs one daemon thread:
    capture_thread  →  deque(maxlen=1)  →  [inference thread reads]

The deque(maxlen=1) is the lock-free single-slot buffer: a new frame evicts
the previous one automatically, so inference always processes the latest image
and frame-drops are bounded to at most 1 old frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
FPS_CLAMP_MAX: float = 120.0          # safety cap so FPS never shows 999+
FPS_WINDOW: int = 30                  # frames in moving-average window
STALE_THRESHOLD_SEC: float = 2.0      # frame older than this → STALE
DEAD_THRESHOLD_SEC: float = 10.0      # no frame for this long → DEAD
RECONNECT_INTERVAL_SEC: float = 5.0   # how often to retry a dead camera


# ═══════════════════════════════════════════════════════════════════════════════
# Camera health state
# ═══════════════════════════════════════════════════════════════════════════════

class CameraHealth(Enum):
    """Lifecycle state of a single camera stream."""
    ACTIVE = auto()   # fresh frame received within stale_threshold_sec
    STALE  = auto()   # no new frame for stale_threshold_sec .. dead_threshold_sec
    DEAD   = auto()   # no new frame for > dead_threshold_sec, or never opened


@dataclass
class CameraFrame:
    """Immutable value object passed through the pipeline."""
    cam_id: Union[int, str]
    name: str
    frame: np.ndarray           # H×W×3 uint8
    timestamp: float            # time.time() of capture
    frame_count: int
    health: CameraHealth = CameraHealth.ACTIVE


@dataclass
class CameraStatus:
    cam_id: Union[int, str]
    name: str
    source: Union[int, str]
    health: CameraHealth
    frame_count: int
    last_ts: float
    fps_avg: float
    last_error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal per-camera worker
# ═══════════════════════════════════════════════════════════════════════════════

class _CameraWorker:
    """
    Background capture thread for one camera.

    The single-slot deque guarantees the inference thread always reads the
    *newest* frame without ever blocking.  Old frames are evicted automatically.
    """

    def __init__(
        self,
        cam_id: Union[int, str],
        name: str,
        source: Union[int, str],
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        stale_threshold_sec: float = STALE_THRESHOLD_SEC,
        dead_threshold_sec: float = DEAD_THRESHOLD_SEC,
        reconnect_interval_sec: float = RECONNECT_INTERVAL_SEC,
        loop_video: bool = True,
        enabled: bool = True,
    ) -> None:
        self.cam_id = cam_id
        self.name = name
        self.source = source
        self.target_fps = target_fps
        self.resize = resize
        self.stale_threshold_sec = stale_threshold_sec
        self.dead_threshold_sec = dead_threshold_sec
        self.reconnect_interval_sec = reconnect_interval_sec
        self.loop_video = loop_video
        self.enabled = enabled

        # ── single-slot buffer (lock-free eviction) ───────────────────────────
        self._slot: Deque[CameraFrame] = deque(maxlen=1)
        self._lock = threading.Lock()           # protects _slot reads

        # ── state ─────────────────────────────────────────────────────────────
        self._frame_count: int = 0
        self._last_ts: float = 0.0
        self._fps_ring: Deque[float] = deque(maxlen=FPS_WINDOW)
        self._last_error: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the capture device and launch the background thread."""
        if not self.enabled:
            return False
        ok = self._open()
        if not ok:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"cap-{self.cam_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release_cap()

    def get_latest(self) -> Optional[CameraFrame]:
        """
        Return the newest frame or None if nothing was ever captured.
        O(1), non-blocking.
        """
        with self._lock:
            if not self._slot:
                return None
            return self._slot[-1]

    @property
    def health(self) -> CameraHealth:
        now = time.time()
        if self._last_ts == 0.0:
            return CameraHealth.DEAD
        age = now - self._last_ts
        if age > self.dead_threshold_sec:
            return CameraHealth.DEAD
        if age > self.stale_threshold_sec:
            return CameraHealth.STALE
        return CameraHealth.ACTIVE

    @property
    def fps(self) -> float:
        """Moving-average capture FPS, clamped to [0, FPS_CLAMP_MAX]."""
        if len(self._fps_ring) < 2:
            return 0.0
        deltas = [b - a for a, b in zip(self._fps_ring, list(self._fps_ring)[1:])]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta <= 0.0:
            return 0.0
        return min(1.0 / avg_delta, FPS_CLAMP_MAX)

    @property
    def status(self) -> CameraStatus:
        return CameraStatus(
            cam_id=self.cam_id,
            name=self.name,
            source=self.source,
            health=self.health,
            frame_count=self._frame_count,
            last_ts=self._last_ts,
            fps_avg=self.fps,
            last_error=self._last_error,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _open(self) -> bool:
        self._release_cap()
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self._last_error = f"cv2.VideoCapture failed: {self.source}"
                logger.warning("[%s] %s", self.cam_id, self._last_error)
                return False
            if self.resize:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize[1])
            # Minimise internal buffer to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
            self._last_error = None
            logger.info("[%s] Camera opened: %s", self.cam_id, self.source)
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("[%s] Error opening camera", self.cam_id)
            return False

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _frame_delay(self) -> float:
        if self._cap is None:
            return 1.0 / max(self.target_fps or 10.0, 1.0)
        src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        eff_fps = self.target_fps if (self.target_fps and self.target_fps > 0) else (src_fps or 10.0)
        return 1.0 / max(eff_fps, 1.0)

    def _loop(self) -> None:
        """Capture loop running in daemon thread."""
        last_reconnect = 0.0
        delay = self._frame_delay()

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            # ── reconnect if dead ──────────────────────────────────────────────
            if self._cap is None or not self._cap.isOpened():
                now = time.time()
                if (now - last_reconnect) >= self.reconnect_interval_sec:
                    last_reconnect = now
                    if self._open():
                        delay = self._frame_delay()
                        logger.info("[%s] Camera reconnected", self.cam_id)
                    else:
                        time.sleep(self.reconnect_interval_sec)
                        continue
                else:
                    time.sleep(0.1)
                    continue

            # ── read frame ────────────────────────────────────────────────────
            assert self._cap is not None
            ret, frame = self._cap.read()

            if not ret or frame is None:
                self._last_error = "Read failed / end of stream"
                if self.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Non-looping source exhausted → mark dead
                self._release_cap()
                continue

            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            ts = time.time()
            self._frame_count += 1
            self._last_ts = ts
            self._fps_ring.append(ts)

            cf = CameraFrame(
                cam_id=self.cam_id,
                name=self.name,
                frame=frame,
                timestamp=ts,
                frame_count=self._frame_count,
                health=CameraHealth.ACTIVE,
            )

            with self._lock:
                self._slot.append(cf)      # evicts old frame automatically

            # ── sleep to match target FPS ──────────────────────────────────────
            elapsed = time.perf_counter() - t0
            sleep_t = max(delay - elapsed, 0.0)
            if sleep_t > 0:
                time.sleep(sleep_t)


# ═══════════════════════════════════════════════════════════════════════════════
# Public manager
# ═══════════════════════════════════════════════════════════════════════════════

class CameraManagerV2:
    """
    Production multi-camera frame manager.

    Usage
    -----
        mgr = CameraManagerV2.from_config("configs/camera_config.json")
        mgr.start()

        # In the inference loop:
        frames = mgr.get_valid_frames()          # only ACTIVE cameras
        if not frames:
            invoke_fallback_policy()
        else:
            run_yolo(frames)
    """

    def __init__(
        self,
        workers: List[_CameraWorker],
        stale_threshold_sec: float = STALE_THRESHOLD_SEC,
        dead_threshold_sec: float = DEAD_THRESHOLD_SEC,
    ) -> None:
        self._workers: Dict[Union[int, str], _CameraWorker] = {
            w.cam_id: w for w in workers
        }
        self.stale_threshold_sec = stale_threshold_sec
        self.dead_threshold_sec = dead_threshold_sec

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        stale_threshold_sec: float = STALE_THRESHOLD_SEC,
        dead_threshold_sec: float = DEAD_THRESHOLD_SEC,
    ) -> "CameraManagerV2":
        import json
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Camera config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        global_cfg = cfg.get("global", {})
        target_fps = float(global_cfg.get("fps", cfg.get("fps", 10.0)) or 10.0)
        fw = int(global_cfg.get("frame_width", cfg.get("frame_width", 0)) or 0)
        fh = int(global_cfg.get("frame_height", cfg.get("frame_height", 0)) or 0)
        global_resize: Optional[Tuple[int, int]] = (fw, fh) if fw and fh else None
        loop_video = bool(cfg.get("loop_video", True))

        workers: List[_CameraWorker] = []
        for idx, cam in enumerate(cfg.get("cameras", [])):
            if not isinstance(cam, dict):
                continue
            source = cam.get("source")
            if source is None:
                continue
            resize_raw = cam.get("resize", global_resize)
            resize: Optional[Tuple[int, int]] = None
            if isinstance(resize_raw, (list, tuple)) and len(resize_raw) == 2:
                resize = (int(resize_raw[0]), int(resize_raw[1]))

            workers.append(
                _CameraWorker(
                    cam_id=cam.get("id", idx),
                    name=str(cam.get("name", f"cam_{idx}")),
                    source=source,
                    target_fps=target_fps,
                    resize=resize,
                    stale_threshold_sec=stale_threshold_sec,
                    dead_threshold_sec=dead_threshold_sec,
                    loop_video=loop_video,
                    enabled=bool(cam.get("enabled", True)),
                )
            )

        if not workers:
            raise ValueError("No camera sources found in config")
        return cls(
            workers=workers,
            stale_threshold_sec=stale_threshold_sec,
            dead_threshold_sec=dead_threshold_sec,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, warmup_sec: float = 1.0) -> Dict[Union[int, str], bool]:
        results = {}
        for cam_id, w in self._workers.items():
            ok = w.start()
            results[cam_id] = ok
            if not ok:
                logger.warning("Camera %s failed to open", cam_id)
        if warmup_sec > 0:
            time.sleep(warmup_sec)
        return results

    def stop(self) -> None:
        for w in self._workers.values():
            w.stop()

    # ── Frame access ───────────────────────────────────────────────────────────

    def get_latest_frame(self, cam_id: Union[int, str]) -> Optional[CameraFrame]:
        """Return the newest frame for one camera, or None."""
        w = self._workers.get(cam_id)
        return w.get_latest() if w is not None else None

    def get_all_frames(self) -> Dict[Union[int, str], Optional[CameraFrame]]:
        """All cameras regardless of health."""
        return {cam_id: w.get_latest() for cam_id, w in self._workers.items()}

    def get_valid_frames(
        self,
        max_age_sec: Optional[float] = None,
    ) -> Dict[Union[int, str], CameraFrame]:
        """
        Return only frames from ACTIVE cameras.

        Parameters
        ----------
        max_age_sec:
            Override the staleness threshold for this call.
            Defaults to ``self.stale_threshold_sec``.

        Returns
        -------
        Dict mapping cam_id → CameraFrame for all healthy cameras.
        Empty dict signals a total camera failure → caller must activate fallback.
        """
        threshold = max_age_sec if max_age_sec is not None else self.stale_threshold_sec
        now = time.time()
        valid: Dict[Union[int, str], CameraFrame] = {}
        for cam_id, w in self._workers.items():
            frame = w.get_latest()
            if frame is None:
                continue
            age = now - frame.timestamp
            if age <= threshold:
                valid[cam_id] = frame
        return valid

    def get_health_map(self) -> Dict[Union[int, str], CameraHealth]:
        return {cam_id: w.health for cam_id, w in self._workers.items()}

    def any_active(self) -> bool:
        return any(w.health == CameraHealth.ACTIVE for w in self._workers.values())

    def status(self) -> List[CameraStatus]:
        return [w.status for w in self._workers.values()]

    @property
    def camera_ids(self) -> List[Union[int, str]]:
        return list(self._workers.keys())

    def wait_until_ready(self, timeout: float = 8.0, min_active: int = 1) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            active = sum(1 for w in self._workers.values() if w.health == CameraHealth.ACTIVE)
            if active >= min_active:
                return True
            time.sleep(0.1)
        return False
