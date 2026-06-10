"""
inference_engine.py
===================
Three-thread pipeline for real-time YOLO + MAPPO inference.

Thread topology
---------------

  ┌──────────────────────┐
  │  CameraManagerV2     │  ← N capture threads (one per camera)
  │  deque(maxlen=1) ×N  │    always holds the freshest frame
  └──────────┬───────────┘
             │  get_valid_frames()
             ▼
  ┌──────────────────────┐
  │  InferenceThread     │  ← runs YOLO + ByteTrack per camera
  │  deque(maxlen=1)     │    puts DetectorSnapshot in single-slot buffer
  └──────────┬───────────┘
             │  get_latest_snapshot()
             ▼
  ┌──────────────────────┐
  │  ControlThread       │  ← VisionToState → VisionBuffer → policy.predict()
  │                      │    → SerialBridge
  └──────────────────────┘

Key design choices
------------------
- `deque(maxlen=1)` between every stage: latest-only, zero blocking.
- Threads communicate only through queues — no shared mutable state.
- The control thread runs at a fixed, user-specified Hz independently of camera FPS.
- If no valid snapshot is available, the control thread invokes the fail-safe policy.
- Graceful shutdown: stop_event signals all threads; join order is
  control → inference → camera.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectorSnapshot:
    """Output of the inference thread."""
    data: Dict[Union[int, str], Dict[str, Any]]     # cam_id → parsed camera dict
    timestamp: float = field(default_factory=time.time)
    inference_ms: float = 0.0


@dataclass
class ControlResult:
    """Output of the control thread."""
    actions: Dict[str, int]                          # tls_id → phase
    serial_phases: List[int]                         # 8 RYG states (0=R,1=Y,2=G)
    timestamp: float = field(default_factory=time.time)
    control_ms: float = 0.0
    fallback: bool = False                           # True if default policy used


# ─────────────────────────────────────────────────────────────────────────────
# Drop counter
# ─────────────────────────────────────────────────────────────────────────────

class DropCounter:
    """Thread-safe frame-drop accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._per_cam: Dict[Union[int, str], int] = {}

    def record(self, cam_id: Union[int, str], n: int = 1) -> None:
        with self._lock:
            self._total += n
            self._per_cam[cam_id] = self._per_cam.get(cam_id, 0) + n

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def per_cam(self, cam_id: Union[int, str]) -> int:
        with self._lock:
            return self._per_cam.get(cam_id, 0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"total": self._total, "per_cam": dict(self._per_cam)}


# ─────────────────────────────────────────────────────────────────────────────
# Moving-average FPS meter
# ─────────────────────────────────────────────────────────────────────────────

class FPSMeter:
    """
    Stable FPS estimator using a sliding window of timestamps.

    Fixes the original spike bug (division by near-zero dt) by:
    1. Using a rolling ring buffer of N timestamps.
    2. Computing fps = (N-1) / (t_last - t_first) over the window.
    3. Clamping to [0, max_fps].
    """

    def __init__(self, window: int = 30, max_fps: float = 120.0) -> None:
        self._window = max(window, 2)
        self._max_fps = max_fps
        self._ts: Deque[float] = deque(maxlen=self._window)

    def tick(self) -> None:
        self._ts.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        if span <= 0.0:
            return 0.0
        return min((len(self._ts) - 1) / span, self._max_fps)

    def reset(self) -> None:
        self._ts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Inference thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread:
    """
    Pulls the latest frames from CameraManagerV2 and runs YOLO+ByteTrack.

    The result is placed in a deque(maxlen=1) for the control thread to consume.
    """

    def __init__(
        self,
        camera_manager: Any,            # CameraManagerV2
        detector_manager: Any,          # DetectorManager (from detector.py)
        drop_counter: DropCounter,
        target_fps: float = 10.0,
        max_age_sec: float = 2.0,
    ) -> None:
        self._cam_mgr = camera_manager
        self._det_mgr = detector_manager
        self._drop_counter = drop_counter
        self._target_fps = max(target_fps, 1.0)
        self._max_age_sec = max_age_sec

        # Output: single-slot snapshot buffer
        self._snapshot_slot: Deque[DetectorSnapshot] = deque(maxlen=1)
        self._slot_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_meter = FPSMeter()
        self._last_frame_counts: Dict[Union[int, str], int] = {}

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="infer-thread", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def get_latest_snapshot(self) -> Optional[DetectorSnapshot]:
        with self._slot_lock:
            if not self._snapshot_slot:
                return None
            return self._snapshot_slot[-1]

    @property
    def fps(self) -> float:
        return self._fps_meter.fps

    def _loop(self) -> None:
        period = 1.0 / self._target_fps
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._step()
                self._fps_meter.tick()
            except Exception:
                logger.exception("InferenceThread error — continuing")

            elapsed = time.perf_counter() - t0
            sleep_t = max(period - elapsed, 0.0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _step(self) -> None:
        t_infer = time.perf_counter()

        # ── 1. Get valid frames (ACTIVE cameras only) ──────────────────────────
        valid_frames = self._cam_mgr.get_valid_frames(max_age_sec=self._max_age_sec)
        if not valid_frames:
            return   # no valid cameras — control thread will fall back

        # ── 2. Update drop counter ─────────────────────────────────────────────
        for cam_id, cf in valid_frames.items():
            prev = self._last_frame_counts.get(cam_id)
            if prev is not None and cf.frame_count > prev + 1:
                self._drop_counter.record(cam_id, cf.frame_count - prev - 1)
            self._last_frame_counts[cam_id] = cf.frame_count

        # ── 3. Run per-camera YOLO inference ──────────────────────────────────
        #  DetectorManager.process_frames() accepts {cam_id: CameraFrame}
        #  and returns a snapshot dict.  This keeps YOLO state encapsulated.
        snapshot_data: Dict[Union[int, str], Dict[str, Any]] = {}

        for cam_id, cf in valid_frames.items():
            try:
                result = self._det_mgr.process_single_frame(
                    cam_id=cam_id,
                    frame=cf.frame,
                    timestamp=cf.timestamp,
                    frame_idx=cf.frame_count,
                )
                if result is not None:
                    snapshot_data[cam_id] = result
            except Exception:
                logger.exception("YOLO inference failed for cam %s", cam_id)

        if not snapshot_data:
            return

        inference_ms = (time.perf_counter() - t_infer) * 1000.0
        snap = DetectorSnapshot(
            data=snapshot_data,
            timestamp=time.time(),
            inference_ms=inference_ms,
        )

        with self._slot_lock:
            self._snapshot_slot.append(snap)


# ─────────────────────────────────────────────────────────────────────────────
# Control thread
# ─────────────────────────────────────────────────────────────────────────────

class ControlThread:
    """
    Runs the MAPPO policy at a fixed control frequency (default 2 Hz).

    If no valid snapshot is available from the inference thread, it invokes
    a fixed-cycle fallback policy and logs clearly.

    Action smoothing
    ----------------
    A phase switch is only committed if:
    1. The new action differs from the current phase.
    2. The current phase has been held for >= min_phase_sec.
    3. (Optional) epsilon-greedy: with probability epsilon, keep current phase.

    This prevents the policy from rapidly toggling phases (hardware stress,
    traffic disruption) and mirrors the min_green_time constraint from training.
    """

    def __init__(
        self,
        inference_thread: InferenceThread,
        policy_interface: Any,           # PolicyInterface
        state_bridge: Any,               # VisionToState
        vision_buffer: Any,              # VisionBuffer
        tls_ids: List[str],
        per_agent_dim: int,
        send_to_hardware: Callable[[List[int]], None],
        control_hz: float = 2.0,
        min_phase_sec: float = 10.0,
        epsilon: float = 0.0,
        fallback_green_sec: int = 30,
        yellow_duration: float = 3.0,
        red_gap: float = 1.0,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._infer = inference_thread
        self._policy = policy_interface
        self._bridge = state_bridge
        self._buffer = vision_buffer
        self._tls_ids = tls_ids
        self._per_agent_dim = per_agent_dim
        self._send_hw = send_to_hardware
        self._period = 1.0 / max(control_hz, 0.1)
        self._min_phase_sec = min_phase_sec
        self._epsilon = epsilon
        self._fallback_green_sec = fallback_green_sec
        self._yellow_duration = yellow_duration
        self._red_gap = red_gap
        self._metrics_cb = metrics_callback

        # ── Runtime state ──────────────────────────────────────────────────────
        self._phase_map: Dict[str, int] = {t: 0 for t in tls_ids}
        self._green_timers: Dict[str, float] = {t: 0.0 for t in tls_ids}
        self._phase_start: Dict[str, float] = {t: time.time() for t in tls_ids}
        # Tracks last-sent per-direction states to know which dirs were GREEN
        self._prev_serial_states: List[int] = [0] * (len(tls_ids) * 4)
        self._consecutive_fallbacks: int = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_meter = FPSMeter()
        self._last_result: Optional[ControlResult] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ctrl-thread", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def get_last_result(self) -> Optional[ControlResult]:
        return self._last_result

    @property
    def fps(self) -> float:
        return self._fps_meter.fps

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                result = self._step()
                self._last_result = result
                self._fps_meter.tick()
            except Exception:
                logger.exception("ControlThread error — continuing")

            elapsed = time.perf_counter() - t0
            sleep_t = max(self._period - elapsed, 0.0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _step(self) -> ControlResult:
        t_ctrl = time.perf_counter()
        now = time.time()

        # Update green timers
        for tls_id in self._tls_ids:
            self._green_timers[tls_id] = now - self._phase_start.get(tls_id, now)

        # ── Pull latest inference snapshot ─────────────────────────────────────
        snap = self._infer.get_latest_snapshot()

        if snap is None or not snap.data:
            return self._fallback_control(now, t_ctrl, reason="no_snapshot")

        # ── Staleness check ────────────────────────────────────────────────────
        snap_age = now - snap.timestamp
        if snap_age > 5.0:
            return self._fallback_control(now, t_ctrl, reason="stale_snapshot")

        # ── Build observation ──────────────────────────────────────────────────
        try:
            packet = self._bridge.build_packet(
                detector_snapshot=snap.data,
                phase_map=self._phase_map,
                green_timers=self._green_timers,
                return_semantic=False,
            )
        except Exception:
            logger.exception("VisionToState failed")
            return self._fallback_control(now, t_ctrl, reason="bridge_error")

        buffered = self._buffer.push(packet)
        obs: np.ndarray = buffered.obs      # shape (obs_dim,), float32

        # Slice per-agent observations
        per_agent_obs: Dict[str, np.ndarray] = {}
        for i, tls_id in enumerate(self._tls_ids):
            lo = i * self._per_agent_dim
            hi = lo + self._per_agent_dim
            per_agent_obs[tls_id] = obs[lo:hi]

        # Assertion: shape must match training exactly
        for tls_id, a_obs in per_agent_obs.items():
            assert a_obs.shape[0] == self._per_agent_dim, (
                f"Obs shape mismatch for {tls_id}: got {a_obs.shape[0]}, "
                f"expected {self._per_agent_dim}"
            )

        # ── Policy inference ───────────────────────────────────────────────────
        try:
            raw_actions = self._policy.predict(per_agent_obs, deterministic=True)
        except Exception:
            logger.exception("Policy predict() failed")
            return self._fallback_control(now, t_ctrl, reason="policy_error")

        # ── Action smoothing + min phase duration ─────────────────────────────
        final_actions = self._smooth_actions(raw_actions, now)

        # ── Update phase map ───────────────────────────────────────────────────
        for tls_id, action in final_actions.items():
            if action != self._phase_map.get(tls_id):
                self._phase_start[tls_id] = now
                logger.debug("TLS %s phase %d → %d", tls_id,
                             self._phase_map.get(tls_id, -1), action)
            self._phase_map[tls_id] = action

        self._consecutive_fallbacks = 0

        # ── Send to hardware ───────────────────────────────────────────────────
        serial_phases = self._actions_to_serial_phases(final_actions)
        self._send_hw(serial_phases)

        ctrl_ms = (time.perf_counter() - t_ctrl) * 1000.0

        if self._metrics_cb:
            self._metrics_cb({
                "control_ms": ctrl_ms,
                "inference_ms": snap.inference_ms,
                "snap_age_ms": snap_age * 1000.0,
                "fallback": False,
            })

        return ControlResult(
            actions=final_actions,
            serial_phases=serial_phases,
            timestamp=now,
            control_ms=ctrl_ms,
            fallback=False,
        )

    def _smooth_actions(
        self,
        raw_actions: Dict[str, int],
        now: float,
    ) -> Dict[str, int]:
        """
        Apply minimum phase duration and optional epsilon-greedy noise.

        A phase switch is allowed only when the current phase has been held for
        at least min_phase_sec seconds.  This mirrors the training constraint
        (min_green_time) and prevents hardware-level rapid switching.
        """
        import random
        smoothed: Dict[str, int] = {}
        for tls_id in self._tls_ids:
            new_action = raw_actions.get(tls_id, 0)
            current = self._phase_map.get(tls_id, 0)
            time_in_phase = now - self._phase_start.get(tls_id, now)

            # Epsilon-greedy: occasionally keep current action
            if self._epsilon > 0.0 and random.random() < self._epsilon:
                smoothed[tls_id] = current
                continue

            # Enforce minimum phase duration
            if new_action != current and time_in_phase < self._min_phase_sec:
                smoothed[tls_id] = current   # hold current phase
            else:
                smoothed[tls_id] = new_action
        return smoothed

    def _fallback_control(
        self,
        now: float,
        t_ctrl: float,
        reason: str,
    ) -> ControlResult:
        """
        Fixed-time-cycle fallback when no valid camera/policy output exists.

        Alternates phases on a simple 30s/30s timer — identical to a legacy
        fixed-time controller — so traffic keeps moving safely.
        """
        self._consecutive_fallbacks += 1
        if self._consecutive_fallbacks == 1 or self._consecutive_fallbacks % 10 == 0:
            logger.warning(
                "FALLBACK policy active (reason=%s, consecutive=%d)",
                reason, self._consecutive_fallbacks,
            )

        CYCLE_SEC = self._fallback_green_sec * 2  # full cycle
        for tls_id in self._tls_ids:
            elapsed = now - self._phase_start.get(tls_id, now)
            new_phase = 0 if (elapsed % CYCLE_SEC) < self._fallback_green_sec else 1
            if new_phase != self._phase_map.get(tls_id):
                self._phase_start[tls_id] = now
            self._phase_map[tls_id] = new_phase

        actions = dict(self._phase_map)
        serial_phases = self._actions_to_serial_phases(actions)
        self._send_hw(serial_phases)

        ctrl_ms = (time.perf_counter() - t_ctrl) * 1000.0
        if self._metrics_cb:
            self._metrics_cb({
                "control_ms": ctrl_ms,
                "inference_ms": 0.0,
                "snap_age_ms": 0.0,
                "fallback": True,
            })

        return ControlResult(
            actions=actions,
            serial_phases=serial_phases,
            timestamp=now,
            control_ms=ctrl_ms,
            fallback=True,
        )

    def _actions_to_serial_phases(self, actions: Dict[str, int]) -> List[int]:
        """Convert {tls_id: phase} to 8-value RYG list for Arduino.

        0=RED  1=YELLOW  2=GREEN
        Phase 0 → dirs 0,1 GREEN / dirs 2,3 RED.
        Phase 1 → dirs 0,1 RED  / dirs 2,3 GREEN.
        """
        _RED, _GREEN = 0, 2
        states: List[int] = []
        for tls_id in self._tls_ids:
            phase = int(actions.get(tls_id, 0))
            phase = 0 if phase <= 0 else 1
            for d in range(4):
                group = d // 2   # dirs 0,1 → group 0; dirs 2,3 → group 1
                states.append(_GREEN if group == phase else _RED)
        return states


# ─────────────────────────────────────────────────────────────────────────────
# Accident thread
# ─────────────────────────────────────────────────────────────────────────────

class AccidentThread:
    """
    Lightweight accident-only detector running independently of InferenceThread.

    Motivation
    ----------
    InferenceThread processes all 8 cameras sequentially for vehicle tracking
    (416 px, ByteTrack).  At ~50 ms/camera on CPU that gives each camera ~2.5 fps
    — fine for traffic-signal control but too slow for accident streak confirmation.

    AccidentThread runs a *separate* YOLO instance at 320 px, no tracker,
    class 0 only.  Each inference is ~20 ms, so the full 8-camera cycle is
    ~160 ms → ~5 fps per camera.  This keeps the 3-frame confirmation window
    below 600 ms, acceptable for alerting.

    The two threads are fully independent: no shared model, no lock.
    AccidentThread feeds results directly into AccidentEventDetector.update().
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        camera_manager: Any,
        accident_detector: Any,
        drop_counter: DropCounter,
        conf_thresh: float = 0.6,
        imgsz: int = 320,
        use_openvino: bool = True,
        target_fps: float = 5.0,
        max_age_sec: float = 5.0,
    ) -> None:
        from pathlib import Path
        from src.vision.detector import _load_yolo
        self._model = _load_yolo(Path(model_path), use_openvino)
        self._cam_mgr = camera_manager
        self._accident_detector = accident_detector
        self._drop_counter = drop_counter
        self._conf_thresh = float(conf_thresh)
        self._imgsz = int(imgsz)
        self._target_fps = max(float(target_fps), 1.0)
        self._max_age_sec = float(max_age_sec)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_meter = FPSMeter()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="accident-thread", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    @property
    def fps(self) -> float:
        return self._fps_meter.fps

    def _loop(self) -> None:
        period = 1.0 / self._target_fps
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._step()
                self._fps_meter.tick()
            except Exception:
                logger.exception("AccidentThread error — continuing")
            elapsed = time.perf_counter() - t0
            sleep_t = max(period - elapsed, 0.0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _step(self) -> None:
        valid_frames = self._cam_mgr.get_valid_frames(max_age_sec=self._max_age_sec)
        if not valid_frames:
            return

        snapshot: Dict[Union[int, str], Dict[str, Any]] = {}
        for cam_id, cf in valid_frames.items():
            try:
                results = self._model.predict(
                    cf.frame,
                    imgsz=self._imgsz,
                    classes=[0],
                    conf=self._conf_thresh,
                    verbose=False,
                )
                boxes = getattr(results[0], "boxes", None)
                accident_found = False
                accident_conf = 0.0
                if boxes is not None and len(boxes) > 0:
                    for cls_id, conf in zip(
                        boxes.cls.cpu().numpy().astype(int),
                        boxes.conf.cpu().numpy(),
                    ):
                        if cls_id == 0:
                            accident_found = True
                            accident_conf = max(accident_conf, float(conf))

                snapshot[cam_id] = {
                    "cam_id": cam_id,
                    "name": cf.name,
                    "timestamp": cf.timestamp,
                    "frame_count": cf.frame_count,
                    "accident_found": accident_found,
                    "accident_conf": accident_conf,
                    "vehicle_count": 0,
                    "tracks": [],
                    "annotated_frame": cf.frame if accident_found else None,
                }
            except Exception:
                logger.exception("AccidentThread inference failed for cam %s", cam_id)

        if snapshot:
            self._accident_detector.update(snapshot)
