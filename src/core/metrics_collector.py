"""
metrics_collector.py
====================
Paper-quality performance metrics for the traffic AI system.

Tracks (per TASKS §10):
  - Latency: capture_ms, inference_ms, control_ms, end_to_end_ms
  - Throughput: control_hz, inference_hz
  - Queue/buffer: vision_buffer_size
  - Drop rate: frames dropped / frames captured
  - System FPS: moving-average over last N steps

All metrics are written to:
  logs/performance.log   — machine-readable JSONL (one record per flush)
  logs/events.log        — human-readable accident/fallback events

Use ``MetricsCollector.snapshot()`` to get the current stats for the paper.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyRecord:
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    control_ms: float = 0.0
    e2e_ms: float = 0.0          # capture→serial

@dataclass
class ThroughputRecord:
    control_hz: float = 0.0
    inference_hz: float = 0.0
    vehicles_detected: int = 0

@dataclass
class ReliabilityRecord:
    frames_captured: int = 0
    frames_dropped: int = 0
    drop_rate: float = 0.0       # [0, 1]
    active_cameras: int = 0
    total_cameras: int = 0
    fallback_steps: int = 0
    total_steps: int = 0
    fallback_rate: float = 0.0

@dataclass
class MetricsSnapshot:
    timestamp: float = field(default_factory=time.time)
    latency: LatencyRecord = field(default_factory=LatencyRecord)
    throughput: ThroughputRecord = field(default_factory=ThroughputRecord)
    reliability: ReliabilityRecord = field(default_factory=ReliabilityRecord)
    buffer_size: int = 0
    accident_count: int = 0
    window_steps: int = 0        # how many steps averaged over


# ─────────────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Thread-safe metrics accumulator with periodic flush to disk.

    Usage
    -----
        mc = MetricsCollector(log_dir="logs", flush_every=20)
        mc.start()

        # In the control loop:
        mc.record_step(
            capture_ms=5.2,
            inference_ms=42.1,
            control_ms=3.4,
            fallback=False,
            frames_dropped=0,
            active_cams=4,
            total_cams=8,
            vehicles=12,
            buffer_size=3,
        )

        # At end:
        snap = mc.snapshot()
        mc.stop()
    """

    def __init__(
        self,
        log_dir: str = "logs",
        flush_every: int = 20,
        window: int = 100,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._window = window

        # Per-step ring buffers
        self._capture_ms:   Deque[float] = deque(maxlen=window)
        self._inference_ms: Deque[float] = deque(maxlen=window)
        self._control_ms:   Deque[float] = deque(maxlen=window)
        self._e2e_ms:       Deque[float] = deque(maxlen=window)
        self._vehicles:     Deque[int]   = deque(maxlen=window)
        self._buffer_sizes: Deque[int]   = deque(maxlen=window)
        self._ctrl_ts:      Deque[float] = deque(maxlen=window)   # for control_hz
        self._infer_ts:     Deque[float] = deque(maxlen=window)   # for inference_hz

        # Accumulators
        self._lock = threading.Lock()
        self._total_steps: int = 0
        self._fallback_steps: int = 0
        self._frames_captured: int = 0
        self._frames_dropped: int = 0
        self._accident_count: int = 0
        self._active_cams: int = 0
        self._total_cams: int = 0

        # Logging
        self._perf_log_path = self._log_dir / "performance.log"
        self._event_log_path = self._log_dir / "events.log"

        self._perf_logger = self._make_file_logger(
            "perf", str(self._perf_log_path)
        )
        self._event_logger = self._make_file_logger(
            "event", str(self._event_log_path)
        )

    # ── Recording API ──────────────────────────────────────────────────────────

    def record_step(
        self,
        capture_ms: float = 0.0,
        inference_ms: float = 0.0,
        control_ms: float = 0.0,
        fallback: bool = False,
        frames_dropped: int = 0,
        frames_captured: int = 0,
        active_cams: int = 0,
        total_cams: int = 0,
        vehicles: int = 0,
        buffer_size: int = 0,
    ) -> None:
        now = time.time()
        e2e = capture_ms + inference_ms + control_ms

        with self._lock:
            self._capture_ms.append(capture_ms)
            self._inference_ms.append(inference_ms)
            self._control_ms.append(control_ms)
            self._e2e_ms.append(e2e)
            self._vehicles.append(vehicles)
            self._buffer_sizes.append(buffer_size)
            self._ctrl_ts.append(now)

            self._total_steps += 1
            if fallback:
                self._fallback_steps += 1
            self._frames_dropped += frames_dropped
            self._frames_captured += max(frames_captured, 0)
            self._active_cams = active_cams
            self._total_cams = total_cams

            if self._total_steps % self._flush_every == 0:
                self._flush_perf(now)

    def record_inference_tick(self) -> None:
        """Call from InferenceThread on each completed inference cycle."""
        with self._lock:
            self._infer_ts.append(time.time())

    def record_accident(self, cam_id: Any, confidence: float, metadata: Dict[str, Any]) -> None:
        with self._lock:
            self._accident_count += 1
        record = {
            "event": "accident",
            "cam_id": str(cam_id),
            "confidence": float(confidence),
            "timestamp": time.time(),
            **{k: str(v) for k, v in metadata.items()},
        }
        self._event_logger.info(json.dumps(record))

    def record_fallback(self, reason: str, consecutive: int) -> None:
        record = {
            "event": "fallback",
            "reason": reason,
            "consecutive": consecutive,
            "timestamp": time.time(),
        }
        self._event_logger.warning(json.dumps(record))

    # ── Snapshot (for paper tables) ────────────────────────────────────────────

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            lat = LatencyRecord(
                capture_ms=_avg(self._capture_ms),
                inference_ms=_avg(self._inference_ms),
                control_ms=_avg(self._control_ms),
                e2e_ms=_avg(self._e2e_ms),
            )
            ctrl_hz = _hz_from_ts(self._ctrl_ts)
            infer_hz = _hz_from_ts(self._infer_ts)

            thr = ThroughputRecord(
                control_hz=ctrl_hz,
                inference_hz=infer_hz,
                vehicles_detected=int(_avg(self._vehicles) * max(len(self._vehicles), 1)),
            )

            fc = max(self._frames_captured, 1)
            rel = ReliabilityRecord(
                frames_captured=self._frames_captured,
                frames_dropped=self._frames_dropped,
                drop_rate=min(self._frames_dropped / fc, 1.0),
                active_cameras=self._active_cams,
                total_cameras=self._total_cams,
                fallback_steps=self._fallback_steps,
                total_steps=self._total_steps,
                fallback_rate=(self._fallback_steps / max(self._total_steps, 1)),
            )

            return MetricsSnapshot(
                timestamp=time.time(),
                latency=lat,
                throughput=thr,
                reliability=rel,
                buffer_size=int(_avg(self._buffer_sizes)),
                accident_count=self._accident_count,
                window_steps=len(self._capture_ms),
            )

    def log_summary(self) -> None:
        s = self.snapshot()
        logger.info(
            "METRICS | steps=%d | ctrl_hz=%.2f | infer_hz=%.2f | "
            "e2e_ms=%.1f | drop_rate=%.3f | fallback_rate=%.3f | accidents=%d",
            s.reliability.total_steps,
            s.throughput.control_hz,
            s.throughput.inference_hz,
            s.latency.e2e_ms,
            s.reliability.drop_rate,
            s.reliability.fallback_rate,
            s.accident_count,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _flush_perf(self, now: float) -> None:
        s = self.snapshot()
        record = {
            "timestamp": now,
            "steps": s.reliability.total_steps,
            **asdict(s.latency),
            **asdict(s.throughput),
            "drop_rate": s.reliability.drop_rate,
            "fallback_rate": s.reliability.fallback_rate,
            "active_cams": s.reliability.active_cameras,
            "buffer_size": s.buffer_size,
            "accidents": s.accident_count,
        }
        self._perf_logger.info(json.dumps(record))

    @staticmethod
    def _make_file_logger(name: str, path: str) -> logging.Logger:
        log = logging.getLogger(f"traffic.{name}")
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(fh)
            log.propagate = False
        return log

    # ── Context manager support ────────────────────────────────────────────────

    def start(self) -> "MetricsCollector":
        return self

    def stop(self) -> None:
        snap = self.snapshot()
        self._flush_perf(time.time())
        self.log_summary()
        logger.info("MetricsCollector stopped. Final snapshot: %s", snap)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _avg(buf: Deque) -> float:
    if not buf:
        return 0.0
    return sum(buf) / len(buf)


def _hz_from_ts(ts_buf: Deque[float]) -> float:
    """Estimate Hz from a ring buffer of timestamps."""
    if len(ts_buf) < 2:
        return 0.0
    span = ts_buf[-1] - ts_buf[0]
    if span <= 0.0:
        return 0.0
    return min((len(ts_buf) - 1) / span, 1000.0)
