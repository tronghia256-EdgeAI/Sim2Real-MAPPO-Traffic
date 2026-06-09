"""
orchestrator_v2.py
==================
Production-grade orchestrator for MAPPO-based intelligent traffic control.

Architecture
------------

  ┌─────────────────────────────────────────────────────────┐
  │                    TrafficOrchestrator                  │
  │                                                         │
  │  CameraManagerV2                                        │
  │    N × capture_thread → deque(maxlen=1)                 │
  │          ↓                                              │
  │  InferenceThread                                        │
  │    YOLO+ByteTrack per camera → DetectorSnapshot         │
  │    deque(maxlen=1)                                      │
  │          ↓                                              │
  │  ControlThread  (@ control_hz)                          │
  │    VisionToState → VisionBuffer → LoadedPolicy          │
  │    → action smoothing + min_phase_sec                   │
  │    → SerialBridge                                       │
  │          ↓                                              │
  │  MetricsCollector  (background flush)                   │
  │  AccidentEventDetector  (3-frame confirmation)          │
  └─────────────────────────────────────────────────────────┘

Fail-safe guarantee
-------------------
- If no ACTIVE cameras exist → ControlThread uses fixed-cycle fallback policy
- If policy predict() raises → ControlThread uses fixed-cycle fallback policy
- Fallback is logged to logs/events.log every time
- System NEVER runs with unvalidated obs_dim

Run
---
    python orchestrator_v2.py \
        --config configs/state_config.json \
        --camera-config configs/camera_config.json \
        --model models/yolo/best.pt \
        --policy models/mappo/20260418_215140/best_model.pt \
        --serial-port COM3 \
        --hz 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Project imports ────────────────────────────────────────────────────────────
# (adjust to your actual package structure)
from camera_manager_v2 import CameraManagerV2, CameraHealth
from inference_engine import InferenceThread, ControlThread, DropCounter
from policy_loader import PolicyLoader
from metrics_collector import MetricsCollector

# ── Configure root logger ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("orchestrator_v2")


# ═════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lane_rois(lane_defs: Dict[str, Any]):
    """Deferred import to avoid circular dependency at module load."""
    from src.vision.state_extractor import LaneROI
    rois = {}
    for lane_id, ldef in lane_defs.items():
        poly = [(float(pt[0]), float(pt[1])) for pt in ldef.get("polygon", [])]
        rois[lane_id] = LaneROI(
            lane_id=lane_id,
            tls_id=str(ldef["tls_id"]),
            polygon=poly,
            cam_id=ldef.get("cam_id"),
            lane_length_px=float(ldef.get("lane_length_px") or 0) or None,
            lane_width_px=float(ldef.get("lane_width_px") or 0) or None,
        )
    return rois


def build_lane_groups(cfg_groups: Dict, tls_ids: List[str]):
    groups = {}
    for tls_id in tls_ids:
        entry = cfg_groups.get(tls_id)
        if isinstance(entry, list) and len(entry) == 2:
            groups[tls_id] = (list(entry[0]), list(entry[1]))
    return groups


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class TrafficOrchestrator:
    """
    Owns component lifecycles and wires the 3-thread pipeline.
    """

    def __init__(
        self,
        state_config_path: str,
        camera_config_path: str,
        model_path: str,
        policy_path: Optional[str] = None,
        tele_config_path: Optional[str] = None,
        serial_port: Optional[str] = None,
        serial_baud: int = 115_200,
        control_hz: float = 2.0,
        inference_fps: float = 10.0,
        buffer_maxlen: int = 5,
        buffer_ema_alpha: float = 0.6,
        min_phase_sec: float = 10.0,
        epsilon: float = 0.0,
        stale_threshold_sec: float = 2.0,
        dead_threshold_sec: float = 10.0,
        log_dir: str = "logs",
        metrics_flush_every: int = 20,
    ) -> None:

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # ── 1. State config ────────────────────────────────────────────────────
        cfg = load_json(state_config_path)
        self._validate_config(cfg)

        tls_ids: List[str] = cfg["tls_ids"]
        sp = cfg.get("system_params", {})
        max_lanes = int(cfg.get("traffic_lights", {}).get("max_lanes_per_tls", 4))

        # Compute per-agent obs dim from config
        # per_agent = max_lanes * lane_feature_dim + tls_feature_dim = 4*5+6 = 26
        vl = cfg.get("vector_layout", {})
        per_agent_dim = int(vl.get("per_agent_observation_dim", 26))
        logger.info("obs layout: per_agent_dim=%d, tls_ids=%s", per_agent_dim, tls_ids)

        # ── 2. Lane ROIs + groups ──────────────────────────────────────────────
        lane_rois = build_lane_rois(cfg["lane_definitions"])
        lane_groups = build_lane_groups(cfg.get("lane_groups", {}), tls_ids)

        # ── 3. StateExtractor + VisionToState ──────────────────────────────────
        from src.vision.state_extractor import StateExtractor
        from src.adapters.vision_to_state import VisionToState

        extractor = StateExtractor(
            tls_ids=tls_ids,
            controlled_lanes_dict=cfg["controlled_lanes_dict"],
            lane_rois=lane_rois,
            lane_groups=lane_groups,
            max_lanes_per_tls=max_lanes,
            speed_cap=float(sp.get("speed_cap", 15.0)),
            max_green_time=float(sp.get("max_green_time", 90.0)),
            px_per_meter=float(sp.get("px_per_meter", 20.0)),
            stop_speed_m_s=float(sp.get("stop_speed_m_s", 0.1)),
            waiting_cap=float(sp.get("waiting_cap", 300.0)),
            vehicle_cap=float(sp.get("vehicle_cap", 20.0)),
            queue_cap=float(sp.get("queue_cap", 50.0)),
            fps=float(sp.get("fps", 10.0)),
            motorbike_class_ids=[3],         # fix C-1: custom model IDs
            heavy_class_ids=[1, 4],
        )

        # Validate obs_dim at startup — hard fail, not silent
        actual_obs_dim = extractor.get_obs_dim()
        expected_obs_dim = per_agent_dim * len(tls_ids)
        if actual_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"obs_dim mismatch: StateExtractor produced {actual_obs_dim}, "
                f"state_config expects {expected_obs_dim}. "
                "Update state_config.json or StateExtractor params."
            )
        logger.info("obs_dim validated: %d (= %d agents × %d)",
                    actual_obs_dim, len(tls_ids), per_agent_dim)

        bridge = VisionToState(
            state_extractor=extractor,
            strict_shape_check=True,
            attach_timestamp=True,
        )

        # ── 4. VisionBuffer ────────────────────────────────────────────────────
        from src.buffer.vision_buffer import VisionBuffer
        vision_buffer = VisionBuffer(
            maxlen=buffer_maxlen,
            ema_alpha=buffer_ema_alpha,
            use_median=False,
            clip_obs=True,
        )

        # ── 5. Camera manager ──────────────────────────────────────────────────
        self._cam_mgr = CameraManagerV2.from_config(
            camera_config_path,
            stale_threshold_sec=stale_threshold_sec,
            dead_threshold_sec=dead_threshold_sec,
        )

        # ── 6. Detector manager ────────────────────────────────────────────────
        from src.vision.detector import DetectorManager
        self._det_mgr = DetectorManager(
            model_path=model_path,
            camera_manager=self._cam_mgr,
            conf_thresh=float(sp.get("conf_thresh", 0.5)),
            accident_conf_thresh=0.7,
            accident_class_id=0,
            vehicle_class_ids=[1, 2, 3, 4],
            tracker_yaml="bytetrack.yaml",
        )

        # ── 7. Accident detector ───────────────────────────────────────────────
        from src.vision.event_detector import AccidentEventDetector
        from src.services.alert_services import TelegramAlertService

        alert_service: Optional[TelegramAlertService] = None
        if tele_config_path and Path(tele_config_path).exists():
            try:
                alert_service = TelegramAlertService.from_config(tele_config_path)
                logger.info("Telegram alerts active")
            except Exception:
                logger.exception("Telegram init failed — alerts disabled")

        self._accident_detector = AccidentEventDetector(
            alert_service=alert_service or TelegramAlertService(
                bot_token="", chat_id="", enabled=False
            ),
            accident_class_id=0,
            accident_conf_thresh=0.7,
            confirm_frames=3,        # ← temporal consistency fix
            cooldown_sec=60.0,
            position=cfg.get("position", "INTERSECTION_A"),
            enabled=True,
            event_dir=f"{log_dir}/events",
        )

        # ── 8. Policy ──────────────────────────────────────────────────────────
        if policy_path:
            self._policy = PolicyLoader(
                checkpoint_path=policy_path,
                obs_dim=per_agent_dim,
                action_dim=2,
            ).load()
            logger.info("Policy loaded: %s", self._policy)
        else:
            self._policy = None
            logger.warning("No policy path provided — fallback-only mode")

        # ── 9. Serial ──────────────────────────────────────────────────────────
        self._serial = None
        if serial_port:
            try:
                from src.utils.serial_bridge import SerialBridge
                self._serial = SerialBridge(port=serial_port, baudrate=serial_baud)
                logger.info("Serial ready on %s", serial_port)
            except Exception:
                logger.warning("Serial unavailable — dry-run mode")

        # ── 10. Drop counter + metrics ─────────────────────────────────────────
        self._drop_counter = DropCounter()
        self._metrics = MetricsCollector(
            log_dir=log_dir,
            flush_every=metrics_flush_every,
        )

        # ── 11. Inference thread ───────────────────────────────────────────────
        self._infer_thread = InferenceThread(
            camera_manager=self._cam_mgr,
            detector_manager=self._det_mgr,
            drop_counter=self._drop_counter,
            target_fps=inference_fps,
            max_age_sec=stale_threshold_sec,
        )

        # ── 12. Control thread ─────────────────────────────────────────────────
        self._ctrl_thread = ControlThread(
            inference_thread=self._infer_thread,
            policy_interface=self._policy,
            state_bridge=bridge,
            vision_buffer=vision_buffer,
            tls_ids=tls_ids,
            per_agent_dim=per_agent_dim,
            send_to_hardware=self._send_to_hardware,
            control_hz=control_hz,
            min_phase_sec=min_phase_sec,
            epsilon=epsilon,
            fallback_green_sec=30,
            metrics_callback=self._on_control_step,
        )

        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=== TrafficOrchestrator starting ===")

        # Start cameras
        open_results = self._cam_mgr.start(warmup_sec=1.0)
        active = [c for c, ok in open_results.items() if ok]
        failed = [c for c, ok in open_results.items() if not ok]
        if failed:
            logger.warning("Cameras that failed to open: %s", failed)

        # Wait for at least one active camera
        if not self._cam_mgr.wait_until_ready(timeout=8.0, min_active=1):
            logger.critical(
                "No camera became ACTIVE within 8s — starting in fallback-only mode"
            )
            # DO NOT crash; ControlThread will use the fallback policy

        # Start detector + inference
        self._det_mgr.start()
        self._infer_thread.start()

        # Wait for first inference snapshot
        time.sleep(1.5)

        # Start control loop
        self._ctrl_thread.start()
        self._metrics.start()
        self._running = True

        logger.info(
            "=== Orchestrator running | cameras=%d/%d active ===",
            len(active), len(open_results),
        )

    def stop(self) -> None:
        logger.info("=== Stopping TrafficOrchestrator ===")
        self._running = False
        # Stop in reverse dependency order
        self._ctrl_thread.stop()
        self._infer_thread.stop()
        self._det_mgr.stop()
        self._cam_mgr.stop()
        self._metrics.stop()
        logger.info("=== Stopped ===")

    def run_until_interrupted(self) -> None:
        """Block the main thread, periodically logging health status."""
        self.start()

        def _sig_handler(sig, frame):
            logger.info("Signal %d received — shutting down", sig)
            self._running = False

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        health_log_interval = 30.0
        last_health = time.time()

        try:
            while self._running:
                time.sleep(0.5)
                now = time.time()
                if now - last_health >= health_log_interval:
                    self._log_health()
                    last_health = now
        finally:
            self.stop()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_control_step(self, step_data: Dict[str, Any]) -> None:
        """Called by ControlThread after every control cycle."""
        health = self._cam_mgr.get_health_map()
        active_cams = sum(1 for h in health.values() if h == CameraHealth.ACTIVE)

        # Run accident detector on the latest snapshot
        snap = self._infer_thread.get_latest_snapshot()
        if snap is not None and snap.data:
            try:
                events = self._accident_detector.update(snap.data)
                for ev in events:
                    self._metrics.record_accident(
                        cam_id=ev.cam_id,
                        confidence=ev.confidence,
                        metadata={"severity": ev.metadata.get("severity", "low")},
                    )
            except Exception:
                logger.exception("AccidentEventDetector error")

        self._metrics.record_step(
            capture_ms=0.0,
            inference_ms=float(step_data.get("inference_ms", 0.0)),
            control_ms=float(step_data.get("control_ms", 0.0)),
            fallback=bool(step_data.get("fallback", False)),
            frames_dropped=self._drop_counter.total,
            active_cams=active_cams,
            total_cams=len(health),
        )

        if step_data.get("fallback"):
            self._metrics.record_fallback(
                reason="see_events_log",
                consecutive=getattr(self._ctrl_thread, "_consecutive_fallbacks", 0),
            )

    def _send_to_hardware(self, phases: List[int]) -> None:
        if self._serial is None:
            return
        if not getattr(self._serial, "connected", False):
            return
        if len(phases) != 8:
            logger.error("Serial expects 8 values, got %d", len(phases))
            return
        try:
            self._serial.pack_and_send_data(phases)
        except Exception:
            logger.exception("Serial send failed")

    def _log_health(self) -> None:
        health = self._cam_mgr.get_health_map()
        ctrl_result = self._ctrl_thread.get_last_result()
        s = self._metrics.snapshot()
        logger.info(
            "HEALTH | cam_active=%d/%d | ctrl_fps=%.2f | infer_fps=%.2f | "
            "e2e_ms=%.1f | drop_rate=%.3f | fallback=%s | accidents=%d",
            sum(1 for h in health.values() if h == CameraHealth.ACTIVE),
            len(health),
            s.throughput.control_hz,
            s.throughput.inference_hz,
            s.latency.e2e_ms,
            s.reliability.drop_rate,
            ctrl_result.fallback if ctrl_result else "N/A",
            s.accident_count,
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_config(cfg: Dict[str, Any]) -> None:
        required = {"tls_ids", "controlled_lanes_dict", "lane_definitions", "system_params"}
        missing = required - cfg.keys()
        if missing:
            raise ValueError(f"state_config missing required keys: {missing}")
        if not cfg.get("tls_ids"):
            raise ValueError("tls_ids must not be empty")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production MAPPO traffic orchestrator v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",          default="configs/state_config.json")
    p.add_argument("--camera-config",   default="configs/camera_config.json")
    p.add_argument("--model",           default="models/yolo/best.pt")
    p.add_argument("--policy",          default=None,   help="MAPPO checkpoint (.pt)")
    p.add_argument("--tele-config",     default="configs/tele.json")
    p.add_argument("--serial-port",     default=None,   help="Arduino serial port")
    p.add_argument("--serial-baud",     type=int, default=115200)
    p.add_argument("--hz",              type=float, default=2.0,  help="Control frequency")
    p.add_argument("--infer-fps",       type=float, default=10.0, help="Inference FPS")
    p.add_argument("--buffer",          type=int,   default=5,    help="VisionBuffer depth")
    p.add_argument("--min-phase-sec",   type=float, default=10.0, help="Min green phase (s)")
    p.add_argument("--epsilon",         type=float, default=0.0,  help="Epsilon-greedy noise")
    p.add_argument("--stale-sec",       type=float, default=2.0,  help="Camera stale threshold")
    p.add_argument("--log-dir",         default="logs")
    p.add_argument("--log-level",       default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    tele = args.tele_config if Path(args.tele_config).exists() else None

    orch = TrafficOrchestrator(
        state_config_path=args.config,
        camera_config_path=args.camera_config,
        model_path=args.model,
        policy_path=args.policy,
        tele_config_path=tele,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        control_hz=args.hz,
        inference_fps=args.infer_fps,
        buffer_maxlen=args.buffer,
        min_phase_sec=args.min_phase_sec,
        epsilon=args.epsilon,
        stale_threshold_sec=args.stale_sec,
        log_dir=args.log_dir,
    )
    orch.run_until_interrupted()


if __name__ == "__main__":
    main()
