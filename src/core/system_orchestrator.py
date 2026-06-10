# DEPRECATED: Use orchestrator_v2.py for production multi-threading deployment.
"""
system_orchestrator.py
======================
Production runtime for MAPPO-based intelligent traffic signal control.

Runtime pipeline
----------------
  MultiCameraManager           ← camera threads
      ↓
  DetectorManager              ← per-camera YOLO + ByteTrack workers
      ↓
  AccidentEventDetector        ← 3-frame confirmation + Telegram alert
      ↓
  VisionToState.build_packet() ← snapshot → (obs, info, semantic)
      ↓
  VisionBuffer.push()          ← EMA-smoothed obs
      ↓
  policy.predict()             ← MAPPO decentralized inference
      ↓
  actions_to_serial_phases()   ← phase → 8-direction RYG states
      ↓
  SerialBridge.pack_and_send() ← Arduino traffic-light controller

Usage
-----
  python src/core/orchestrator_v2.py --config configs/state_config.json \
                                      --camera-config configs/camera_config.json \
                                      --model models/yolo/yolov11.pt \
                                      --policy models/mappo/20260418_215140/best_model.pt \
                                      --serial-port COM3

Inference-only.  No SUMO / traci import.  No training code.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# stdlib
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# third-party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# project modules  (import failures are fatal – no silent swallowing)
# ─────────────────────────────────────────────────────────────────────────────
from src.vision.multi_camera import MultiCameraManager
from src.vision.detector import DetectorManager
from src.vision.event_detector import AccidentEventDetector, AccidentEvent
from src.vision.state_extractor import StateExtractor, LaneROI
from src.adapters.vision_to_state import VisionToState
from src.buffer.vision_buffer import VisionBuffer
from src.utils.serial_bridge import SerialBridge
from src.services.alert_services import TelegramAlertService
from src.core.config_validator import validate_state_config

# ─────────────────────────────────────────────────────────────────────────────
# logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")


class FatalOrchestratorError(RuntimeError):
    """Raised when runtime state/policy is invalid and execution must stop."""


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_state_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate state_config.json with comprehensive checks."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"state_config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Comprehensive config validation (CRITICAL-1, HIGH-1, HIGH-5 fixes)
    logger.info("Validating state configuration...")
    try:
        validate_state_config(cfg)
    except Exception as e:
        raise FatalOrchestratorError(f"State config validation failed: {e}") from e
    
    if "runtime_to_training_tls" in cfg:
        rt_to_train = cfg["runtime_to_training_tls"]
        if not isinstance(rt_to_train, dict):
            raise ValueError("runtime_to_training_tls must be a dict")
        missing_map = [tls for tls in cfg["tls_ids"] if tls not in rt_to_train]
        if missing_map:
            raise ValueError(f"runtime_to_training_tls missing keys: {missing_map}")
    return cfg


def build_lane_rois(lane_definitions: Dict[str, Any]) -> Dict[str, LaneROI]:
    """Convert lane_definitions from state_config.json → LaneROI objects."""
    rois: Dict[str, LaneROI] = {}
    for lane_id, ldef in lane_definitions.items():
        raw_poly = ldef.get("polygon", [])
        polygon: List[Tuple[float, float]] = [
            (float(pt[0]), float(pt[1])) for pt in raw_poly
        ]
        rois[lane_id] = LaneROI(
            lane_id=lane_id,
            tls_id=str(ldef["tls_id"]),
            polygon=polygon,
            cam_id=ldef.get("cam_id"),
            lane_length_px=float(ldef.get("lane_length_px") or 0) or None,
            lane_width_px=float(ldef.get("lane_width_px") or 0) or None,
        )
    return rois


def build_lane_groups(
    lane_groups_cfg: Dict[str, Any],
    tls_ids: List[str],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Convert lane_groups section → {tls_id: (group_a, group_b)}."""
    groups: Dict[str, Tuple[List[str], List[str]]] = {}
    for tls_id in tls_ids:
        entry = lane_groups_cfg.get(tls_id)
        if isinstance(entry, list) and len(entry) == 2:
            groups[tls_id] = (list(entry[0]), list(entry[1]))
    return groups


# ═════════════════════════════════════════════════════════════════════════════
# 2.  STATE EXTRACTOR BUILDER
#     NOTE: The custom YOLO model uses class IDs {1:bus, 2:car, 3:motorcycle,
#     4:truck}, NOT standard COCO {5:bus, 7:truck}.  We patch the heavy-vehicle
#     and motorbike sets here to match the deployment model.
#
#     ── REQUIRED MINIMAL CHANGE IN state_extractor.py ──────────────────────
#     Add two constructor params (see AUDIT REPORT at bottom of this file):
#         motorbike_class_ids: Optional[Sequence[int]] = None,
#         heavy_class_ids:     Optional[Sequence[int]] = None,
#     and inside __init__:
#         if motorbike_class_ids is not None:
#             self._MOTORBIKE_CLASSES = frozenset(motorbike_class_ids)
#         if heavy_class_ids is not None:
#             self._HEAVY_CLASSES = frozenset(heavy_class_ids)
#     ────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

#: Custom model class IDs — change if you retrain with a different label set.
CUSTOM_MOTORBIKE_CLASS_IDS: List[int] = [3]       # motorcycle
CUSTOM_HEAVY_CLASS_IDS:     List[int] = [1, 4]    # bus, truck


def build_state_extractor(
    cfg: Dict[str, Any],
    lane_rois: Dict[str, LaneROI],
    lane_groups: Dict[str, Tuple[List[str], List[str]]],
) -> StateExtractor:
    tls_ids: List[str] = cfg["tls_ids"]
    ctrl_lanes: Dict[str, List[str]] = cfg["controlled_lanes_dict"]
    sp = cfg.get("system_params", {})

    extractor = StateExtractor(
        tls_ids=tls_ids,
        controlled_lanes_dict=ctrl_lanes,
        lane_rois=lane_rois,
        lane_groups=lane_groups,
        max_lanes_per_tls=int(cfg.get("traffic_lights", {}).get("max_lanes_per_tls", 4)),
        speed_cap=float(sp.get("speed_cap", 15.0)),
        max_green_time=float(sp.get("max_green_time", 90.0)),
        px_per_meter=float(sp.get("px_per_meter", 20.0)),
        stop_speed_m_s=float(sp.get("stop_speed_m_s", 0.1)),
        waiting_cap=float(sp.get("waiting_cap", 300.0)),
        vehicle_cap=float(sp.get("vehicle_cap", 20.0)),
        queue_cap=float(sp.get("queue_cap", 50.0)),
        fps=float(sp.get("fps", 10.0)),
        # --- FIX: custom model class IDs (see note above) ---
        motorbike_class_ids=CUSTOM_MOTORBIKE_CLASS_IDS,
        heavy_class_ids=CUSTOM_HEAVY_CLASS_IDS,
    )
    return extractor


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# RYG state constants matching Arduino firmware
_S_RED    = 0
_S_YELLOW = 1
_S_GREEN  = 2


def actions_to_serial_phases(
    tls_ids:      List[str],
    action_map:   Dict[str, int],
    dirs_per_tls: int = 4,
) -> List[int]:
    """
    Convert per-TLS phase decisions to an 8-value RYG list for SerialBridge.

    Layout (for 2 TLS × 4 directions):
        [tls_0_dir0, tls_0_dir1, tls_0_dir2, tls_0_dir3,
         tls_1_dir0, tls_1_dir1, tls_1_dir2, tls_1_dir3]

    Each value is 0=RED, 1=YELLOW, 2=GREEN.
    Phase 0 → group A (dirs 0,1) GREEN, group B (dirs 2,3) RED.
    Phase 1 → group B (dirs 2,3) GREEN, group A (dirs 0,1) RED.
    """
    half = dirs_per_tls // 2
    states: List[int] = []
    for tls_id in tls_ids:
        phase = int(action_map.get(tls_id, 0))
        phase = 0 if phase <= 0 else 1
        for d in range(dirs_per_tls):
            group = d // half   # 0 or 1
            states.append(_S_GREEN if group == phase else _S_RED)
    return states


# ═════════════════════════════════════════════════════════════════════════════
# 4.  POLICY INTERFACE (stub — replace with actual MAPPO / PPO inference)
# ═════════════════════════════════════════════════════════════════════════════

class Actor:
    """MAPPO actor network wrapper matching train_ppo architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        import torch.nn as nn

        self._nn_module = nn.Module()
        # Keep exact attribute name "net" to match checkpoint keys: net.0.*, net.2.*, net.4.*
        self._nn_module.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @property
    def net(self):
        return self._nn_module.net

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load at module level so "net.*" keys resolve exactly.
        self._nn_module.load_state_dict(state_dict, strict=True)

    def eval(self) -> None:
        self._nn_module.eval()

    def forward(self, obs):
        return self._nn_module.net(obs)

    def get_action(self, obs, deterministic: bool = True):
        import torch
        from torch.distributions import Categorical

        logits = self.forward(obs)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        return Categorical(logits=logits).sample()


class PolicyInterface:
    """
    Strict MAPPO actor loader + decentralized inference.
    """

    def __init__(
        self,
        policy_path: Union[str, Path],
        tls_ids: List[str],
        obs_dim: int,
        action_dim: int = 2,
    ) -> None:
        self.tls_ids = tls_ids
        self.obs_dim = obs_dim
        self.action_dim = int(action_dim)
        self._model = self._load(Path(policy_path))

    def _load(self, path: Path):
        import torch

        if not path.exists():
            raise FatalOrchestratorError(f"Policy checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
            raise FatalOrchestratorError(
                "Invalid checkpoint format: expected dict with 'actor_state_dict'"
            )
        model = Actor(obs_dim=self.obs_dim, action_dim=self.action_dim)
        try:
            model.load_state_dict(ckpt["actor_state_dict"])
        except Exception as exc:
            raise FatalOrchestratorError(f"Actor state_dict incompatible: {exc}") from exc
        model.eval()
        logger.info("Policy loaded from %s", path)
        return model

    def predict(
        self,
        per_agent_obs: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> Dict[str, int]:
        """
        Return {tls_id: phase_action (0 or 1)} for each agent.

        Strict mode: raises on any mismatch.
        """
        actions: Dict[str, int] = {}
        import torch

        for tls_id in self.tls_ids:
            if tls_id not in per_agent_obs:
                raise FatalOrchestratorError(f"Missing per-agent obs for tls_id={tls_id}")
            obs = np.asarray(per_agent_obs[tls_id], dtype=np.float32).reshape(-1)
            if obs.shape[0] != self.obs_dim:
                raise FatalOrchestratorError(
                    f"State dimension mismatch for {tls_id}: got {obs.shape[0]}, expected {self.obs_dim}"
                )
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t = self._model.get_action(t, deterministic=deterministic)
            action = int(action_t.item())
            if not (0 <= action < self.action_dim):
                raise FatalOrchestratorError(f"Invalid action {action} for tls_id={tls_id}")
            actions[tls_id] = action
        return actions


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ORCHESTRATOR CLASS
# ═════════════════════════════════════════════════════════════════════════════

class TrafficSystemOrchestrator:
    """
    Top-level orchestrator.  Owns component lifetime (start/stop) and drives
    the real-time control loop.

    Parameters
    ----------
    state_config_path : str | Path
        Path to state_config.json.
    camera_config_path : str | Path
        Path to camera_config.json.
    model_path : str | Path
        YOLO detection model (*.pt).
    policy_path : str | Path
        MAPPO policy checkpoint (required). Invalid/missing checkpoint is fatal.
    tele_config_path : str | Path, optional
        Telegram credentials.  If None or missing, alerts are silently disabled.
    serial_port : str, optional
        Arduino serial port (e.g. "COM3", "/dev/ttyUSB0").
    serial_baud : int
        Baud rate for Arduino.
    control_hz : float
        Target control loop frequency in Hz.
    buffer_maxlen : int
        VisionBuffer depth for temporal smoothing.
    buffer_ema_alpha : float
        EMA alpha for VisionBuffer.
    """

    def __init__(
        self,
        state_config_path: Union[str, Path],
        camera_config_path: Union[str, Path],
        model_path: Union[str, Path],
        policy_path: Optional[Union[str, Path]] = None,
        tele_config_path: Optional[Union[str, Path]] = None,
        serial_port: Optional[str] = "COM3",
        serial_baud: int = 115200,
        yellow_duration: float = 3.0,
        red_gap: float = 1.0,
        control_hz: float = 2.0,
        buffer_maxlen: int = 5,
        buffer_ema_alpha: float = 0.6,
    ) -> None:
        # ── 1. Load config ──────────────────────────────────────────────────
        logger.info("Loading state config from %s", state_config_path)
        self.cfg = load_state_config(state_config_path)
        self.tls_ids: List[str] = self.cfg["tls_ids"]
        self.runtime_to_training_tls: Dict[str, str] = self.cfg.get(
            "runtime_to_training_tls", {tls: tls for tls in self.tls_ids}
        )
        self.training_tls_ids: List[str] = self.cfg.get(
            "training_tls_ids",
            [self.runtime_to_training_tls[tls] for tls in self.tls_ids],
        )
        train_set = set(self.training_tls_ids)
        mapped_set = {self.runtime_to_training_tls[tls] for tls in self.tls_ids}
        if train_set != mapped_set:
            raise FatalOrchestratorError(
                "training_tls_ids mismatch with runtime_to_training_tls mapping"
            )
        self.policy_tls_order: List[str] = sorted(
            self.tls_ids,
            key=lambda tls: self.training_tls_ids.index(self.runtime_to_training_tls[tls]),
        )
        sp = self.cfg.get("system_params", {})
        # Override buffer_ema_alpha from config if present (avoids pre-loading in main())
        _smoothing = sp.get("temporal_smoothing", {})
        if "buffer_ema_alpha" in _smoothing:
            buffer_ema_alpha = float(_smoothing["buffer_ema_alpha"])
            logger.info("EMA alpha overridden from config: %.2f", buffer_ema_alpha)
        self.metrics_log_every = int(sp.get("metrics_log_every", 20))
        self.control_period = 1.0 / max(float(control_hz), 0.1)
        self._phase_action_map: Dict[str, List[int]] = {
            tls: [0, 2] for tls in self.tls_ids
        }
        for tls in self.tls_ids:
            raw = self.cfg.get("phase_action_map", {}).get(tls, [0, 2])
            if isinstance(raw, list) and len(raw) >= 2:
                self._phase_action_map[tls] = [int(raw[0]), int(raw[1])]

        # ── 2. Lane ROIs + groups ────────────────────────────────────────────
        lane_rois   = build_lane_rois(self.cfg["lane_definitions"])
        lane_groups = build_lane_groups(self.cfg.get("lane_groups", {}), self.tls_ids)

        # ── 3. State extractor + VisionToState bridge ────────────────────────
        self._extractor = build_state_extractor(self.cfg, lane_rois, lane_groups)
        self._bridge = VisionToState(
            state_extractor=self._extractor,
            camera_order=None,
            strict_shape_check=True,
            attach_timestamp=True,
        )
        self._per_agent_dim: int = self._extractor.get_obs_dim() // len(self.tls_ids)
        if self._extractor.get_obs_dim() % len(self.tls_ids) != 0:
            raise FatalOrchestratorError("Global obs_dim is not divisible by num TLS agents")
        logger.info(
            "StateExtractor: %d TLS | obs_dim=%d | per_agent_dim=%d",
            len(self.tls_ids), self._extractor.get_obs_dim(), self._per_agent_dim,
        )

        # ── 4. Vision buffer ─────────────────────────────────────────────────
        self._vision_buffer = VisionBuffer(
            maxlen=buffer_maxlen,
            ema_alpha=buffer_ema_alpha,
            use_median=False,
            clip_obs=True,
        )

        # ── 5. Camera manager ────────────────────────────────────────────────
        logger.info("Initialising camera manager from %s", camera_config_path)
        self._camera_mgr = MultiCameraManager(str(camera_config_path))

        # ── 6. Detector manager ──────────────────────────────────────────────
        self._detector_mgr = DetectorManager(
            model_path=model_path,
            camera_manager=self._camera_mgr,
            conf_thresh=float(sp.get("conf_thresh", 0.5)),
            accident_conf_thresh=0.7,
            accident_class_id=0,
            vehicle_class_ids=[1, 2, 3, 4],   # custom model class IDs
            tracker_yaml="bytetrack.yaml",
        )

        # ── 7. Accident event detector ───────────────────────────────────────
        alert_service: Optional[TelegramAlertService] = None
        if tele_config_path is not None and Path(tele_config_path).exists():
            try:
                alert_service = TelegramAlertService.from_config(tele_config_path)
                logger.info("Telegram alert service loaded from %s", tele_config_path)
            except Exception:
                logger.exception("Failed to load Telegram config; alerts disabled")
        else:
            logger.warning("No Telegram config supplied; accident alerts will be suppressed")

        self._accident_detector = AccidentEventDetector(
            alert_service=alert_service,
            tele_config_path=tele_config_path or "configs/tele.json",
            accident_class_id=0,
            accident_conf_thresh=0.7,
            confirm_frames=3,
            cooldown_sec=60.0,
            position=self.cfg.get("position", "INTERSECTION_A"),
            enabled=True,
            event_dir=self.cfg.get("event_dir", "logs/events"),
        )

        # ── 8. Policy ─────────────────────────────────────────────────────────
        if not policy_path:
            raise FatalOrchestratorError("Policy checkpoint is required in strict mode (--policy)")
        self._policy = PolicyInterface(
            policy_path=policy_path,
            tls_ids=self.policy_tls_order,
            obs_dim=self._per_agent_dim,
            action_dim=2,
        )

        # ── 9. Serial bridge ──────────────────────────────────────────────────
        self._serial: Optional[SerialBridge] = None
        if serial_port:
            try:
                self._serial = SerialBridge(port=serial_port, baudrate=serial_baud)
                logger.info("Serial bridge connected on %s @ %d baud", serial_port, serial_baud)
            except Exception:
                logger.exception(
                    "Serial bridge failed to open %s — running in dry-run mode", serial_port
                )

        # ── 10. Runtime state ─────────────────────────────────────────────────
        self._phase_map:    Dict[str, int]   = {t: 0   for t in self.tls_ids}
        self._action_map:   Dict[str, int]   = {t: 0   for t in self.tls_ids}
        self._green_timers: Dict[str, float] = {t: 30.0 for t in self.tls_ids}
        self._phase_start:  Dict[str, float] = {t: time.time() for t in self.tls_ids}
        self._prev_serial_states: List[int]  = [_S_RED] * (len(self.tls_ids) * 4)
        self._yellow_duration: float = yellow_duration
        self._red_gap: float = red_gap
        self._running: bool = False
        self._last_frame_count: Dict[Union[str, int], int] = {}
        self._dropped_frames_total: int = 0
        self._metrics_accum: Dict[str, float] = {
            "capture_ms": 0.0,
            "infer_ms": 0.0,
            "postprocess_ms": 0.0,
            "action_ms": 0.0,
            "fps": 0.0,
            "steps": 0.0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("Starting camera manager …")
        open_results = self._camera_mgr.start(warmup_sec=1.0)
        opened = [cam for cam, ok in open_results.items() if ok]
        failed = [cam for cam, ok in open_results.items() if not ok]
        if failed:
            logger.warning("Cameras failed to open: %s", failed)
        if not opened:
            raise RuntimeError("No cameras opened — cannot start detector.")

        logger.info("Starting detector manager on cameras: %s …", opened)
        self._detector_mgr.start()

        # Allow detector workers to process at least one frame each.
        time.sleep(1.0)
        self._running = True
        logger.info("Orchestrator started — entering control loop at %.1f Hz", 1.0 / self.control_period)

    def stop(self) -> None:
        self._running = False
        logger.info("Stopping detector manager …")
        self._detector_mgr.stop()
        logger.info("Stopping camera manager …")
        self._camera_mgr.stop()
        logger.info("Orchestrator stopped.")

    # ── Main control loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        if not self._running:
            self.start()

        loop_idx = 0
        t_prev = time.time()

        while self._running:
            t_loop_start = time.time()

            try:
                self._step(loop_idx, t_prev)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received — shutting down.")
                break
            except FatalOrchestratorError:
                logger.exception("Fatal runtime error in control loop at step %d", loop_idx)
                raise
            except Exception:
                logger.exception("Unhandled non-fatal error in control loop at step %d", loop_idx)

            # ── Rate limiting ────────────────────────────────────────────────
            elapsed = time.time() - t_loop_start
            sleep_t = max(0.0, self.control_period - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

            t_prev = t_loop_start
            loop_idx += 1

        self.stop()

    def _step(self, loop_idx: int, t_prev: float) -> None:
        """Execute one control cycle."""
        t_step_start = time.perf_counter()
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)

        # ── 1. Pull detector snapshot ─────────────────────────────────────────
        snapshot = self._detector_mgr.get_latest_snapshot()
        if not snapshot:
            logger.debug("[%05d] Empty snapshot — skipping", loop_idx)
            return

        # ── 2. Check for stale cameras (HIGH-2 fix: proper handling) ──────────
        stale_cams = self._check_stale_cameras(snapshot, stale_threshold_sec=5.0)
        if stale_cams:
            logger.warning(
                "[%05d] Stale/missing cameras: %s (stale_threshold=5.0s)",
                loop_idx, stale_cams
            )
            # HIGH-2 FIX: Skip control cycle if too many cameras are stale
            # (keep at least 75% of cameras active)
            total_cameras = len(self._camera_mgr.camera_ids)
            stale_count = len(stale_cams)
            if total_cameras > 0 and stale_count / total_cameras > 0.25:
                logger.critical(
                    "[%05d] Too many stale cameras (%d/%d). "
                    "Entering SAFE MODE: keeping current phase, no action.",
                    loop_idx, stale_count, total_cameras
                )
                # Log but don't send new actions
                self._log_degraded_mode(loop_idx, stale_cams)
                return

        # ── 3. Accident detection (3-frame confirmation + Telegram) ───────────
        accident_events: List[AccidentEvent] = []
        try:
            accident_events = self._accident_detector.update(snapshot)
        except Exception:
            logger.exception("[%05d] AccidentEventDetector raised", loop_idx)

        # ── 4. Build PPO observation via VisionToState bridge ─────────────────
        t_post_start = time.perf_counter()
        packet = self._bridge.build_packet(
            detector_snapshot=snapshot,
            phase_map=self._phase_map,
            green_timers=self._green_timers,
            return_semantic=False,
        )

        if packet is None:
            raise FatalOrchestratorError(f"[{loop_idx:05d}] No valid observation packet")

        # ── 5. Temporal smoothing via VisionBuffer ────────────────────────────
        buffered = self._vision_buffer.push(packet)
        smoothed_obs: np.ndarray = buffered.obs     # shape (obs_dim,), already float32

        # ── 6. Slice obs per TLS agent for MAPPO decentralized execution ──────
        #
        #   StateExtractor concatenates TLS blocks in order of self.tls_ids.
        #   Each block is self._per_agent_dim wide.
        #   For 2 agents: obs[0:26] → agent_0, obs[26:52] → agent_1.
        #
        per_agent_obs: Dict[str, np.ndarray] = {}
        for i, tls_id in enumerate(self.tls_ids):
            lo = i * self._per_agent_dim
            hi = lo + self._per_agent_dim
            block = smoothed_obs[lo:hi]
            if block.shape[0] != self._per_agent_dim:
                raise FatalOrchestratorError(
                    f"Per-agent obs block mismatch for {tls_id}: got {block.shape[0]}, expected {self._per_agent_dim}"
                )
            per_agent_obs[tls_id] = block
        postprocess_ms = (time.perf_counter() - t_post_start) * 1000.0

        # ── 7. Policy inference ───────────────────────────────────────────────
        t_infer_start = time.perf_counter()
        new_phases: Dict[str, int] = self._policy.predict(per_agent_obs, deterministic=True)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # ── 8. Update phase map and green timers ──────────────────────────────
        for tls_id, raw_action in new_phases.items():
            action_idx = int(raw_action)
            action_idx = 0 if action_idx <= 0 else 1
            mapped_phase = self._phase_action_map.get(tls_id, [0, 2])[action_idx]
            self._action_map[tls_id] = action_idx
            if mapped_phase != self._phase_map.get(tls_id):
                self._phase_start[tls_id] = now
                logger.debug("[%05d] %s phase change: %d → %d",
                             loop_idx, tls_id, self._phase_map.get(tls_id, -1), mapped_phase)
            self._phase_map[tls_id] = mapped_phase

        # Compute elapsed green time for each TLS.
        for tls_id in self.tls_ids:
            self._green_timers[tls_id] = now - self._phase_start.get(tls_id, now)

        # ── 9. Format actions → 8-value RYG list for Arduino ─────────────────
        t_action_start = time.perf_counter()
        serial_phases = actions_to_serial_phases(
            tls_ids=self.tls_ids,
            action_map=self._action_map,
        )
        self._send_to_arduino(serial_phases)
        action_ms = (time.perf_counter() - t_action_start) * 1000.0

        # ── 10. Collect per-TLS vehicle counts for logging ────────────────────
        total_vehicles = sum(
            int(cam_data.get("vehicle_count", 0)) for cam_data in snapshot.values()
        )

        # ── 11. Log loop stats ─────────────────────────────────────────────────
        dropped_now = self._update_drop_counters(snapshot)
        capture_ms = (time.perf_counter() - t_step_start) * 1000.0
        self._update_latency_metrics(capture_ms, infer_ms, postprocess_ms, action_ms, fps)
        self._log_step(
            loop_idx=loop_idx,
            fps=fps,
            total_vehicles=total_vehicles,
            accident_events=accident_events,
            phase_map=self._phase_map,
            serial_phases=serial_phases,
            buffer_size=buffered.buffer_size,
            dropped_now=dropped_now,
            dropped_total=self._dropped_frames_total,
        )

    def _update_latency_metrics(
        self,
        capture_ms: float,
        infer_ms: float,
        postprocess_ms: float,
        action_ms: float,
        fps: float,
    ) -> None:
        self._metrics_accum["capture_ms"] += capture_ms
        self._metrics_accum["infer_ms"] += infer_ms
        self._metrics_accum["postprocess_ms"] += postprocess_ms
        self._metrics_accum["action_ms"] += action_ms
        self._metrics_accum["fps"] += fps
        self._metrics_accum["steps"] += 1.0

        steps = int(self._metrics_accum["steps"])
        if steps > 0 and steps % self.metrics_log_every == 0:
            logger.info(
                "Perf[%d]: capture_ms=%.2f infer_ms=%.2f postprocess_ms=%.2f action_ms=%.2f fps=%.2f dropped_frames=%d",
                steps,
                self._metrics_accum["capture_ms"] / steps,
                self._metrics_accum["infer_ms"] / steps,
                self._metrics_accum["postprocess_ms"] / steps,
                self._metrics_accum["action_ms"] / steps,
                self._metrics_accum["fps"] / steps,
                self._dropped_frames_total,
            )

    def _update_drop_counters(self, snapshot: Dict[Any, Dict[str, Any]]) -> int:
        dropped_now = 0
        for cam_id, data in snapshot.items():
            frame_count = int(data.get("frame_count", 0))
            prev = self._last_frame_count.get(cam_id)
            if prev is not None and frame_count > prev + 1:
                dropped_now += (frame_count - prev - 1)
            self._last_frame_count[cam_id] = frame_count
        self._dropped_frames_total += dropped_now
        return dropped_now

    # ── Sub-step helpers ──────────────────────────────────────────────────────

    def _check_stale_cameras(
        self,
        snapshot: Dict[Any, Dict[str, Any]],
        stale_threshold_sec: float = 5.0,
    ) -> List[Any]:
        """Check for stale (old or missing) camera frames."""
        now = time.time()
        stale = []
        for cam_id, data in snapshot.items():
            ts = float(data.get("timestamp", 0.0))
            if ts == 0.0 or (now - ts) > stale_threshold_sec:
                stale.append(cam_id)
        return stale

    def _log_degraded_mode(self, loop_idx: int, stale_cams: List[Any]) -> None:
        """Log when system enters degraded/safe mode due to stale cameras."""
        logger.critical(
            "[%05d] DEGRADED MODE: Stale cameras %s. "
            "Keeping current phase, no new actions sent.",
            loop_idx, stale_cams
        )
        # Could extend to record to separate degraded-mode log file
        # or alert ops team

    def _send_to_arduino(self, phases: List[int]) -> None:
        """Send RYG phase states to Arduino (0=RED, 1=YELLOW, 2=GREEN)."""
        if self._serial is None:
            logger.debug("Serial bridge not initialized")
            return

        expected = len(self.tls_ids) * 4   # 4 directions per TLS
        if len(phases) != expected:
            logger.error(
                "serial_phases length %d ≠ expected %d; not sending", len(phases), expected
            )
            return

        if any(int(v) not in {0, 1, 2} for v in phases):
            logger.error("Invalid RYG state values (must be 0/1/2): %s", phases)
            return

        try:
            ok = self._serial.pack_and_send_data([int(v) for v in phases], retry_on_fail=True)
            if not ok:
                logger.error("Serial send failed. Health: %s", self._serial.get_stats())
            else:
                logger.debug("Serial send succeeded: %s", phases)
        except Exception as e:
            logger.exception("SerialBridge.pack_and_send_data() raised: %s", e)


    @staticmethod
    def _log_step(
        loop_idx:        int,
        fps:             float,
        total_vehicles:  int,
        accident_events: List[AccidentEvent],
        phase_map:       Dict[str, int],
        serial_phases:   List[int],
        buffer_size:     int,
        dropped_now:     int,
        dropped_total:   int,
    ) -> None:
        accident_flag = bool(accident_events)
        phase_str  = " | ".join(f"{k}→{v}" for k, v in phase_map.items())
        # Render as R/Y/G labels for readability
        _lbl = {0: "R", 1: "Y", 2: "G"}
        phases_str = ",".join(_lbl.get(v, "?") for v in serial_phases)

        if accident_flag:
            logger.warning(
                "[%05d] FPS=%.1f | vehicles=%d | ACCIDENT CONFIRMED (%d events) | phases=[%s] | hw=%s | buf=%d",
                loop_idx, fps, total_vehicles, len(accident_events), phase_str, phases_str, buffer_size,
            )
        else:
            logger.info(
                "[%05d] FPS=%.1f | vehicles=%d | accident=False | phases=[%s] | hw=%s | buf=%d | drop=%d/%d",
                loop_idx, fps, total_vehicles, phase_str, phases_str, buffer_size, dropped_now, dropped_total,
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production traffic-control orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",         type=str, default="configs/state_config.json",  help="State config (JSON)")
    p.add_argument("--camera-config",  type=str, default="configs/camera_config.json", help="Camera config (JSON)")
    p.add_argument("--model",          type=str, default="models/yolo/yolov11.pt",  help="YOLO model path")
    p.add_argument("--policy",         type=str, default=None,                          help="MAPPO policy checkpoint (.pt)")
    p.add_argument("--tele-config",    type=str, default="configs/tele.json",           help="Telegram config (JSON)")
    p.add_argument("--serial-port",    type=str, default="COM3",                        help="Arduino serial port")
    p.add_argument("--serial-baud",    type=int, default=115200,                        help="Serial baud rate")
    p.add_argument("--hz",             type=float, default=2.0,                         help="Control loop frequency (Hz)")
    p.add_argument("--buffer",         type=int,   default=5,                           help="VisionBuffer length")
    p.add_argument("--no-serial",      action="store_true",                             help="Disable serial output (dry-run)")
    p.add_argument("--log-level",      type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    serial_port: Optional[str] = None if args.no_serial else args.serial_port
    tele_path:   Optional[str] = args.tele_config if Path(args.tele_config).exists() else None

    # Load yellow_duration and red_gap from serial.json if available
    serial_cfg_path = Path("configs/serial.json")
    yellow_duration = 3.0
    red_gap = 1.0
    if serial_cfg_path.exists():
        import json as _json
        with open(serial_cfg_path) as _f:
            _scfg = _json.load(_f)
        yellow_duration = float(_scfg.get("yellow_duration", yellow_duration))
        red_gap = float(_scfg.get("red_gap", red_gap))

    # buffer_ema_alpha is read from system_params.temporal_smoothing in __init__
    # if present in config; default 0.6 is used otherwise.
    orchestrator = TrafficSystemOrchestrator(
        state_config_path=args.config,
        camera_config_path=args.camera_config,
        model_path=args.model,
        policy_path=args.policy,
        tele_config_path=tele_path,
        serial_port=serial_port,
        serial_baud=args.serial_baud,
        yellow_duration=yellow_duration,
        red_gap=red_gap,
        control_hz=args.hz,
        buffer_maxlen=args.buffer,
    )

    # Graceful SIGINT / SIGTERM shutdown
    def _shutdown(sig, frame):  # noqa: ANN001
        logger.info("Received signal %d — stopping …", sig)
        orchestrator._running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    orchestrator.run()


if __name__ == "__main__":
    main()


# ═════════════════════════════════════════════════════════════════════════════
# ── AUDIT REPORT ─────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     PHASE 1 — CODEBASE AUDIT REPORT                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ─────────────────────────────────────────────────────────
# CRITICAL ISSUES  (will break inference if not fixed)
# ─────────────────────────────────────────────────────────
#
# [C-1] CLASS ID MISMATCH — state_extractor.py vs detector.py / tracker_parser.py
#   File: src/vision/state_extractor.py  Lines 157-159
#         src/vision/detector.py         Line 93, 101-110
#         src/vision/tracker_parser.py   Line 55-59
#
#   Problem:
#     - Custom YOLO model label set: {0:accident, 1:bus, 2:car, 3:motorcycle, 4:truck}
#     - TrackerParser.class_map (and DetectorManager defaults) use exactly these IDs.
#     - StateExtractor FIX-2 hardcodes COCO class IDs:
#         _MOTORBIKE_CLASSES = frozenset({3})     # motorcycle ← coincidentally correct
#         _HEAVY_CLASSES     = frozenset({5, 7})  # bus=5, truck=7  ← WRONG for custom model
#     - As a result heavy_vehicle_share will always be 0.0 at inference because
#       no track ever has cls_id ∈ {5, 7}.
#
#   Fix (2 lines in state_extractor.py __init__):
#     Add constructor params:
#         motorbike_class_ids: Optional[Sequence[int]] = None,
#         heavy_class_ids:     Optional[Sequence[int]] = None,
#     Inside __init__, after super().__init__():
#         if motorbike_class_ids is not None:
#             self._MOTORBIKE_CLASSES = frozenset(motorbike_class_ids)
#         if heavy_class_ids is not None:
#             self._HEAVY_CLASSES = frozenset(heavy_class_ids)
#     Then in build_state_extractor() above, pass:
#         motorbike_class_ids=[3],
#         heavy_class_ids=[1, 4],
#
#   (The orchestrator already calls build_state_extractor() with the correct values;
#   only the StateExtractor constructor param needs to be added.)
#
#
# [C-2] state_config.json DIMENSION DOCUMENTATION IS WRONG
#   File: configs/state_config.json
#
#   Problem:
#     The JSON documents "lane_feature_dim: 5" but then:
#       - Lists 6 lane_features (indices 0-5, including throughput).
#       - Calculates "4 * 6 + 4 = 28" and "global = 28 * 2 = 56".
#       - Sets "per_agent_observation_dim: 29".
#     None of these match the runtime.
#
#   Actual runtime values (from state_extractor.py audit-fixed v2):
#       lane_feature_dim  = 5  (effective_queue_norm, occupancy_norm,
#                               avg_speed_norm, motorbike_share, heavy_vehicle_share)
#       tls_feature_dim   = 6  (4-phase one-hot + green_timer_norm + pressure_norm)
#       per_agent_obs_dim = 4*5 + 6 = 26
#       global_obs_dim    = 26 * 2  = 52
#
#   Fix: update state_config.json fields:
#       "lane_feature_dim": 5,
#       "tls_feature_dim": 6,
#       "per_agent_observation_dim": 26,
#       "global_observation_dim": 52,
#       "calculation": "per_agent = max_lanes*5 + 6 = 20+6 = 26; global = 26*2 = 52"
#   The JSON is documentation only; no runtime code reads these fields directly.
#
#
# ─────────────────────────────────────────────────────────
# MEDIUM ISSUES  (degrade correctness; fix before production)
# ─────────────────────────────────────────────────────────
#
# [M-1] MAPPO PER-AGENT OBS SPLIT NOT IMPLEMENTED IN VISION PIPELINE
#   Files: src/adapters/vision_to_state.py, src/vision/state_extractor.py
#
#   Problem:
#     StateExtractor.build_state() returns ONE flat vector for ALL TLS agents
#     (shape (52,) for 2 TLS).  MAPPO decentralized execution requires a
#     separate local obs for each agent (shape (26,) each).
#     VisionToState.build_packet() propagates this flat obs unchanged.
#     There is no API to obtain per-agent obs from the bridge directly.
#
#   Fix (implemented in orchestrator):
#     In _step(), after vision_buffer.push():
#         per_agent_obs = {tls_id: obs[i*26:(i+1)*26] for i, tls_id in enumerate(tls_ids)}
#     This is valid because StateExtractor assembles TLS blocks sequentially in
#     the same order as self.tls_ids.  No change to state_extractor.py is needed.
#     If StateExtractor.build_per_agent_states() is later added for clarity,
#     it should just do this same slicing.
#
#
# [M-2] AccidentEventDetector WILL CRASH IF tele.json IS MISSING
#   File: src/vision/event_detector.py  Line 48
#
#   Problem:
#     If alert_service=None is passed, the constructor calls
#     TelegramAlertService.from_config(tele_config_path) which raises
#     FileNotFoundError when the file doesn't exist.  This is fatal at init.
#
#   Fix (implemented in orchestrator):
#     Always pass a pre-constructed alert_service (or one with enabled=False)
#     to AccidentEventDetector so it never attempts to read a missing file.
#     The orchestrator does this: it loads the Telegram service with a try/except
#     and passes None only when the file doesn't exist, which means the
#     AccidentEventDetector constructor will still try to load the path.
#
#   Correct fix:
#     In event_detector.py AccidentEventDetector.__init__(), change:
#         self.alert_service = alert_service or TelegramAlertService.from_config(tele_config_path)
#     to:
#         if alert_service is not None:
#             self.alert_service = alert_service
#         else:
#             try:
#                 self.alert_service = TelegramAlertService.from_config(tele_config_path)
#             except FileNotFoundError:
#                 logger.warning("tele.json not found; Telegram alerts disabled.")
#                 self.alert_service = TelegramAlertService(
#                     bot_token="", chat_id="", enabled=False
#                 )
#
#
# [M-3] NO RL POLICY FILE PROVIDED
#   Files: (absent)
#
#   Problem:
#     There is no policy.py, inference.py, or checkpoint file in the repo.
#     The orchestrator cannot perform real MAPPO inference without one.
#
#   Fix: PolicyInterface._load() expects a PyTorch checkpoint (model.pt).
#     Implement predict() for your specific policy format
#     (StableBaselines3, RLlib, custom nn.Module, etc.).
#     Until then the pressure-heuristic fallback keeps hardware running.
#
#
# [M-4] VisionToState STRICT_SHAPE_CHECK RAISES UNHANDLED RuntimeError
#   File: src/adapters/vision_to_state.py  Line ~110
#
#   Problem:
#     VisionToState.build_packet() raises RuntimeError if the obs shape
#     doesn't match the expected dimension.  In the orchestrator this is
#     caught by the broad try/except in _step(), but the error is silently
#     absorbed and the control cycle is skipped.  Under high latency or
#     partial snapshots this can cause extended control gaps.
#
#   Fix (implemented in orchestrator):
#     The try/except in _step() logs the exception and returns early.
#     Additionally, consider setting strict_shape_check=False during a
#     warmup period until VisionBuffer has accumulated min_len frames.
#
#
# ─────────────────────────────────────────────────────────
# LOW-RISK ISSUES  (clean up when convenient)
# ─────────────────────────────────────────────────────────
#
# [L-1] vision_buffer.py push_obs() CREATES A DYNAMIC ANONYMOUS CLASS
#   File: src/buffer/vision_buffer.py  Line ~57
#   Impact: Fragile duck-typing.  Use a dataclass or NamedTuple instead.
#   Fix: Replace with a proper lightweight @dataclass or simple object.
#
# [L-2] serial_bridge.py HAS NO RECONNECTION LOGIC
#   File: src/utils/serial_bridge.py
#   Impact: A USB disconnect will crash the serial write silently or raise.
#   Fix: Wrap ser.write() in try/except; attempt serial.Serial() reconnect
#   after N consecutive failures.
#
# [L-3] detector.py CONTAINS argparse + demo/test CODE AT MODULE LEVEL
#   File: src/vision/detector.py  Lines 459–525
#   Impact: Low; guarded by if __name__ == "__main__".  Not a runtime risk.
#   Fix: Move demo code to scripts/run_detector_demo.py.
#
# [L-4] state_config.json lane_groups SECTION UNDER "lane_groups" KEY
#        BUT CONTAINS tls_0/tls_1 ENTRIES AS LISTS, NOT AS
#        {"group_a": [...], "group_b": [...]} DICTS.
#   File: configs/state_config.json
#   Impact: build_lane_groups() in this orchestrator reads it correctly
#   as list[0] = group_a, list[1] = group_b.  No code change needed.
#
# [L-5] state_config.json THROUGHPUT FEATURE (index 5) IS DOCUMENTED
#        BUT NOT IMPLEMENTED IN StateExtractor
#   File: configs/state_config.json feature[5] = throughput
#         src/vision/state_extractor.py FEATURE_NAMES has only 5 features
#   Impact: Documentation drift.  No runtime impact.
#   Fix: Remove throughput from state_config.json feature list OR implement
#   throughput estimation (track exit events) in state_extractor.
#
# ─────────────────────────────────────────────────────────
# SUPPORT CHANGES REQUIRED IN OTHER FILES (minimal)
# ─────────────────────────────────────────────────────────
#
# 1. src/vision/state_extractor.py  — ADD 4 LINES to __init__:
#
#    def __init__(
#        self,
#        ...
#        motorbike_class_ids: Optional[Sequence[int]] = None,   # <-- ADD
#        heavy_class_ids:     Optional[Sequence[int]] = None,   # <-- ADD
#    ) -> None:
#        ...
#        # after existing init logic:
#        if motorbike_class_ids is not None:                     # <-- ADD
#            self._MOTORBIKE_CLASSES = frozenset(motorbike_class_ids)  # <-- ADD
#        if heavy_class_ids is not None:                         # <-- ADD
#            self._HEAVY_CLASSES = frozenset(heavy_class_ids)   # <-- ADD
#
# 2. src/vision/event_detector.py  — GUARD tele.json LOAD (see [M-2] above)
#    Change the single line in __init__ to a try/except block (4 lines).
#
# ─────────────────────────────────────────────────────────
# FILES THAT DO NOT NEED ANY CHANGES
# ─────────────────────────────────────────────────────────
#   src/vision/multi_camera.py    — correct and complete
#   src/vision/detector.py        — correct; demo code is harmless
#   src/vision/tracker_parser.py  — correct for custom model
#   src/adapters/vision_to_state.py — correct; per-agent slicing done in orchestrator
#   src/buffer/vision_buffer.py   — functional; cosmetic cleanup optional
#   src/services/alert_services.py — correct and complete
#   src/utils/serial_bridge.py     — functional; reconnection logic is optional hardening
#
# ═════════════════════════════════════════════════════════════════════════════