"""Multi-camera detector — CPU-optimised, single shared YOLO instance.

Architecture
------------
- One YOLO model shared across all cameras (was: one per camera → 8×)
- Cameras processed sequentially; per-camera ByteTrack state saved/restored
- OpenVINO INT8 quantized model loaded by default (~3× faster than .pt on CPU)
  Priority: INT8 OpenVINO → FP32 OpenVINO → .pt
- process_single_frame() is the primary API consumed by InferenceThread
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.multi_camera import MultiCameraManager
from src.vision.tracker_parser import ParsedFrame, TrackRecord, TrackerParser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO-aware model loader
# ─────────────────────────────────────────────────────────────────────────────

def _read_ov_task(ov_dir: Path) -> str:
    """Read task from OpenVINO metadata.yaml; fall back to 'detect'."""
    meta = ov_dir / "metadata.yaml"
    if meta.exists():
        try:
            import yaml
            with open(meta, encoding="utf-8") as f:
                return yaml.safe_load(f).get("task", "detect")
        except Exception:
            pass
    return "detect"


def _load_yolo(model_path: Union[str, Path], use_openvino: bool = True) -> YOLO:
    """
    Load YOLO, preferring OpenVINO INT8 → FP32 → .pt (in that order).
    Exports FP32 OpenVINO on first run if neither INT8 nor FP32 exists (~30 s).
    Falls back to .pt silently on any failure.
    """
    pt_path = Path(model_path)
    if not use_openvino:
        return YOLO(str(pt_path))

    int8_dir = pt_path.parent / f"{pt_path.stem}_int8_openvino_model"
    fp32_dir = pt_path.parent / f"{pt_path.stem}_openvino_model"

    # INT8 takes priority — fastest on CPU
    if int8_dir.exists():
        task = _read_ov_task(int8_dir)
        logger.info("Loading OpenVINO INT8 model from %s (task=%s)", int8_dir, task)
        return YOLO(str(int8_dir), task=task)

    # FP32 fallback
    if not fp32_dir.exists():
        logger.info("OpenVINO model not found — exporting %s (one-time, ~30 s)…", pt_path.name)
        try:
            tmp = YOLO(str(pt_path))
            tmp.export(format="openvino", imgsz=640)
            del tmp
            logger.info("OpenVINO FP32 export complete → %s", fp32_dir)
        except Exception as exc:
            logger.warning("OpenVINO export failed (%s) — falling back to .pt", exc)
            return YOLO(str(pt_path))

    if fp32_dir.exists():
        task = _read_ov_task(fp32_dir)
        logger.info("Loading OpenVINO FP32 model from %s (task=%s)", fp32_dir, task)
        return YOLO(str(fp32_dir), task=task)

    return YOLO(str(pt_path))


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraDetectionState:
    cam_id: Union[int, str]
    name: str
    timestamp: float = 0.0
    frame_count: int = 0
    accident_found: bool = False
    accident_conf: float = 0.0
    vehicle_count: int = 0
    vehicle_ids: List[Union[int, str]] = field(default_factory=list)
    tracks: List[TrackRecord] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    last_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Detector manager
# ─────────────────────────────────────────────────────────────────────────────

class DetectorManager:
    """
    Single-YOLO sequential detector for all cameras.

    Replaces the original architecture (one YOLO instance per camera thread)
    with a single model that processes cameras one at a time.  Per-camera
    ByteTrack state is saved/restored via deepcopy so track IDs never bleed
    across camera streams.

    Public API is unchanged from the original DetectorManager so orchestrators
    need no modifications.
    """

    _CLASS_MAP = {0: "accident", 1: "bus", 2: "car", 3: "motorcycle", 4: "truck"}

    def __init__(
        self,
        model_path: Union[str, Path],
        camera_manager: MultiCameraManager,
        conf_thresh: float = 0.5,
        accident_conf_thresh: float = 0.7,
        accident_class_id: int = 0,
        vehicle_class_ids: Optional[List[int]] = None,
        tracker_yaml: str = "bytetrack.yaml",
        imgsz: int = 416,
        use_openvino: bool = True,
        enable_annotation: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.camera_manager = camera_manager
        self.conf_thresh = float(conf_thresh)
        self.accident_conf_thresh = float(accident_conf_thresh)
        self.accident_class_id = int(accident_class_id)
        self.vehicle_class_ids = list(vehicle_class_ids or [1, 2, 3, 4])
        self.tracker_yaml = tracker_yaml
        self.imgsz = int(imgsz)
        self.enable_annotation = enable_annotation
        self.running = False

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = _load_yolo(self.model_path, use_openvino)

        self._parsers: Dict[Union[int, str], TrackerParser] = {}
        self._tracker_states: Dict[Union[int, str], Any] = {}
        self._active_cam_id: Optional[Union[int, str]] = None
        self._cam_states: Dict[Union[int, str], CameraDetectionState] = {}
        self._lock = threading.Lock()

        self._init_per_camera_state()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_per_camera_state(self) -> None:
        # Parsers are registered lazily in process_single_frame().
        # Nothing to do here — cameras may not be open yet at __init__ time.
        pass

    def _register_camera(self, cam_id: Union[int, str], name: str) -> None:
        if cam_id in self._parsers:
            return
        self._parsers[cam_id] = TrackerParser(
            class_map=self._CLASS_MAP,
            vehicle_class_ids=self.vehicle_class_ids,
            accident_class_id=self.accident_class_id,
            accident_conf_thresh=self.accident_conf_thresh,
        )
        self._cam_states[cam_id] = CameraDetectionState(cam_id=cam_id, name=name)

    # ── ByteTrack state management ─────────────────────────────────────────────

    def _swap_tracker_state(self, cam_id: Union[int, str]) -> None:
        """
        Save the departing camera's ByteTrack state and restore the arriving
        camera's state.  If the arriving camera is new, reset the tracker so
        Kalman filters start fresh for that stream.
        """
        pred = getattr(self.model, "predictor", None)
        if pred is None:
            return
        trackers = getattr(pred, "trackers", None)
        if not trackers:
            return

        if self._active_cam_id is not None and self._active_cam_id != cam_id:
            try:
                self._tracker_states[self._active_cam_id] = copy.deepcopy(trackers[0])
            except Exception:
                pass

        if cam_id in self._tracker_states:
            try:
                trackers[0] = copy.deepcopy(self._tracker_states[cam_id])
                self._active_cam_id = cam_id
                return
            except Exception:
                pass

        try:
            trackers[0].reset()
        except Exception:
            pass
        self._active_cam_id = cam_id

    # ── Primary inference method ───────────────────────────────────────────────

    def process_single_frame(
        self,
        cam_id: Union[int, str],
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        cam_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO + ByteTrack on one frame from one camera.
        Called sequentially by InferenceThread for each camera in turn.
        Returns a structured dict compatible with get_latest_snapshot().
        """
        name = cam_name or str(self._cam_states.get(cam_id, CameraDetectionState(cam_id, str(cam_id))).name)
        if cam_id not in self._parsers:
            self._register_camera(cam_id, name)

        try:
            self._swap_tracker_state(cam_id)
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_thresh,
                tracker=self.tracker_yaml,
                imgsz=self.imgsz,
                verbose=False,
            )
            parsed = self._parsers[cam_id].parse_ultralytics_result(
                cam_id=cam_id,
                camera_name=name,
                result=results[0],
                timestamp=timestamp,
                frame_idx=frame_idx,
            )
            annotated = self._annotate(frame, parsed) if self.enable_annotation else None
            state = CameraDetectionState(
                cam_id=cam_id,
                name=name,
                timestamp=parsed.timestamp,
                frame_count=parsed.frame_idx,
                accident_found=parsed.accident_found,
                accident_conf=float(parsed.accident_conf),
                vehicle_count=int(parsed.vehicle_count),
                vehicle_ids=list(parsed.vehicle_ids),
                tracks=list(parsed.tracks),
                annotated_frame=annotated,
                last_error=None,
            )
            with self._lock:
                self._cam_states[cam_id] = state
            return self._state_to_dict(state)

        except Exception as exc:
            logger.exception("Inference failed for cam %s", cam_id)
            with self._lock:
                if cam_id in self._cam_states:
                    self._cam_states[cam_id].last_error = str(exc)
            return None

    # ── Snapshot / query API ───────────────────────────────────────────────────

    def get_latest_snapshot(self) -> Dict[Union[int, str], Dict[str, Any]]:
        with self._lock:
            return {cam_id: self._state_to_dict(s) for cam_id, s in self._cam_states.items()}

    def step(self) -> Dict[Union[int, str], Dict[str, Any]]:
        return self.get_latest_snapshot()

    def get_accident_flags(self) -> Dict[Union[int, str], bool]:
        return {k: bool(v["accident_found"]) for k, v in self.get_latest_snapshot().items()}

    def get_vehicle_counts(self) -> Dict[Union[int, str], int]:
        return {k: int(v["vehicle_count"]) for k, v in self.get_latest_snapshot().items()}

    def build_semantic_state(self) -> Dict[Union[int, str], Dict[str, Any]]:
        snap = self.get_latest_snapshot()
        return {
            cam_id: {
                "vehicle_count": d["vehicle_count"],
                "accident": d["accident_found"],
                "accident_conf": d["accident_conf"],
                "track_ids": list(d["vehicle_ids"]),
                "timestamp": d["timestamp"],
                "frame_count": d["frame_count"],
                "last_error": d["last_error"],
            }
            for cam_id, d in snap.items()
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    # ── Annotation (dashboard / preview only) ──────────────────────────────────

    def _annotate(self, frame: np.ndarray, parsed: ParsedFrame) -> np.ndarray:
        canvas = frame.copy()
        for track in parsed.tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = self._track_color(track.track_id, track.cls_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{track.label} v={track.speed_px_s:.1f}px/s"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 3, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"Vehicles: {parsed.vehicle_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if parsed.accident_found:
            cv2.putText(canvas, f"ACCIDENT {parsed.accident_conf:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ts_str = datetime.fromtimestamp(parsed.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, ts_str, (10, max(canvas.shape[0] - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return canvas

    def _track_color(self, track_id: Optional[int], cls_id: int) -> Tuple[int, int, int]:
        if cls_id == self.accident_class_id:
            return (0, 0, 255)
        if cls_id in self.vehicle_class_ids:
            if track_id is not None and track_id >= 0:
                return (int((track_id * 37) % 255),
                        int((track_id * 17) % 255),
                        int((track_id * 29) % 255))
            return (255, 0, 0)
        return (200, 200, 200)

    # ── Serialisation helper ───────────────────────────────────────────────────

    @staticmethod
    def _state_to_dict(state: CameraDetectionState) -> Dict[str, Any]:
        return {
            "cam_id": state.cam_id,
            "name": state.name,
            "timestamp": state.timestamp,
            "frame_count": state.frame_count,
            "accident_found": state.accident_found,
            "accident_conf": state.accident_conf,
            "vehicle_count": state.vehicle_count,
            "vehicle_ids": list(state.vehicle_ids),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "cls_id": t.cls_id,
                    "label": t.label,
                    "conf": t.conf,
                    "bbox": list(t.bbox),
                    "center": t.center,
                    "width": t.width,
                    "height": t.height,
                    "area": t.area,
                    "speed_px_s": t.speed_px_s,
                    "motion_dx": t.motion_dx,
                    "motion_dy": t.motion_dy,
                    "age_frames": t.age_frames,
                }
                for t in state.tracks
            ],
            "annotated_frame": state.annotated_frame,
            "last_error": state.last_error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Config loader (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_detector_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
