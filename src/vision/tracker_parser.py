from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import math
import numpy as np


@dataclass
class TrackRecord:
    track_id: Optional[int]
    cls_id: int
    label: str
    conf: float
    bbox: List[float]              # [x1, y1, x2, y2]
    center: Tuple[float, float]
    width: float
    height: float
    area: float
    speed_px_s: float = 0.0        # tốc độ theo pixel/giây
    motion_dx: float = 0.0
    motion_dy: float = 0.0
    age_frames: int = 0


@dataclass
class ParsedFrame:
    cam_id: Union[int, str]
    camera_name: str
    timestamp: float
    frame_idx: int
    vehicle_count: int
    accident_found: bool
    accident_conf: float
    vehicle_ids: List[Union[int, str]] = field(default_factory=list)
    tracks: List[TrackRecord] = field(default_factory=list)


class TrackerParser:
    """
    Parse Ultralytics YOLO + ByteTrack output into a stable, structured format.

    Responsibilities:
    - Convert boxes to TrackRecord
    - Compute bbox center / size / area
    - Maintain per-camera, per-track temporal memory
    - Estimate simple motion features
    - Keep object schema stable for state_extractor / alert logic / PPO
    """

    def __init__(
        self,
        class_map: Optional[Dict[int, str]] = None,
        vehicle_class_ids: Optional[List[int]] = None,
        accident_class_id: int = 0,
        accident_conf_thresh: float = 0.7,
    ) -> None:
        self.class_map = class_map or {
            0: "accident",
            1: "bus",
            2: "car",
            3: "motorcycle",
            4: "truck",
        }
        self.vehicle_class_ids = set(vehicle_class_ids or [1, 2, 3, 4])
        self.accident_class_id = int(accident_class_id)
        self.accident_conf_thresh = float(accident_conf_thresh)

        self.prev_state: Dict[Union[int, str], Dict[int, Dict[str, Any]]] = {}

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _bbox_wh(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (max(0.0, x2 - x1), max(0.0, y2 - y1))

    def _estimate_motion(
        self,
        cam_id: Union[int, str],
        track_id: Optional[int],
        center: Tuple[float, float],
        timestamp: float,
        frame_idx: int,
    ) -> Tuple[float, float, float, int]:
        """
        Returns:
            speed_px_s, dx, dy, age_frames
        """
        if track_id is None:
            return 0.0, 0.0, 0.0, 0
        # cleanup old tracks
        MAX_AGE = 30  
        cam_mem = self.prev_state.setdefault(cam_id, {})
        prev = cam_mem.get(track_id)
        to_delete = [tid for tid, v in cam_mem.items() if v.get("age", 0) > MAX_AGE]

        for tid in to_delete:
            del cam_mem[tid]

        if prev is None:
            cam_mem[track_id] = {
                "center": center,
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "age": 0,
            }
            return 0.0, 0.0, 0.0, 0

        px, py = prev["center"]
        pt = float(prev["timestamp"])
        dx = float(center[0] - px)
        dy = float(center[1] - py)

        dist = math.hypot(dx, dy)
        dt = float(timestamp) - pt
        if dt <= 0 or dt > 1.0:   # ignore abnormal jumps
            return 0.0, 0.0, 0.0, prev.get("age", 0)
        speed_px_s = dist / dt

        age = int(prev.get("age", 0)) + 1
        cam_mem[track_id] = {
            "center": center,
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "age": age,
        }
        return speed_px_s, dx, dy, age

    def _safe_track_id(self, raw_id: Any) -> Optional[int]:
        try:
            if raw_id is None:
                return None
            tid = int(raw_id)
            return tid if tid >= 0 else None
        except Exception:
            return None

    def parse_ultralytics_result(
        self,
        cam_id: Union[int, str],
        camera_name: str,
        result: Any,
        timestamp: Optional[float] = None,
        frame_idx: int = 0,
    ) -> ParsedFrame:
        """
        Parse a single Ultralytics result object from model.track(...)[0].
        """
        ts = float(timestamp if timestamp is not None else time.time())

        tracks: List[TrackRecord] = []
        vehicle_count = 0
        vehicle_ids: List[Union[int, str]] = []
        accident_found = False
        accident_conf = 0.0

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ParsedFrame(
                cam_id=cam_id,
                camera_name=camera_name,
                timestamp=ts,
                frame_idx=frame_idx,
                vehicle_count=0,
                accident_found=False,
                accident_conf=0.0,
                vehicle_ids=[],
                tracks=[],
            )

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                track_ids = boxes.id.cpu().numpy()
            except Exception:
                track_ids = None

        for i in range(len(xyxy)):
            bbox = xyxy[i].tolist()
            conf = float(confs[i])
            cls_id = int(clss[i])
            track_id = None

            if track_ids is not None and i < len(track_ids):
                track_id = self._safe_track_id(track_ids[i])

            label_base = self.class_map.get(cls_id, f"cls{cls_id}")
            center = self._bbox_center(bbox)
            width, height = self._bbox_wh(bbox)
            area = width * height

            speed_px_s, dx, dy, age = self._estimate_motion(
                cam_id=cam_id,
                track_id=track_id,
                center=center,
                timestamp=ts,
                frame_idx=frame_idx,
            )

            if cls_id in self.vehicle_class_ids:
                vehicle_count += 1
                if track_id is not None:
                    vehicle_ids.append(track_id)
                    label = f"{label_base} ID:{track_id} {conf:.2f}"
                else:
                    vehicle_ids.append(f"{cam_id}_box_{i}")
                    label = f"{label_base} {conf:.2f}"

            elif cls_id == self.accident_class_id and conf >= self.accident_conf_thresh:
                accident_found = True
                accident_conf = max(accident_conf, conf)
                label = f"{label_base} {conf:.2f}"
            else:
                label = f"{label_base} {conf:.2f}"

            tracks.append(
                TrackRecord(
                    track_id=track_id,
                    cls_id=cls_id,
                    label=label,
                    conf=conf,
                    bbox=bbox,
                    center=center,
                    width=width,
                    height=height,
                    area=area,
                    speed_px_s=speed_px_s,
                    motion_dx=dx,
                    motion_dy=dy,
                    age_frames=age,
                )
            )

        return ParsedFrame(
            cam_id=cam_id,
            camera_name=camera_name,
            timestamp=ts,
            frame_idx=frame_idx,
            vehicle_count=vehicle_count,
            accident_found=accident_found,
            accident_conf=float(accident_conf),
            vehicle_ids=vehicle_ids,
            tracks=tracks,
        )

    def reset_camera(self, cam_id: Union[int, str]) -> None:
        self.prev_state.pop(cam_id, None)

    def reset_all(self) -> None:
        self.prev_state.clear()

    def to_semantic_dict(self, parsed: ParsedFrame) -> Dict[str, Any]:
        """
        Compact dict for downstream state extraction.
        """
        return {
            "cam_id": parsed.cam_id,
            "camera_name": parsed.camera_name,
            "timestamp": parsed.timestamp,
            "frame_idx": parsed.frame_idx,
            "vehicle_count": parsed.vehicle_count,
            "accident_found": parsed.accident_found,
            "accident_conf": parsed.accident_conf,
            "vehicle_ids": parsed.vehicle_ids,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "cls_id": t.cls_id,
                    "label": t.label,
                    "conf": t.conf,
                    "bbox": t.bbox,
                    "center": t.center,
                    "width": t.width,
                    "height": t.height,
                    "area": t.area,
                    "speed_px_s": t.speed_px_s,
                    "motion_dx": t.motion_dx,
                    "motion_dy": t.motion_dy,
                    "age_frames": t.age_frames,
                }
                for t in parsed.tracks
            ],
        }