from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

from src.services.alert_services import TelegramAlertService

logger = logging.getLogger(__name__)


@dataclass
class AccidentEvent:
    cam_id: Union[int, str]
    camera_name: str
    timestamp: float
    confidence: float
    vehicle_count: int
    frame_count: int
    position: str = "XXXXXXX"
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccidentEventDetector:
    """
    Accident detector with 3-frame confirmation and Telegram alerting.

    Logic:
    - Detect accident from top-level flag or accident class in tracks
    - Require 3 consecutive positive frames before confirming
    - Send full-frame image (annotated_frame if available) to Telegram
    - Add time and position in caption
    - Use cooldown to avoid spam
    """

    def __init__(
        self,
        alert_service: Optional[TelegramAlertService] = None,
        tele_config_path: str | Path = "configs/tele.json",
        accident_class_id: int = 0,
        accident_conf_thresh: float = 0.7,
        confirm_frames: int = 3,
        cooldown_sec: float = 60.0,
        position: str = "XXXXXXX",
        enabled: bool = True,
        event_dir: str | Path = "logs/events",
    ) -> None:
        self.accident_class_id = int(accident_class_id)
        self.accident_conf_thresh = float(accident_conf_thresh)
        self.confirm_frames = max(1, int(confirm_frames))
        self.cooldown_sec = float(cooldown_sec)
        self.position = str(position)
        self.enabled = bool(enabled)
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.event_dir / "events.jsonl"

        self._streak: Dict[Union[int, str], int] = {}
        self._last_alert_ts: Dict[Union[int, str], float] = {}
        if alert_service is not None:
            self.alert_service = alert_service
        else:
            try:
                self.alert_service = TelegramAlertService.from_config(tele_config_path)
            except FileNotFoundError:
                logger.warning("tele.json not found; Telegram alerts disabled.")
                self.alert_service = TelegramAlertService(bot_token="", chat_id="", enabled=False)
            except Exception:
                logger.exception("Failed to initialize Telegram service; alerts disabled.")
                self.alert_service = TelegramAlertService(bot_token="", chat_id="", enabled=False)
    def reset(self, cam_id: Optional[Union[int, str]] = None) -> None:
        if cam_id is None:
            self._streak.clear()
            self._last_alert_ts.clear()
        else:
            self._streak.pop(cam_id, None)
            self._last_alert_ts.pop(cam_id, None)

    def update(self, snapshot: Any) -> List[AccidentEvent]:
        """
        Process one snapshot from DetectorManager.
        Supports:
        - multi-camera dict: {cam_id: { ... }, ...}
        - single camera dict: {cam_id, name, tracks, accident_found, ...}
        """
        if not self.enabled:
            return []

        camera_items = self._normalize_snapshot(snapshot)
        events: List[AccidentEvent] = []

        for cam_id, item in camera_items.items():
            ev = self._evaluate_camera(cam_id, item)
            if ev is not None:
                events.append(ev)

        return events

    def _normalize_snapshot(self, snapshot: Any) -> Dict[Union[int, str], Dict[str, Any]]:
        if snapshot is None:
            return {}

        if isinstance(snapshot, dict):
            # single camera dict
            if "tracks" in snapshot or "accident_found" in snapshot:
                cam_id = snapshot.get("cam_id", "cam0")
                return {cam_id: snapshot}

            # multi camera dict
            out: Dict[Union[int, str], Dict[str, Any]] = {}
            for cam_id, item in snapshot.items():
                if isinstance(item, dict):
                    out[cam_id] = item
            return out

        return {}

    def _evaluate_camera(
        self,
        cam_id: Union[int, str],
        item: Dict[str, Any],
    ) -> Optional[AccidentEvent]:
        accident_found, confidence = self._detect_accident(item)
        streak = self._streak.get(cam_id, 0)

        if accident_found:
            streak += 1
        else:
            streak = 0

        self._streak[cam_id] = streak

        # not yet confirmed
        if streak < self.confirm_frames:
            return None

        now_ts = float(item.get("timestamp", time.time()))
        last_alert = self._last_alert_ts.get(cam_id, 0.0)
        if (now_ts - last_alert) < self.cooldown_sec:
            return None

        camera_name = str(item.get("name", cam_id))
        frame_count = int(item.get("frame_count", 0))
        vehicle_count = int(item.get("vehicle_count", 0))

        full_frame = item.get("annotated_frame", None)
        if full_frame is None:
            # fallback: no crop, no bbox
            full_frame = item.get("frame", None)

        message = self._build_message(
            cam_id=cam_id,
            camera_name=camera_name,
            timestamp=now_ts,
            confidence=confidence,
            vehicle_count=vehicle_count,
            frame_count=frame_count,
            position=self.position,
        )

        sent = False
        event_image_path: Optional[str] = None
        try:
            if full_frame is not None:
                event_image_path = self._save_event_image(cam_id=cam_id, frame=full_frame, timestamp=now_ts)
                sent = bool(self.alert_service.send_accident_alert(message, frame=full_frame))
            else:
                sent = bool(self.alert_service.send_accident_alert(message))
        except Exception:
            logger.exception("Failed to send accident alert")

        self._last_alert_ts[cam_id] = now_ts
        self._streak[cam_id] = 0

        event = AccidentEvent(
            cam_id=cam_id,
            camera_name=camera_name,
            timestamp=now_ts,
            confidence=confidence,
            vehicle_count=vehicle_count,
            frame_count=frame_count,
            position=self.position,
            message=message,
            metadata={
                "sent": sent,
                "accident_found": True,
                "confirm_frames": self.confirm_frames,
                "cooldown_sec": self.cooldown_sec,
                "event_type": "collision",
                "severity": self._severity_from_conf(confidence),
                "event_image_path": event_image_path,
            },
        )
        self._append_event_log(event)
        return event

    def _detect_accident(self, item: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Return (accident_found, confidence).

        Uses:
        - top-level accident_found / accident_conf from DetectorManager
        - tracks with cls_id == accident_class_id and conf >= threshold
        """
        top_flag = bool(item.get("accident_found", False))
        top_conf = float(item.get("accident_conf", 0.0) or 0.0)

        best_conf = top_conf
        found = top_flag and top_conf >= self.accident_conf_thresh

        tracks = item.get("tracks", []) or []
        for t in tracks:
            if not isinstance(t, dict):
                continue

            try:
                cls_id = int(t.get("cls_id", -1))
                conf = float(t.get("conf", 0.0))
            except Exception:
                continue

            if cls_id == self.accident_class_id and conf >= self.accident_conf_thresh:
                found = True
                best_conf = max(best_conf, conf)

        return found, best_conf

    @staticmethod
    def _build_message(
        cam_id: Union[int, str],
        camera_name: str,
        timestamp: float,
        confidence: float,
        vehicle_count: int,
        frame_count: int,
        position: str = "XXXXXXX",
    ) -> str:
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"<b>⚠️ Traffic Accident Detected</b>\n"
            f"Camera: <b>{camera_name}</b> ({cam_id})\n"
            f"Time: <b>{time_str}</b>\n"
            f"Position: <b>{position}</b>\n"
            f"Confidence: <b>{confidence:.2f}</b>\n"
            f"Vehicles: <b>{vehicle_count}</b>\n"
            f"Frame: <b>{frame_count}</b>"
        )

    def process_snapshot(self, snapshot: Any) -> List[AccidentEvent]:
        return self.update(snapshot)

    @staticmethod
    def _severity_from_conf(confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.8:
            return "medium"
        return "low"

    def _save_event_image(
        self,
        cam_id: Union[int, str],
        frame: np.ndarray,
        timestamp: float,
    ) -> Optional[str]:
        try:
            ts = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            cam = str(cam_id).replace(" ", "_")
            out_path = self.event_dir / f"event_{cam}_{ts}.jpg"
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                return None
            return str(out_path)
        except Exception:
            logger.exception("Failed to save event image")
            return None

    def _append_event_log(self, event: AccidentEvent) -> None:
        record = {
            "event_type": event.metadata.get("event_type", "collision"),
            "cam_id": event.cam_id,
            "camera_name": event.camera_name,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "position": event.position,
            "confidence": float(event.confidence),
            "severity": event.metadata.get("severity", "low"),
            "vehicle_count": int(event.vehicle_count),
            "frame_count": int(event.frame_count),
            "alert_sent": bool(event.metadata.get("sent", False)),
            "event_image_path": event.metadata.get("event_image_path"),
            "message": event.message,
        }
        try:
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append event log")