"""Multi-camera detector manager for traffic vision.

Responsibilities:
- Pull latest frames from MultiCameraManager.
- Run YOLO + ByteTrack per camera independently.
- Keep per-camera tracking state isolated so IDs do not leak across streams.
- Expose structured outputs for downstream state extraction, alerting, and PPO.

Recommended usage:

    from src.vision.multi_camera import MultiCameraManager
    from src.vision.detector import DetectorManager

    cam_mgr = MultiCameraManager("configs/camera_config.json")
    cam_mgr.start()

    det_mgr = DetectorManager(
        model_path="models/best.pt",
        camera_manager=cam_mgr,
        accident_class_id=0,
        vehicle_class_ids=[1, 2, 3, 4],
        conf_thresh=0.5,
        accident_conf_thresh=0.7,
        tracker_yaml="bytetrack.yaml",
    )
    det_mgr.start()

    while True:
        snapshot = det_mgr.get_latest_snapshot()
        # snapshot[cam_id] -> structured dict with tracks, counts, accident flag, etc.

Press q in the preview window if you enable preview.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.multi_camera import MultiCameraManager, _draw_status_panel
from src.vision.tracker_parser import ParsedFrame, TrackRecord, TrackerParser


@dataclass
class CameraDetectionState:
    """Latest structured detection state for one camera."""

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


class _CameraDetectorWorker:
    """Per-camera inference worker with its own YOLO instance and tracking state."""

    def __init__(
        self,
        cam_id: Union[int, str],
        camera_name: str,
        model_path: Union[str, Path],
        camera_manager: MultiCameraManager,
        conf_thresh: float = 0.5,
        accident_conf_thresh: float = 0.7,
        accident_class_id: int = 0,
        vehicle_class_ids: Optional[List[int]] = None,
        tracker_yaml: str = "bytetrack.yaml",
        enable_preview: bool = False,
        preview_scale: float = 1.0,
    ) -> None:
        self.cam_id = cam_id
        self.camera_name = camera_name
        self.camera_manager = camera_manager
        self.conf_thresh = float(conf_thresh)
        self.accident_conf_thresh = float(accident_conf_thresh)
        self.accident_class_id = int(accident_class_id)
        self.vehicle_class_ids = [1, 2, 3, 4] if vehicle_class_ids is None else list(vehicle_class_ids)
        self.tracker_yaml = tracker_yaml
        self.enable_preview = enable_preview
        self.preview_scale = float(preview_scale)

        self.model = YOLO(str(model_path))
        self.parser = TrackerParser(
            class_map={
                0: "accident",
                1: "bus",
                2: "car",
                3: "motorcycle",
                4: "truck",
            },
            vehicle_class_ids=self.vehicle_class_ids,
            accident_class_id=self.accident_class_id,
            accident_conf_thresh=self.accident_conf_thresh,
        )

        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_state = CameraDetectionState(cam_id=cam_id, name=camera_name)
        self.is_running = False
        self.last_processed_frame_idx = -1
        self.max_idle_sleep = 0.01

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name=f"det-{self.cam_id}", daemon=True)
        self.thread.start()
        self.is_running = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.is_running = False

    @staticmethod
    def _make_color(track_id: Optional[int], cls_id: int, accident_class_id: int, vehicle_class_ids: List[int]) -> Tuple[int, int, int]:
        if cls_id == accident_class_id:
            return (0, 0, 255)
        if cls_id in vehicle_class_ids:
            if track_id is not None and track_id >= 0:
                return (
                    int((track_id * 37) % 255),
                    int((track_id * 17) % 255),
                    int((track_id * 29) % 255),
                )
            return (255, 0, 0)
        return (200, 200, 200)

    def _annotate(self, frame: np.ndarray, parsed: ParsedFrame) -> np.ndarray:
        canvas = frame.copy()

        for track in parsed.tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = self._make_color(track.track_id, track.cls_id, self.accident_class_id, self.vehicle_class_ids)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{track.label} v={track.speed_px_s:.1f}px/s"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - 20)), (x1 + w + 6, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            canvas,
            f"Vehicles: {parsed.vehicle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if parsed.accident_found:
            cv2.putText(
                canvas,
                f"Accident: {parsed.accident_conf:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            canvas,
            datetime.fromtimestamp(parsed.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            (10, max(canvas.shape[0] - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        if self.enable_preview and self.preview_scale != 1.0:
            canvas = cv2.resize(
                canvas,
                None,
                fx=self.preview_scale,
                fy=self.preview_scale,
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            latest = self.camera_manager.get_latest_frames(copy=False).get(self.cam_id)
            if not latest:
                time.sleep(self.max_idle_sleep)
                continue

            frame = latest.get("frame")
            if frame is None:
                time.sleep(self.max_idle_sleep)
                continue

            frame_idx = int(latest.get("frame_count", 0))
            if frame_idx == self.last_processed_frame_idx:
                time.sleep(self.max_idle_sleep)
                continue

            self.last_processed_frame_idx = frame_idx
            ts = float(latest.get("timestamp", time.time()))
            cam_name = str(latest.get("name", self.camera_name))

            try:
                # YOLO track with ByteTrack per camera worker.
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.conf_thresh,
                    tracker=self.tracker_yaml,
                    verbose=False,
                )
                result = results[0]

                parsed = self.parser.parse_ultralytics_result(
                    cam_id=self.cam_id,
                    camera_name=cam_name,
                    result=result,
                    timestamp=ts,
                    frame_idx=frame_idx,
                )
                annotated = self._annotate(frame, parsed)

                with self.lock:
                    self.latest_state = CameraDetectionState(
                        cam_id=self.cam_id,
                        name=cam_name,
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
            except Exception as e:
                with self.lock:
                    self.latest_state.last_error = str(e)
                time.sleep(self.max_idle_sleep)

    def get_state(self) -> CameraDetectionState:
        with self.lock:
            return CameraDetectionState(
                cam_id=self.latest_state.cam_id,
                name=self.latest_state.name,
                timestamp=self.latest_state.timestamp,
                frame_count=self.latest_state.frame_count,
                accident_found=self.latest_state.accident_found,
                accident_conf=self.latest_state.accident_conf,
                vehicle_count=self.latest_state.vehicle_count,
                vehicle_ids=list(self.latest_state.vehicle_ids),
                tracks=list(self.latest_state.tracks),
                annotated_frame=None if self.latest_state.annotated_frame is None else self.latest_state.annotated_frame.copy(),
                last_error=self.latest_state.last_error,
            )


class DetectorManager:
    """High-level detector manager for multiple cameras."""

    def __init__(
        self,
        model_path: Union[str, Path],
        camera_manager: MultiCameraManager,
        conf_thresh: float = 0.5,
        accident_conf_thresh: float = 0.7,
        accident_class_id: int = 0,
        vehicle_class_ids: Optional[List[int]] = None,
        tracker_yaml: str = "bytetrack.yaml",
        enable_preview: bool = False,
        preview_scale: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.camera_manager = camera_manager
        self.conf_thresh = float(conf_thresh)
        self.accident_conf_thresh = float(accident_conf_thresh)
        self.accident_class_id = int(accident_class_id)
        self.vehicle_class_ids = [1, 2, 3, 4] if vehicle_class_ids is None else list(vehicle_class_ids)
        self.tracker_yaml = tracker_yaml
        self.enable_preview = enable_preview
        self.preview_scale = float(preview_scale)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.workers: Dict[Union[int, str], _CameraDetectorWorker] = {}
        self.running = False
        self._build_workers()

    def _build_workers(self) -> None:
        self.workers.clear()
        latest = self.camera_manager.get_latest_frames(copy=False)
        for cam_id, item in latest.items():
            worker = _CameraDetectorWorker(
                cam_id=cam_id,
                camera_name=str(item.get("name", cam_id)),
                model_path=self.model_path,
                camera_manager=self.camera_manager,
                conf_thresh=self.conf_thresh,
                accident_conf_thresh=self.accident_conf_thresh,
                accident_class_id=self.accident_class_id,
                vehicle_class_ids=self.vehicle_class_ids,
                tracker_yaml=self.tracker_yaml,
                enable_preview=self.enable_preview,
                preview_scale=self.preview_scale,
            )
            self.workers[cam_id] = worker

    def start(self) -> None:
        if self.running:
            return
        if not self.camera_manager.running:
            raise RuntimeError("camera_manager must be started before detector_manager.")
        for worker in self.workers.values():
            worker.start()
        self.running = True

    def stop(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        self.running = False

    def get_latest_snapshot(self) -> Dict[Union[int, str], Dict[str, Any]]:
        snapshot: Dict[Union[int, str], Dict[str, Any]] = {}
        for cam_id, worker in self.workers.items():
            state = worker.get_state()
            snapshot[cam_id] = {
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
        return snapshot

    def step(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """Compatibility alias for synchronous polling style."""
        return self.get_latest_snapshot()

    def get_accident_flags(self) -> Dict[Union[int, str], bool]:
        snap = self.get_latest_snapshot()
        return {cam_id: bool(data["accident_found"]) for cam_id, data in snap.items()}

    def get_vehicle_counts(self) -> Dict[Union[int, str], int]:
        snap = self.get_latest_snapshot()
        return {cam_id: int(data["vehicle_count"]) for cam_id, data in snap.items()}

    def build_semantic_state(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """Compact state suitable for a downstream adapter (e.g., PPO state builder)."""
        snap = self.get_latest_snapshot()
        out: Dict[Union[int, str], Dict[str, Any]] = {}
        for cam_id, data in snap.items():
            out[cam_id] = {
                "vehicle_count": data["vehicle_count"],
                "accident": data["accident_found"],
                "accident_conf": data["accident_conf"],
                "track_ids": list(data["vehicle_ids"]),
                "timestamp": data["timestamp"],
                "frame_count": data["frame_count"],
                "last_error": data["last_error"],
            }
        return out

    def preview_loop(self, window_name: str = "Detector Manager Preview", max_width: int = 1800) -> None:
        """Optional interactive preview. Press q to quit."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                snap = self.get_latest_snapshot()
                frames = []
                labels = []
                for cam_id in self.camera_manager.camera_ids:
                    item = snap.get(cam_id, {})
                    frame = item.get("annotated_frame")
                    name = item.get("name", str(cam_id))
                    if frame is None:
                        frame = np.zeros((270, 480, 3), dtype=np.uint8)
                    frames.append(frame)
                    labels.append(
                        f"{name} | vehicles={item.get('vehicle_count', 0)} | accident={item.get('accident_found', False)}"
                    )

                if not frames:
                    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                else:
                    tile_size = (480, 270)
                    tiles = []
                    for frame, label in zip(frames, labels):
                        tile = cv2.resize(frame, tile_size)
                        tile = _draw_status_panel(tile, [label])
                        tiles.append(tile)

                    cols = 4 if len(tiles) >= 4 else len(tiles)
                    rows = int(np.ceil(len(tiles) / cols))
                    blank = np.zeros_like(tiles[0])
                    while len(tiles) < rows * cols:
                        tiles.append(blank.copy())

                    row_imgs = []
                    for r in range(rows):
                        row_imgs.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
                    canvas = np.vstack(row_imgs)
                    if canvas.shape[1] > max_width:
                        scale = max_width / canvas.shape[1]
                        canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))

                cv2.imshow(window_name, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()


def load_detector_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_demo(
    camera_config: Union[str, Path],
    model_path: Union[str, Path],
    duration: Optional[float] = None,
    enable_preview=False,
) -> None:
    cam_mgr = MultiCameraManager(camera_config)
    cam_mgr.start(warmup_sec=0.7)

    det_mgr = DetectorManager(
        model_path=model_path,
        camera_manager=cam_mgr,
        conf_thresh=0.5,
        accident_conf_thresh=0.7,
        accident_class_id=0,
        vehicle_class_ids=[1, 2, 3, 4],
        tracker_yaml="bytetrack.yaml",
        enable_preview=enable_preview,
    )
    det_mgr.start()

    print("Detector manager is running.")

    if enable_preview:
        print("Press 'q' to quit preview window")
        try:
            det_mgr.preview_loop()
        finally:
            det_mgr.stop()
            cam_mgr.stop()
        return

    start = time.time()

    try:
        while True:
            snap = det_mgr.get_latest_snapshot()
            total_vehicle = sum(int(v["vehicle_count"]) for v in snap.values())
            any_accident = any(bool(v["accident_found"]) for v in snap.values())
            print(f"vehicles={total_vehicle} | accident={any_accident}", end="\r")

            if duration is not None and duration > 0 and (time.time() - start) >= duration:
                break
            time.sleep(0.1)
    finally:
        det_mgr.stop()
        cam_mgr.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera detector manager demo")
    parser.add_argument("--camera-config", type=str, default="configs/camera_config.json")
    parser.add_argument("--model", type=str, default="models/yolo/best.pt")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--preview", action="store_true", help="Enable OpenCV preview window.")

    args = parser.parse_args()

    run_demo(
        args.camera_config,
        args.model,
        duration=args.duration,
        enable_preview=args.preview   
    )


if __name__ == "__main__":
    main()