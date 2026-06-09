"""Multi-camera frame manager for traffic vision.

Design goals:
- Support multiple video sources / RTSP / webcam / video files.
- Keep only the latest frame per camera to minimize latency.
- Read sources in independent background threads.
- Provide a simple API for downstream YOLO / tracking / state extraction.

Expected config formats (flexible):

1) Minimal:
{
  "cameras": [
    {"id": 0, "name": "north", "source": "data/cam1.mp4"},
    {"id": 1, "name": "east",  "source": "data/cam2.mp4"}
  ]
}

2) Shorthand list:
{
  "sources": ["data/cam1.mp4", "data/cam2.mp4"]
}

3) With additional per-camera settings:
{
  "cameras": [
    {"id": "cam_0", "name": "north", "source": "rtsp://...", "resize": [1280, 720]}
  ],
  "frame_width": 1280,
  "frame_height": 720,
  "fps": 15
}

Run directly for a smoke test:
    python src/vision/multi_camera.py --config configs/camera_config.json --duration 15

Controls in test window:
- q : quit
- s : save a collage snapshot of the current latest frames
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class CameraSource:
    cam_id: Union[int, str]
    name: str
    source: Union[int, str]
    resize: Optional[Tuple[int, int]] = None
    enabled: bool = True


class _CameraWorker:
    def __init__(self, spec, target_fps=None, loop_video=True):
        self.spec = spec
        self.target_fps = target_fps
        self.loop_video = loop_video

        self.capture = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_ts = 0.0
        self.frame_count = 0
        self.is_opened = False
        self.last_error = None

        self.source_fps = 0.0
        self.frame_delay = 0.0

    def open(self) -> bool:
        if not self.spec.enabled:
            self.last_error = "Camera disabled in config."
            return False

        src = self.spec.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        self.capture = cv2.VideoCapture(src)

        if self.capture.isOpened():
            self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.source_fps <= 0.0:
                self.source_fps = float(self.target_fps or 10.0)
            effective_fps = float(self.target_fps) if self.target_fps and self.target_fps > 0 else self.source_fps
            self.frame_delay = 1.0 / max(effective_fps, 1.0)

        if self.spec.resize is not None and self.capture.isOpened():
            width, height = self.spec.resize
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self.is_opened = bool(self.capture is not None and self.capture.isOpened())
        if not self.is_opened:
            self.last_error = f"Unable to open source: {self.spec.source}"
        return self.is_opened

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name=f"cam-{self.spec.cam_id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
        self.is_opened = False

    def _loop(self) -> None:
        assert self.capture is not None
        while not self.stop_event.is_set():
            loop_start = time.time()

            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.last_error = "Frame read failed or end of stream reached."
                if self.loop_video and isinstance(self.spec.source, (str, Path)):
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            if self.spec.resize is not None:
                frame = cv2.resize(frame, self.spec.resize)

            with self.lock:
                self.latest_frame = frame
                self.latest_ts = time.time()
                self.frame_count += 1

            elapsed = time.time() - loop_start
            sleep_time = max(self.frame_delay - elapsed, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest(self, copy: bool = True) -> Tuple[Optional[np.ndarray], float, int]:
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, self.frame_count
            frame = self.latest_frame.copy() if copy else self.latest_frame
            return frame, self.latest_ts, self.frame_count


class MultiCameraManager:
    """Threaded multi-camera reader that exposes the latest frame from each camera."""

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        self.config_path: Optional[Path] = None
        self.raw_config: Dict[str, Any] = {}
        self.camera_specs: List[CameraSource] = []
        self.workers: List[_CameraWorker] = []
        self.running: bool = False
        self.target_fps: Optional[float] = None
        self.loop_video: bool = True
        self._load_config(config)
        self._build_workers()

    def _load_config(self, config: Union[str, Path, Dict[str, Any]]) -> None:
        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Camera config not found: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.raw_config = json.load(f)
        elif isinstance(config, dict):
            self.raw_config = dict(config)
        else:
            raise TypeError("config must be a path or a dict")

        self.target_fps = self._parse_target_fps(self.raw_config)
        self.loop_video = bool(self.raw_config.get("loop_video", True))
        self.camera_specs = self._parse_camera_specs(self.raw_config)
        if not self.camera_specs:
            raise ValueError("No camera sources found in config.")

    def _parse_target_fps(self, cfg: Dict[str, Any]) -> Optional[float]:
        global_cfg = cfg.get("global", {}) if isinstance(cfg.get("global", {}), dict) else {}
        fps = global_cfg.get("fps", cfg.get("fps"))
        try:
            if fps is None:
                return None
            fps = float(fps)
            return fps if fps > 0 else None
        except Exception:
            return None

    def _parse_camera_specs(self, cfg: Dict[str, Any]) -> List[CameraSource]:
        specs: List[CameraSource] = []

        global_resize = None
        if "frame_width" in cfg and "frame_height" in cfg:
            global_resize = (int(cfg["frame_width"]), int(cfg["frame_height"]))

        cameras = cfg.get("cameras")
        if isinstance(cameras, list) and cameras:
            for idx, cam in enumerate(cameras):
                if not isinstance(cam, dict):
                    continue
                cam_id = cam.get("id", idx)
                name = str(cam.get("name", f"cam_{cam_id}"))
                source = cam.get("source")
                if source is None:
                    continue
                enabled = bool(cam.get("enabled", True))
                resize = cam.get("resize", global_resize)
                if isinstance(resize, list) and len(resize) == 2:
                    resize = (int(resize[0]), int(resize[1]))
                elif isinstance(resize, tuple) and len(resize) == 2:
                    resize = (int(resize[0]), int(resize[1]))
                else:
                    resize = None
                specs.append(CameraSource(cam_id=cam_id, name=name, source=source, resize=resize, enabled=enabled))
            return specs

        sources = cfg.get("sources")
        if isinstance(sources, list) and sources:
            for idx, source in enumerate(sources):
                if source is None:
                    continue
                specs.append(
                    CameraSource(
                        cam_id=idx,
                        name=f"cam_{idx}",
                        source=source,
                        resize=global_resize,
                        enabled=True,
                    )
                )
            return specs

        # Fallback: support direct key like cam_0, cam_1, ...
        direct_keys = [k for k in cfg.keys() if k.lower().startswith("cam")]
        if direct_keys:
            for idx, key in enumerate(sorted(direct_keys)):
                source = cfg.get(key)
                if source is None:
                    continue
                specs.append(
                    CameraSource(
                        cam_id=idx,
                        name=key,
                        source=source,
                        resize=global_resize,
                        enabled=True,
                    )
                )
        return specs

    def _build_workers(self) -> None:
        self.workers = [_CameraWorker(spec, target_fps=self.target_fps, loop_video=self.loop_video) for spec in self.camera_specs]

    @property
    def camera_ids(self) -> List[Union[int, str]]:
        return [w.spec.cam_id for w in self.workers]

    @property
    def camera_names(self) -> List[str]:
        return [w.spec.name for w in self.workers]

    def start(self, warmup_sec: float = 0.5) -> Dict[Union[int, str], bool]:
        results: Dict[Union[int, str], bool] = {}
        for worker in self.workers:
            opened = worker.open()
            results[worker.spec.cam_id] = opened
            if opened:
                worker.start()
        self.running = any(results.values())
        if warmup_sec > 0:
            time.sleep(warmup_sec)
        return results

    def stop(self) -> None:
        for worker in self.workers:
            worker.stop()
        self.running = False

    def get_latest_frames(self, copy: bool = True) -> Dict[Union[int, str], Dict[str, Any]]:
        out: Dict[Union[int, str], Dict[str, Any]] = {}
        for worker in self.workers:
            frame, ts, count = worker.get_latest(copy=copy)
            out[worker.spec.cam_id] = {
                "name": worker.spec.name,
                "frame": frame,
                "timestamp": ts,
                "frame_count": count,
                "opened": worker.is_opened,
                "error": worker.last_error,
            }
        return out

    def get_collage(
        self,
        max_width: int = 1600,
        tile_size: Tuple[int, int] = (320, 180),
        show_labels: bool = True,
    ) -> Optional[np.ndarray]:
        """Build a preview grid from the latest frames."""
        latest = self.get_latest_frames(copy=False)
        frames = []
        for cam_id in self.camera_ids:
            item = latest.get(cam_id, {})
            frame = item.get("frame")
            name = item.get("name", str(cam_id))
            if frame is None:
                tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
                label = f"{name} (no frame)"
                cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                tile = cv2.resize(frame, tile_size)
                if show_labels:
                    label = f"{name} | {item.get('frame_count', 0)}"
                    cv2.rectangle(tile, (0, 0), (tile_size[0], 35), (0, 0, 0), -1)
                    cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            frames.append(tile)

        if not frames:
            return None

        cols = 4 if len(frames) >= 4 else len(frames)
        rows = int(np.ceil(len(frames) / cols))
        blank = np.zeros_like(frames[0])
        while len(frames) < rows * cols:
            frames.append(blank.copy())

        row_imgs = []
        for r in range(rows):
            row = np.hstack(frames[r * cols:(r + 1) * cols])
            row_imgs.append(row)
        collage = np.vstack(row_imgs)

        if collage.shape[1] > max_width:
            scale = max_width / collage.shape[1]
            collage = cv2.resize(collage, (int(collage.shape[1] * scale), int(collage.shape[0] * scale)))
        return collage

    def wait_until_ready(self, timeout: float = 5.0, min_frames: int = 1) -> bool:
        """Wait until each opened camera has produced at least `min_frames` frames."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = 0
            for worker in self.workers:
                _, _, count = worker.get_latest(copy=False)
                if worker.is_opened and count >= min_frames:
                    ready += 1
            if ready > 0:
                return True
            time.sleep(0.05)
        return False

    def status(self) -> List[Dict[str, Any]]:
        items = []
        for worker in self.workers:
            _, ts, count = worker.get_latest(copy=False)
            items.append(
                {
                    "cam_id": worker.spec.cam_id,
                    "name": worker.spec.name,
                    "source": worker.spec.source,
                    "opened": worker.is_opened,
                    "frames": count,
                    "last_timestamp": ts,
                    "last_error": worker.last_error,
                }
            )
        return items


def _draw_status_panel(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    panel = img.copy()
    y = 24
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 26
    return panel


def smoke_test(config_path: Union[str, Path], duration = None) -> None:
    manager = MultiCameraManager(config_path)
    results = manager.start(warmup_sec=0.7)

    print("Open results:")
    for cam_id, opened in results.items():
        print(f"  {cam_id}: {'OK' if opened else 'FAILED'}")

    if not any(results.values()):
        raise RuntimeError("No camera/video source could be opened.")

    print("Press 'q' to quit, 's' to save collage snapshot.")
    print(f"Target FPS: {manager.target_fps if manager.target_fps else 'source FPS'} | Loop video: {manager.loop_video}")
    start = time.time()
    snap_idx = 0

    try:
        while True:
            collage = manager.get_collage()
            status = manager.status()
            if duration is None:
                duration_text = "∞"
            else:
                duration_text = f"{duration:.1f}s"
            lines = [
                f"Running: {manager.running}",
                f"Sources: {len(status)}",
                f"Elapsed: {time.time() - start:.1f}s / {duration_text}",
            ]
            for item in status[:6]:
                lines.append(
                    f"{item['name']}: open={item['opened']} frames={item['frames']} err={str(item['last_error'])[:28] if item['last_error'] else 'None'}"
                )

            if collage is None:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                canvas = _draw_status_panel(canvas, lines)
            else:
                canvas = collage
                canvas = _draw_status_panel(canvas, lines)

            cv2.imshow("MultiCamera Smoke Test", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                out_path = Path("multi_camera_snapshot.png")
                cv2.imwrite(str(out_path), canvas)
                print(f"Saved snapshot to {out_path.resolve()}")
                snap_idx += 1

            if duration is not None and duration > 0 and (time.time() - start) >= duration:
                break
    finally:
        manager.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Threaded multi-camera frame manager")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/camera_config.json",
        help="Path to camera configuration JSON",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Smoke test duration in seconds (<=0 means run until q)",
    )
    args = parser.parse_args()
    smoke_test(args.config, duration=args.duration)


if __name__ == "__main__":
    main()