# Paper 2 — Perception (YOLOv11 + ByteTrack: accident detection + vehicle recognition)

**Status: SCOPE TBD.** This page is a template to fill once the scope is settled.

## Code (owned)
- `src/vision/`
  - `detector.py` — YOLOv11 (OpenVINO INT8) detection wrapper.
  - `event_detector.py` — accident-event confirmation (3-frame) logic.
  - `tracker_parser.py` — ByteTrack track parsing.
  - `multi_camera.py` — multi-camera frame management.
  - `state_extractor.py` — detections → features (shared: also the P1-aligned obs bridge used by P3).

## Artifacts
- `models/yolo/` — `yolov11.pt`, `yolov11_int8_openvino_model/`.
- `data/` — `raw_video/` (8 clips), `dataset/`, `processed/`.
- `configs/camera_config.json` — 8 cameras: source, direction, phase group, TLS, ROI.
- `results/paper2_perception/` — (empty) detection accuracy, P/R, alert latency.

## Custom class IDs (NOT COCO) — see NOTES.md §4.1
`0=accident, 1=bus, 2=car, 3=motorcycle, 4=truck`.

## TODO before scope lock
- [ ] Define the contribution (accident-detection accuracy? vehicle-mix recognition? INT8 latency on edge?).
- [ ] Dataset + annotation protocol; train/val split; confusion matrix (`scripts/dump_confusion_matrix.py`).
- [ ] Metrics: mAP/precision/recall per class, accident-event latency, ROI calibration error.
- [ ] Relationship to Paper 1's III-E (feature-recovery validation reuses this stack).
- [ ] Decide what lands in `results/paper2_perception/` and which figures.
