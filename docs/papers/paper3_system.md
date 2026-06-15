# Paper 3 — System (full real-time deployment)

**Status: SCOPE TBD.** This page is a template to fill once the scope is settled.
This is the **companion deployment study** referenced by Paper 1 (closed-loop field operation).

## Code (owned)
- `src/core/`
  - `system_orchestrator.py` — primary production entry point.
  - `orchestrator_v2.py` — refactored (InferenceThread + ControlThread + AccidentThread).
  - `inference_engine.py`, `camera_manager_v2.py`, `metrics_collector.py`, `config_validator.py`.
- `src/adapters/vision_to_state.py` — DetectorSnapshot → VisionPacket.
- `src/buffer/vision_buffer.py` — EMA temporal smoothing → policy obs.
- `src/services/alert_services.py` — Telegram + JPEG accident alerts.
- `src/utils/serial_bridge.py` — Arduino serial protocol.
- `src/visualization/display_manager.py` — live display.
- `hardware/` — `mega_controller/mega_controller.ino`, `schematic_traffic_system.png`.

## Artifacts
- `scripts/run_real_deployment.py`, `scripts/dashboard.py` (Streamlit).
- `configs/{camera_config,serial,tele}.json` (`tele.json` gitignored; copy from `tele_example.json`).
- `results/paper3_system/` — (empty) end-to-end latency/throughput, fail-safe behaviour.

## Depends on (shared)
- Paper 1 policy via `src/core/policy_loader.py`; obs bridge via `src/vision/state_extractor.py`;
  obs schema `configs/state_config.json`. Pre-deploy checklist: NOTES.md §13.

## TODO before scope lock
- [ ] Define the contribution (end-to-end pipeline? real-time latency budget? fail-safe design? hardware-in-the-loop?).
- [ ] Metrics: per-stage latency (capture→detect→obs→policy→serial), throughput, safe-mode triggers, uptime.
- [ ] Field vs sim gap framing (this is Paper 1's "companion field study").
- [ ] Decide what lands in `results/paper3_system/` and which figures.
