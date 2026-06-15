# Module Index

Complete reference for every folder and file in this repository.
Each entry includes a one-sentence technical description and the paper(s) it supports.

> **Paper mapping**
> - **Paper 1** — Multi-agent reinforcement learning for adaptive traffic signal control (MAPPO vs baselines, SUMO simulation)
> - **Paper 2** — Real-time vision pipeline for vehicle detection, tracking, and accident alerting (YOLOv8 + ByteTrack + Telegram)
> - **Paper 3** — Sim-to-real transfer: bridging SUMO-trained MAPPO policy to a live 8-camera + Arduino deployment

---

## Root

| File | Description | Paper |
|------|-------------|-------|
| `train_ppo.py` | Main MAPPO training script: rollout collection, PPO update loop, checkpoint saving to `models/mappo/`. | 1 |
| `README.md` | Public-facing project overview with architecture diagram, installation, and run instructions. | 1, 2, 3 |
| `NOTES.md` | AI coding assistant instructions: architecture constraints, class IDs, obs layout, fail-safes. | — |
| `requirements.txt` | Python dependency manifest for the entire project. | — |
| `pytest.ini` | pytest configuration: test markers (`unit`, `integration`, `slow`, `serial`, `state`, `visualization`). | — |

---

## `configs/`

Runtime configuration files. None contain hard-coded logic — all are loaded at startup.

| File | Description | Paper |
|------|-------------|-------|
| `state_config.json` | Single source of truth for obs schema, lane polygon ROIs, normalization caps, and runtime-to-training TLS ID mapping. | 1, 3 |
| `camera_config.json` | 8-camera source definitions (cam_0…cam_7); `source` fields point to `data/raw_video/*.mp4` for offline testing. | 2, 3 |
| `serial.json` | Arduino serial port (`COM3`), baud rate (115 200), yellow duration, and inter-phase red gap. | 3 |
| `tele.json` | Telegram Bot `API_TOKEN` + `CHAT_ID`; missing file silently suppresses alerts without crashing. | 2 |
| `tele_example.json` | Template for `tele.json`; committed to repo so new contributors know the required schema. | 2 |

---

## `src/`

Production source code. No experiment logic lives here.

### `src/core/`

| File | Description | Paper |
|------|-------------|-------|
| `system_orchestrator.py` | Primary production entry point: initialises all subsystems, runs the main control loop at a configurable Hz rate. | 3 |
| `orchestrator_v2.py` | Refactored orchestrator using a 3-thread model (InferenceThread + ControlThread + MainThread) for lower latency. | 3 |
| `policy_loader.py` | Robust checkpoint loader: reads `.pt` files with `weights_only=False`, auto-strips common key prefixes (`net.`, `actor.`, `module.`), restores `RunningMeanStd` obs normaliser. | 1, 3 |
| `camera_manager_v2.py` | Manages 8 camera threads; detects stale feeds (>5 s) and triggers SAFE MODE. | 2, 3 |
| `inference_engine.py` | Batches per-agent obs vectors and calls the MAPPO actor network; optimised for low-latency repeated inference. | 3 |
| `metrics_collector.py` | Aggregates and logs runtime traffic metrics (queue length, throughput, latency) from the live production loop. | 2, 3 |
| `config_validator.py` | Validates all config files at startup and raises descriptive errors for missing or malformed fields. | 3 |

### `src/vision/`

| File | Description | Paper |
|------|-------------|-------|
| `detector.py` | Per-camera YOLOv8s + ByteTrack worker: runs inference on each frame and returns a `DetectorSnapshot` of tracked objects. | 2 |
| `event_detector.py` | Accident detection module: requires 3 consecutive frame confirmations at confidence ≥ 0.7 with a 60 s cooldown before firing a Telegram alert and saving a JPEG. | 2 |
| `multi_camera.py` | Spawns one thread per camera, reads frames, and feeds them into `DetectorManager`; implements the `>25% stale` SAFE MODE trigger. | 2, 3 |
| `state_extractor.py` | Converts ByteTrack output (bounding boxes, class IDs) into a normalised 26-dim observation vector using custom YOLO class IDs (1=bus, 2=car, 3=motorcycle, 4=truck). | 3 |
| `tracker_parser.py` | Parses raw ByteTrack track objects into structured `VehicleTrack` dataclasses consumed by `StateExtractor`. | 2 |

### `src/adapters/`

| File | Description | Paper |
|------|-------------|-------|
| `vision_to_state.py` | Bridge adapter: converts a `DetectorSnapshot` (camera-indexed dict) into a `VisionPacket` (flat 52-dim global obs + per-TLS semantic state). | 3 |

### `src/buffer/`

| File | Description | Paper |
|------|-------------|-------|
| `vision_buffer.py` | Implements EMA temporal smoothing (α=0.6) over successive `VisionPacket` observations to reduce per-frame noise before policy inference. | 3 |

### `src/services/`

| File | Description | Paper |
|------|-------------|-------|
| `alert_services.py` | Telegram Bot client: sends annotated JPEG images and text messages to a configured chat ID when an accident event is confirmed. | 2 |

### `src/traffic_env/`

The SUMO-based training environment. Mirrors the production obs vector exactly for sim-to-real consistency.

| File | Description | Paper |
|------|-------------|-------|
| `config.py` | Dataclass-based single source of truth for all training hyperparameters (`SimConfig`, `ObservationConfig`, `RewardConfig`); `build_default_config()` factory ensures consistent env construction. | 1, 3 |
| `components/observations.py` | `ObservationBuilder` reads lane metrics from SUMO via TraCI and assembles the 26-dim per-agent obs vector; `VisionLaneMetrics` mirrors the same logic for vision-based obs. | 1, 3 |
| `components/rewards.py` | `RewardCalculator` computes a weighted combination of queue length, throughput, waiting time, and phase-switch penalty. | 1 |
| `envs/base_sumo.py` | Base class managing SUMO/TraCI process lifecycle (start, step, close) and yellow-phase enforcement. | 1 |
| `envs/multi_agent.py` | PettingZoo-style `MappoTrafficEnv`: exposes `reset()`, `step()`, and `close()` with per-agent obs, rewards, and info dicts compatible with the MAPPO trainer. | 1 |

### `src/utils/`

| File | Description | Paper |
|------|-------------|-------|
| `serial_bridge.py` | Encodes 8 green-time integers and writes them to the Arduino over USB serial at 115 200 baud; no-op when port is unavailable (dry-run mode). | 3 |

### `src/visualization/`

| File | Description | Paper |
|------|-------------|-------|
| `display_manager.py` | Renders live annotated camera feeds with bounding boxes, class labels, and traffic state overlays for operator monitoring. | 2 |

---

## `experiment/`

All scripts that run inside the SUMO simulator for training support, evaluation, and plotting. No production logic.

### `experiment/baselines/`

Classical ATSC algorithms used as comparison baselines in Paper 1.

| File | Description | Paper |
|------|-------------|-------|
| `__init__.py` | Package marker for `experiment.baselines`. | — |
| `max_pressure.py` | Stateless Max Pressure controller (Varaiya 2013): selects the signal group with higher total `effective_queue_norm` pressure each step; includes 10-episode evaluator with 95% CI reporting. | 1 |
| `sotl.py` | Stateful Self-Organizing Traffic Light controller (Cools 2013): switches phase based on minimum green window (`phi_min`) and red-approach vehicle threshold (`kappa`); includes 10-episode evaluator with 95% CI reporting. | 1 |

### `experiment/runners/`

Episode execution scripts that instantiate an environment and run a full evaluation loop.

| File | Description | Paper |
|------|-------------|-------|
| `__init__.py` | Package marker for `experiment.runners`. | — |
| `eval_ppo_gui.py` | Interactive SUMO GUI runner for a single MAPPO policy episode; intended for visual inspection and debugging. (was `test_ppo.py`) | 1 |
| `eval_baseline_gui.py` | Interactive SUMO GUI runner for a single fixed-time baseline episode (60 s green / 3 s yellow per phase). (was `test_baseline.py`) | 1 |
| `_deprecated/evaluate.py` | **DEPRECATED** — use `eval_compare.py`. Unified comparative runner: MAPPO/MaxPressure/SOTL for N episodes; proxy metrics. | 1 |
| `_deprecated/compare_traffic_metrics.py` | **DEPRECATED** — use `eval_compare.py`. Gutted to a deprecation stub (was a PPO-vs-fixed 3-panel figure). | 1 |

### `experiment/ablation/`

Reserved for Paper 1 ablation study runners (reward component ablation, EMA alpha sweep, multi-seed training).

| File | Description | Paper |
|------|-------------|-------|
| `__init__.py` | Package marker for `experiment.ablation`. | — |

### `experiment/plots/`

Post-processing and visualisation utilities that operate on saved result files.

| File | Description | Paper |
|------|-------------|-------|
| `__init__.py` | Package marker for `experiment.plots`. | — |
| `parse_tripinfo.py` | Parses SUMO `tripinfo.xml` output files to extract vehicle throughput and total CO₂ emissions (mg → kg); supports both baseline and PPO output files. | 1 |

---

## `sumo_configs/`

SUMO network and demand files. Each sub-directory is a self-contained simulation scenario.

### `sumo_configs/training/`

Training demand scenario (period=1.2 s ≈ 3 000 veh/h). Lower density encourages early exploration.

| File | Description | Paper |
|------|-------------|-------|
| `intersections.net.xml` | SUMO road network: 2-intersection arterial topology used for all training and evaluation. | 1 |
| `sumo_config.sumocfg` | Training scenario entry point: references `traffic_train.rou.xml`, no simulation end time (episode-length controlled by env). | 1 |
| `traffic_train.rou.xml` | Randomised vehicle routes (period=1.2 s) generated by `randomTrips.py` with `mixed_traffic` vehicle type. | 1 |
| `trips.trips.xml` | Raw trip definitions used to generate `traffic_train.rou.xml`. | 1 |
| `vtypes.add.xml` | Vehicle type definitions: `mixed_traffic` distribution over car, bus, motorcycle, truck. | 1 |

### `sumo_configs/evaluation/medium/`

Medium-density evaluation scenario (period=0.8 s ≈ 4 500 veh/h, end=5 400 s). Primary benchmark scenario for Paper 1 tables.

| File | Description | Paper |
|------|-------------|-------|
| `intersections.net.xml` | Same road network as training (shared topology). | 1 |
| `sumo_config.sumocfg` | Evaluation entry point: references `traffic_test.rou.xml`, end=5 400 s, emissions output enabled. | 1 |
| `traffic_test.rou.xml` | Evaluation vehicle routes (period=0.8 s) — higher density than training for out-of-distribution testing. | 1 |
| `vtypes.add.xml` | Vehicle type definitions (same as training). | 1 |
| `trips.trips.xml` | Raw trips used to generate `traffic_test.rou.xml`. | 1 |
| `tripinfo.xml` | SUMO output: per-vehicle trip completion times (most recent run). | 1 |
| `tripinfo_baseline.xml` | SUMO output: tripinfo for the fixed-time baseline run. | 1 |
| `tripinfo_ppo.xml` | SUMO output: tripinfo for the MAPPO policy run. | 1 |
| `summary.xml` | SUMO step-level network summary (most recent run). | 1 |
| `summary_baseline.xml` | SUMO step-level summary for the fixed-time baseline run. | 1 |
| `summary_ppo.xml` | SUMO step-level summary for the MAPPO policy run. | 1 |

### `sumo_configs/evaluation/low/`

Low-density scenario placeholder (target period ≈ 2.0 s, ~1 800 veh/h). Required for Paper 1 robustness section.

### `sumo_configs/evaluation/high/`

High-density scenario placeholder (target period ≈ 0.5 s, ~7 200 veh/h). Required for Paper 1 robustness section.

### `sumo_configs/evaluation/asymmetric/`

Asymmetric-demand scenario placeholder (80/20 directional split). Required for Paper 1 robustness section.

---

## `data/`

| Path | Description | Paper |
|------|-------------|-------|
| `raw_video/video1–8.mp4` | Eight recorded traffic camera feeds used for offline production system testing. | 2, 3 |
| `raw_video/accident_video.mp4` | Annotated accident recording used to validate `AccidentEventDetector` trigger logic. | 2 |
| `dataset/` | YOLO training annotations (images + labels) for the custom 5-class vehicle + accident model. | 2 |
| `processed/` | Reserved for pre-extracted feature arrays or compressed clips (currently empty). | 2, 3 |

---

## `models/`

| Path | Description | Paper |
|------|-------------|-------|
| `mappo/20260418_215140/last_model.pt` | Most recent MAPPO checkpoint from run `20260418_215140`; contains `actor_state_dict`, `obs_rms`, and `obs_dim`. | 1, 3 |
| `mappo/20260418_215140/best_model.pt` | Best-performing MAPPO checkpoint (highest mean eval reward) from the same run. | 1, 3 |
| `mappo/20260418_215140/checkpoint_*.pt` | Periodic intermediate checkpoints saved every 50 000 training steps for learning curve reconstruction. | 1 |
| `mappo/20260418_215140/run_config.json` | Frozen hyperparameter snapshot for the training run (seed, lr, γ, λ, clip ratio, etc.). | 1 |
| `yolo/best.pt` | Primary YOLOv8s checkpoint trained on the custom 5-class dataset (0=accident, 1=bus, 2=car, 3=motorcycle, 4=truck). | 2 |
| `yolo/best_yolo_model.pt` | Alternative YOLOv8 checkpoint (backup / earlier training run). | 2 |

---

## `logs/`

Runtime logs. Not committed to git except for reproducibility artifacts.

| Path | Description | Paper |
|------|-------------|-------|
| `events/events.jsonl` | Append-only JSONL event log: one record per confirmed accident with camera ID, timestamp, confidence, and JPEG path. | 2 |
| `events/event_cam_*.jpg` | Annotated JPEG frames saved at accident detection time; sent to Telegram and archived locally. | 2 |
| `rl/20260418_215140/events.out.tfevents.*` | TensorBoard event file for the MAPPO training run; contains reward, loss, entropy, and fps curves. | 1 |
| `rl/20260418_215140/train_episodes.csv` | Per-episode training metrics CSV (episode reward, length, halted vehicles). | 1 |
| `rl/20260418_215140/train_updates.csv` | Per-update training metrics CSV (policy loss, value loss, entropy, KL divergence). | 1 |
| `rl/20260418_215140/run_config.json` | Duplicate of `models/mappo/.../run_config.json`; written by the logger for convenience. | 1 |
| `yolo/` | YOLO training logs directory (populated by Ultralytics during fine-tuning). | 2 |

---

## `results/`

Artefacts produced by evaluation runs. Organised by paper.

| Path | Description | Paper |
|------|-------------|-------|
| `paper1_mappo/training_curves/mean_reward.png` | Mean episode reward over training steps — primary learning curve figure. | 1 |
| `paper1_mappo/training_curves/loss.png` | Total policy + value loss over training updates. | 1 |
| `paper1_mappo/training_curves/value_loss.png` | Value function loss (Huber) over training updates. | 1 |
| `paper1_mappo/training_curves/entropy_loss.png` | Policy entropy over training updates (exploration indicator). | 1 |
| `paper1_mappo/training_curves/fps.png` | Simulation throughput (frames per second) over training. | 1 |
| `paper1_mappo/training_curves/Screenshot*.png` | TensorBoard screenshots of the training dashboard. | 1 |
| `paper1_mappo/eval_tables/metrics_comparison.csv` | Structured comparison table: waiting time, queue length, speed, throughput for baseline vs MAPPO across Normal and Hardcore demand modes. | 1 |
| `paper2_perception/` | Reserved for Paper 2 (Perception) detection accuracy, precision/recall, and alert-latency results. | 2 |
| `paper3_system/` | Reserved for Paper 3 (System) end-to-end latency/throughput and fail-safe results. | 3 |
| `graphs/Figure_1_normal.png` | Time-series comparison plot (normal traffic demand): halted vehicles and speed for PPO vs fixed-time. | 1 |
| `graphs/Figure_2_hardcore.png` | Time-series comparison plot (high-demand scenario): halted vehicles and speed for PPO vs fixed-time. | 1 |

---

## `scripts/`

Convenience entry-point launchers. Thin wrappers — no business logic.

| File | Description | Paper |
|------|-------------|-------|
| `run_smoke_test.py` | Smoke test launcher: initialises `MappoTrafficEnv`, resets, takes one step, and prints obs/reward keys to confirm the SUMO+Python stack is working. | 1 |

---

## `hardware/`

| File | Description | Paper |
|------|-------------|-------|
| `mega_controller/mega_controller.ino` | Arduino Mega firmware: receives 8 green-time integers over serial (115 200 baud) and drives 2 × 4-direction traffic light arrays with configurable yellow gap. | 3 |
| `schematic_traffic_system_circuit.png` | Wiring schematic for the Arduino + relay + LED traffic light hardware assembly. | 3 |

---

## `docs/`

| File | Description | Paper |
|------|-------------|-------|
| `MODULE_INDEX.md` | This file: comprehensive module index with paper attributions. | — |
| `architecture_diagram.drawio` | Editable draw.io source for the full system architecture diagram. | 1, 2, 3 |
| `architecture_diagram.png` | Exported PNG of the architecture diagram for embedding in papers and README. | 1, 2, 3 |
| `system_diagram.drawio` | Editable draw.io source for the production pipeline data-flow diagram. | 3 |
| `system_diagram.png` | Exported PNG of the production pipeline diagram. | 3 |

---

## `notebooks/`

Reserved for Jupyter notebooks (exploratory analysis, figure generation). Currently empty.

---

## `tests/`

| File | Description | Paper |
|------|-------------|-------|
| `__init__.py` | Package marker for the test suite. | — |
| `test_state_consistency.py` | Unit + integration tests verifying that the 26-dim obs vector layout is identical between `ObservationBuilder` (SUMO) and `StateExtractor` (vision). | 1, 3 |
| `test_end_to_end.py` | End-to-end integration test: spins up a SUMO episode, runs the full obs→policy→action→step loop, and asserts no exceptions. | 1 |
| `test_serial_communication.py` | Tests `SerialBridge` encoding logic with a mocked COM port; marked `serial` to exclude from CI. | 3 |
| `test_visualization.py` | Tests `DisplayManager` rendering with synthetic frames; marked `visualization`. | 2 |
