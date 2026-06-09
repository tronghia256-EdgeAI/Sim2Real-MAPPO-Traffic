# Traffic-Guard-AI

**MAPPO-based intelligent traffic signal control** with real-time vehicle detection, accident alerting, and Arduino hardware integration.

| Component | Technology |
|-----------|-----------|
| Multi-agent RL | MAPPO (decentralized actors, shared critic) |
| Simulation | SUMO + TraCI |
| Detection | YOLOv8s + ByteTrack |
| Hardware | Arduino (serial, 115200 baud) |
| Alerting | Telegram Bot API |

---

## System Architecture

```
┌─────────────────── PRODUCTION PIPELINE ────────────────────┐
│                                                             │
│  8 Cameras ──► MultiCameraManager (8 threads)              │
│                        │                                   │
│                DetectorManager                             │
│              (YOLOv8s + ByteTrack, 1 worker/cam)           │
│                        │                                   │
│           ┌────────────┴─────────────┐                     │
│           │                          │                     │
│  AccidentEventDetector        VisionToState.build_packet() │
│  (3-frame confirm)            (snapshot → obs[52,])        │
│  Telegram alert + JPEG               │                     │
│                               VisionBuffer.push()          │
│                               (EMA α=0.6, smoothed)        │
│                                      │                     │
│                         obs[0:26] ──► Actor tls_0          │
│                         obs[26:52] ─► Actor tls_1          │
│                                      │                     │
│                         actions_to_serial_times()          │
│                         [t0_d0,t0_d1,t0_d2,t0_d3,         │
│                          t1_d0,t1_d1,t1_d2,t1_d3]         │
│                                      │                     │
│                         SerialBridge ──► Arduino (COM3)    │
└─────────────────────────────────────────────────────────────┘

┌────────────── TRAINING PIPELINE ──────────────┐
│                                               │
│  SUMO ──► MappoTrafficEnv (PettingZoo-style)  │
│           ObservationBuilder (26-dim/agent)   │
│           RewardCalculator                    │
│                    │                          │
│              train_ppo.py ──► last_model.pt   │
└───────────────────────────────────────────────┘
```

### Observation Vector (per agent, 26-dim)

```
Lane ×4:  [effective_queue_norm, occupancy_norm, avg_speed_norm,
           motorbike_share, heavy_vehicle_share]        → 20 dims
TLS:      [phase_0, phase_1, phase_2, phase_3,
           green_timer_norm, pressure_norm]             →  6 dims
```

### Custom YOLO Class IDs (not COCO)

| ID | Class |
|----|-------|
| 0 | accident |
| 1 | bus |
| 2 | car |
| 3 | motorcycle |
| 4 | truck |

---

## Prerequisites

- Python ≥ 3.9
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) — set `SUMO_HOME` environment variable
- CUDA-capable GPU recommended for YOLOv8 inference

---

## Installation

```bash
git clone <repo-url>
cd Traffic-Guard-AI-main

pip install -r requirements.txt
```

**Key dependencies installed by `requirements.txt`:**

```bash
pip install torch torchvision          # PyTorch
pip install ultralytics                # YOLOv8
pip install pyserial                   # Arduino serial bridge
pip install python-telegram-bot        # Telegram alerting
pip install traci eclipse-sumo         # SUMO Python bindings (fallback)
```

> **SUMO must be installed separately.** Verify with `sumo --version` and confirm `$SUMO_HOME` is set.

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/state_config.json` | Obs schema, lane ROI polygons, TLS ID mapping |
| `configs/camera_config.json` | 8 camera sources (`data/raw_video/*.mp4` for testing) |
| `configs/serial.json` | Arduino port (`COM3`), baud rate, yellow/red gap |
| `configs/tele.json` | Telegram `API_TOKEN` + `CHAT_ID` |

```bash
# Copy Telegram template and fill in credentials
cp configs/tele_example.json configs/tele.json
```

---

## How to Run

### Step 1 — Train the MAPPO Policy (SUMO Simulation)

```bash
# Windows: set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
# Linux:   export SUMO_HOME=/usr/share/sumo

python train_ppo.py
# Checkpoint saved to: models/rl/<timestamp>/last_model.pt
```

Evaluate trained policy with SUMO GUI:

```bash
python experiment/test_ppo.py
```

Compare against fixed-time baseline:

```bash
python experiment/test_baseline.py
```

---

### Step 2 — Run the Production System (Vision + Hardware Bridge)

**With Arduino hardware:**

```bash
python src/core/system_orchestrator.py \
  --config        configs/state_config.json \
  --camera-config configs/camera_config.json \
  --model         models/yolo/best.pt \
  --policy        models/rl/<timestamp>/last_model.pt \
  --serial-port   COM3
```

**Dry-run (no Arduino — software only):**

```bash
python src/core/system_orchestrator.py \
  --model  models/yolo/best.pt \
  --policy models/rl/<timestamp>/last_model.pt \
  --no-serial
```

**Key CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hz` | `2.0` | Control loop frequency (Hz) |
| `--buffer` | `5` | VisionBuffer depth (frames) |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `--no-serial` | — | Disable Arduino output |

---

### Step 3 — Telegram Accident Alerts

Alerts fire automatically when `AccidentEventDetector` confirms an accident across **3 consecutive frames** with confidence ≥ 0.7, subject to a 60-second cooldown per camera.

- **Alert sent**: annotated JPEG + message to Telegram chat
- **No `tele.json`**: alerts silently suppressed; event JPEGs still saved to `logs/events/`
- **Event log**: `logs/events/events.jsonl` (append-only JSONL)

---

## Fail-Safe Behaviors

| Condition | Behavior |
|-----------|----------|
| >25% cameras stale (>5 s) | SAFE MODE — no new actions sent to Arduino |
| Policy `predict()` failure | Falls back to fixed 30 s green cycle |
| Serial port unavailable | Dry-run mode — all logic runs, nothing sent |
| `tele.json` missing | Accident detection runs; alerts suppressed |

---

## Project Structure

```
Traffic-Guard-AI-main/
├── src/
│   ├── core/
│   │   ├── system_orchestrator.py   # Production entry point
│   │   ├── policy_loader.py         # Checkpoint loader (auto key-strip)
│   │   └── orchestrator_v2.py       # Refactored v2 (3-thread model)
│   ├── vision/
│   │   ├── multi_camera.py          # 8-camera thread manager
│   │   ├── detector.py              # YOLOv8s + ByteTrack
│   │   ├── state_extractor.py       # Tracks → obs vector
│   │   └── event_detector.py        # Accident detection + Telegram
│   ├── adapters/
│   │   └── vision_to_state.py       # DetectorSnapshot → VisionPacket
│   ├── buffer/
│   │   └── vision_buffer.py         # EMA temporal smoothing
│   ├── traffic_env/
│   │   ├── envs/multi_agent.py      # MAPPO PettingZoo env
│   │   └── components/observations.py
│   ├── utils/
│   │   └── serial_bridge.py         # Arduino serial protocol
│   └── services/
│       └── alert_services.py        # Telegram client
├── experiment/
│   ├── test_ppo.py                  # Evaluate policy in SUMO
│   ├── test_baseline.py             # Fixed-time baseline
│   └── compare_traffic_metrics.py
├── sumo_configs/
│   ├── training/
│   └── evaluating/
├── configs/
├── models/
├── train_ppo.py
└── requirements.txt
```

---

## Running Tests

```bash
pytest                                      # all tests
pytest tests/test_state_consistency.py     # single file
pytest -m "not slow and not serial"        # exclude hardware tests
```

Test markers: `unit`, `integration`, `slow`, `serial`, `state`, `visualization`

---

## Serial Protocol (Arduino)

8 integers sent per control cycle:

```
[tls_0_dir0, tls_0_dir1, tls_0_dir2, tls_0_dir3,
 tls_1_dir0, tls_1_dir1, tls_1_dir2, tls_1_dir3]
```

- **Phase 0** → dirs 0,1 get green time; dirs 2,3 = 0
- **Phase 1** → dirs 2,3 get green time; dirs 0,1 = 0
- Minimum green time: **5 s** | Default cycle: **60 s**

---

## License

MIT
