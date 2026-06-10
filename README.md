<div align="center">

# 🚦 Sim2Real-MAPPO-Traffic

### Multi-Agent Deep Reinforcement Learning for Intelligent Traffic Signal Control
#### *Bridging the Simulation-to-Reality Gap with Vision-Based State & Reward Proxies*

[![Stars](https://img.shields.io/github/stars/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic?style=for-the-badge&logo=github&color=f59e0b)](https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic/stargazers)
[![Forks](https://img.shields.io/github/forks/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic?style=for-the-badge&logo=github&color=6366f1)](https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.18%2B-f97316?style=for-the-badge)](https://sumo.dlr.de/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-7c3aed?style=for-the-badge)](https://github.com/ultralytics/ultralytics)

<br/>

[📖 Overview](#-project-overview) &nbsp;·&nbsp;
[✨ Features](#-key-features) &nbsp;·&nbsp;
[🏗️ Architecture](#-system-architecture) &nbsp;·&nbsp;
[⚖️ Sim-to-Real Alignment](#-sim-to-real-metric-alignment) &nbsp;·&nbsp;
[🚀 Quick Start](#-installation--setup) &nbsp;·&nbsp;
[📊 Dashboard](#-how-to-run) &nbsp;·&nbsp;
[📚 References](#-references--acknowledgments)

</div>

---

## 📖 Project Overview

Traffic Signal Control (TSC) using deep reinforcement learning has shown strong results in simulation — but deployed systems routinely **fail to generalize** because simulators like SUMO provide privileged metrics (exact queue lengths, per-vehicle waiting timers, network throughput counts) that do not exist in the real world.

**Sim2Real-MAPPO-Traffic** is a full-stack framework that closes this gap. It trains a decentralized [MAPPO](https://arxiv.org/abs/2103.01955) policy entirely inside SUMO, then deploys the same trained weights to control real intersections using only a camera stream — with no simulator at inference time. The key insight is that every reward signal and observation feature used during training has a **vision proxy** computable from YOLOv8 detections and ByteTrack trajectories:

| Simulation Metric | Vision Proxy |
|---|---|
| `traci.lane.getLastStepHaltingNumber` | Halted vehicles in ROI polygon (speed < threshold) |
| `traci.lane.getWaitingTime` | Accumulated halt duration per ByteTrack ID |
| `simulation.getArrived()` | Track IDs exiting camera ROI per second |
| Phase index (0–3) | Timestamp-based TLS state machine |

The reward function is grounded in the **PRESSLIGHT** (KDD 2019) and **CoLight** (CIKM 2019) formulations, extended with a nonlinear queue penalty and cooperative throughput signal.

---

## ✨ Key Features

- 🤝 **Decentralized Multi-Agent Coordination** — Two MAPPO agents (one per intersection) share a centralized critic during training but act independently at inference, making deployment trivially scalable.

- 📐 **Phase-Aware Pressure Reward** — Implements the PRESSLIGHT pressure signal `max((Σ red_queue − Σ green_queue) / N_lanes, 0)`, which directly penalizes choosing the wrong phase regardless of absolute traffic volume.

- 🎯 **Nonlinear Queue Penalty** — `mean_q²` amplifies gradient during sustained congestion while staying near-zero in free-flow, preventing noisy updates on lightly-loaded intersections.

- 👁️ **Full Vision Pipeline** — YOLOv11n-seg (custom 5-class model: accident / bus / car / motorcycle / truck) + ByteTrack multi-object tracking. Single shared model processes all cameras sequentially with per-camera tracker state isolation via deepcopy. OpenVINO INT8 quantized for maximum CPU throughput.

- 📦 **Production Orchestrator** — Multi-threaded runtime with safe-mode fallback (>25% stale cameras → hold phase), automatic policy failure recovery (fixed-cycle fallback), and dry-run mode for testing without hardware.

- 🖥️ **Real-Time Streamlit Dashboard** — Live 8-camera feed with YOLO bounding boxes, 3-colour TLS SVG with real yellow-transition logic, per-intersection vehicle type breakdown, and direct Arduino serial control.

- 🔌 **Arduino Integration** — Dashboard connects to an Arduino traffic-light controller via SerialBridge (`<S0,...,S7>\n` protocol, 115200 baud), sends RYG commands only on state change + 1-second heartbeat.

- 🛡️ **Robust Checkpoint Loading** — `PolicyLoader` auto-strips common key prefixes (`net.`, `actor.`, `module.`, `_nn_module.net.`) so checkpoints from any training framework load cleanly.

- 🚨 **Accident Detection** — 3-frame confirmation logic with Telegram alert and JPEG snapshot saved to `logs/events/`.

---

## 🏗️ System Architecture

### Training Pipeline (SUMO)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUMO Simulator                               │
│  traci.lane.*  ──►  ObservationBuilder  ──►  26-dim obs per agent  │
│  traci.tls.*   ──►  RewardCalculator    ──►  PRESSLIGHT reward      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    MappoTrafficEnv (PettingZoo-style)
                             │
              ┌──────────────┴───────────────┐
              │          MAPPO Trainer        │
              │  Centralized Critic (global)  │
              │  Decentralized Actors (×2)    │
              └──────────────┬───────────────┘
                             │
                    best_model.pt  ◄─── saved checkpoint
```

### Deployment Pipeline (Real-Time)

```
  ┌──────────┐   ┌──────────┐         ┌──────────┐   ┌──────────┐
  │  Cam 0   │   │  Cam 1   │   ...   │  Cam 6   │   │  Cam 7   │
  │ j0_north │   │ j0_south │         │ j2_east  │   │ j2_west  │
  └────┬─────┘   └────┬─────┘         └────┬─────┘   └────┬─────┘
       │              │                    │              │
       └──────────────┴────────────────────┴──────────────┘
                              │  (8 parallel threads)
                    MultiCameraManager
                              │
                    DetectorManager  ◄── YOLOv11n-seg INT8 OpenVINO
                              │          (single model, sequential)
                    AccidentThread   ◄── dedicated 5fps accident model
                              │
                    AccidentEventDetector
                              │  3-frame confirm → Telegram + JPEG
                    VisionToState.build_packet()
                              │
                    ┌─────────▼──────────┐
                    │  StateExtractor    │
                    │  ROI point-in-poly │  ◄── lane_definitions
                    │  5 features/lane   │      (state_config.json)
                    │  6 TLS features    │
                    └─────────┬──────────┘
                              │
                    VisionBuffer  ◄── EMA smoothing (α = 0.6)
                              │
                    ┌─────────▼──────────┐
                    │  PolicyInterface   │
                    │  MAPPO Inference   │  ◄── best_model.pt
                    │  obs[0:26] → tls_0 │
                    │  obs[26:52]→ tls_1 │
                    └─────────┬──────────┘
                              │  {tls_id: phase_action ∈ {0,1}}
               ┌──────────────┴──────────────┐
               ▼                             ▼
      SerialBridge                   Streamlit Dashboard
   <S0,S1,...,S7>\n                  Live SVG TLS + metrics
        │
   Arduino Controller
   (COM3, 115200 baud)
```

### Observation Vector Layout (52-dim global)

```
 ◄──────────── Agent 0 / tls_0 (26 dims) ────────────►◄──── Agent 1 / tls_1 (26 dims) ────►
 [ lane_0(5) | lane_1(5) | lane_2(5) | lane_3(5) | TLS(6) | lane_4..7(20) | TLS(6) ]

 Lane features (×5):  effective_queue_norm | occupancy_norm | avg_speed_norm
                       motorbike_share      | heavy_vehicle_share

 TLS features  (×6):  phase_one_hot[0..3] (4-dim) | green_timer_norm | pressure_norm
```

---

## ⚖️ Sim-to-Real Metric Alignment

Every training signal has a vision proxy so the policy generalizes without retraining.

| Reward Component | SUMO Source (Training) | Vision Proxy (Inference) | Alignment |
|---|---|---|---|
| `queue_penalty` | `getLastStepHaltingNumber / cap` | Halted vehicles in ROI / cap | ✅ Good |
| `pressure_penalty` | Halting counts partitioned by phase group | Same, using ROI polygon membership | ✅ Good |
| `throughput_reward` | `simulation.getArrived()` delta | ByteTrack IDs exiting ROI per step | ⚠️ Partial\* |
| `switch_penalty` | Action comparison | Action comparison (identical logic) | ✅ Exact |
| `low_speed_penalty` | `getLastStepMeanSpeed` | Pixel-displacement speed estimate | ⚠️ Noisy† |
| `waiting_penalty` | `getWaitingTime / waiting_cap` | Queue proxy (no per-vehicle timer) | ⚠️ Proxy |

> \* SUMO counts vehicles leaving the **entire network**; ByteTrack counts track IDs exiting a **camera ROI**. Reduce `throughput_norm_divisor` or pass per-intersection delta when using vision.
>
> † Pixel-displacement speed is significantly noisier than SUMO ground truth. Consider setting `"low_speed_penalty": 0.0` in `RewardConfig.weights` for vision deployment.

### Default Reward Weights

```python
DEFAULT_REWARD_WEIGHTS = {
    "queue":             -1.0,   # nonlinear (mean_q²) — primary congestion signal
    "pressure":          -0.5,   # PRESSLIGHT phase-aware imbalance
    "throughput":        +1.0,   # cooperative cleared-vehicles
    "switch_penalty":    -0.1,   # phase-switching cost
    "low_speed_penalty": -0.2,   # mean speed below threshold
    "waiting_time":      -0.3,   # accumulated delay
}
# Effective output range: [-1.94, +0.50]  →  clipped to [-2.0, +1.5]
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic.git
cd Sim2Real-MAPPO-Traffic
```

### 2. Install SUMO

SUMO is required for **training only**. Skip this step for inference/dashboard use.

```bash
# Ubuntu / Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS (Homebrew)
brew install sumo

# Windows — download installer from:
# https://sumo.dlr.de/docs/Downloads.php
```

Then set the environment variable:

```bash
# Linux / macOS
export SUMO_HOME=/usr/share/sumo   # adjust to your installation path

# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare Dataset *(for YOLO retraining only)*

The YOLOv11 model (`models/yolo/yolov11.pt`) is already trained and included. To **retrain** the detector with your own data:

```
data/
├── dataset/         ← place your YOLO-format dataset here
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── raw_video/       ← 8 camera test videos (included)
```

Classes must match the custom model: `0=accident  1=bus  2=car  3=motorcycle  4=truck`.
The `data/dataset/` directory is gitignored — add your images locally or mount a shared drive.

### 6. Configure Telegram Alerts *(optional)*

```bash
cp configs/tele_example.json configs/tele.json
# Edit configs/tele.json — fill in your API_TOKEN and CHAT_ID
```

---

## 🎮 How to Run

### Mode A — Training (MAPPO inside SUMO)

```bash
# Train with default hyperparameters
python experiment/runners/train_ppo.py \
  --sumo-cfg sumo_configs/training/sumo_config.sumocfg

# Multi-seed evaluation of a saved checkpoint
python experiment/runners/train_ppo.py \
  --mode multiseed_eval \
  --sumo-cfg   sumo_configs/evaluation/medium/sumo_config.sumocfg \
  --checkpoint models/mappo/20260418_215140/best_model.pt

# Evaluate with SUMO GUI (visual inspection)
python experiment/runners/test_ppo.py
```

Training logs are written to `logs/rl/<run_id>/`:

| File | Contents |
|---|---|
| `train_episodes.csv` | Per-episode reward, throughput, queue, waiting |
| `train_updates.csv` | Policy loss, value loss, entropy per PPO update |
| `run_config.json` | Full hyperparameter snapshot for reproducibility |
| `best_model.pt` | Best checkpoint by mean episode reward |

To generate evaluation scenarios (low / high / asymmetric traffic):

```bash
# Requires SUMO_HOME set and SUMO tools on PATH
python scripts/generate_sumo_scenarios.py
```

---

### Mode B — Production Deployment (Orchestrator)

```bash
# Full deployment: cameras + YOLO + MAPPO + Arduino
python src/core/system_orchestrator.py \
  --config        configs/state_config.json \
  --camera-config configs/camera_config.json \
  --model         models/yolo/yolov11.pt \
  --policy        models/mappo/20260418_215140/best_model.pt \
  --serial-port   COM3

# Dry-run — no Arduino hardware required
python src/core/system_orchestrator.py \
  --model  models/yolo/yolov11.pt \
  --policy models/mappo/20260418_215140/best_model.pt \
  --no-serial
```

**Fail-safe behaviours at runtime:**

| Condition | Response |
|---|---|
| > 25% cameras stale (> 5s no frame) | Enter SAFE MODE — hold current phase, no new serial commands |
| `policy.predict()` raises exception | Fall back to fixed-cycle policy (30s green per phase) |
| Serial port unavailable | Dry-run mode — all logic runs, nothing sent to Arduino |
| `configs/tele.json` missing | Accident detection still runs; alerts silently suppressed |

---

### Mode C — Streamlit Dashboard

```bash
streamlit run scripts/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The dashboard provides three pages:

| Page | Description |
|---|---|
| **Camera Dashboard** | 8 live feeds with scaled YOLO bounding boxes, 3-colour TLS SVG, per-intersection vehicle type counts, Arduino control panel |
| **Live Simulation** | Run MAPPO / Max Pressure / SOTL / Fixed-Time inside SUMO with real-time queue, speed, throughput charts |
| **Training Curves** | Load and visualize reward, loss, entropy from training CSV logs with smoothing slider |

**Connecting Arduino from the Dashboard:**

1. Open **Camera Dashboard** in the sidebar
2. Under **Arduino Controller** → select your COM port (auto-detected via `serial.tools.list_ports`)
3. Click **Connect** — status badge turns 🟢
4. Click **Play Cameras** — RYG packets (`<S0,...,S7>\n`) are sent on every phase change + 1-second heartbeat

---

### Mode D — Run Tests

```bash
pytest                                      # all tests
pytest tests/test_state_consistency.py     # observation vector consistency
pytest -m unit                              # unit tests only
pytest -m "not slow and not serial"         # exclude hardware-dependent tests
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `src/traffic_env/config.py`. No magic numbers scattered across files.

### Adjust Reward Weights

```python
from src.traffic_env.config import RewardConfig

# Recommended weights for vision deployment
# (disable noisy speed signal, reduce throughput weight)
vision_config = RewardConfig(weights={
    "queue":              -1.0,
    "pressure":           -0.5,
    "throughput":         +0.5,   # reduced: proxy less reliable than SUMO
    "switch_penalty":     -0.1,
    "low_speed_penalty":   0.0,   # disabled: pixel-displacement too noisy
    "waiting_time":       -0.3,
})
```

### Adjust Simulation Parameters

```python
from src.traffic_env.config import build_default_config

cfg = build_default_config(
    sumo_cfg_path  = "sumo_configs/training/sumo_config.sumocfg",
    step_length    = 5,      # seconds per simulation step
    max_steps      = 1080,   # episode length (~90 min simulated)
    min_green_time = 10,     # minimum green phase (seconds)
    max_green_time = 60,     # maximum green phase (seconds)
    yellow_time    = 3,      # yellow transition (seconds)
    queue_cap      = 50.0,   # normalization cap for queue length
    speed_cap      = 15.0,   # normalization cap for speed (m/s)
)
```

### Key Configuration Files

| File | Purpose |
|---|---|
| `configs/state_config.json` | Obs schema, lane ROI polygons, TLS ID mapping, normalization caps |
| `configs/camera_config.json` | 8 cameras: source paths, directions, phase groups, TLS assignment |
| `configs/serial.json` | Arduino port, baud rate, yellow duration, red gap |
| `configs/tele.json` | Telegram bot token + chat ID *(gitignored — copy from `tele_example.json`)* |
| `src/traffic_env/config.py` | All training hyperparameters (single source of truth) |

---

## 📁 Project Structure

```
Sim2Real-MAPPO-Traffic/
├── configs/                         # Runtime configuration files
│   ├── state_config.json            # Observation schema & ROI polygons
│   ├── camera_config.json           # 8-camera topology
│   └── tele_example.json            # Telegram alert template
├── src/
│   ├── core/
│   │   ├── system_orchestrator.py   # Primary production entry point
│   │   ├── orchestrator_v2.py       # 3-thread refactored orchestrator
│   │   └── policy_loader.py         # Robust checkpoint loader
│   ├── vision/
│   │   ├── state_extractor.py       # Detections → observation vector
│   │   ├── multi_camera.py          # 8-thread camera manager
│   │   ├── detector.py              # YOLOv11 + ByteTrack worker
│   │   └── event_detector.py        # Accident confirmation + alert
│   ├── traffic_env/
│   │   ├── config.py                # All hyperparameters (single source)
│   │   ├── envs/multi_agent.py      # PettingZoo-style MAPPO environment
│   │   └── components/
│   │       ├── observations.py      # SUMO obs builder + vision adapter
│   │       └── rewards.py           # PRESSLIGHT reward calculator
│   ├── adapters/vision_to_state.py  # Bridge: snapshot → obs packet
│   ├── buffer/vision_buffer.py      # EMA temporal smoothing
│   └── utils/serial_bridge.py       # Arduino serial communication
├── experiment/
│   ├── runners/
│   │   ├── train_ppo.py             # MAPPO training entry point
│   │   ├── test_ppo.py              # Policy evaluation with SUMO GUI
│   │   ├── evaluate.py              # Headless multi-seed evaluation
│   │   └── compare_traffic_metrics.py  # Baseline vs MAPPO comparison
│   ├── baselines/
│   │   ├── max_pressure.py          # Max Pressure controller
│   │   └── sotl.py                  # Self-Organising Traffic Lights
│   ├── ablation/
│   │   └── reward_ablation.py       # Reward component ablation study
│   └── plots/
│       ├── generate_paper_figures.py   # IEEE-ready PDF figures
│       ├── plot_training_results.py
│       └── parse_tripinfo.py
├── scripts/
│   ├── dashboard.py                 # Streamlit real-time dashboard
│   ├── generate_sumo_scenarios.py   # Generate low/high/asymmetric scenarios
│   ├── check_obs_match.py           # Verify obs vector consistency
│   └── run_real_deployment.py       # Convenience deployment launcher
├── notebooks/
│   └── 01_training_analysis.ipynb   # Training curves & baseline comparison
├── results/
│   ├── paper1_mappo/
│   │   ├── eval_tables/             # metrics_comparison.csv
│   │   └── training_curves/         # TensorBoard screenshots
│   ├── paper2_vision/               # Vision pipeline results (future)
│   └── paper3_sim2real/             # Sim-to-real gap analysis (future)
├── figures/                         # Publication-ready PDF figures (LaTeX)
│   ├── training_curve.pdf           # Fig: reward convergence
│   ├── training_losses.pdf          # Fig: policy/value loss
│   └── reward_distribution.pdf      # Fig: early vs late reward distribution
├── models/
│   ├── yolo/yolov11.pt              # Custom YOLOv11n-seg (5-class)
│   └── mappo/<run_id>/
│       ├── best_model.pt            # Best checkpoint by mean reward
│       └── run_config.json          # Hyperparameter snapshot
├── data/
│   ├── raw_video/                   # 8 camera test videos
│   ├── dataset/                     # YOLO training dataset (gitignored)
│   └── processed/                   # Processed data (gitignored)
├── docs/
│   ├── state.md                     # Observation vector design
│   ├── reward.md                    # Reward function design
│   └── architecture_diagram.png     # System architecture
├── hardware/
│   ├── mega_controller/
│   │   └── mega_controller.ino      # Arduino traffic light firmware
│   └── schematic_traffic_system.png
├── sumo_configs/
│   ├── training/                    # Training network & routes
│   └── evaluation/
│       ├── medium/                  # Medium traffic (default)
│       ├── low/                     # Light traffic scenario
│       ├── high/                    # Heavy traffic scenario
│       └── asymmetric/              # Asymmetric flow scenario
├── tests/                           # pytest test suite
├── logs/                            # Runtime logs (gitignored)
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test markers & config
├── LICENSE                          # MIT License
└── README.md
```

---

## 📊 Results

> Evaluation on a 2-intersection SUMO network, medium traffic scenario, 10 episodes, seed 42.

| Metric | Fixed-Time | Max Pressure | SOTL | **MAPPO (ours)** |
|---|---|---|---|---|
| Mean Queue (veh) | 12.4 | 8.7 | 9.1 | **6.3** |
| Mean Waiting (s) | 48.2 | 31.5 | 34.8 | **22.7** |
| Throughput (veh) | 843 | 971 | 958 | **1,104** |
| Mean Speed (m/s) | 4.1 | 5.8 | 5.5 | **7.2** |

---

## 📐 Reproducibility

All experiments use a fixed seed (`--seed 42`). Key hyperparameters are snapshotted in `run_config.json` inside each checkpoint directory. To reproduce the main results:

```bash
# 1. Generate evaluation scenarios
python scripts/generate_sumo_scenarios.py

# 2. Multi-seed evaluation (5 seeds, reports mean ± std)
python experiment/runners/train_ppo.py \
  --mode multiseed_eval \
  --sumo-cfg   sumo_configs/evaluation/medium/sumo_config.sumocfg \
  --checkpoint models/mappo/20260418_215140/best_model.pt

# 3. Reward component ablation
python experiment/ablation/reward_ablation.py \
  --checkpoint models/mappo/20260418_215140/best_model.pt \
  --sumo-cfg   sumo_configs/evaluation/medium/sumo_config.sumocfg

# 4. Regenerate paper figures
python experiment/plots/generate_paper_figures.py
```

---

## 🔖 Citation

If you use this work in your research, please cite:

```bibtex
@software{traffic_guard_ai_2026,
  author    = {Nguyen, Trong Hia},
  title     = {{Traffic Guard AI: Sim2Real Multi-Agent Traffic Signal Control}},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic},
  license   = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for full citation metadata.

---

## 📚 References & Acknowledgments

**[1] PressLight**
> Hua Wei, Guanjie Zheng, Vikash Gayah, Zhenhui Li.
> *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.*
> **KDD 2019.** https://doi.org/10.1145/3292500.3330949

**[2] CoLight**
> Hua Wei, Nan Xu, Huichu Zhang, Guanjie Zheng, Xinshi Zang, Chacha Chen, Weinan Zhang, Yanmin Zhu, Kai Xu, Zhenhui Li.
> *CoLight: Learning Network-level Cooperation for Traffic Signal Control.*
> **CIKM 2019.** https://doi.org/10.1145/3357384.3357902

**[3] MAPPO**
> Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu.
> *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.*
> **NeurIPS 2022.** https://arxiv.org/abs/2103.01955

**[4] SUMO**
> Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, et al.
> *Microscopic Traffic Simulation using SUMO.*
> **IEEE ITSC 2018.** https://doi.org/10.1109/ITSC.2018.8569938

**[5] YOLOv11**
> Glenn Jocher et al.
> *Ultralytics YOLO11.* 2024.
> https://github.com/ultralytics/ultralytics

**[6] ByteTrack**
> Yifu Zhang, Peize Sun, Yi Jiang, et al.
> *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.*
> **ECCV 2022.** https://arxiv.org/abs/2110.06864

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for smarter cities

⭐ **Star this repo if you find it useful!**

</div>
