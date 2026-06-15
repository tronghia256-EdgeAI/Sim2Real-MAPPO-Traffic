# Papers — code & artifact map

This repository is **one shared codebase** that produces **three papers**. The source
library (`src/`) is shared infrastructure; this page maps every part to the paper it
serves so you can tell "phần nào của paper nào" at a glance.

| # | Paper | Status | Owned `src/` | Experiments / scripts | Configs | Results | Docs |
|---|---|---|---|---|---|---|---|
| **1** | **TSC** — camera-observable MAPPO traffic-signal control | **Active** (training campaign in progress; eval→tables→figures pipeline turnkey) | `traffic_env/**`, `core/policy_loader.py`\* | `experiment/**`, `scripts/{generate_networks,generate_demand,generate_ood_scenarios,run_pilot,parallel_launcher,build_paper_artifacts,dump_confusion_matrix,check_obs_match}.py` | `configs/state_config.json`\*, `configs/noise_config.json` | `results/paper1_mappo/`, `figures/`, `models/mappo/`, `logs/` | `docs/paper1_tsc/` |
| **2** | **Perception** — YOLOv11 + ByteTrack: accident detection + vehicle recognition | **Scope TBD** | `vision/**` (`detector`, `event_detector`, `tracker_parser`, `multi_camera`, `state_extractor`\*) | `scripts/dump_confusion_matrix.py`\* | `configs/camera_config.json` | `results/paper2_perception/` | `docs/papers/paper2_perception.md` |
| **3** | **System** — full real-time deployment (8 cams → policy → Arduino) | **Scope TBD** | `core/**` (orchestrators, `inference_engine`, `camera_manager_v2`, `metrics_collector`, `config_validator`), `adapters/`, `buffer/`, `services/`, `utils/`, `visualization/` | `scripts/{run_real_deployment,dashboard}.py` | `configs/{camera_config,serial,tele}.json` | `results/paper3_system/` | `docs/papers/paper3_system.md` |

\* = shared across papers (see below).

## Shared infrastructure (no single owner)
- `src/traffic_env/config.py` — env/obs/reward config dataclasses (P1 training truth).
- `configs/state_config.json` — observation schema; **bridges P1 training ↔ P2/P3 deployment** (must stay in sync with `config.py`).
- `src/vision/state_extractor.py` — P2 detections → P1-aligned 26-dim obs; consumed by P3 at deployment.
- `src/core/policy_loader.py` — loads the P1 MAPPO actor; consumed by P3 orchestrators.
- `tests/`, `requirements.txt`, `pytest.ini`, `NOTES.md` — repo-wide.

## Why the code is not physically split per paper
The three papers share the perception→obs→policy→actuation chain, so the same modules
(`state_extractor`, `policy_loader`, `config`/`state_config`) are reused across them.
Physically separating `src/` would duplicate or break these. Instead `src/` stays a shared
library and ownership is documented here. Per-paper *artifacts* (docs, results) are grouped.

## Per-paper detail
- [Paper 1 — TSC](paper1_tsc.md)
- [Paper 2 — Perception](paper2_perception.md)
- [Paper 3 — System](paper3_system.md)

> Naming note: `results/paper1_mappo/` keeps its name (hardcoded in 10 scripts + written by
> the live campaign). "mappo" = the MAPPO method of Paper 1 (TSC). A rename to `paper1_tsc`
> is a deferred optional step (see the restructure plan) to run after the campaign finishes.
