# State Space Review — Train/Deploy Gap Analysis

**System:** MAPPO Traffic Signal Control  
**Reviewed:** 2026-06-08; updated 2026-06-12 for **obs schema 1.1.0** (S1/S2/S3 fixes — see §12)  
**Reviewer scope:** Every feature in the 26-dim per-agent observation vector, cross-checked between
`src/traffic_env/components/observations.py` (SUMO/training) and
`src/vision/state_extractor.py` (YOLO+ByteTrack/deployment).

> **LEGACY WARNING:** checkpoint `20260418_215140` was trained on the
> pre-1.1.0 schema (per-lane slots, unsigned pressure) and is invalid for
> paper results. All results require retraining with `env_cfg.version >= 1.1.0`.

---

## 1. Observation Vector Layout (schema 1.1.0)

```
Per agent (26 dims):
  [0..4]   approach_0 → effective_queue_norm, occupancy_norm, avg_speed_norm,
                        motorbike_share, heavy_vehicle_share
  [5..9]   approach_1 → same 5 features
  [10..14] approach_2 → same 5 features
  [15..19] approach_3 → same 5 features
  [20]     phase_one_hot_0
  [21]     phase_one_hot_1
  [22]     phase_one_hot_2
  [23]     phase_one_hot_3
  [24]     green_timer_norm
  [25]     pressure_norm

Global critic state (52 dims): concat(agent_0_obs, agent_1_obs)
```

Agents: `tls_0` (intersection J0, approaches 0–3), `tls_1` (intersection J2, approaches 0–3)

**S1 fix (schema 1.1.0):** each 5-feature slot is one **APPROACH** (all lanes
of one incoming edge, aggregated), not one raw lane. Pre-1.1.0 the builder
sliced `getControlledLanes()[:4]`, leaving 4 of the 8 controlled lanes — two
entire approaches — invisible to the policy.

**Aggregation rules** (`ObservationBuilder._aggregate_approach_metrics`):
- `effective_queue_norm`, `occupancy_norm` — length-ratio features → **unweighted mean** across the approach's lanes
- `avg_speed_norm`, `motorbike_share`, `heavy_vehicle_share` — per-vehicle statistics → **vehicle-count-weighted mean**, so empty lanes do not dilute them

Lanes are grouped by parent edge (`lane_id.rsplit("_", 1)[0]`) in
`getControlledLanes()` order. In deployment each camera ROI already covers one
approach, so slot k = ROI k — **ROI order must match the training approach
order**; verify with `python scripts/check_obs_match.py --sumo`.

---

## 2. System Parameters That Affect All Features

| Parameter | Configured Value | File |
|---|---|---|
| `fps` | 10 Hz | `configs/state_config.json` |
| `speed_cap` | 15.0 m/s | `state_config.json`, `config.py` |
| `max_green_time` | 60 s | `state_config.json`, `config.py` |
| `px_per_meter` | 20.0 px/m | `state_config.json` |
| `halt_threshold` | 0.1 m/s | `state_extractor.py`, `observations.py` |
| `max_lanes_per_tls` | 4 | `state_config.json`, `config.py` |
| SUMO step length | 5 s | `config.py:SimConfig.step_length` |
| EMA alpha (VisionBuffer) | 0.6 | `state_config.json`, `vision_buffer.py` |
| Vehicle class IDs (YOLO) | 1=bus, 2=car, 3=motorcycle, 4=truck | `state_config.json` |

---

## 3. Per-Feature Deep Review

> **Note (schema 1.1.0):** the SUMO methods below describe the **per-lane**
> computation in `_lane_metrics()`. Since S1, per-lane values are aggregated
> into one approach-level value per obs slot using the rules in §1
> (queue/occupancy: unweighted mean; speed/shares: count-weighted mean). The
> camera side is unchanged — each ROI was always approach-level.

---

### Feature 1: `effective_queue_norm`
**Vector position:** indices 0, 5, 10, 15 (one per approach)

#### SUMO (training)
- **Source:** `observations.py:_lane_metrics()`
- **Method:**
  1. `lane_length = lane.getLength(lane_id)` — exact road length from network file (metres)
  2. For each vehicle on lane: `speed = vehicle.getSpeed(vehicle_id)` (m/s, exact simulation value)
  3. `halted = speed <= 0.1 m/s` (hardcoded threshold at line 572)
  4. `halted_effective_length += vehicle.getLength(vehicle_id)` — exact per-vehicle length
  5. `effective_queue_norm = halted_effective_length / lane_length`
- **Unit:** dimensionless ratio ∈ [0, 1]
- **Update freq:** every 5 s (SimConfig.step_length)
- **Noise:** none — deterministic simulation

#### Camera (deployment)
- **Source:** `state_extractor.py:_lane_metrics_from_tracks()`
- **Method:**
  1. `lane_length_px` — from `LaneROI.lane_length_px` (manually configured per camera)
  2. For each track in lane ROI: `speed_m_s = speed_px_s / px_per_meter`
  3. `halted = speed_m_s <= stop_speed_m_s (0.1 m/s)`
  4. `halted_length_px += CLASS_LENGTH_M[cls_id] * px_per_meter` (derived at init)
  5. `effective_queue_norm = halted_length_px / lane_length_px`
- **Unit:** dimensionless ratio ∈ [0, 1]
- **Update freq:** every 0.1 s (10 fps), then EMA-smoothed in VisionBuffer
- **Noise:** ByteTrack speed estimates, px_per_meter calibration error, occlusion

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| `px_per_meter` calibration error | Scales everything. ±10% error in px/m → ±10% error in speed/length | Yes — measure at install |
| ByteTrack velocity noise (slow vehicles) | High. At 0.5 m/s a 1-pixel jitter = ±0.5 m/s at 10fps/20px_per_m | Partially — EMA helps |
| Occlusion (vehicles hidden behind others) | Stopped vehicles behind queue front are invisible | No — inherent limit |
| Camera perspective (near/far scaling) | Vehicles at different depths appear different sizes | Partially — calibration |
| New track initialisation (v=0 on first frame) | Creates false "stopped" for 1–3 frames | Yes — min_track_age |
| YOLO miss-detection (false negative) | Queue underestimated when vehicles not detected | No — model quality |
| Lane ROI polygon miscalibration | Wrong `lane_length_px` shifts normalization | Yes — measure at install |

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL after fix to derive pixel lengths from `CLASS_LENGTH_M × px_per_meter`
- **Absolute values:** APPROXIMATE — depends on calibration accuracy
- **Risk level:** MEDIUM
- **Dominant risk:** `px_per_meter` calibration and ByteTrack speed noise on slow/stopped vehicles

---

### Feature 2: `occupancy_norm`
**Vector position:** indices 1, 6, 11, 16

#### SUMO (training)
- **Source:** `observations.py:_lane_metrics()` line 534
- **Method:** `lane.getLastStepOccupancy(lane_id) / 100.0`
- **SUMO definition:** fraction of lane LENGTH covered by vehicle lengths (1D linear density), averaged over last time step
  - Formula: `Σ(vehicle_length_i) / lane_length × 100`
- **Unit:** ∈ [0, 1]
- **Update freq:** every 5 s

#### Camera (deployment)
- **Source:** `state_extractor.py:_lane_metrics_from_tracks()` line 436
- **Method:** `Σ(bbox_width × bbox_height) / lane_roi_area_px`
- **Definition:** fraction of 2D ROI pixel area covered by 2D bounding boxes
- **Unit:** ∈ [0, 1]
- **Update freq:** every 0.1 s

#### Fundamental Measurement Difference
SUMO measures **1D linear occupancy** (vehicle lengths along lane).  
Camera measures **2D area occupancy** (bounding box pixels in ROI).

They are correlated (more vehicles = higher both) but not equal:
- Camera occupancy depends on camera angle and vehicle-to-camera distance
- Vehicles farther from the camera appear smaller → lower bbox area → lower occupancy
- Vehicles in perspective view appear longer horizontally than in top-down view
- Standard SUMO lanes (e.g., 3.2 m wide) mapped to a camera ROI polygon will have angle distortion

| Source | SUMO | Camera |
|---|---|---|
| Denominator | lane_length (metres) | ROI area (pixels²) |
| Numerator | Σ vehicle_length_m | Σ bbox_area_px |
| Projection | 1D along road axis | 2D image projection |

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| 3D perspective distortion | Medium — near vehicles inflate area | Partially — homography |
| YOLO bbox over-/under-size | Low–medium | No |
| Overlapping bboxes (double-counting area) | Low | No |

#### Deployment Mismatch Assessment
- **Formula:** DIFFERENT — 1D length fraction vs 2D area fraction
- **Correlation:** High (same monotone direction) but systematic offset expected
- **Risk level:** MEDIUM — values will be correlated but not equal under heavy or sparse load
- **Mitigation:** Accept systematic offset; policy learns relative ordering, not absolute values

---

### Feature 3: `avg_speed_norm`
**Vector position:** indices 2, 7, 12, 17

#### SUMO (training)
- **Source:** `observations.py:_lane_metrics()` line 553–556
- **Method:** `mean(vehicle.getSpeed(v) for v in vehicle_ids) / speed_cap`
- **Speed source:** SUMO's own kinematic model (exact position differentiation, no noise)
- **Unit:** ∈ [0, 1], cap = 15.0 m/s
- **Update freq:** every 5 s

#### Camera (deployment)
- **Source:** `state_extractor.py:_lane_metrics_from_tracks()` line 406
- **Method:** `mean(speed_px_s / px_per_meter for t in tracks) / speed_cap`
- **Speed source:** ByteTrack optical flow estimate = `Δposition / Δtime`
  - `speed_px_s` = `||velocity||` (px/frame) × fps
- **Unit:** ∈ [0, 1], cap = 15.0 m/s
- **Update freq:** every 0.1 s

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| ByteTrack velocity noise at low speeds | High (±1 m/s for vehicles moving at 0–3 m/s) | Partially — EMA |
| ID switch resets velocity to 0 | Medium — occurs in dense traffic | No |
| `px_per_meter` calibration error | Proportional — scales all speeds | Yes — measure at install |
| Frame drop (variable fps) | Medium — speed computed as Δx / Δt, fps assumed constant | Yes — use timestamps |
| New track (1st frame, v=0) | Low — affects 1 frame per detection | Yes — min_track_age |
| Occlusion breaking tracks | Medium — tracked vehicle disappears, new detection has v=0 | Partially |

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL (mean/speed_cap)
- **Absolute values:** APPROXIMATE — ByteTrack speed noise is significant for slow traffic
- **Risk level:** MEDIUM
- **Note:** Low-speed regime (congestion, queues) has worst relative error. At near-zero speed, a 0.5 px jitter at 10fps/20px_m = 0.25 m/s noise = 1.7% of speed_cap — tolerable after EMA.

---

### Feature 4: `motorbike_share`
**Vector position:** indices 3, 8, 13, 18

#### SUMO (training)
- **Source:** `observations.py:_classify_vehicle_type()` line 497–516
- **Method:** string match on SUMO vehicle type ID, or length ≤ 2.2 m
  - Keyword match: `motor/bike/moto/scooter/twowheel`
- `motorbike_share = motorbike_count / vehicle_count`
- **Unit:** ∈ [0, 1]

#### Camera (deployment)
- **Source:** `state_extractor.py:_lane_metrics_from_tracks()`
- **Method:** YOLO class 3 (motorcycle) count / total count
- **Unit:** ∈ [0, 1]

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| YOLO miss-detection (small vehicle, distant) | Medium — motorcycles are small | No — model quality |
| Occlusion (motorcycle behind bus/truck) | Medium–High in Vietnamese traffic | No |
| Scooter/bicycle classification overlap | Low–Medium | Improve YOLO training data |
| SUMO scenario uses mismatched type names | Low — controllable in training | Yes — fix SUMO routes |

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL
- **Classification method:** DIFFERENT (SUMO type-string vs YOLO class ID) — same intent
- **Risk level:** LOW–MEDIUM
- **Note:** Vietnamese traffic has high motorcycle fraction (60–80%). Small absolute errors have large relative impact on this feature.

---

### Feature 5: `heavy_vehicle_share`
**Vector position:** indices 4, 9, 14, 19

#### SUMO (training)
- **Source:** `observations.py:_classify_vehicle_type()`
- **Method:** type contains `bus/coach` → bus; type contains `truck/lorry/heavy` → truck
  - OR length > 9.0 m → bus; 5.5 < length ≤ 9.0 m → truck
- `heavy_share = (bus+truck)_count / vehicle_count`
- **Unit:** ∈ [0, 1]

#### Camera (deployment)
- **Source:** `state_extractor.py`
- **Method:** YOLO class 1 (bus) + class 4 (truck) count / total count
- **Unit:** ∈ [0, 1]

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| Van/minibus borderline (SUMO: car; YOLO: bus?) | Medium | Align SUMO type definitions |
| YOLO confusion bus/truck at distance | Low | No |
| Length-based fallback inconsistency in SUMO | Low | Fix SUMO vehicle types explicitly |

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL
- **Classification method:** DIFFERENT (length-based vs class ID) — borderline vehicle types differ
- **Risk level:** LOW

---

### Feature 6: `phase_one_hot_0..3`
**Vector position:** indices 20–23

#### SUMO (training)
- **Source:** `observations.py:_phase_one_hot(tls_id)` line 461
- **Method:** `phase = trafficlight.getPhase(tls_id)` → integer 0..3
- **SUMO phase cycle:**
  - Phase 0: Group A green (`GGrr…`)
  - Phase 1: Yellow transition A→B
  - Phase 2: Group B green (`rrGG…`)
  - Phase 3: Yellow transition B→A
- **Encoding:** `vec[phase] = 1.0`
  - Group A green → `[1, 0, 0, 0]`
  - Yellow A→B   → `[0, 1, 0, 0]`
  - Group B green → `[0, 0, 1, 0]`
  - Yellow B→A   → `[0, 0, 0, 1]`
- **Update freq:** every 5 s (but yellow is 3 s within a step)

#### Camera (deployment) — PATCHED (Patch 1, see §11)
- **Source:** `state_extractor.py:_phase_one_hot(action: int)`
- **Method:** MAPPO action ∈ {0, 1} mapped to SUMO phase index via class-level
  `_ACTION_TO_PHASE = {0: 0, 1: 2}` before one-hot encoding
- **Encoding (current, correct):**
  - Action 0 (Group A green) → SUMO phase 0 → `[1, 0, 0, 0]` ✓
  - Action 1 (Group B green) → SUMO phase 2 → `[0, 0, 1, 0]` ✓

#### Historical Train-Deploy Mismatch — PHASE ENCODING (Bug B3, FIXED)
Before Patch 1, `action=1` was one-hot-encoded directly at index 1:
```
Training:    Group B green = [0, 0, 1, 0]   (SUMO phase index = 2)
Deployment:  Group B green = [0, 1, 0, 0]   (raw action index = 1)  ← was WRONG
```

The policy learned `[0,1,0,0]` as the yellow A→B transition, so every Group B
step in deployment was misconditioned — the policy "thought" it was in a
transient yellow while Group B was serving traffic. This was rated HIGH risk
because `phase_one_hot` drives the switch/hold decision.

**Regression guard:** if any deployment log shows `[0,1,0,0]` for Group B
green, the patch is missing. `check_obs_match.py` Check 2 covers this.

**Yellow phases during training:** `yellow_time=3s < step_length=5s`, so yellow transitions are brief but *do* appear in the training obs as `[0,1,0,0]` / `[0,0,0,1]` — the policy learned they are transient, which is why the old mis-encoding was so damaging.

---

### Feature 7: `green_timer_norm`
**Vector position:** index 24

#### SUMO (training)
- **Source:** `multi_agent.py` `green_timers` dict
- **Method:** timer starts at 0 on phase switch; incremented by `step_length` (5 s) each step; capped at `max_green_time` (60 s)
- `green_timer_norm = green_timer / max_green_time`
- **Unit:** ∈ [0, 1]
- **Update freq:** every 5 s

#### Camera (deployment)
- **Source:** orchestrator `green_timers` dict
- **Method:** same logic — reset on phase switch, incremented by elapsed real time, capped at 60 s
- `green_timer_norm = green_timer / max_green_time`
- **Unit:** ∈ [0, 1]

#### Noise Sources
| Source | Magnitude | Mitigable? |
|---|---|---|
| Wall-clock drift vs simulation time | Negligible — real-time control | No |
| System startup: timer initialises to 0 for both TLS | Harmless — same as SUMO episode reset | — |
| `max_green_time` default (90 s in StateExtractor) vs training (60 s) | HIGH if orchestrator does not override | Yes — load from state_config.json |

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL
- **Risk level:** LOW (if orchestrator loads `max_green_time` from config)
- **Risk level:** HIGH (if orchestrator uses `StateExtractor` default of 90 s)
- **Verified:** `state_config.json` has `max_green_time: 60`. Orchestrators must pass this explicitly to `StateExtractor`.

---

### Feature 8: `pressure_norm` *(SIGNED — S2 fix, schema 1.1.0)*
**Vector position:** index 25

#### SUMO (training)
- **Source:** `observations.py:_tls_features()` → `_tls_pressure_proxy()`
- **Method:**
  - `q̄_A = mean(effective_queue_norm(lane) for lane in group_a)`
  - `q̄_B = mean(effective_queue_norm(lane) for lane in group_b)`
  - `pressure_norm = clip(0.5 × (1 + q̄_A − q̄_B), 0, 1)`
- Groups from `manual_lane_groups` in config
- **Semantics:** 0.5 = balanced; > 0.5 = group A (phase-0 lanes) more queued;
  < 0.5 = group B more queued. Group **means** make the value robust to
  unequal group sizes.
- **Unit:** ∈ [0, 1]
- **Update freq:** every 5 s

#### Camera (deployment)
- **Source:** `state_extractor.py:_tls_pressure_proxy()`
- **Method:** identical formula
- Groups from `lane_groups` parameter to `StateExtractor`
- **Unit:** ∈ [0, 1]
- **Update freq:** every 0.1 s

#### Pre-1.1.0 form (Bug S2, FIXED 2026-06-12)
The old formula was `|Σq_A − Σq_B| / max_lanes_per_tls` — **unsigned**, so the
policy could see *that* the groups were imbalanced but not *which* group was
congested, making the feature nearly useless for the switch decision. Both
pipelines were changed together to the signed group-mean form.

#### Deployment Mismatch Assessment
- **Formula:** IDENTICAL (verified — `check_obs_match.py` Check 2 shows 100% match)
- **Risk level:** LOW — inherits whatever noise exists in `effective_queue_norm`
- **Critical dependency:** Lane group assignments must match between SUMO training config (`DEFAULT_MANUAL_LANE_GROUPS`) and production `StateExtractor(lane_groups=...)`. If they differ, `pressure_norm` is wrong. **Bug S3** (J2 group B listed `-E1_*` — the approach to J0 — instead of `E1_*`) corrupted exactly this feature for J2 until 2026-06-12; `check_obs_match.py` CHECK 6 now guards the lane-group topology.

---

## 4. Rejected Features

These features exist in `configs/state_config.json:features` or the SUMO pipeline but are
**excluded from the actual 26-dim obs vector**. If re-introduced they must pass the camera
reliability bar defined below.

### 4.1 `waiting_time`
- **SUMO source:** `vehicle.getWaitingTime(vehicle_id)` — simulator's exact accumulated waiting time
- **Camera equivalent:** none reliable. Would require tracking the same track ID across hundreds of frames without ID switch. ByteTrack loses IDs in occluded/dense scenarios.
- **Verdict: REJECT.** Cannot be measured reliably from camera. `StateExtractor` accumulates per-track stop memory but it resets on track loss. Not in active obs vector — do not add.

### 4.2 `throughput` (vehicles exiting per second)
- **SUMO source:** vehicles teleported off edge / exiting the lane polygon per time step
- **Camera equivalent:** count tracks whose centroid crosses the ROI boundary (exit event). Extremely noise-prone — occlusion causes track loss that looks identical to a true exit.
- **Verdict: REJECT.** False-positive exit rate from tracking loss would inject large noise. Not in active obs vector. Documented in `state_config.json:features[5]` but NOT implemented anywhere in code — confirmed documentation drift.

### 4.3 `queue_length` (stopped vehicle count)
- **SUMO source:** count of vehicles with speed ≤ 0.1 m/s
- **Camera equivalent:** `queue_count` computed inside `StateExtractor` but not in the 5 policy features
- **Why excluded:** Raw count is a redundant and noisier version of `effective_queue_norm`. The normalised queue length already captures queue severity. Adding count adds multicollinearity.
- **Verdict: REJECT.** Not in active obs vector. Keep as diagnostic only.

### 4.4 `vehicle_count_norm`
- **SUMO source:** `vehicle_count / vehicle_cap`
- **Camera equivalent:** track count inside ROI
- **Why excluded:** Highly occlusion-sensitive in dense Vietnamese traffic. Close to `occupancy_norm` in information content for uncongested lanes. Adds noise without adding signal for the RL task.
- **Verdict: REJECT.** Not in active obs vector.

### 4.5 `co2_mg_per_s`, `fuel_ml_per_s`
- **SUMO source:** `vehicle.getCO2Emission()`, `vehicle.getFuelConsumption()` — SUMO internal emission models
- **Camera equivalent:** none. These are physics/chemistry models with no camera equivalent.
- **Verdict: REJECT.** SUMO-only. Computed in `_lane_metrics()` and passed in `LaneMetrics` dict but correctly zeroed out in `VisionLaneMetrics.compute()`. Do not promote to obs features.

### 4.6 `accident_flag`
- **SUMO source:** not natively modelled; would require custom SUMO scripting
- **Camera equivalent:** YOLO class 0 detections → confirmed by `AccidentEventDetector` (3-frame rule)
- **Why excluded:** Accident detection is a separate subsystem (`AccidentEventDetector`) that triggers Telegram alerts. Including it in the RL obs space would require training on scenarios with accident injection, which the SUMO training environment does not model.
- **Verdict: REJECT for RL obs.** Keep as external event signal feeding into safe-mode behaviour only.

---

## 5. Temporal Alignment Problem

| | SUMO Training | Camera Deployment |
|---|---|---|
| Obs update frequency | 5 s (step_length) | 0.1 s (10 fps) |
| Policy action frequency | 5 s | 5 s (control cycle) |
| Smoothing | None (discrete steps) | EMA alpha=0.6 (VisionBuffer) |

The policy was trained on observations that change at most every 5 seconds.  
In deployment, the policy receives EMA-smoothed observations at 10 Hz.

**Consequence:**  
The effective obs at time t in deployment is `0.6 × obs_t + 0.4 × ema_{t-1}`, where `ema` integrates ~1–2 s of history at alpha=0.6. This is *shorter* than the 5 s SUMO step but provides some smoothing. The policy is not distracted by individual frame jitter.

**Residual risk:**  
If a sudden density change occurs (e.g., a queue dissolves in one green cycle), SUMO training would see a clean step-change at the next obs update. The camera + EMA path would show a gradual slope over ~0.5–1.5 s. This is unlikely to cause policy instability but may cause slightly delayed phase-switching decisions.

**Mitigation already in place:** `VisionBuffer.clip_obs=True` prevents out-of-range values. `ema_alpha=0.6` gives reasonable tracking lag.

---

## 6. Summary Table

| State Feature | SUMO Source | Camera Source | Match | Risk | Fix |
|---|---|---|---|---|---|
| `effective_queue_norm` | `lane.getLastStepVehicleIDs` → speed ≤ 0.1 → Σ(vehicle_length) / lane_length | ByteTrack speed ≤ 0.1 m/s → Σ(CLASS_LENGTH_M×px_m) / lane_length_px | **PARTIAL** — formula identical; absolute value depends on px_per_meter calibration | MEDIUM | Measure px_per_meter per camera at install; confirm `lane_length_px` matches real-world distance |
| `occupancy_norm` | `lane.getLastStepOccupancy() / 100` (1D length fraction) | `Σ(bbox_area) / ROI_area_px` (2D area fraction) | **APPROXIMATE** — correlated but different measurement basis | MEDIUM | Accept systematic offset; consider homography correction for exact match |
| `avg_speed_norm` | `vehicle.getSpeed()` per vehicle / speed_cap | ByteTrack `‖velocity‖ × fps / px_per_meter` / speed_cap | **APPROXIMATE** — same formula; ByteTrack noise at low speeds | MEDIUM | Tune EMA alpha; enforce min_track_age ≥ 3 |
| `motorbike_share` | SUMO type string matching or length ≤ 2.2 m | YOLO class 3 count / total | **APPROXIMATE** — same intent, different classifier | LOW–MEDIUM | Ensure SUMO training scenarios have representative motorbike fraction |
| `heavy_vehicle_share` | SUMO type string or length > 5.5 m | YOLO class 1 + class 4 count / total | **APPROXIMATE** — borderline vans may differ | LOW | Define explicit SUMO type IDs for bus/truck to avoid length-based ambiguity |
| `phase_one_hot_0` | SUMO `getPhase()=0` → `[1,0,0,0]` | action=0 → `[1,0,0,0]` | **MATCH** | LOW | None |
| `phase_one_hot_1` | SUMO `getPhase()=1` (yellow A→B) → `[0,1,0,0]` | never produced in deployment (yellow handled by env/serial timing, not policy) | **MATCH** (fixed — Patch 1; was MISMATCH: action=1 encoded here) | LOW | Patch applied: `_ACTION_TO_PHASE = {0: 0, 1: 2}` |
| `phase_one_hot_2` | SUMO `getPhase()=2` (Group B green) → `[0,0,1,0]` | action=1 → `[0,0,1,0]` | **MATCH** (fixed — Patch 1) | LOW | None — regression-guarded by check_obs_match.py |
| `phase_one_hot_3` | SUMO `getPhase()=3` (yellow B→A) → `[0,0,0,1]` | never produced in deployment | LOW — yellow B→A is brief; policy rarely acts on it | LOW | None |
| `green_timer_norm` | `green_timer / max_green_time` (60 s), stepped every 5 s | `green_timer / max_green_time` (60 s), real-time clock | **MATCH** — formula identical; continuous vs discrete update | LOW | Orchestrators must load `max_green_time=60` from `state_config.json`, not legacy StateExtractor default (90 s — patched to 60) |
| `pressure_norm` | `clip(0.5·(1 + q̄_A − q̄_B), 0, 1)` signed group-mean (S2) | same formula from camera queue | **MATCH** (formula verified by check_obs_match.py Check 2 = 100%) | LOW | Lane group assignments in `StateExtractor(lane_groups=...)` must match SUMO `DEFAULT_MANUAL_LANE_GROUPS` (S3 J2 topology fixed; CHECK 6 guards) |

---

## 7. State Config Documentation Drift

`configs/state_config.json:features` documents a **different, older observation space** that is not implemented anywhere in the current codebase:

| JSON field | JSON says | Reality |
|---|---|---|
| `lane_features[0]` | `vehicle_count` | Not in obs vector |
| `lane_features[1]` | `queue_length` | Not in obs vector |
| `lane_features[2]` | `waiting_time` | Not in obs vector |
| `lane_features[3]` | `avg_speed` | Not the same as `avg_speed_norm` (index 2) |
| `lane_features[4]` | `occupancy` | Corresponds to `occupancy_norm` (index 1) — wrong index |
| `lane_features[5]` | `throughput` | Not implemented anywhere |
| `tls_features[0]` | `pressure` (tanh-normalised) | Actual normalisation is `/max_lanes`, not tanh |
| `tls_features[1]` | `phase_id` | Actual is 4-dim one-hot, not scalar |
| `tls_features[2]` | `time_in_phase` | Corresponds to `green_timer_norm` — wrong name and wrong max (90 s vs 60 s) |
| `tls_features[3]` | `accident_flag` | Not in obs vector |

**Action required:** Update `state_config.json:features` to reflect the actual implemented obs space.
The authoritative source is `DEFAULT_LANE_FEATURE_NAMES` and `DEFAULT_TLS_FEATURE_NAMES` in `config.py`.

---

## 8. Critical Bugs Identified and Fixed

The following bugs were found and patched during this review. See `scripts/check_obs_match.py` for regression testing.

### Bug 1 — Halt Threshold Divergence (FIXED)
- **File:** `src/traffic_env/components/observations.py:63`
- **Before:** `VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS = 0.5`
- **After:** `VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS = 0.1`
- **Impact:** Vehicles creeping at 0.1–0.5 m/s (common in slow-rolling queues) were classified as MOVING by VisionLaneMetrics but STOPPED by SUMO and StateExtractor. This caused systematic underestimation of `effective_queue_norm` in the VisionLaneMetrics path.

### Bug 2 — Vehicle Length Calibration Mismatch (FIXED)
- **File:** `src/vision/state_extractor.py:164`
- **Before:** `_CLASS_LENGTH_PX = {3: 10.0, 2: 30.0, 1: 80.0, 4: 55.0}` hardcoded (calibrated for ~6.7 px/m)
- **After:** `_CLASS_LENGTH_PX` computed at init as `CLASS_LENGTH_M[cls] × px_per_meter`
- **Impact:** At configured `px_per_meter=20`, car length was 30 px instead of correct 4.5×20=90 px. This caused `effective_queue_norm` to be ~3× too low in production vs training.

### Bug 3 — Phase Encoding (FIXED — Patch 1, §11)
- **File:** `src/vision/state_extractor.py:_phase_one_hot()`
- **Issue:** `action=1` (Group B green) → encoded as `[0,1,0,0]` instead of `[0,0,1,0]`
- **Fix applied:** class-level `_ACTION_TO_PHASE = {0: 0, 1: 2}` maps MAPPO action to SUMO phase index before one-hot encoding

### Bug S1 — Per-Lane Slicing (FIXED 2026-06-12 — schema 1.1.0, §12)
- **File:** `src/traffic_env/components/observations.py:build_local_obs`
- **Issue:** obs sliced `getControlledLanes()[:4]` — only 4 of 8 controlled lanes observed; 2 entire approaches invisible to the policy
- **Fix applied:** lanes grouped by parent edge into approaches and aggregated (§1)

### Bug S2 — Unsigned Pressure (FIXED 2026-06-12 — schema 1.1.0, §12)
- **Files:** `observations.py` + `state_extractor.py` (both pipelines together)
- **Issue:** `pressure_norm = |Σq_A − Σq_B| / max_lanes` — direction of imbalance unobservable
- **Fix applied:** signed group-mean form `clip(0.5·(1 + q̄_A − q̄_B), 0, 1)`

### Bug S3 — J2 Lane-Group Topology (FIXED 2026-06-12)
- **File:** `src/traffic_env/config.py:DEFAULT_MANUAL_LANE_GROUPS`
- **Issue:** J2 group B listed `-E1_*` (the approach to **J0**) instead of `E1_*`; corrupted J2 `pressure_norm` obs, pressure reward, and the MaxPressure baseline
- **Fix applied:** corrected lane IDs; `check_obs_match.py` CHECK 6 guards the topology

### Bug B1b — `stop_speed_m_s` Config Remnant (FIXED 2026-06-12)
- **File:** `configs/state_config.json`
- **Issue:** `stop_speed_m_s: 0.5` remained after Bug 1's code fix (training uses 0.1 m/s) → deployed queue estimation diverged from training
- **Fix applied:** set to 0.1 to match `HALT_SPEED_THRESHOLD_MPS`

---

## 9. Actionable Checklist

**Before deployment:**

- [x] Fix `_phase_one_hot()` in `state_extractor.py` to map action=1 → phase_index=2 → `[0,0,1,0]` *(Patch 1 — verify `_ACTION_TO_PHASE = {0: 0, 1: 2}` present)*
- [ ] Measure `px_per_meter` for every camera installation; update `state_config.json` per camera
- [ ] Verify `lane_length_px` in ROI definitions matches actual surveyed distance / px_per_meter
- [ ] Confirm all orchestrators pass `max_green_time=60` from `state_config.json`
- [ ] Confirm `lane_groups` passed to `StateExtractor` matches SUMO training `DEFAULT_MANUAL_LANE_GROUPS` topology (CHECK 6)
- [ ] Confirm camera ROI order matches the training approach order (`check_obs_match.py --sumo`)
- [ ] Run `python scripts/check_obs_match.py` and confirm all checks pass (1 structural, 2 assembly, 3 metric, 4 params, 6 lane-group topology)

**After deployment:**

- [ ] Log raw (pre-EMA) lane metrics per step for the first 24 h; compare distributions against SUMO episode stats
- [ ] Monitor `avg_speed_norm` — if consistently higher than training distribution, revisit `px_per_meter`
- [ ] Monitor `effective_queue_norm` during peak hours — if consistently lower than expected, check occlusion rate
- [ ] Add `effective_queue_norm` and `avg_speed_norm` distribution shift alerts to dashboard

---

## 10. Test Coverage

`scripts/check_obs_match.py` provides automated verification:

| Check | What it tests | Pass criterion |
|---|---|---|
| 1. Structural | obs_dim, feature names, halt_threshold, class IDs, normalization caps | 0 failures |
| 2. Assembly equivalence | Same metrics injected into both pipelines → identical obs | 100% match |
| 3. Metric computation | Same YOLO tracks → StateExtractor vs VisionLaneMetrics per-feature | ≥ 95% match |
| 4. Parameter alignment | state_config.json vs TrafficEnvConfig caps | 0 mismatches |
| 5. Live SUMO (optional, `--sumo`) | SUMO obs vs re-assembled from same SUMO metrics; approach/ROI order | cosine ≥ 0.95 |
| 6. Lane-group topology | `DEFAULT_MANUAL_LANE_GROUPS` lane IDs are real approaches of their own junction (guards Bug S3) | 0 mismatches |

Run with: `python scripts/check_obs_match.py [--sumo] [--scenarios N]`

---

## 11. Applied Patches — Sim-to-Real Refactor (2026-06-08)

All four mandatory patches have been applied to
`src/vision/state_extractor.py` and `src/traffic_env/components/observations.py`.

### Patch 1 — Phase Encoding (CRITICAL)

**File:** `src/vision/state_extractor.py`

**Problem:** `_phase_one_hot(action=1)` produced `[0,1,0,0]`; policy was trained expecting `[0,0,1,0]` for Group B green (SUMO phase 2). Every Group B step was misconditioned.

**Fix:** Added class-level `_ACTION_TO_PHASE = {0: 0, 1: 2}` mapping. `_phase_one_hot` now maps MAPPO action → SUMO phase index before encoding.

```python
# Before
def _phase_one_hot(self, phase: int) -> List[float]:
    phase = int(np.clip(int(phase), 0, 3))   # action=1 → index 1 → [0,1,0,0]  WRONG

# After
_ACTION_TO_PHASE: Dict[int, int] = {0: 0, 1: 2}
def _phase_one_hot(self, action: int) -> List[float]:
    sumo_phase = self._ACTION_TO_PHASE.get(int(action), 0)  # action=1 → phase 2 → [0,0,1,0]  CORRECT
```

**Impact:** All Group B green observations now carry the correct encoding that matches training.

---

### Patch 2 — Occupancy Geometry (SIM2REAL)

**Files:** `src/vision/state_extractor.py`, `src/traffic_env/components/observations.py`

**Problem:**
- **SUMO** `occupancy_norm` = `getLastStepOccupancy() / 100` = `Σ(vehicle_length_m) / lane_length_m` (1D linear coverage)
- **Camera** `occupancy_norm` = `Σ(bbox_width × bbox_height) / lane_ROI_area_px` (2D bbox area fraction)

These are geometrically incompatible. Perspective projection makes near-field vehicles inflate the camera value relative to far-field vehicles, creating a distance-dependent systematic offset.

**Fix:**
Both pipelines now use the same 1D formula: `Σ(class_length) / lane_length`.

```python
# state_extractor.py — Before
total_bbox_area += w * h
"occupancy_norm": total_bbox_area / lane_area_px

# state_extractor.py — After
total_vehicle_length_px += veh_len_px          # CLASS_LENGTH_M[cls] × px_per_meter
"occupancy_norm": total_vehicle_length_px / lane_length_px

# observations.py VisionLaneMetrics — Before
total_bbox_area += w * h
occupancy = total_bbox_area / lane_roi_area

# observations.py VisionLaneMetrics — After
total_vehicle_length_m += CLASS_LENGTH_M.get(cls_id, DEFAULT_LENGTH_M)
occupancy = total_vehicle_length_m / lane_length_m
```

**Side effect:** `tlwh` (bounding box dimensions) is no longer extracted in `VisionLaneMetrics.compute()` — removed from the `try` block entirely.

---

### Patch 3 — Timer Normalization (SCALE)

**File:** `src/vision/state_extractor.py`

**Problem:** `StateExtractor.__init__` defaulted `max_green_time=90.0` while training used `SimConfig.max_green_time=60`. This caused `green_timer_norm` to be scaled by 60/90 = 0.667× in deployment vs training — the policy would perceive phase durations as 33% shorter than they actually were.

**Fix:**
```python
# Before
max_green_time: float = 90.0

# After
max_green_time: float = 60.0
```

**Note:** Orchestrators that explicitly pass `max_green_time` are unaffected. This fixes the silent failure case where `StateExtractor` is instantiated with no override.

---

### Patch 4 — Zero-Division and Clip Guards (VERIFIED)

**Files:** Both

All features verified to have correct guards. No changes required — existing code was already safe:

| Guard | Location | Mechanism |
|---|---|---|
| Empty lane | `_lane_metrics_from_tracks` | Early return `_empty_lane_metrics()` when `not lane_tracks` |
| Empty lane | `VisionLaneMetrics.compute` | Early return `empty` dict when `not tracks` |
| Lane length zero | Both | `max(lane_length_px, 1.0)` and `max(lane_length_m, 1.0)` denominators |
| Vehicle count zero | `_lane_metrics_from_tracks` | `n = max(total_vehicle_count, 1.0)` |
| Feature out-of-range | All 5 lane features | `float(np.clip(value, 0.0, 1.0))` on every return |
| Speed cap zero | `_normalize()` | `max(float(cap), 1e-6)` |

---

### Post-Patch State Quality

| | Before | After |
|---|---|---|
| Phase encoding | BROKEN — Group B misconditioned every step | FIXED |
| Occupancy sim2real | APPROXIMATE — 2D vs 1D geometric mismatch | FIXED — both 1D |
| Timer scale | SKEWED — 0.667× in default case | FIXED |
| Guards | Already correct | Verified correct |
| **Overall obs quality** | **4 / 10** | **7 / 10** |

Remaining gaps (not addressable by code alone):
- `px_per_meter` must be measured per camera at installation
- `avg_speed_norm` noise from ByteTrack at low speeds (EMA partially mitigates)
- SUMO training scenarios should include representative Vietnamese vehicle type distributions

---

## 12. Schema 1.1.0 — Audit v3 Patches (2026-06-12)

A second audit pass found three structural observation bugs that survived the
2026-06-08 review. All are fixed; the obs schema version was bumped to
**1.1.0**. The obs vector remains 26-dim — the *meaning* of the slots changed,
which is why every pre-1.1.0 checkpoint is invalid.

### S1 — Per-Approach Aggregation (CRITICAL)

**File:** `src/traffic_env/components/observations.py:build_local_obs`

**Problem:** the builder sliced `getControlledLanes()[:4]`, so each obs slot
was one raw lane and only 4 of the 8 controlled lanes per junction were
observed — two entire approaches were invisible to the policy. The vision side
was *already* approach-level (one ROI per approach), so training and
deployment slots had different semantics.

**Fix:** `_approach_groups_for_tls()` groups controlled lanes by parent edge;
`_aggregate_approach_metrics()` aggregates per-lane metrics into one
approach-level feature set (queue/occupancy: unweighted mean; speed/class
shares: vehicle-count-weighted mean). Each slot now covers one full approach
in both pipelines.

### S2 — Signed Pressure

**Files:** `observations.py:_tls_pressure_proxy` + `state_extractor.py:_tls_pressure_proxy`

**Problem:** `pressure_norm = |Σq_A − Σq_B| / max_lanes_per_tls` was unsigned —
the policy could not tell WHICH group was congested.

**Fix:** `pressure_norm = clip(0.5 × (1 + q̄_A − q̄_B), 0, 1)` with group
**means**. 0.5 = balanced; > 0.5 = group A more queued. Both pipelines changed
together; equivalence verified by `check_obs_match.py` Check 2.

### S3 — J2 Lane-Group Topology

**File:** `src/traffic_env/config.py:DEFAULT_MANUAL_LANE_GROUPS`

**Problem:** J2's group B listed `-E1_*` lanes — the approach to **J0** — so
J2's `pressure_norm` (and the pressure reward term, and the MaxPressure
baseline) mixed in another junction's queues.

**Fix:** corrected to `E1_*`; guarded by `check_obs_match.py` CHECK 6.

### Related same-day fixes outside the obs vector

| ID | Area | Summary |
|---|---|---|
| R1 | reward | `green_lanes_by_tls` now wired in `multi_agent.py` — pressure reward was silently 0.0 before (see `docs/reward.md` §7) |
| R2 | training | per-agent value heads + per-agent GAE in `train_ppo.py` (MAPPO v3) |
| E1 | eval | `eval_max_steps` default 400 → 1080 (full episode) |
| B1b | config | `state_config.json:stop_speed_m_s` 0.5 → 0.1 (matches training halt threshold) |

Reward revision **1.2.0** (mean-of-squares queue, signed pressure reward,
local lane-exit throughput) followed the same day — documented in
`docs/reward.md`. The obs schema was deliberately left unchanged at 26-dim
for the W5 campaign.
