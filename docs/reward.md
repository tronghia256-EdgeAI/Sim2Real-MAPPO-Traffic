# Reward Function — Design Reference

**File:** `src/traffic_env/components/rewards.py`  
**Config:** `src/traffic_env/config.py` → `RewardConfig`, `DEFAULT_REWARD_WEIGHTS`  
**Revision:** **1.2.0** (2026-06-12, W5 campaign — see §6 for the change log)

> **LEGACY WARNING:** checkpoint `20260418_215140` was trained on a pre-1.1.0
> reward (no pressure term active — bug R1) and must not be used for paper
> results. All results require retraining with `env_cfg.version >= 1.1.0`.

---

## 1. Reward Equation

$$
R(t) = w_q \cdot \text{queue}(t)
     + w_{pr} \cdot \text{pressure}(t)
     + w_t \cdot \text{throughput}(t)
     + w_s \cdot \text{switch}(t)
     + w_{sp} \cdot \text{low\_speed}(t)
     + w_w \cdot \text{waiting}(t)
$$

**Final output:**

$$
r_{\text{agent}} = \text{clip}(R(t) \times \text{scale},\ \text{clip\_low},\ \text{clip\_high})
$$

All component functions return values in `[0, 1]` except `pressure` which is **signed** in `[−1, 1]` (1.2.0) and `switch_penalty` which is unbounded above. Signs are applied via weights. **Penalty weights are negative; reward weights are positive** — for the signed pressure term this means wrong-side green is penalised AND correct-side green is rewarded.

---

## 2. Component Reference

### 2.1 `queue_penalty` — Nonlinear Congestion *(1.2.0: mean of squares)*

$$
\text{queue}(t) = \text{clip}\!\left(\frac{1}{|L|}\sum_{l \in L} q_l^{\,2},\; 0,\; 1\right)
\quad\text{where}\quad
q_l = \text{effective\_queue\_norm}(l)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−1.0` |
| SUMO source | `traci.lane.getLastStepHaltingNumber` / cap |
| Vision proxy | halted-vehicle fraction in ROI polygon |

**Rationale:** Squaring provides near-zero gradient in free-flow conditions and strong gradient during sustained congestion (PRESSLIGHT's saturation-aware design).

**1.2.0 change (bug R4):** the formula is `mean(q²)` — mean of squares, **not**
`mean(q)²`. By Jensen's inequality `mean(q²) ≥ mean(q)²` with equality only
when all lanes are equal, so a single saturated lane among many empty ones is
penalised heavily instead of being averaged away (anti-starvation guard). The
pre-1.2.0 `mean(q)²` form let one starved lane hide behind a low average.

---

### 2.2 `pressure_penalty` — Phase-Aware Queue Imbalance *(PRESSLIGHT / CoLight; 1.2.0: SIGNED)*

$$
\text{pressure}(t) = \text{clip}\!\left(\frac{\sum_{l \in L_{\text{red}}} q_l - \sum_{l \in L_{\text{green}}} q_l}{|L_{\text{red}}| + |L_{\text{green}}|},\; -1,\; 1\right)
$$

| Property | Value |
|---|---|
| Range | **`[−1, 1]`** (signed, 1.2.0) |
| Default weight | `−0.5` |
| Requires | `green_lanes_by_tls` in `action_info` (wired in `multi_agent.py` since 2026-06-12) |
| Returns | `0.0` (neutral) if phase-lane info is absent |
| Vision proxy | same `effective_queue_norm` partitioned by phase group |

**Rationale:** This is the core signal from PRESSLIGHT (Wei et al., KDD 2019). Positive value = red-side lanes carry more queue than green-side lanes (wrong phase held); with the negative weight this is a penalty. Negative value = green correctly serves the more congested side; the negative weight turns this into a **positive reward**.

**1.2.0 change (bug R3):** the pre-1.2.0 form was one-sided
`max(red − green, 0)`, which gave **zero gradient whenever the allocation was
already correct** — the policy was never rewarded for the right decision. The
signed form restores gradient on both sides.

**Bug R1 (FIXED 2026-06-12):** `MappoTrafficEnv.step` previously never passed
`green_lanes_by_tls` to the `RewardCalculator`, so this term was silently
`0.0` in **all** training runs before that date (including checkpoint
`20260418_215140`). It is now populated every step from the manual lane groups
(see §7); red lanes are derived automatically as
`all_controlled_lanes − green_lanes`.

---

### 2.3 `throughput_reward` — LOCAL Lane-Exit Count *(1.2.0)*

**Primary path (1.2.0, bug R5):** per-agent count of vehicles that left this
agent's controlled lanes since the previous decision step, computed from
per-lane vehicle-ID set differences:

$$
\text{throughput}(t) = \text{clip}\!\left(\frac{1}{D}\sum_{l \in L} \bigl|\,\text{ids}_{t-1}(l) \setminus \text{ids}_t(l)\,\bigr|,\; 0,\; 1\right)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `+1.0` |
| `D` (divisor) | `config.reward.throughput_norm_divisor` = 20 |
| SUMO source | `lane_cache[lane_id]["vehicle_ids"]` set differences (training-only key, populated by `ObservationBuilder._lane_metrics`) |
| Vision proxy | ByteTrack IDs exiting ROI per step — **exact semantic match** |

Each agent is credited only for traffic it actually served — no cross-agent
splitting. This also makes the SUMO training signal semantically identical to
the documented vision proxy (ROI exit events).

**Fallback path (pre-1.2.0 behaviour):** when the lane cache carries no
`vehicle_ids` (synthetic test fixtures, external callers), the global
`simulation.getArrived()` delta is split equally across agents:

$$
\text{throughput}_{\text{fallback}}(t) = \frac{\text{clip}(\Delta V / D,\; 0,\; 1)}{N_{\text{agents}}}
\quad\text{where}\quad
\Delta V = V_{\text{current}} - V_{\text{prev}}
$$

The global path has high variance and diluted credit at scale (every agent
receives the same signal regardless of contribution) — that is why it was
demoted to fallback. The switch is automatic per step (`has_lane_ids` in
`calculate_rewards`).

---

### 2.4 `switch_penalty` — Phase-Switching Cost

$$
\text{switch}(t) = \mathbb{1}[\text{phase changed at step } t]
$$

| Property | Value |
|---|---|
| Range | `[0, ∞)` — typically `{0, 1}` |
| Default weight | `−0.1` |
| Source | `action_info["switches_by_tls"]` |

Prevents rapid phase cycling. The weight magnitude controls the minimum effective green time the policy will voluntarily hold.

---

### 2.5 `low_speed_penalty` — Mean Speed Below Threshold

$$
\text{low\_speed}(t) = \max(\tau_s - \overline{v},\; 0)
\quad\text{where}\quad
\overline{v} = \frac{1}{|L|}\sum_{l \in L} \text{avg\_speed\_norm}(l)
$$

| Property | Value |
|---|---|
| Range | `[0, τ_s]` |
| Default weight | `−0.2` |
| `τ_s` | `config.reward.low_speed_threshold` = 0.20 |
| Vision proxy | ByteTrack pixel-displacement speed estimate |

> **Vision deployment note:** Speed estimated from ByteTrack pixel displacement is significantly noisier than SUMO speed. Consider setting `"low_speed_penalty": 0.0` in `RewardConfig.weights` when deploying with YOLO + ByteTrack to avoid reward noise corrupting the policy.

---

### 2.6 `waiting_penalty` — Accumulated Delay

$$
\text{waiting}(t) = \text{clip}\!\left(\overline{w},\; 0,\; 1\right)
\quad\text{where}\quad
\overline{w} = \frac{1}{|L|}\sum_{l \in L} s(l)
$$

$$
s(l) = \begin{cases}
\text{waiting\_time\_norm}(l) & \text{if available in lane\_cache} \\
\text{effective\_queue\_norm}(l) & \text{(vision proxy)}
\end{cases}
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−0.3` |
| SUMO source | `traci.lane.getWaitingTime` / `waiting_cap` |
| Vision proxy | halted-vehicle fraction (same as queue) |

**Previous bug (fixed):** The old implementation multiplied by `step_length / max_green_time` (= `5/60` = 0.083), capping the output at ≤ 0.083 and making the term functionally negligible. This division is removed.

---

## 3. Value Ranges with Default Weights

| Component | Raw range | Weight | Contribution |
|---|---|---|---|
| `queue_penalty` | `[0, 1]` | `−1.0` | `[−1.0, 0]` |
| `pressure_penalty` | `[−1, 1]` | `−0.5` | `[−0.5, +0.5]` |
| `throughput_reward` | `[0, 1]` | `+1.0` | `[0, +1.0]` |
| `switch_penalty` | `[0, 1]` | `−0.1` | `[−0.1, 0]` |
| `low_speed_penalty` | `[0, 0.2]` | `−0.2` | `[−0.04, 0]` |
| `waiting_penalty` | `[0, 1]` | `−0.3` | `[−0.3, 0]` |
| **Total (raw)** | | | **`[−1.94, +1.50]`** |
| **After clip** | | | **`[−2.0, +1.5]`** |

With the 1.2.0 signed pressure and local throughput, the positive end of the
raw range genuinely reaches the `+1.5` clip bound (was capped at `+0.50`
pre-1.2.0); the lower bound remains tight against the clip.

---

## 4. `action_info` Protocol

```python
action_info = {
    # Required
    "lane_ids_by_tls": {
        "J0": ["lane_id_0", "lane_id_1", ...],
        "J2": ["lane_id_4", "lane_id_5", ...],
    },

    # Recommended — enables pressure_penalty
    "green_lanes_by_tls": {
        "J0": ["lane_id_0", "lane_id_1"],   # currently green
        "J2": ["lane_id_4", "lane_id_5"],
    },

    # Throughput FALLBACK (used only when lane_cache lacks "vehicle_ids");
    # one of these three:
    "delta_passed": 5,                # preferred: pre-computed delta
    "current_passed": 120,            # OR: current cumulative + prev below
    "prev_passed": 115,
    # (if neither, internal _prev_passed_total is used)

    # Switch info (one of these)
    "switches_by_tls": {"J0": True, "J2": False},  # preferred
    "num_switches": 1,                              # OR: distributed equally
    "switched_tls": {"J0": True, "J2": False},      # OR: boolean map

    # Optional
    "step_length_seconds": 5,  # kept for compatibility; no longer used
}
```

**Local throughput (1.2.0) does not use `action_info`** — it reads
`lane_cache[lane_id]["vehicle_ids"]` (a training-only key populated by
`ObservationBuilder._lane_metrics`) and diffs the per-lane ID sets across
consecutive `calculate_rewards` calls. `RewardCalculator.reset()` clears the
internal `_prev_lane_vehicle_ids` snapshot at episode boundaries.

---

## 5. Sim-to-Real Alignment

The reward was designed so every component has a direct vision proxy computable from YOLO + ByteTrack detections. However, the reward is **only used during training in SUMO**. During deployment, only the trained policy runs — no reward is computed.

The alignment matters because if SUMO training metrics and vision inference metrics diverge significantly, the policy will not generalise. The table below summarises alignment quality:

| Component | SUMO metric | Vision proxy | Alignment |
|---|---|---|---|
| `queue_penalty` | `getLastStepHaltingNumber` / cap | Halted vehicles in ROI / cap | ✓ Good |
| `pressure_penalty` | Same, partitioned by phase | Same | ✓ Good |
| `throughput_reward` | Per-lane vehicle-ID exits (local, 1.2.0) | Track IDs exiting ROI | ✓ Good* |
| `switch_penalty` | Action logic | Action logic | ✓ Exact |
| `low_speed_penalty` | `getLastStepMeanSpeed` | Pixel-displacement estimate | ✗ Noisy |
| `waiting_penalty` | `getWaitingTime` / cap | Queue proxy (no per-vehicle timer) | ⚠ Proxy |

\* The 1.2.0 local lane-exit formulation is the **same event semantics** as
ByteTrack ROI exits (was ⚠ Partial pre-1.2.0, when SUMO counted network-level
arrivals). Residual gap: ByteTrack track loss under occlusion looks identical
to a true exit, so the vision-side count over-fires — hence the reduced
throughput weight recommended below.

**Recommended weights for vision deployment:**

```python
from src.traffic_env.config import RewardConfig

vision_weights = RewardConfig(weights={
    "queue":            -1.0,
    "pressure":         -0.5,
    "throughput":        0.5,   # reduced: proxy less reliable
    "switch_penalty":   -0.1,
    "low_speed_penalty": 0.0,   # disabled: speed estimate too noisy
    "waiting_time":     -0.3,
})
```

---

## 6. Change Log

### Revision 1.2.0 (2026-06-12, W5 campaign)

| Bug | Old behaviour | Fixed behaviour |
|---|---|---|
| R3 | Pressure one-sided `max(red−green, 0)` — zero gradient whenever the allocation was correct | **Signed** `(Σq_red − Σq_green)/n_total ∈ [−1, 1]` — penalty for wrong-side green AND reward for correct-side green |
| R4 | Queue penalty `mean(q)²` — single-lane starvation averaged away | `mean(q²)` — mean of squares (Jensen, anti-starvation) |
| R5 | Global `getArrived()` delta split equally across agents — variance + diluted credit | **Local** per-agent lane-exit count from vehicle-ID set differences; global path kept as automatic fallback |

### Revision 1.1.0 (2026-06-12, audit v3)

| Bug | Old behaviour | Fixed behaviour |
|---|---|---|
| R1 | `green_lanes_by_tls` never passed by `MappoTrafficEnv.step` → pressure term silently `0.0` in ALL training runs to date | Wired in `multi_agent.py` from manual lane groups (see §7) |

### Earlier revision

| Issue | Old behaviour | Fixed behaviour |
|---|---|---|
| `waiting_penalty` formula | Multiplied by `step_length/max_green` → max ≤ 0.083 | Direct `waiting_time_norm` or queue proxy; range `[0, 1]` |
| `_prev_passed_total` side-effect | Always mutated even when caller supplied `prev_passed` | Only mutated when caller does **not** supply `prev_passed` |
| Double-counting queue | `queue_penalty` + `high_queue_penalty` both used `effective_queue_norm` | `high_queue_penalty` removed; replaced by `pressure_penalty` |
| Linear queue penalty | `0.85 × queue + 0.15 × occupancy` (magic numbers) | `mean_queue ** 2` (nonlinear, no magic numbers) |
| Pressure signal | Absent | `pressure_penalty` (PRESSLIGHT formulation) added |
| `tls_id` unused in private methods | Accepted but silently ignored | Used for `logger.debug` output in every method |
| Dead code | `_weight_with_default` never called | Removed |
| Reward clip | `[−2.0, +2.0]` — never fired (actual range `[−1.24, +0.50]`) | `[−2.0, +1.5]` — both bounds now active |

---

## 7. `pressure_penalty` Wiring in `multi_agent.py` (DONE — bug R1)

The pressure component returns `0.0` until the environment populates
`green_lanes_by_tls` in `action_info`. **This is wired since 2026-06-12** —
`MappoTrafficEnv.step` maps the applied green phase back to its manual lane
group every step (`src/traffic_env/envs/multi_agent.py`, reward section):

```python
# multi_agent.py step() — actual implementation
lane_groups = self.config.get_lane_groups()
green_lanes_by_tls: Dict[str, List[str]] = {}
for tls_id in self.tls_ids:
    groups    = lane_groups.get(tls_id)
    phase_map = self._green_phase_map.get(tls_id, [])
    target    = target_phases.get(tls_id)
    if groups and target in phase_map:
        group_idx = phase_map.index(target)
        green_lanes_by_tls[tls_id] = list(groups[0] if group_idx == 0 else groups[1])

action_info["green_lanes_by_tls"] = green_lanes_by_tls
```

Before this fix the pressure reward term was silently inactive in every
training run — any checkpoint trained before 2026-06-12 (including
`20260418_215140`) never received the pressure signal.

---

## 8. Configuration Reference

```python
# src/traffic_env/config.py

DEFAULT_REWARD_WEIGHTS = {
    "queue":            -1.0,
    "pressure":         -0.5,
    "throughput":        1.0,
    "switch_penalty":   -0.1,
    "low_speed_penalty":-0.2,
    "waiting_time":     -0.3,
}

@dataclass
class RewardConfig:
    weights:                  dict  = DEFAULT_REWARD_WEIGHTS
    reward_scale:             float = 1.0
    reward_clip_low:          float = -2.0
    reward_clip_high:         float = 1.5
    throughput_norm_divisor:  float = 20.0   # vehicles per step at saturation
    low_speed_threshold:      float = 0.20   # normalised speed below which penalty applies
    jam_speed_threshold:      float = 0.50   # informational; not used in reward
    jam_queue_threshold:      float = 0.50   # informational; not used in reward
```

---

## 9. References

- Wei et al. (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.* KDD 2019.
- Wei et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control.* CIKM 2019.
- Chen et al. (2020). *Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control.* AAAI 2020.
