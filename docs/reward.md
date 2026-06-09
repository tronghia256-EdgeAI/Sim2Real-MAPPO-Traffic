# Reward Function — Design Reference

**File:** `src/traffic_env/components/rewards.py`  
**Config:** `src/traffic_env/config.py` → `RewardConfig`, `DEFAULT_REWARD_WEIGHTS`

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

All component functions return **non-negative** values in `[0, 1]` (except `switch_penalty` which is unbounded above). Signs are applied via weights. **Penalty weights are negative; reward weights are positive.**

---

## 2. Component Reference

### 2.1 `queue_penalty` — Nonlinear Congestion

$$
\text{queue}(t) = \text{clip}\!\left(\overline{q}^{\,2},\; 0,\; 1\right)
\quad\text{where}\quad
\overline{q} = \frac{1}{|L|}\sum_{l \in L} \text{effective\_queue\_norm}(l)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−1.0` |
| SUMO source | `traci.lane.getLastStepHaltingNumber` / cap |
| Vision proxy | halted-vehicle fraction in ROI polygon |

**Rationale:** Squaring `mean_q` provides near-zero gradient in free-flow conditions and strong gradient during sustained congestion. This matches PRESSLIGHT's saturation-aware design and reduces reward noise in lightly loaded intersections.

---

### 2.2 `pressure_penalty` — Phase-Aware Queue Imbalance *(PRESSLIGHT / CoLight)*

$$
\text{pressure}(t) = \max\!\left(\frac{\sum_{l \in L_{\text{red}}} q_l - \sum_{l \in L_{\text{green}}} q_l}{|L_{\text{red}}| + |L_{\text{green}}|},\; 0\right)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−0.5` |
| Requires | `green_lanes_by_tls` in `action_info` |
| Returns | `0.0` if phase-lane info is absent |
| Vision proxy | same `effective_queue_norm` partitioned by phase group |

**Rationale:** This is the core signal from PRESSLIGHT (Wei et al., KDD 2019). Penalises the policy when red-side lanes are more congested than green-side lanes — i.e. the policy chose the wrong phase. The normalisation by total lane count keeps the output in `[0, 1]` regardless of intersection size.

**To enable:** the environment must populate `action_info["green_lanes_by_tls"]`:

```python
action_info["green_lanes_by_tls"] = {
    "J0": ["-E5_0", "-E5_1", "-E6_0", "-E6_1"],   # lanes currently green
    "J2": ["-E3_0", "-E3_1", "-E4_0", "-E4_1"],
}
```

Red lanes are derived automatically as `all_controlled_lanes − green_lanes`.

---

### 2.3 `throughput_reward` — Cooperative Cleared-Vehicles

$$
\text{throughput}(t) = \frac{\text{clip}(\Delta V / D,\; 0,\; 1)}{N_{\text{agents}}}
\quad\text{where}\quad
\Delta V = V_{\text{current}} - V_{\text{prev}}
$$

| Property | Value |
|---|---|
| Range | `[0, 1/N_agents]` |
| Default weight | `+1.0` |
| `D` (divisor) | `config.reward.throughput_norm_divisor` = 20 |
| SUMO source | `simulation.getArrived()` delta |
| Vision proxy | ByteTrack IDs exiting ROI per step |

Splitting by `N_agents` preserves cooperative incentive while preventing free-riding. The signal is global (network-level) in SUMO but should be per-intersection when using vision — pass `delta_passed` per-TLS if available.

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
| `pressure_penalty` | `[0, 1]` | `−0.5` | `[−0.5, 0]` |
| `throughput_reward` | `[0, 0.5]` | `+1.0` | `[0, +0.5]` |
| `switch_penalty` | `[0, 1]` | `−0.1` | `[−0.1, 0]` |
| `low_speed_penalty` | `[0, 0.2]` | `−0.2` | `[−0.04, 0]` |
| `waiting_penalty` | `[0, 1]` | `−0.3` | `[−0.3, 0]` |
| **Total (raw)** | | | **`[−1.94, +0.50]`** |
| **After clip** | | | **`[−2.0, +1.5]`** |

The clip `[−2.0, +1.5]` now actively bounds the upper range and is tight against the worst-case lower bound.

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

    # Throughput (one of these three)
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

---

## 5. Sim-to-Real Alignment

The reward was designed so every component has a direct vision proxy computable from YOLO + ByteTrack detections. However, the reward is **only used during training in SUMO**. During deployment, only the trained policy runs — no reward is computed.

The alignment matters because if SUMO training metrics and vision inference metrics diverge significantly, the policy will not generalise. The table below summarises alignment quality:

| Component | SUMO metric | Vision proxy | Alignment |
|---|---|---|---|
| `queue_penalty` | `getLastStepHaltingNumber` / cap | Halted vehicles in ROI / cap | ✓ Good |
| `pressure_penalty` | Same, partitioned by phase | Same | ✓ Good |
| `throughput_reward` | `simulation.getArrived()` | Track IDs exiting ROI | ⚠ Partial* |
| `switch_penalty` | Action logic | Action logic | ✓ Exact |
| `low_speed_penalty` | `getLastStepMeanSpeed` | Pixel-displacement estimate | ✗ Noisy |
| `waiting_penalty` | `getWaitingTime` / cap | Queue proxy (no per-vehicle timer) | ⚠ Proxy |

\* SUMO counts vehicles leaving the **entire network**; ByteTrack counts track IDs exiting a **camera ROI**. Reduce `throughput_norm_divisor` or pass per-intersection delta when using vision.

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

## 6. Changes from Previous Version

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

## 7. Enabling `pressure_penalty` in `multi_agent.py`

The pressure component returns `0.0` until the environment populates `green_lanes_by_tls` in `action_info`. Add the following to the `step` method in `MappoTrafficEnv`:

```python
# After applying green phases, before calling calculate_rewards:
green_lanes_by_tls = {}
for tls_id in self.tls_ids:
    current_phase = self._get_phase(tls_id)
    lane_groups   = self.config.get_lane_groups().get(tls_id, ([], []))
    # phase 0 → group A green; phase 2 → group B green (SUMO phase index)
    green_lanes_by_tls[tls_id] = (
        lane_groups[0] if current_phase == 0 else lane_groups[1]
    )

action_info["green_lanes_by_tls"] = green_lanes_by_tls
```

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
