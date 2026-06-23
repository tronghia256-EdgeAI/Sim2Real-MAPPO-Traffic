# LEGACY CHECKPOINT — DO NOT USE FOR PAPER RESULTS

**Run:** `20260418_215140` (trained 2026-04-18, seed 42, 2M steps)
**Status:** Archived development artifact. Incompatible with code at schema **1.1.0** (2026-06-12).

## Why this checkpoint is invalid for publication

### 1. Trained with a different reward function than the codebase documents

This run's `run_config.json` shows the reward it was actually trained with:

```
queue=-1.0, throughput=+1.0, switch_penalty=-0.1,
high_queue_penalty=-0.15, low_speed_penalty=-0.25, waiting_time=-0.5
clip = [-2.0, +2.0]
```

There is **no `pressure` term** — the "Extended PRESSLIGHT reward" in
`src/traffic_env/config.py` (`pressure=-0.5`, clip `[-2.0, +1.5]`) was added
after this run. Additionally, even for runs after the pressure weight was
added, the env never passed `green_lanes_by_tls` to `RewardCalculator`, so the
pressure penalty evaluated to 0.0 on every step (fixed in `multi_agent.py` v3).

### 2. Trained on a pre-1.1.0 observation schema with three correctness bugs

| Bug | Description |
|---|---|
| S1 | Obs sliced `getControlledLanes()[:4]` — only 4 of 8 approach lanes observed; two approaches per junction were invisible to the policy. Now: per-approach aggregation. |
| S2 | `pressure_norm = abs(qa-qb)/4` — unsigned; the policy could not tell which group was congested. Now: signed, `0.5*(1 + mean_qA - mean_qB)`. |
| S3 | `DEFAULT_MANUAL_LANE_GROUPS["J2"]` listed `-E1_*` (an approach to **J0**) instead of `E1_*`. J2's pressure feature/reward partition used another junction's lanes. |

### 3. Critic architecture mismatch

This checkpoint has a scalar critic head (team-mean reward). Code v3 uses one
value head per agent (per-agent advantages). `critic_state_dict` will not load;
training cannot be resumed from this checkpoint.

## What it may still be used for

- Pipeline smoke tests (actor loads fine; its decisions are meaningless under
  the new obs semantics).
- Historical reference for the training-infrastructure validation.

All paper results must come from runs with `env_cfg.version >= 1.1.0` in their
`run_config.json`.
