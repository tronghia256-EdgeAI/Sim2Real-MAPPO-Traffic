# Paper 1 — Master Execution Plan (IEEE T-ITS)

> **Deadline:** 2026-06-28 · **Created:** 2026-06-16 · **Status:** compute GREEN-LIT, ready to launch
> Shared checklist / single source of truth. Paper content lives in
> [`T-ITS_skeleton.md`](T-ITS_skeleton.md); artifact build steps in
> [`POST_TRAINING_RUNBOOK.md`](POST_TRAINING_RUNBOOK.md). This file is the *project plan*:
> what to run, when, and what is still missing.

This plan supersedes the original Phase-8 schedule. Phases 1–7 of the original
plan (item checklist, figure/table roadmaps, fact collection, related-work
harvest, evidence map, reviewer-attack map) are adopted as-is; the schedule and
compute below are the corrected, feasibility-checked version.

---

## 0. Locked Decisions (2026-06-16)

| Decision | Value | Rationale |
|---|---|---|
| **Training budget** | **0.5M steps/seed — everything (main + ablation)** | Pilot held-out eval flat 100k→500k (135.8→141.5 s wait); converges <200k. 0.5M = full plateau + ~6-day deadline buffer. |
| **VM** | **32 vCPU, 1 machine, ~19 workers**, compute-optimized x86 (AMD Genoa / Intel SPR), ~2 GB RAM/worker, **no GPU**, Linux, **on-demand (not spot)** | Chosen for cost (2026-06-16): higher utilization (19/19 busy) vs 64 vCPU (idle slots) → ~30% cheaper, still hits deadline. Spot rejected: `train()` starts `global_step=0` with no checkpoint load → a preempted job restarts from 0 (no mid-train resume). |
| **III-E / VI-E split** | III-E = sensing-noise model **overview only** (defines ε); **VI-E parameterizes** it in **TABLE-3a** on a **documented basis** (state.md + ByteTrack/YOLOv11 lit). `class_flip_rate` from the YOLO confusion matrix is an **optional** strengthening of one TABLE-3a row; measured feature-recovery (and **FIG-1a**) **deferred to the companion study**. | No gt/est trajectory CSVs exist; full feature-recovery infeasible in window. Honest framing for reviewer Q2. |
| **Seeds** | 42, 123, 456, 789, 1337 (main); 42, 123, 456 (ablation) | 5-seed for headline, 3-seed for ablation comparisons. |
| **Networks** | n2_corridor, n3_grid | Ablations: n3 only. |

**Compute math (verified):** 0.5M ≈ 54 h/seed @ 2.58 SPS. ~59 jobs on 19 workers →
main results **~Day 5 (20/6)**, ablations **~Day 8 (23/6)**, thin buffer (≈1 day at D12).
The **24/6 fallback trigger is load-bearing** at this size — watch it.

---

## 1. Pre-flight (already GREEN 2026-06-16 — re-run on the VM)

```bash
python scripts/check_obs_match.py            # checks 1–4 + 6  (was: 0 failures)
python scripts/check_obs_match.py --sumo     # check 5 live cosine (was: 1.0000)
pytest -m "not slow and not serial"          # (was: 60/60 passed)
```
Verified clean on 2026-06-16: all checks pass, smoke-train of all 6 arms
(mappo/ippo × proxy/privileged × n2/n3) exit 0, resume works, obs_rms present,
proxy=26-dim / privileged=38-dim. See memory `precampaign-greenlight`.

---

## 2. Compute Launch (the long pole)

> ⚠️ **Launcher gotcha:** `--reward-presets` × `--obs-ablations` is a CROSS
> PRODUCT. Pass them in **separate** invocations or you generate spurious combos.
> Ablation arms are **n3 / proxy / mappo only**; `full,none` is the main arm
> (Stage 1), never re-trained.

### Stage 1 — Main campaign (30 jobs), launch Day 1

```bash
python scripts/parallel_launcher.py \
  --networks n2_corridor n3_grid --algos mappo ippo --obs-modes proxy privileged \
  --seeds 42 123 456 789 1337 --total-timesteps 500000 \
  --max-workers 19 --stagger 45 --campaign-id main_05M
```
30 jobs = (2 nets × {mappo-proxy, mappo-priv, ippo-proxy} × 5 seeds); ippo-privileged auto-skipped.
30 jobs on 19 workers ≈ 1.6 waves (machine fully saturated — that is the point) → **done ~Day 4–5**.

### Stage 2 — Ablations (concurrent, launch when Stage 1 finishes ~Day 3–4)

```bash
# 2a — reward ablations (18 jobs)  — run CONCURRENTLY with 2b
#      no_delay is REQUIRED to fill the VI-F "No Delay Term (w_w=0)" table row.
#      Optionally append no_low_speed (-> 21 jobs) — near-inert/vision-noisy term,
#      nice-to-have only if the deadline allows.
python scripts/parallel_launcher.py --networks n3_grid --algos mappo --obs-modes proxy \
  --reward-presets no_pressure no_throughput queue_only unsigned_pressure mean_then_square no_delay \
  --seeds 42 123 456 --total-timesteps 500000 \
  --max-workers 12 --stagger 45 --campaign-id ablation_reward_05M

# 2b — obs ablations (9 jobs)
python scripts/parallel_launcher.py --networks n3_grid --algos mappo --obs-modes proxy \
  --obs-ablations no_class_shares no_pressure_feature lane_truncated \
  --seeds 42 123 456 --total-timesteps 500000 \
  --max-workers 7 --stagger 45 --campaign-id ablation_obs_05M
```
12 + 7 = 19 workers total (no oversubscription). 2a is now 18 jobs (~1.5 waves on
12 workers) + 2b 9 jobs → **done ~Day 8** (no_delay added; if no_low_speed too → 21
jobs/~1.75 waves, still within buffer). Launch after Stage 1 frees the machine (~Day 4–5).

### Stage 3 — Sublane cross-eval for TABLE-9 (separate, ~Day 1–4, cheap)

`--networks` only maps to `networks/<n>/sumo_config.sumocfg`, so the lane-based
arm goes through `train_ppo.py` directly, then cross-eval:
```bash
python experiment/runners/train_ppo.py --mode train --algo mappo --obs-mode proxy \
  --sumo-cfg   sumo_configs/networks/n3_grid/sumo_config_lanebased.sumocfg \
  --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
  --seed 42 --total-timesteps 500000 --ckpt-dir results/paper1_mappo/sublane_lanebased
# then cross-eval lane-trained vs sublane-trained (both directions) via eval_compare.py
```

**Fallback trigger:** if any stage is <80% done by **24/6**, cut ablation seeds
3→2 or drop the `no_throughput` preset. Per-arm convergence is visible in FIG-3;
extend only a non-converged arm from its 50k checkpoint.

---

## 3. Day-by-Day Schedule (0.5M)

| Day | Date | Compute milestone (32 vCPU / 19 workers) | Writing / artifacts |
|---|---|---|---|
| **D1** | Mon 16/6 | Launch Stage 1 (19w) + Stage 3 sublane. (optional) measure `class_flip` (§4). | Literature Group 1 + 3. |
| **D2** | Tue 17/6 | Stage 1 running. **Baselines eval** (Webster/Actuated/MaxP, no training) N2+N3 +`--emissions`. | Write §II Related Work + TABLE-1 (real cites); polish §I. |
| **D3** | Wed 18/6 | Stage 1 running. | Write §III–V; III-E overview prose; VI-E TABLE-3a (documented-basis σ + provenance); FIG-1 (arch), FIG-2 (net+demand). |
| **D4** | Thu 19/6 | Stage 1 finishing (~Day 4–5). | Writing buffer; literature Group 2/4/5. |
| **D5** | Fri 20/6 | **MAIN RESULTS in.** Launch Stage 2 (12+7w). | `eval_compare`→TABLE-4/5/6; `campaign_figures`→FIG-3/4/5; **redo noise sweep + investigate anomaly**→FIG-6; write VI-A/B/C; fill Abstract X/Y/Z/W%. |
| **D6** | Sat 21/6 | Stage 2 running. | OOD/moto eval→TABLE-7/FIG-6a/VI-D; emissions→TABLE-10/VI-G. |
| **D7** | Sun 22/6 | Stage 2 finishing. | Self-review prose; figure polish. |
| **D8** | Mon 23/6 | **ABLATIONS in.** ⚠ if slipping → **24/6 fallback** (3→2 seeds / drop `no_throughput`). | Eval ablations→TABLE-8/9; write VI-F; first complete draft. |
| **D9** | Tue 24/6 | — | Draft self-review (reviewer-attack pass §7). |
| **D10** | Wed 25/6 | — | IEEE LaTeX formatting; import .tex tables, embed PDF figures; ≤14 pages. |
| **D11** | Thu 26/6 | — | LaTeX + proofread (style, acronyms, no overclaim). |
| **D12** | Fri 27/6 | — | Pre-submission gate (§8); cover letter; supplementary; **repo freeze + `git tag v1.0-submission`**. |
| **D13** | Sat 28/6 | — | **SUBMIT** (ScholarOne) + confirm acknowledgment. |

Writing (D2–3) does **not** depend on results — front-load it while compute runs.

---

## 4. VI-E Noise Parameterization — TABLE-3a (Day 1, mostly no compute)

**III-E is overview-only prose** (defines the projection perturbation ε); the
numbers live in **VI-E / TABLE-3a**. The base (1×) σ magnitudes are on a
**documented basis** (state.md + ByteTrack/YOLOv11 literature) — **writable now,
no compute**. **No FIG-1a**; measured feature-recovery is **deferred to the
companion study**.

**Optional** (strengthens only the `class_flip` row of TABLE-3a — *not* a submit gate):
```bash
python scripts/dump_confusion_matrix.py \
  --model models/yolo/yolov11.pt --data data/dataset/data.yaml --out confusion_matrix.csv
python experiment/runners/calibrate_noise.py \
  --confusion-matrix confusion_matrix.csv --set occupancy_bias_sigma=0.08
# -> configs/noise_config.json  (sets calibrated=true)
```
> Path note: use `models/yolo/yolov11.pt` + `data/dataset/data.yaml` (present);
> the original plan's `runs/detect/train/weights/best.pt` does not exist. If run,
> `class_flip_rate` becomes **measured** while queue/occupancy/speed σ stay
> documented-basis (state.md) — TABLE-3a must label provenance per row honestly
> either way.

---

## 5. Noise Anomaly — Investigate, do not assume away

Existing 5-seed sweep on the 500k mid-train ckpt is **non-monotonic both ways**:
`scale_0=323s > scale_0.5=300s` and `scale_1=372s > scale_2=349s`. "Redo as
5-seed" is **not** the fix (already 5-seed). On Day 4:
1. Re-run on the converged 0.5M campaign checkpoints (5 seeds).
2. If anomaly persists: check (a) per-condition eval demand seed parity, (b)
   noise-as-regularization, (c) eval variance. **Report honestly** in VI-E /
   footnote — a persistent effect is a finding, not a bug to hide.

Defends reviewer **Q3** (the single most dangerous attack on this paper).

---

## 6. Deliverables Checklist

### 🔴 Critical (cannot submit without)
- [ ] C1 TABLE-4/5/6 filled (`eval_compare`→`make_tables --auto`)
- [ ] C2 MAPPO proxy 5-seed N2+N3
- [ ] C3 MAPPO privileged 5-seed N2+N3 → TABLE-6 restriction cost
- [ ] C4 IPPO 5-seed N2+N3
- [ ] C5 Webster / Actuated / MaxPressure evaluated
- [ ] C6 Abstract X/Y/Z/W% filled (traceable to tables)
- [ ] C7 III-E overview prose (sensing-noise model / ε definition only — numbers live in VI-E)
- [ ] C8 ~~FIG-1a~~ DROPPED — measured feature-recovery deferred to companion study
- [ ] C9 VI-E TABLE-3a σ magnitudes + provenance labels (documented basis; `class_flip` measurement optional, §4)
- [ ] C10 Noise robustness 5-seed redo + anomaly resolution (FIG-6)
- [ ] C11 TABLE-1 real citation names (no "Vision-RL A/B")
- [ ] C12 Related Work prose, ≥15 real cites
- [ ] C13 VI-A prose · C14 VI-B + FIG-3 · C15 VI-C restriction-cost prose
- [ ] C16 TABLE-9 sublane-vs-lane cross-eval
- [ ] C17 TABLE-8 reward ablations retrained (≥3 seeds, no eval-time reweighting)
- [ ] C18 Conclusion filled · C19 Vietnamese fleet citation (Le et al.) · C20 repo URL + tagged commit

### 🟡 Important (raises accept probability)
- [ ] I1 OOD demand TABLE-7 · I2 moto-share sweep FIG-6a (configs already exist)
- [ ] I3 FIG-4 entropy/KL · I4 FIG-5 comparison bars · I7 significance ★ · I8 Cohen's d
- [ ] I5 TABLE-10 CO₂/fuel (`--emissions`) · I6 VI-G prose
- [ ] I9 FIG-2 network panel · I10 saturation-flow cite · I11/I12 obs ablations · I13 companion footnote · I14 wall-clock cost reported

### Already complete
TABLE-2, TABLE-3, TABLE-V-A1 (max-green pilot), Dec-POMDP/φ/reward/MAPPO math,
ALGORITHM-1, demand_profile.pdf. OOD/moto/lanebased `.sumocfg` already generated.
SOTL dropped (not in skeleton).

---

## 7. Reviewer Attack Map — Top Risks (full list in original plan §7)

| Q | Risk | Defense |
|---|---|---|
| **Q1** TABLE-1 placeholder names | FATAL | Phase-5 lit review → real papers (2 camera-state/privileged-reward rows) |
| **Q2** "camera-observable" asserted not shown | Critical | III-B/III-E noise model + VI-E TABLE-3a (documented-basis σ + provenance); measured feature-recovery scoped to companion study (no FIG-1a) |
| **Q3** noise non-monotonic (0.5×<0×) | Critical | §5 investigation on converged ckpts; report honestly |
| **Q4** only 2 synthetic nets | High | State real-map deferred to companion; N3=16-junction realistic scale |
| **Q5** significance vs MaxP? | High | MWU p-values + ★ + Cohen's d in TABLE-4/5 |
| Q6 signed-pressure guarantees | High | Already: "reactive heuristic gradient, not throughput-optimal" |
| Q17 max-green=60 1-seed | Low | TABLE-V-A1 design-choice framing, applied to all methods |

---

## 8. Pre-Submission Gate

- [ ] Every Abstract number traceable to a table
- [ ] 5 seeds × 10 eval seeds everywhere (3 seeds for ablations, stated)
- [ ] Ablations retrained, **not** eval-time reweighted
- [ ] No "novel / first / sim-to-real" overclaim; count-vs-PCU sentence in V-A
- [ ] III-E overview present; VI-E TABLE-3a σ on documented basis + honest provenance (measured feature-recovery deferred to companion)
- [ ] VI-C labeled "information-restriction cost"; VI-C & VI-E both present
- [ ] Companion study = footnote, not field-validation claim
- [ ] Repo frozen at tagged commit; schema version 1.2.0 stamped in run_config
- [ ] Figures ≥8pt, column widths (3.45in / 7.16in), captions present

---

*Maintainers: update checkboxes as items land. Next decision point: Day 4 (main results).*
