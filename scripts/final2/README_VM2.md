# FINAL2 campaign — N2 retrain + obs ablation + n=10 significance extension

Runbook for the second cloud VM. Goal: fill the last three experimental holes
of Paper 1 and convert the n=5 "large-effect trends" into Holm-certified
p < 0.05 results.

## Why n=5 could never reach p < 0.05 (the math)

With 5 training seeds vs 5 baseline samples, the exact two-sided Mann-Whitney
minimum is p = 2/C(10,5) = 0.0079 **even with perfect separation**. Holm across
the 7-method family multiplies the smallest p by 7 → floor ≈ 0.056 > 0.05.
The observed best (mappo vs fixed, p_holm = 0.079) sat at that floor: the
result was *mathematically uncertifiable*, not weak. With 10 training seeds ×
10 route seeds the floor drops to ~7.6e-5, and the observed effect sizes
(|d| ≥ 1.13 on every deployable baseline) give a realistic shot at
certification for fixed / SOTL / Webster / max-pressure / IPPO.

**Honest warning:** more power cuts both ways. The actuated comparison
(d = +1.83 *in actuated's favor*) will likely ALSO certify at n=10 — the
"statistically indistinguishable from actuated" sentence in the abstract/sec6
may have to become "a certified single-digit concession to a loop-instrumented
reference". The instrumented-reference framing already absorbs this; only the
wording changes. Do not be surprised by it.

## Job matrix (14 jobs; +10 optional)

> **ALL steps = 300k.** Every existing checkpoint (proxy / privileged / ippo /
> reward ablation) was trained to `total_timesteps=300000` — the old
> `final_main_05M` dir name is a misnomer. `check_run_config_parity.py`
> compares `total_timesteps`, so a 0.5M arm can never be pooled with them.

| Campaign id        | Arm                                            | Jobs | Steps |
|--------------------|------------------------------------------------|------|-------|
| `final2_obsabl_03M`| n3_grid × mappo × {no_class_shares, no_pressure_feature, lane_truncated} × seeds 42/123/456 | 9 | 0.3M |
| `final2_n2_03M`    | n2_corridor × mappo × seeds 42/123/456/789/1337 (auto `--upstream-phase-obs --shared-reward`) | 5 | 0.3M |
| `final2_main_ext`  | OPTIONAL (`RUN_MAIN_EXT=1`): n3_grid × {mappo, ippo} × seeds 2024–2028 | 10 | 0.3M |

The n=10 significance extension is now opt-in: current strategy replaces it
with two eval-time protocols run LOCALLY against the existing n=5 checkpoints
(`scripts/final2/eval_tricks_local.ps1`): OOD-high-demand comparison (VI-D)
and the decision-frequency boost (`eval_compare --learned-step-length 3`).

Privileged stays n=5 (old seeds only): its claims (restriction cost 3.2%,
descriptive) don't need certification. Optional extension if a reviewer
demands it: +5 privileged seeds ≈ 5 × ~50h extra.

## VM configuration (recommendation)

- **Machine:** GCP `c2d-highcpu-56` (56 vCPU / 112 GB) — all 24 jobs run
  unthrottled (traci job ≈ 1 python + 1 sumo ≈ 1.7–2 vCPU). Budget option:
  `c2d-highcpu-32` with `WORKERS_MAIN=8 WORKERS_OBS=6` (≈ +25–30% wall-clock).
- **Disk:** 100 GB pd-balanced. **No GPU** (CPU-only torch, as before).
- **On-demand, not spot,** given deadline + past campaign failures. If you do
  take spot (~65% cheaper), `resume_all.sh` + the `@reboot` crontab line in it
  make preemption survivable (auto-resume from `last_model.pt` is proven).
- **Cost estimate:** 56 vCPU ≈ $1.9–2.1/h × ~2.5 days training ≈ $115–125.
  After training, **downsize to 8 vCPU for the eval day** (evals are 3
  single-core pipelines): ≈ +$8. Total ≈ **$125–140**.
- **Timeline (56 vCPU):** 0.3M jobs ~28–34 h; N2 jobs are much lighter
  (2.8k veh vs 15k) and finish well before the obs-ablation arms. Eval A
  ~8–15 h (n=5) / ~15–24 h (n=10 with `RUN_MAIN_EXT=1`).
  **End-to-end ≈ 2–2.5 days** for the 14-job scope; a 32-vCPU VM suffices
  (WORKERS_OBS=6) if main_ext stays off.

## Order of operations

**LOCAL (Windows, before creating the VM):**
1. `git add`/commit is already done for `scripts/final2/` + the train_ppo.py
   N2 auto-enable — **`git push`** so the VM clone has them.
2. `powershell -File scripts\final2\pack_for_vm.ps1` → `vm2_payload.tar.gz`
   (clean old checkpoints ×15 + n2 demand REGENERATED 2837 veh + n3 demand
   byte-identical to the old campaign, eval = 14888 veh).
3. `scp vm2_payload.tar.gz <user>@<vm>:~/Sim2Real-MAPPO-Traffic/`

**VM:**
4. `git clone <repo> && cd Sim2Real-MAPPO-Traffic` (+ scp payload here)
5. `bash scripts/final2/setup_vm.sh` — installs SUMO/venv, **hard-gates** on
   demand integrity (n3 eval = 14888, n2 < 5000), check_obs_match, 2 smoke
   runs incl. the "N2 Corridor detected" auto-flag print. Record
   `sumo --version` for the sec5 TODO.
6. `bash scripts/final2/train_stage1_gate_n2.sh` — starts the 9 obs-ablation
   jobs + the n2 seed-42 gate job (add `RUN_MAIN_EXT=1` for the 10 extension
   jobs; parity vs old seeds is then auto-checked after 7 min — kill early if
   it screams).
7. After ~6–10 h (n2 seed-42 ≥ ~150k steps):
   `bash scripts/final2/train_stage1_check_gate.sh`
   - **PASS** → `bash scripts/final2/train_stage2_n2_remaining.sh`
   - **FAIL** → do not spend 4 seeds. Paper fallback: keep the grid as the
     headline (already how sec6 is written), report the corridor as a
     coordination-limitation result; C4/abstract wording drops the corridor
     improvement claim. Re-check once near 300k before final verdict.
8. After ALL bench.json present: `bash scripts/final2/eval_all.sh`
   (evals A/B/C in parallel; A = merged n=10 Table I with `--emissions`).
9. `bash scripts/final2/pack_results.sh` → scp home → extract at local repo
   root → `python experiment/runners/make_tables.py --auto` +
   `python experiment/plots/campaign_figures.py --comparison`.

## Eval protocol changes vs the old MERGED table

- Route seeds 5 → **10** (42 123 456 789 1337 2024 2025 2026 2027 2028):
  baselines get n=10 samples; learned sample stays = checkpoint mean.
- MAPPO/IPPO n = **10** checkpoints (5 old + 5 new); privileged n = 5.
- Everything (old ckpts, new ckpts, all baselines) re-runs on ONE SUMO
  version on this VM → the new table is internally consistent and REPLACES
  the old MERGED numbers wholesale.
- Poolability is enforced twice: `check_run_config_parity.py` at launch
  (+7 min) and again in `eval_all.sh` before any episode runs.

## Post-eval paper edits (local)

- Refill headline macros in `bare_jrnl.tex` (\ImproveFixed, \ImproveStrongest,
  \RetainPriv, \RestrictionCost) from the new table.csv.
- Table I: n=10 numbers + † markers for p_holm < 0.05; caption: ten route
  seeds, ten training seeds (privileged: five).
- sec5 statistical protocol: "five" → "ten" (both seed kinds), keep the
  power caveat only for the privileged column.
- sec7 limitation #3 (n=5 underpowered): rewrite or delete depending on how
  many comparisons certify.
- If actuated certifies in its favor: adjust "statistically indistinguishable"
  → quantified-concession wording (abstract, C4, sec6-A, sec8).
- N2 gate PASS: fill the sec6 N2 block + Fig. 4 N2 panel + sec4 corridor
  coordination paragraph. FAIL: rescope C4/abstract to grid-only.
- Obs-ablation table (VI-F) from the three arm dirs in
  `final2_evallogs/MANIFEST.txt` (protocol identical to reward Table VII, n=3,
  descriptive — no significance claims at n=3).
