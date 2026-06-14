# Post-Training Runbook — Paper 1

Everything from a finished training campaign to paper-ready tables + figures.
Once the campaign is done you run **two commands** (train → build); steps 3–6
are what `build_paper_artifacts.py` does for you.

> Prereq: `pip install -r requirements.txt`, `SUMO_HOME` set, networks +
> demand generated (`scripts/generate_networks.py`, `scripts/generate_demand.py`).
> Schema/reward version is `1.2.0` (post lane-dedup fix, 2026-06-14) — do **not**
> pool results with the pre-fix pilots.

---

## 0. Pre-flight (once)

```bash
python scripts/check_obs_match.py          # obs consistency (checks 1–4 + 6)
pytest -m "not slow and not serial"        # code health
```

## 1. Train the campaign (the long part)

```bash
# {n2_corridor, n3_grid} x MAPPO x {proxy, privileged} x 5 seeds  (+ IPPO proxy)
python scripts/parallel_launcher.py --algos mappo ippo --max-workers 10
# artifacts: results/paper1_mappo/<campaign_id>/{models,tblogs}/<job>/<run_id>/
# resume after a kill: --campaign-id <campaign_id>
```
Each finished job leaves `bench.json`; the launcher skips those on resume.
`<job> = <network>_<algo>_<obs>_seed<seed>` (e.g. `n3_grid_mappo_proxy_seed42`).

## 2. Build all paper artifacts (one command)

```bash
python scripts/build_paper_artifacts.py            # uses the latest campaign
# or pin it:  --campaign-id campaign_2026XXXX_XXXXXX
# add the sensing-noise sweep (only once NoiseConfig is calibrated, see step 5):
python scripts/build_paper_artifacts.py --noise
```
This discovers per-seed checkpoints and runs steps 3–6 below. Re-run with
`--skip-eval` to rebuild tables/figures from existing evaluations.

---

### 3. Evaluation (what build does) — `eval_compare.py`
Tripinfo-based, multi-seed, per network. Unit of variation for learned methods
is the **training seed** (each checkpoint averaged over the eval route-seeds);
baselines vary over route-seeds. Stats: mean ± 95% CI, Mann-Whitney U,
Holm-Bonferroni, Cohen's d vs `--reference` (default MAPPO proxy).

```bash
python experiment/runners/eval_compare.py \
  --network n3_grid \
  --checkpoint results/paper1_mappo/<camp>/models/n3_grid_mappo_proxy_seed*  \
  --ippo-checkpoint results/paper1_mappo/<camp>/models/n3_grid_ippo_proxy_seed* \
  --privileged-checkpoint results/paper1_mappo/<camp>/models/n3_grid_mappo_privileged_seed* \
  --methods mappo ippo privileged webster actuated maxpressure sotl fixed \
  --seeds 42 123 456 789 1337
# -> results/paper1_mappo/eval_tables/n3_grid_<ts>/{summary.json, table.csv}
```
(`--checkpoint` accepts a file, a directory, or a glob; multiple → per-seed CI.)

### 4. LaTeX tables — `make_tables.py`
```bash
python experiment/runners/make_tables.py --auto
# -> figures/tables/<scenario>_main.tex            (TABLE-4 / TABLE-5)
#    figures/tables/<scenario>_restriction_cost.tex (TABLE-6, if priv arm present)
# \usepackage{booktabs};  \input{figures/tables/n3_grid_main.tex}
```

### 5. Figures — `campaign_figures.py`
```bash
python experiment/plots/campaign_figures.py --all --campaign-id <camp>
# figures/convergence.pdf          reward vs steps, mean±std over seeds (per net)
# figures/training_losses.pdf      policy/value loss + entropy/KL
# figures/comparison_<scen>.pdf    grouped bars per metric, 95% CI
# figures/restriction_cost_<scen>.pdf  proxy vs privileged
# figures/noise_robustness.pdf     travel vs noise scale (needs step 6)
```

### 6a. III-E noise calibration — `calibrate_noise.py` (flips `is_calibrated`→True)
A real detector cannot read SUMO-GUI renders (total domain gap), so calibration
is **source-agnostic**: feed measured numbers, not screenshots.

**Path 1 + 3 (no new labeling — recommended start).** `class_flip_rate` from your
YOLO validation confusion matrix; the rest from documented basis / manual set:
```bash
# (1) dump the confusion matrix from your trained detector's val split
python scripts/dump_confusion_matrix.py \
  --model runs/detect/train/weights/best.pt --data data.yaml --out confusion_matrix.csv
# (2) calibrate: class_flip from the matrix; occupancy assumed/cited (path 3)
python experiment/runners/calibrate_noise.py \
  --confusion-matrix confusion_matrix.csv \
  --set occupancy_bias_sigma=0.08
# -> configs/noise_config.json   (eval_noise_robustness reads it automatically)
```
queue/speed/pressure σ keep their documented basis (docs/state.md: px_per_meter
±10%, ByteTrack ±1 m/s). Provenance per parameter is recorded in the JSON.

**Path 2 (fuller — measures all 5 features).** Two row-aligned CSVs (ground-truth
vs detector-estimated) with columns from `effective_queue_norm, occupancy_norm,
avg_speed_norm, motorbike_share, heavy_vehicle_share`; speed/queue need
track-level GT (MOT/CVAT-track), class/occupancy need only per-frame boxes:
```bash
python experiment/runners/calibrate_noise.py \
  --gt-csv gt.csv --est-csv est.csv --confusion-matrix confusion_matrix.csv
# also writes results/paper1_mappo/iii_e/feature_recovery.csv  (FIG-1a: bias/σ)
```

### 6b. Sensing-noise robustness — `eval_noise_robustness.py`
⚠ Only paper-ready once step 6a has produced `configs/noise_config.json`
(otherwise it runs with placeholders and prints a loud warning).
```bash
python experiment/runners/eval_noise_robustness.py \
  --checkpoint results/paper1_mappo/<camp>/models/n3_grid_mappo_proxy_seed42/<run>/best_model.pt \
  --network n3_grid --seeds 42 123 456 789 1337
# -> results/paper1_mappo/noise_robustness/<ckpt_tag>/{summary.json, sweep.csv}
```

### 7. Ablations (VI-F) — RETRAINED arms (gate: no eval-time reweighting)
Train the ablation arms (reward + observation variants), then eval them. The
launcher names ablation jobs with a suffix so they stay out of the main set.
```bash
# reward ablations (≥3 seeds) on n3, retrained:
python scripts/parallel_launcher.py --networks n3_grid --seeds 42 123 456 \
  --reward-presets full no_pressure no_throughput queue_only unsigned_pressure mean_then_square
# observation ablations:
python scripts/parallel_launcher.py --networks n3_grid --seeds 42 123 456 \
  --obs-ablations no_class_shares no_pressure_feature lane_truncated
# eval one ablation family (point --checkpoint at the suffixed job dirs):
python experiment/runners/eval_compare.py --network n3_grid --methods mappo \
  --checkpoint "results/paper1_mappo/<camp>/models/n3_grid_mappo_proxy_r-no_pressure_seed*"
```
Sublane-vs-lane cross-eval (C4): train a lane-based policy on
`sumo_config_lanebased.sumocfg`, then eval it on the sublane `*_eval.sumocfg`
(and vice versa) — the degradation is the evidence the mixed-traffic model matters.
```bash
python experiment/runners/train_ppo.py --sumo-cfg sumo_configs/networks/n3_grid/sumo_config_lanebased.sumocfg \
  --lane-groups sumo_configs/networks/n3_grid/lane_groups.json --seed 42 --total-timesteps <N>
```

### 8. Generalization (VI-D) — OOD demand + motorcycle-share sweep
```bash
python scripts/generate_ood_scenarios.py            # n2 + n3: OOD + moto sweep + lane-based cfgs
# eval the trained policy on each held-out scenario:
python experiment/runners/eval_compare.py \
  --sumo-cfg sumo_configs/networks/n3_grid/sumo_config_ood_high.sumocfg \
  --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
  --checkpoint "results/paper1_mappo/<camp>/models/n3_grid_mappo_proxy_seed*" --methods mappo webster actuated maxpressure sotl fixed
# moto-share curve (FIG-6a): repeat for sumo_config_moto{50,73,90}.sumocfg
```

### 9. Emissions (VI-G) — post-hoc CO2/fuel
Add `--emissions` to `build_paper_artifacts.py` or `eval_compare.py`; CO2/fuel are
parsed from tripinfo and land in `figures/tables/<scen>_emissions.tex` (TABLE-10).

---

## Paper mapping

| Paper item | Source |
|---|---|
| TABLE-4 / TABLE-5 (main comparison) | `figures/tables/<scen>_main.tex` ← eval_compare |
| TABLE-6 (information-restriction cost) | `figures/tables/<scen>_restriction_cost.tex` |
| FIG-3 convergence, FIG-4 losses/KL | `figures/convergence.pdf`, `figures/training_losses.pdf` |
| VI-A bars, VI-C bars | `figures/comparison_*.pdf`, `figures/restriction_cost_*.pdf` |
| FIG-1a feature recovery (III-E) | `figures/feature_recovery.pdf` + `results/paper1_mappo/iii_e/feature_recovery.csv` |
| FIG-2 demand profile | `figures/demand_profile.pdf` (+ a manual network diagram) |
| FIG-6 noise robustness | `figures/noise_robustness.pdf` (after steps 6a + 6b) |
| TABLE-7 OOD, FIG-6a moto-share | eval_compare on `sumo_config_ood_*` / `sumo_config_moto*` (step 8) |
| TABLE-8/9 ablations, sublane-vs-lane | retrained arms (step 7) → eval_compare |
| TABLE-10 emissions | `figures/tables/<scen>_emissions.tex` (step 9, `--emissions`) |

## Still needs human work before submission (not automatable)
- **III-E measurement input** — the calibration *tool* exists (step 6a); you still
  produce the gt-vs-estimate CSVs from an annotated clip, or just read your YOLO
  confusion matrix. Detector quality only sets the noise-envelope magnitude — it
  does not gate paper 1 (sim-only); the detector itself is the companion paper.
- **Writing** (I–VIII prose), **TABLE-1 positioning matrix** + literature review.
- **Citations** (V-A): Vietnamese vehicle mix 73/17/7/3, saturation-flow field
  range, sublane vType parameters, count-vs-PCU sentence.
- **FIG-1 architecture diagram** + **ALGORITHM-1** pseudocode (drawn by hand).
- **FIG-2 network diagram** panel (SUMO-GUI / netedit screenshot; the demand-profile panel is auto).

## Deprecated (do not use — proxy metrics)
`experiment/runners/evaluate.py`, `experiment/runners/compare_traffic_metrics.py`
— both print a deprecation notice; superseded by `eval_compare.py`.
