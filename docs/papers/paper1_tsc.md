# Paper 1 — TSC (camera-observable MAPPO traffic-signal control)

**Status: ACTIVE.** Title/narrative settled (camera-observable RL, restriction cost measured
in simulation; field deployment = the companion System paper). Full skeleton:
[`docs/paper1_tsc/T-ITS_skeleton.md`](../paper1_tsc/T-ITS_skeleton.md).

## Code (owned)
- `src/traffic_env/`
  - `config.py` — all hyperparameters + obs/reward schema (also the shared env truth).
  - `envs/{base_sumo,multi_agent}.py` — SUMO wrapper + PettingZoo MARL env.
  - `components/{observations,rewards,obs_noise}.py` — 26-dim obs, 6-term reward, VI-E noise model.
- `src/core/policy_loader.py` — checkpoint loader (shared with the System paper).
- `experiment/`
  - `runners/` — `train_ppo` (MAPPO/IPPO + ablation presets), `eval_compare` (unified tripinfo harness),
    `make_tables`, `calibrate_noise`, `eval_noise_robustness`, `eval_ppo_gui`; `_deprecated/{evaluate,compare_traffic_metrics}` (retired).
  - `baselines/` — `webster`, `actuated`, `max_pressure`, `sotl`.
  - `ablation/reward_ablation.py` (eval-time only — VI-F arms are RETRAINED via `train_ppo --reward-preset/--obs-ablation`).
  - `plots/` — `campaign_figures` (multi-seed), `generate_paper_figures` (legacy), `parse_tripinfo`, `plot_training_results`.
  - `common/tripinfo.py` — shared tripinfo parser (travel/waiting/timeloss/p95 + CO2/fuel).
- `scripts/` — `generate_networks`, `generate_demand`, `generate_ood_scenarios` (VI-D), `run_pilot`,
  `parallel_launcher` (campaign + ablation fan-out), `build_paper_artifacts` (turnkey eval→tables→figures),
  `dump_confusion_matrix` (III-E, shared with P2), `check_obs_match`, `generate_sumo_scenarios`.

## Artifacts
- `sumo_configs/{networks,training,evaluation}/` — N1/N2/N3 nets, demand, OOD + moto + lane-based scenarios.
- `models/mappo/`, `logs/` — checkpoints + training logs (**live writes during the campaign**).
- `results/paper1_mappo/` — eval tables, pilots, noise sweeps, III-E feature recovery.
- `figures/` — paper PDFs (`campaign_figures`/`make_tables` output).
- `configs/state_config.json` (shared), `configs/noise_config.json` (from `calibrate_noise`).

## Run order (post-training)
See [`docs/paper1_tsc/POST_TRAINING_RUNBOOK.md`](../paper1_tsc/POST_TRAINING_RUNBOOK.md):
`parallel_launcher` → `build_paper_artifacts` → (`dump_confusion_matrix`+`calibrate_noise`) → `eval_noise_robustness`.

## Remaining (human / compute)
- Run the full multi-seed campaign (wall-clock bottleneck ~2 SPS on the 16-agent grid).
- III-E gt/est measurement input; prose, citations (V-A), architecture diagram (FIG-1), ALGORITHM-1.
