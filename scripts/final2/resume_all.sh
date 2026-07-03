#!/usr/bin/env bash
# resume_all.sh — restart all three campaigns after a reboot / spot preemption.
#
# Safe to run any time: parallel_launcher skips jobs whose run dir already has
# bench.json (completed), and train_ppo auto-resumes an interrupted job from
# the newest last_model.pt under its --ckpt-dir (resume segments are stitched
# by campaign_figures at plot time).
#
# For a SPOT VM, add to crontab (crontab -e):
#   @reboot sleep 60 && cd /home/<user>/Sim2Real-MAPPO-Traffic && bash scripts/final2/resume_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS_MAIN="${WORKERS_MAIN:-10}"
WORKERS_OBS="${WORKERS_OBS:-9}"
WORKERS_N2="${WORKERS_N2:-5}"

mkdir -p results/paper1_mappo/final2_launchlogs

nohup python scripts/parallel_launcher.py --campaign-id final2_main_ext \
    --networks n3_grid --algos mappo ippo --obs-modes proxy \
    --seeds 2024 2025 2026 2027 2028 \
    --total-timesteps 500000 --max-workers "${WORKERS_MAIN}" \
    -- --no-libsumo \
    >> results/paper1_mappo/final2_launchlogs/main_ext.launcher.log 2>&1 &

nohup python scripts/parallel_launcher.py --campaign-id final2_obsabl_03M \
    --networks n3_grid --algos mappo --obs-modes proxy \
    --obs-ablations no_class_shares no_pressure_feature lane_truncated \
    --seeds 42 123 456 \
    --total-timesteps 300000 --max-workers "${WORKERS_OBS}" \
    -- --no-libsumo \
    >> results/paper1_mappo/final2_launchlogs/obsabl.launcher.log 2>&1 &

# includes seed 42: if it already finished, bench.json makes it a no-op skip.
# ONLY resume n2 with all 5 seeds if the gate PASSED (otherwise keep --seeds 42).
nohup python scripts/parallel_launcher.py --campaign-id final2_n2_05M \
    --networks n2_corridor --algos mappo --obs-modes proxy \
    --seeds 42 123 456 789 1337 \
    --total-timesteps 500000 --max-workers "${WORKERS_N2}" \
    -- --no-libsumo \
    >> results/paper1_mappo/final2_launchlogs/n2.launcher.log 2>&1 &

echo "all three campaign launchers resumed."
