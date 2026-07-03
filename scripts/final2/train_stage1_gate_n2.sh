#!/usr/bin/env bash
# train_stage1_gate_n2.sh — launch EVERYTHING except the 4 remaining N2 seeds.
#
# Rationale: N2 previously lost to every baseline (gridlocked stale demand +
# no coordination signal). Before spending 4 more seeds we gate on seed 42:
# after ~150k steps (~4-6h) run train_stage1_check_gate.sh; only if MAPPO
# beats max-pressure do we fan out the rest (train_stage2_n2_remaining.sh).
#
# ALL jobs train to 300k, matching the existing checkpoints EXACTLY: every
# old_main run_config.json has total_timesteps=300000 (the "final_main_05M"
# dir name is a misnomer — runs were cut to 300k). check_run_config_parity.py
# compares total_timesteps, so any 0.5M extension seed would fail poolability.
#
# Stages:
#   final2_obsabl_03M : n3_grid x mappo x 3 obs-ablation arms x seeds {42,123,456} @ 0.3M (9 jobs)
#   final2_n2_03M     : n2_corridor x mappo x seed 42 @ 0.3M (gate job)
#   final2_main_ext   : OPTIONAL (RUN_MAIN_EXT=1) n=10 significance extension,
#                       n3_grid x {mappo, ippo} x seeds {2024..2028} @ 0.3M (10 jobs)
#
# Worker budget assumes 56 vCPU. For a 32-vCPU VM use: MAIN=8 OBS=6 (N2 gate stays 1).
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS_MAIN="${WORKERS_MAIN:-10}"
WORKERS_OBS="${WORKERS_OBS:-9}"
RUN_MAIN_EXT="${RUN_MAIN_EXT:-0}"   # 1 = also launch the n=10 extension seeds

mkdir -p results/paper1_mappo/final2_launchlogs

echo "[launch] final2_obsabl_03M (9 jobs @ 0.3M, workers=${WORKERS_OBS})"
nohup python scripts/parallel_launcher.py \
    --campaign-id final2_obsabl_03M \
    --networks n3_grid --algos mappo --obs-modes proxy \
    --obs-ablations no_class_shares no_pressure_feature lane_truncated \
    --seeds 42 123 456 \
    --total-timesteps 300000 --max-workers "${WORKERS_OBS}" \
    -- --no-libsumo \
    > results/paper1_mappo/final2_launchlogs/obsabl.launcher.log 2>&1 &
echo "  launcher pid $!"

echo "[launch] final2_n2_03M gate job (seed 42 only @ 0.3M)"
nohup python scripts/parallel_launcher.py \
    --campaign-id final2_n2_03M \
    --networks n2_corridor --algos mappo --obs-modes proxy \
    --seeds 42 \
    --total-timesteps 300000 --max-workers 1 \
    -- --no-libsumo \
    > results/paper1_mappo/final2_launchlogs/n2_gate.launcher.log 2>&1 &
echo "  launcher pid $!"

if [ "${RUN_MAIN_EXT}" = "1" ]; then
    echo "[launch] final2_main_ext (10 jobs @ 0.3M, workers=${WORKERS_MAIN})"
    nohup python scripts/parallel_launcher.py \
        --campaign-id final2_main_ext \
        --networks n3_grid --algos mappo ippo --obs-modes proxy \
        --seeds 2024 2025 2026 2027 2028 \
        --total-timesteps 300000 --max-workers "${WORKERS_MAIN}" \
        -- --no-libsumo \
        > results/paper1_mappo/final2_launchlogs/main_ext.launcher.log 2>&1 &
    echo "  launcher pid $!"
else
    echo "[skip] final2_main_ext (set RUN_MAIN_EXT=1 to launch the n=10 extension)"
fi

sleep 420  # let jobs create their run dirs + run_config.json

if [ "${RUN_MAIN_EXT}" = "1" ]; then
    echo ""
    echo "=== early parity check: new main seeds vs old campaign (poolability) ==="
    NEW_RC=$(ls results/paper1_mappo/final2_main_ext/models/n3_grid_mappo_proxy_seed*/*/run_config.json 2>/dev/null | head -1 || true)
    if [ -n "${NEW_RC}" ] && [ -d results/paper1_mappo/ckpts_clean/old_main ]; then
        python scripts/final2/check_run_config_parity.py \
            --ref results/paper1_mappo/ckpts_clean/old_main/n3_grid_mappo_proxy_seed42/run_config.json \
            --others "${NEW_RC}" \
        || echo "!!!!!! PARITY MISMATCH — new seeds may NOT be poolable with old ones. INVESTIGATE NOW (cheap to kill, expensive to discover at eval). !!!!!!"
    else
        echo "(run_config not there yet or old_main missing — rerun this check manually in a few minutes)"
    fi
fi

echo ""
echo "=== confirm N2 coordination flags fired ==="
grep -l "N2 Corridor detected" results/paper1_mappo/final2_n2_03M/*.log \
    && echo "[OK] N2 auto-enable confirmed in job log" \
    || echo "(not in log yet — check again shortly: grep 'N2 Corridor detected' results/paper1_mappo/final2_n2_03M/*.log)"

echo ""
echo "All launchers up. Monitor:  tail -f results/paper1_mappo/final2_*/campaign_manifest.json"
echo "Gate check after ~150k n2 steps (~4-6h):  bash scripts/final2/train_stage1_check_gate.sh"
