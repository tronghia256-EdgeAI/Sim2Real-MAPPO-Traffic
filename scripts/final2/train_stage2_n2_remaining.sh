#!/usr/bin/env bash
# train_stage2_n2_remaining.sh — after GATE PASS, fan out the 4 remaining N2 seeds.
#
# Same campaign dir as the gate (final2_n2_03M). The gate launcher (seed 42)
# keeps running independently; note campaign_manifest.json is rewritten by
# whichever launcher updated last — judge job state by tblogs/bench.json, not
# by the manifest, while both launchers are alive (cosmetic only).
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS_N2="${WORKERS_N2:-4}"

echo "[launch] final2_n2_03M remaining seeds 123 456 789 1337 (workers=${WORKERS_N2})"
nohup python scripts/parallel_launcher.py \
    --campaign-id final2_n2_03M \
    --networks n2_corridor --algos mappo --obs-modes proxy \
    --seeds 123 456 789 1337 \
    --total-timesteps 300000 --max-workers "${WORKERS_N2}" \
    -- --no-libsumo \
    > results/paper1_mappo/final2_launchlogs/n2_rest.launcher.log 2>&1 &
echo "  launcher pid $!"
