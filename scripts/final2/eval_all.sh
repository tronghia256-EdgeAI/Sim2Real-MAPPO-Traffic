#!/usr/bin/env bash
# eval_all.sh — run after ALL training campaigns finished (bench.json present
# for every job). Produces the paper's definitive tables:
#
#   A) N3 main, n=10 x 10       mappo(10 ckpt = 5 old + 5 new) vs ippo(10) vs
#                               privileged(5 old) vs 5 classical baselines,
#                               10 route seeds, --emissions (Table I + TABLE-10)
#   B) N2 corridor, n=5 x 10    mappo(5 new coord ckpts) vs 5 classical baselines
#   C) Obs ablation, 3 x 3      3 arms x 3 ckpts x 3 route seeds (mirrors the
#                               reward-ablation Table VII protocol)
#
# Wall-clock: A ~15-24h (long pole), B ~2-4h, C ~2-3h; A/B/C run concurrently.
# After training you can DOWNSIZE the VM (e.g. 8 vCPU) before running this —
# eval is 3 sequential single-core pipelines.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

ROUTE_SEEDS="42 123 456 789 1337 2024 2025 2026 2027 2028"
ABL_SEEDS="42 123 456"
CLEAN=results/paper1_mappo/ckpts_clean
LOGS=results/paper1_mappo/final2_evallogs
mkdir -p "${LOGS}"

echo "=== [1/4] collect clean checkpoints (bench.json run dir per job) ==="
python scripts/final2/collect_checkpoints.py --require-bench \
    --models-dir results/paper1_mappo/final2_main_ext/models   --out "${CLEAN}/final2_main_ext"
python scripts/final2/collect_checkpoints.py --require-bench \
    --models-dir results/paper1_mappo/final2_obsabl_03M/models --out "${CLEAN}/final2_obsabl_03M"
python scripts/final2/collect_checkpoints.py --require-bench \
    --models-dir results/paper1_mappo/final2_n2_05M/models     --out "${CLEAN}/final2_n2_05M"

echo "=== [2/4] checkpoint counts ==="
count() { ls $1 2>/dev/null | wc -l; }
N_MAPPO=$(( $(count "${CLEAN}/old_main/n3_grid_mappo_proxy_seed*/best_model.pt") \
          + $(count "${CLEAN}/final2_main_ext/n3_grid_mappo_proxy_seed*/best_model.pt") ))
N_IPPO=$((  $(count "${CLEAN}/old_main/n3_grid_ippo_proxy_seed*/best_model.pt") \
          + $(count "${CLEAN}/final2_main_ext/n3_grid_ippo_proxy_seed*/best_model.pt") ))
N_PRIV=$(count "${CLEAN}/old_main/n3_grid_mappo_privileged_seed*/best_model.pt")
N_N2=$(count "${CLEAN}/final2_n2_05M/n2_corridor_mappo_proxy_seed*/best_model.pt")
echo "mappo=${N_MAPPO}/10  ippo=${N_IPPO}/10  privileged=${N_PRIV}/5  n2=${N_N2}/5"
[ "${N_MAPPO}" -eq 10 ] || { echo "FATAL: expected 10 mappo checkpoints"; exit 1; }
[ "${N_IPPO}"  -eq 10 ] || { echo "FATAL: expected 10 ippo checkpoints"; exit 1; }
[ "${N_PRIV}"  -eq 5 ]  || { echo "FATAL: expected 5 privileged checkpoints"; exit 1; }
[ "${N_N2}"    -ge 1 ]  || echo "WARN: no n2 checkpoints — eval B will be skipped"

echo "=== [3/4] pooling parity (old vs new seeds MUST match exactly) ==="
python scripts/final2/check_run_config_parity.py \
    --ref "${CLEAN}/old_main/n3_grid_mappo_proxy_seed42/run_config.json" \
    --others "${CLEAN}/final2_main_ext/n3_grid_mappo_proxy_seed*/run_config.json"
python scripts/final2/check_run_config_parity.py \
    --ref "${CLEAN}/old_main/n3_grid_ippo_proxy_seed42/run_config.json" \
    --others "${CLEAN}/final2_main_ext/n3_grid_ippo_proxy_seed*/run_config.json"

echo "=== [4/4] launching evals (A long pole; tail ${LOGS}/*.log) ==="

# A) N3 main merged, n=10, emissions on
nohup python experiment/runners/eval_compare.py \
    --network n3_grid \
    --checkpoint "${CLEAN}/old_main/n3_grid_mappo_proxy_seed*/best_model.pt" \
                 "${CLEAN}/final2_main_ext/n3_grid_mappo_proxy_seed*/best_model.pt" \
    --ippo-checkpoint "${CLEAN}/old_main/n3_grid_ippo_proxy_seed*/best_model.pt" \
                      "${CLEAN}/final2_main_ext/n3_grid_ippo_proxy_seed*/best_model.pt" \
    --privileged-checkpoint "${CLEAN}/old_main/n3_grid_mappo_privileged_seed*/best_model.pt" \
    --methods mappo ippo privileged webster actuated maxpressure sotl fixed \
    --seeds ${ROUTE_SEEDS} --emissions \
    > "${LOGS}/A_n3_main.log" 2>&1 &
PID_A=$!
echo "  [A] n3 main merged (pid ${PID_A})"

# B) N2 corridor (only if checkpoints exist)
PID_B=""
if [ "${N_N2}" -ge 1 ]; then
    nohup python experiment/runners/eval_compare.py \
        --network n2_corridor \
        --checkpoint "${CLEAN}/final2_n2_05M/n2_corridor_mappo_proxy_seed*/best_model.pt" \
        --methods mappo webster actuated maxpressure sotl fixed \
        --seeds ${ROUTE_SEEDS} \
        > "${LOGS}/B_n2.log" 2>&1 &
    PID_B=$!
    echo "  [B] n2 corridor (pid ${PID_B})"
fi

# C) obs-ablation arms, sequential inside one background shell
nohup bash -c '
    set -e
    CLEAN=results/paper1_mappo/ckpts_clean
    for arm in no_class_shares no_pressure_feature lane_truncated; do
        echo "=== arm ${arm} ==="
        python experiment/runners/eval_compare.py \
            --network n3_grid \
            --checkpoint "${CLEAN}/final2_obsabl_03M/n3_grid_mappo_proxy_o-${arm}_seed*/best_model.pt" \
            --methods mappo \
            --seeds '"${ABL_SEEDS}"'
        newest=$(ls -dt results/paper1_mappo/eval_tables/n3_grid_* | head -1)
        echo "ARM ${arm} -> ${newest}" >> results/paper1_mappo/final2_evallogs/MANIFEST.txt
    done
' > "${LOGS}/C_obsabl.log" 2>&1 &
PID_C=$!
echo "  [C] obs ablations (pid ${PID_C})"

wait ${PID_A} ${PID_B} ${PID_C}
echo ""
echo "ALL EVALS DONE. Output dirs (newest first):"
ls -dt results/paper1_mappo/eval_tables/* | head -8
echo "obs-ablation arm mapping: ${LOGS}/MANIFEST.txt"
echo "Next: bash scripts/final2/pack_results.sh"
