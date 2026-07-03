#!/usr/bin/env bash
# pack_results.sh — verify + archive everything the paper needs, before VM shutdown.
# Produces final2_results_<date>.tar.gz at the repo root; scp it back and extract
# at the local repo root.
#
#   PACK_TB=1 bash scripts/final2/pack_results.sh   # also pack tensorboard logs
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -f scripts/verify_before_shutdown.sh ]; then
    bash scripts/verify_before_shutdown.sh || echo "WARN: verify_before_shutdown reported problems — review before deleting the VM"
fi

OUT="final2_results_$(date +%Y%m%d_%H%M).tar.gz"
LIST=$(mktemp)

# eval tables (the paper numbers) + eval logs + clean checkpoints
find results/paper1_mappo/eval_tables -type f                >> "${LIST}" 2>/dev/null || true
find results/paper1_mappo/final2_evallogs -type f            >> "${LIST}" 2>/dev/null || true
find results/paper1_mappo/ckpts_clean -type f                >> "${LIST}" 2>/dev/null || true

# per-campaign: manifests, job logs, and per-run artifacts needed to reproduce
# figures/tables locally (checkpoints, run_config, bench, CSV logs)
for camp in final2_main_ext final2_obsabl_03M final2_n2_05M; do
    d="results/paper1_mappo/${camp}"
    [ -d "${d}" ] || continue
    find "${d}" -maxdepth 1 -type f                          >> "${LIST}"
    find "${d}/models" -type f \( -name "best_model.pt" -o -name "last_model.pt" \
        -o -name "run_config.json" -o -name "bench.json" -o -name "*.csv" \) >> "${LIST}"
    if [ "${PACK_TB:-0}" = "1" ]; then
        find "${d}/tblogs" -type f                           >> "${LIST}" 2>/dev/null || true
    fi
done

sort -u "${LIST}" -o "${LIST}"
tar -czf "${OUT}" -T "${LIST}"
rm -f "${LIST}"
du -h "${OUT}"
echo "scp this file home, extract at the local repo root, then run make_tables/campaign_figures."
