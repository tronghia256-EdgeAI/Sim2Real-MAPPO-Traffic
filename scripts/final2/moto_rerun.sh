#!/bin/bash
# Moto-share sweep RERUN (after the resolved-vtypes fix, commit f143c2f).
# Regenerates per-mix route files, verifies the mix, launches 3 concurrent
# eval_compare runs. Run on the VM from anywhere:
#   cd ~/Sim2Real-MAPPO-Traffic && git pull && bash scripts/final2/moto_rerun.sh
set -e
cd "$(dirname "$0")/../.."

pgrep -f eval_compare >/dev/null && { echo "FATAL: eval_compare dang chay"; exit 1; }

# activate venv if python is not already on PATH
if ! command -v python >/dev/null; then
  ACT=$(find ~ -maxdepth 4 -path "*/bin/activate" -name activate 2>/dev/null | head -1)
  [ -n "$ACT" ] && source "$ACT" && echo "venv: $ACT"
fi
command -v python >/dev/null || { echo "FATAL: khong tim thay python/venv"; exit 1; }
python -c "import numpy, traci" || { echo "FATAL: venv thieu dependency"; exit 1; }

python scripts/generate_ood_scenarios.py --network n3_grid --skip-ood --skip-lanebased

ND=sumo_configs/networks/n3_grid
for pct in 50 73 90; do
  m=$(grep -c 'type="moto"' "$ND/demand_moto${pct}.rou.xml")
  t=$(grep -c '<vehicle ' "$ND/demand_moto${pct}.rou.xml")
  p=$((100 * m / t))
  echo "VERIFY moto${pct}: $m/$t = ${p}%"
  lo=$((pct - 5)); hi=$((pct + 5))
  if [ "$p" -lt "$lo" ] || [ "$p" -gt "$hi" ]; then
    echo "FATAL: moto${pct} mix sai (${p}% vs ${pct}%)"; exit 1
  fi
done

L=results/paper1_mappo/final2_evallogs
mkdir -p "$L"
CKPT="results/paper1_mappo/ckpts_clean/old_main/n3_grid_mappo_proxy_seed*/best_model.pt"
for pct in 50 73 90; do
  nohup python experiment/runners/eval_compare.py \
    --sumo-cfg "$ND/sumo_config_moto${pct}.sumocfg" \
    --lane-groups "$ND/lane_groups.json" \
    --checkpoint "$CKPT" \
    --methods mappo webster actuated maxpressure sotl fixed \
    --seeds 42 123 456 789 1337 \
    --tmp-dir "/tmp/eval_moto${pct}_v2" \
    > "$L/A2_moto${pct}.log" 2>&1 &
  echo "launched moto${pct} pid $!"
done
echo "DONE — theo doi: tail -1 $L/A2_moto*.log"
