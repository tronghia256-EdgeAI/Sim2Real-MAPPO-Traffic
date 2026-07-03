#!/usr/bin/env bash
# setup_vm.sh — one-shot environment setup on a fresh Ubuntu 22.04/24.04 VM.
#
# Usage (after cloning the repo and scp-ing vm2_payload.tar.gz to the repo root):
#     cd ~/Sim2Real-MAPPO-Traffic
#     bash scripts/final2/setup_vm.sh
#
# If shell scripts arrive with CRLF endings, first run:
#     sed -i 's/\r$//' scripts/final2/*.sh scripts/*.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== [1/6] system packages + SUMO (ppa:sumo/stable) ==="
sudo apt-get update -y
sudo apt-get install -y software-properties-common git tmux htop python3-venv python3-pip
sudo add-apt-repository -y ppa:sumo/stable
sudo apt-get update -y
sudo apt-get install -y sumo sumo-tools

export SUMO_HOME=/usr/share/sumo
grep -q "SUMO_HOME" ~/.bashrc || echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
echo "SUMO version (RECORD THIS for the paper's sec5 TODO):"
sumo --version | head -2

echo "=== [2/6] python venv + deps (CPU torch first to avoid the CUDA wheel) ==="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import traci" 2>/dev/null || pip install traci sumolib

echo "=== [3/6] unpack payload (old checkpoints + demand route files) ==="
if [ -f vm2_payload.tar.gz ]; then
    tar -xzf vm2_payload.tar.gz
else
    echo "WARNING: vm2_payload.tar.gz not found at repo root."
    echo "         Training can start without it, but the merged n=10 eval"
    echo "         needs ckpts_clean/old_main and the demand files are REQUIRED."
    exit 1
fi

echo "=== [4/6] demand-file integrity gates ==="
n3_eval=$(grep -c '<vehicle ' sumo_configs/networks/n3_grid/demand_eval.rou.xml)
n2_train=$(grep -c '<vehicle ' sumo_configs/networks/n2_corridor/demand_train.rou.xml)
n2_eval=$(grep -c '<vehicle ' sumo_configs/networks/n2_corridor/demand_eval.rou.xml)
echo "n3 demand_eval vehicles:  ${n3_eval} (MUST be 14888 — the old campaign's exact file)"
echo "n2 demand_train vehicles: ${n2_train} (MUST be ~2837 capped; stale pre-fix file had 11272)"
echo "n2 demand_eval vehicles:  ${n2_eval} (MUST be ~2769 capped)"
[ "${n3_eval}" -eq 14888 ] || { echo "FATAL: n3 eval demand mismatch — do NOT proceed"; exit 1; }
[ "${n2_train}" -lt 5000 ]  || { echo "FATAL: n2 train demand is the STALE gridlock file"; exit 1; }
[ "${n2_eval}" -lt 5000 ]   || { echo "FATAL: n2 eval demand is the STALE gridlock file"; exit 1; }

echo "=== [5/6] obs-consistency check ==="
python scripts/check_obs_match.py

echo "=== [6/6] smoke tests (~15-20 min total) ==="
mkdir -p /tmp/smoke
python experiment/runners/train_ppo.py --mode train --no-libsumo \
    --sumo-cfg sumo_configs/networks/n2_corridor/sumo_config.sumocfg \
    --lane-groups sumo_configs/networks/n2_corridor/lane_groups.json \
    --seed 42 --total-timesteps 2048 \
    --ckpt-dir /tmp/smoke/n2_ckpt --log-dir /tmp/smoke/n2_log 2>&1 | tee /tmp/smoke/n2.log
grep -q "N2 Corridor detected" /tmp/smoke/n2.log \
    || { echo "FATAL: N2 auto-coordination flags did NOT fire — is train_ppo.py up to date?"; exit 1; }
echo "[OK] N2 auto-enable (--upstream-phase-obs + --shared-reward) confirmed."

python experiment/runners/train_ppo.py --mode train --no-libsumo \
    --sumo-cfg sumo_configs/networks/n3_grid/sumo_config.sumocfg \
    --lane-groups sumo_configs/networks/n3_grid/lane_groups.json \
    --seed 42 --total-timesteps 2048 \
    --ckpt-dir /tmp/smoke/n3_ckpt --log-dir /tmp/smoke/n3_log
echo "[OK] n3 smoke passed."

echo ""
echo "SETUP COMPLETE. Next: bash scripts/final2/train_stage1_gate_n2.sh"
