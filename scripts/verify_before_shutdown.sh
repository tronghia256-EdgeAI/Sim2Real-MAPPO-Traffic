#!/usr/bin/env bash
# verify_before_shutdown.sh
# Chạy trên GCP VM sau khi campaign xong.
# Kiểm tra toàn bộ artifacts rồi in GO / NO-GO để pull results + xóa VM.
#
# Usage:
#   bash scripts/verify_before_shutdown.sh
#
# Exit 0 = GO, Exit 1 = NO-GO (có vấn đề)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results/paper1_mappo"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GRN}[OK  ]${NC} $*"; }
warn() { echo -e "${YLW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

ISSUES=0

echo -e "${BLD}"
echo "========================================================"
echo "  PRE-SHUTDOWN VERIFICATION — $(date)"
echo "========================================================"
echo -e "${NC}"

# ── 1. Launchers / training còn chạy? ───────────────────────
echo -e "${BLD}── 1. PROCESSES ─────────────────────────────────────────${NC}"
RUNNING=$(pgrep -fc train_ppo.py 2>/dev/null || echo 0)
LAUNCHERS=$(pgrep -fc parallel_launcher.py 2>/dev/null || echo 0)
if [ "$LAUNCHERS" -gt 0 ] || [ "$RUNNING" -gt 0 ]; then
    warn "Vẫn còn $LAUNCHERS launcher + $RUNNING train_ppo.py đang chạy!"
    warn "Hãy đợi campaign xong trước — bench.json chưa đầy đủ."
    ISSUES=$((ISSUES + 1))
else
    ok "Không còn launcher / training process nào đang chạy."
fi
echo ""

# ── 2. bench.json counts ─────────────────────────────────────
echo -e "${BLD}── 2. BENCH.JSON (hoàn thành) ───────────────────────────${NC}"
TOTAL_BENCH=0
for c in main_05M ablation_reward_05M ablation_obs_05M; do
    COUNT=$(find "$RESULTS/$c" -name bench.json 2>/dev/null | wc -l)
    TOTAL_BENCH=$((TOTAL_BENCH + COUNT))
    echo "    $c : $COUNT"
done
echo "    TOTAL : $TOTAL_BENCH"
echo ""

# ── 3. Failed jobs ───────────────────────────────────────────
echo -e "${BLD}── 3. FAILED JOBS ───────────────────────────────────────${NC}"
FAILED_LINES=()
for log in campaign_main.log camp_2a.log camp_2b.log; do
    [ -f "$ROOT/$log" ] || continue
    while IFS= read -r line; do
        FAILED_LINES+=("$line")
    done < <(grep "FAILED exit" "$ROOT/$log" 2>/dev/null || true)
done
if [ ${#FAILED_LINES[@]} -eq 0 ]; then
    ok "0 job FAILED."
else
    warn "${#FAILED_LINES[@]} job(s) FAILED (lost permanently):"
    for f in "${FAILED_LINES[@]}"; do
        echo "    $f"
    done
fi
echo ""

# ── 4. best_model.pt tồn tại? ────────────────────────────────
echo -e "${BLD}── 4. CHECKPOINT FILES ──────────────────────────────────${NC}"
# Count ALL best_model.pt (mid-run jobs save best_model.pt before bench.json)
ALL_PT=$(find "$RESULTS" -name best_model.pt 2>/dev/null | wc -l)
echo "    best_model.pt hiện có: $ALL_PT"

# When campaign is done: every bench.json dir must have best_model.pt
MISSING_PT=0
FOUND_PT=0
while IFS= read -r bench; do
    RUN_DIR=$(dirname "$bench")
    PT="$RUN_DIR/best_model.pt"
    if [ ! -f "$PT" ]; then
        JOB=$(basename "$(dirname "$RUN_DIR")")
        fail "Missing best_model.pt: $JOB / $(basename "$RUN_DIR")"
        MISSING_PT=$((MISSING_PT + 1))
    else
        FOUND_PT=$((FOUND_PT + 1))
    fi
done < <(find "$RESULTS" -name bench.json 2>/dev/null)

if [ "$TOTAL_BENCH" -eq 0 ] && [ "$RUNNING" -gt 0 ]; then
    # Campaign mid-run: bench.json not yet written — not an error
    ok "Training in progress ($ALL_PT best_model.pt saved so far, bench.json pending)."
elif [ "$MISSING_PT" -eq 0 ] && [ "$FOUND_PT" -gt 0 ]; then
    ok "$FOUND_PT best_model.pt tìm thấy — tất cả completed jobs có checkpoint."
elif [ "$FOUND_PT" -eq 0 ] && [ "$ALL_PT" -eq 0 ]; then
    fail "Không có best_model.pt nào — campaign chưa bắt đầu hoặc bị lỗi?"
    ISSUES=$((ISSUES + 1))
else
    fail "$MISSING_PT jobs thiếu best_model.pt dù có bench.json!"
    ISSUES=$((ISSUES + 1))
fi
echo ""

# ── 5. Spot-check 3 checkpoint (actor + obs_rms) ─────────────
echo -e "${BLD}── 5. CHECKPOINT INTEGRITY SPOT-CHECK ───────────────────${NC}"
SAMPLE_PTS=$(find "$RESULTS" -name best_model.pt 2>/dev/null | shuf | head -3)
if [ -z "$SAMPLE_PTS" ]; then
    warn "Không tìm thấy best_model.pt để kiểm tra."
else
    SPOT_FAIL=0
    while IFS= read -r pt; do
        JOB=$(basename "$(dirname "$(dirname "$pt")")")
        RESULT=$(python3 - "$pt" <<'PYEOF'
import sys, torch, os
path = sys.argv[1]
try:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ck.keys())
    has_actor  = any(k for k in keys if any(x in k for x in ("actor","net.","0.weight")))
    has_obs    = "obs_rms" in ck or "obs_rms_mean" in ck
    ver        = ck.get("run_config", {}).get("env_cfg", {}).get("version", "?")
    status = "OK" if (has_actor and has_obs) else "FAIL"
    print(f"{status}|actor={'Y' if has_actor else 'N'}|obs_rms={'Y' if has_obs else 'N'}|ver={ver}")
except Exception as e:
    print(f"FAIL|error={e}")
PYEOF
)
        STATUS=$(echo "$RESULT" | cut -d'|' -f1)
        DETAIL=$(echo "$RESULT" | cut -d'|' -f2-)
        if [ "$STATUS" = "OK" ]; then
            ok "$JOB — $DETAIL"
        else
            fail "$JOB — $DETAIL"
            SPOT_FAIL=$((SPOT_FAIL + 1))
        fi
    done <<< "$SAMPLE_PTS"
    if [ "$SPOT_FAIL" -gt 0 ]; then
        ISSUES=$((ISSUES + 1))
    fi
fi
echo ""

# ── 6. run_config version == 1.2.0? ─────────────────────────
echo -e "${BLD}── 6. SCHEMA VERSION (run_config.json) ──────────────────${NC}"
BAD_VER=0; CHECKED=0
for cfg in $(find "$RESULTS" -name run_config.json 2>/dev/null | head -30); do
    VER=$(python3 -c "
import json,sys
try:
    d=json.load(open('$cfg'))
    print(d.get('env_cfg',{}).get('version',d.get('version','?')))
except:
    print('parse_error')
" 2>/dev/null)
    CHECKED=$((CHECKED + 1))
    if [ "$VER" != "1.2.0" ]; then
        warn "$(basename "$(dirname "$cfg")"): version=$VER"
        BAD_VER=$((BAD_VER + 1))
    fi
done
if [ "$CHECKED" -eq 0 ]; then
    warn "Không tìm thấy run_config.json nào."
elif [ "$BAD_VER" -eq 0 ]; then
    ok "Tất cả $CHECKED run_config.json kiểm tra đều version=1.2.0."
else
    fail "$BAD_VER / $CHECKED config có version sai!"
    ISSUES=$((ISSUES + 1))
fi
echo ""

# ── 7. Kích thước artifacts ──────────────────────────────────
echo -e "${BLD}── 7. RESULTS SIZE ──────────────────────────────────────${NC}"
echo "    results/paper1_mappo/ : $(du -sh "$RESULTS" 2>/dev/null | cut -f1)"
echo "    best_model.pt count   : $(find "$RESULTS" -name best_model.pt 2>/dev/null | wc -l)"
echo "    tfevents count        : $(find "$RESULTS" -name '*tfevents*' 2>/dev/null | wc -l)"
echo "    bench.json total      : $(find "$RESULTS" -name bench.json 2>/dev/null | wc -l)"
if [ -d "$ROOT/figures" ]; then
    echo "    figures/              : $(du -sh "$ROOT/figures" 2>/dev/null | cut -f1)"
fi
echo ""

# ── 8. Lệnh pull về local ────────────────────────────────────
echo -e "${BLD}── 8. PULL COMMANDS (chạy từ máy LOCAL) ────────────────${NC}"
IP=$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo "<IP>")
echo "    scp -r ubuntu@${IP}:~/Sim2Real-MAPPO-Traffic/results/paper1_mappo ./results_cloud"
echo "    scp -r ubuntu@${IP}:~/Sim2Real-MAPPO-Traffic/figures ./figures_cloud"
echo "    (hoặc commit + push trực tiếp trên VM nếu muốn)"
echo ""

# ── VERDICT ──────────────────────────────────────────────────
echo "========================================================"
if [ "$ISSUES" -eq 0 ]; then
    echo -e "${GRN}${BLD}  VERDICT: ✓ GO${NC}"
    echo -e "  An toàn kéo results về + DELETE VM."
else
    echo -e "${RED}${BLD}  VERDICT: ✗ NO-GO — $ISSUES vấn đề cần xử lý trước.${NC}"
fi
echo "========================================================"

exit $ISSUES
