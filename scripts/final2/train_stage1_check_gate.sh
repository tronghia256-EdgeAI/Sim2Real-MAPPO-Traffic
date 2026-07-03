#!/usr/bin/env bash
# train_stage1_check_gate.sh — N2 go/no-go gate.
#
# Run after the n2 seed-42 job has passed ~150k steps (memory: n3 converged by
# ~100-150k; corridor should show its trend by then). Evaluates the CURRENT
# best_model.pt of the gate job against max-pressure / webster / fixed on the
# held-out n2 eval scenario (3 route seeds, fast — n2 episodes are small).
#
# PASS = MAPPO mean travel time < max-pressure  ->  bash train_stage2_n2_remaining.sh
# FAIL = corridor coordination flags did not rescue N2 -> do NOT spend 4 seeds;
#        fall back to grid-only headline + corridor-as-limitation (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

GATE_DIR=results/paper1_mappo/final2_n2_03M/models/n2_corridor_mappo_proxy_seed42
ls "${GATE_DIR}"/*/best_model.pt >/dev/null 2>&1 || { echo "no best_model.pt under ${GATE_DIR} yet"; exit 1; }

echo "=== gate eval (n2, seeds 42 123 456, ~15-30 min) ==="
python experiment/runners/eval_compare.py \
    --network n2_corridor \
    --checkpoint "${GATE_DIR}" \
    --methods mappo maxpressure webster fixed \
    --seeds 42 123 456

LATEST=$(ls -dt results/paper1_mappo/eval_tables/n2_corridor_* | head -1)
echo ""
echo "=== verdict (from ${LATEST}/table.csv) ==="
python - "$LATEST" <<'EOF'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(Path(sys.argv[1]) / "table.csv", encoding="utf-8")))
tt = {r["method"]: float(r["mean"]) for r in rows
      if r.get("metric") == "mean_travel_time_s"}
print("mean travel time (s):", {k: round(v, 1) for k, v in tt.items()})
m, mp = tt.get("mappo"), tt.get("maxpressure")
if m is None or mp is None:
    print("could not read mappo/maxpressure rows — inspect table.csv manually")
    sys.exit(2)
if m < mp:
    print(f"\nGATE PASS: mappo {m:.1f}s < maxpressure {mp:.1f}s "
          f"-> bash scripts/final2/train_stage2_n2_remaining.sh")
else:
    print(f"\nGATE FAIL: mappo {m:.1f}s >= maxpressure {mp:.1f}s "
          f"-> do NOT launch remaining N2 seeds; see README fallback."
          f" (mid-train checkpoint: consider re-checking near 300k before final verdict)")
    sys.exit(3)
EOF
