#!/bin/bash
# Check whether the 3 moto-sweep evals (moto_rerun.sh) are finished. Read-only.
# Run on the VM (pull once, then re-run anytime):
#   cd ~/Sim2Real-MAPPO-Traffic && git pull && bash scripts/final2/moto_check.sh
cd "$(dirname "$0")/../.."
L=results/paper1_mappo/final2_evallogs

echo "== processes =="
if pgrep -af eval_compare >/dev/null; then
  pgrep -af eval_compare | sed 's/^/  RUNNING: /'
else
  echo "  (khong con eval_compare nao dang chay)"
fi

echo
echo "== last line of each log =="
for pct in 50 73 90; do
  f="$L/A2_moto${pct}.log"
  if [ -f "$f" ]; then
    echo "moto${pct}: $(tail -1 "$f")"
  else
    echo "moto${pct}: LOG MISSING ($f)"
  fi
done

echo
echo "== errors =="
if grep -l Traceback "$L"/A2_moto*.log 2>/dev/null | sed 's/^/  TRACEBACK IN: /' | grep .; then
  :
else
  echo "  khong co Traceback"
fi

echo
echo "== finished summaries (mappo must be in aggregates) =="
done_n=0
DIRS=""
for pct in 50 73 90; do
  f="$L/A2_moto${pct}.log"
  d=$(grep -o '[^ ]*summary\.json' "$f" 2>/dev/null | tail -1)
  if [ -n "$d" ] && [ -f "$d" ]; then
    if grep -q '"mappo"' "$d"; then
      echo "  moto${pct}: DONE-OK  $d"
      done_n=$((done_n + 1))
      DIRS="$DIRS ${d%/summary.json}"
    else
      echo "  moto${pct}: summary.json co nhung THIEU mappo -- INVALID: $d"
    fi
  else
    echo "  moto${pct}: chua xong"
  fi
done

echo
if [ "$done_n" -eq 3 ]; then
  echo "ALL 3 DONE -- pack ket qua de tai ve:"
  echo "  cd ~/Sim2Real-MAPPO-Traffic && tar -czf ~/moto_delta.tar.gz$DIRS $L/A2_moto50.log $L/A2_moto73.log $L/A2_moto90.log && ls -lh ~/moto_delta.tar.gz"
else
  echo "$done_n/3 done -- cho them roi chay lai script nay"
fi
