"""
collect_checkpoints.py
======================
Select ONE clean checkpoint per campaign job and copy it (+ run_config.json)
into a flat eval-ready layout:  <out>/<job_name>/best_model.pt

Why: crash/resume leaves multiple timestamped run dirs per job. Only the run
dir containing bench.json is a COMPLETED run; earlier dirs hold mid-train
best_model.pt files that must never enter an eval (they silently deflate the
policy). eval_compare's dir-glob (**/best_model.pt) would pick up all of them.

Usage:
    python scripts/final2/collect_checkpoints.py \
        --models-dir results/paper1_mappo/final_main_05M/models \
        --out        results/paper1_mappo/ckpts_clean/old_main \
        [--require-bench]   # exit 1 if any job lacks a completed run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def pick_run_dir(job_dir: Path) -> tuple[Path | None, str]:
    runs = sorted(d for d in job_dir.iterdir() if d.is_dir())
    with_bench = [d for d in runs if (d / "bench.json").exists()
                  and (d / "best_model.pt").exists()]
    if with_bench:
        note = "" if len(with_bench) == 1 else f" (WARN: {len(with_bench)} bench dirs, took latest)"
        return with_bench[-1], f"bench{note}"
    with_best = [d for d in runs if (d / "best_model.pt").exists()]
    if with_best:
        return with_best[-1], "NO-BENCH (incomplete run — latest best_model.pt)"
    return None, "no best_model.pt at all"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-bench", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_root = Path(args.out)
    if not models_dir.is_dir():
        print(f"[ERR] not a directory: {models_dir}")
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok, n_nobench, n_missing = 0, 0, 0
    for job_dir in sorted(d for d in models_dir.iterdir() if d.is_dir()):
        run_dir, status = pick_run_dir(job_dir)
        if run_dir is None:
            print(f"[MISS] {job_dir.name:<48} {status}")
            n_missing += 1
            continue
        dst = out_root / job_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "best_model.pt", dst / "best_model.pt")
        rc = run_dir / "run_config.json"
        if rc.exists():
            shutil.copy2(rc, dst / "run_config.json")
            ver = json.loads(rc.read_text(encoding="utf-8")).get("env_cfg", {}).get("version", "?")
        else:
            ver = "no-run-config!"
        flag = "OK  " if status.startswith("bench") else "WARN"
        if not status.startswith("bench"):
            n_nobench += 1
        else:
            n_ok += 1
        print(f"[{flag}] {job_dir.name:<48} run={run_dir.name} env={ver} ({status})")

    print(f"\ncollected -> {out_root}   complete={n_ok} incomplete={n_nobench} missing={n_missing}")
    if n_missing or (args.require_bench and n_nobench):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
