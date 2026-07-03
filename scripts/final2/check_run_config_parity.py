"""
check_run_config_parity.py
==========================
Verify that checkpoints trained in DIFFERENT campaigns are statistically
poolable: their run_config.json must be identical except for run-identity
fields (seed, run_id, absolute paths, timestamps).

Pooling old (seeds 42..1337) + new (seeds 2024..2028) training seeds into one
n=10 sample is only valid if env version, reward weights, obs schema, timing,
and PPO hyperparameters all match exactly. This script fails loudly if not.

Usage:
    python scripts/final2/check_run_config_parity.py \
        --ref    results/paper1_mappo/ckpts_clean/old_main/n3_grid_mappo_proxy_seed42/run_config.json \
        --others "results/paper1_mappo/ckpts_clean/final2_main_ext/n3_grid_mappo_proxy_seed*/run_config.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# leaf keys that legitimately differ between runs of the same arm
IGNORED_LEAVES = {
    "run_id", "seed", "gui", "debug", "sumo_cfg_path", "lane_groups_file",
    "ckpt_dir", "log_dir", "run_dir", "device", "hostname", "started_at",
    "created_at", "timestamp", "resumed_from", "tb_dir",
}


def flatten(obj, prefix="") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        out[prefix.rstrip(".")] = json.dumps(obj)
    else:
        out[prefix.rstrip(".")] = obj
    return out


def compare(ref_path: Path, other_path: Path) -> list[str]:
    ref = flatten(json.loads(ref_path.read_text(encoding="utf-8")))
    oth = flatten(json.loads(other_path.read_text(encoding="utf-8")))
    diffs = []
    for key in sorted(set(ref) | set(oth)):
        leaf = key.rsplit(".", 1)[-1]
        if leaf in IGNORED_LEAVES:
            continue
        a, b = ref.get(key, "<absent>"), oth.get(key, "<absent>")
        if a != b:
            diffs.append(f"    {key}: ref={a!r}  vs  {b!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--others", nargs="+", required=True,
                    help="paths or glob patterns of run_config.json to compare")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"[ERR] ref not found: {ref_path}")
        return 1

    others: list[Path] = []
    for pat in args.others:
        hits = [Path(p) for p in glob.glob(pat)]
        others.extend(hits if hits else [Path(pat)])

    n_bad = 0
    for other in others:
        if not other.exists():
            print(f"[ERR ] missing: {other}")
            n_bad += 1
            continue
        if other.resolve() == ref_path.resolve():
            continue
        diffs = compare(ref_path, other)
        if diffs:
            print(f"[DIFF] {other}")
            print("\n".join(diffs))
            n_bad += 1
        else:
            print(f"[OK  ] {other}")

    if n_bad:
        print(f"\nPARITY FAILED for {n_bad} config(s) — these checkpoints must "
              f"NOT be pooled with the reference arm until explained.")
        return 1
    print("\nPARITY OK — all configs poolable with the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
