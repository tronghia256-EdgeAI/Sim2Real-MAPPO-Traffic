from __future__ import annotations

"""
build_paper_artifacts.py
========================
One command to turn a finished training campaign into all paper deliverables:
discover per-seed checkpoints -> run the unified tripinfo evaluation per network
-> emit LaTeX tables -> render the multi-seed figures. Optionally runs the
sensing-noise robustness sweep.

Each step is launched as an isolated subprocess (so one network's SUMO failure
cannot abort the rest) and is resumable: pass --skip-eval to only (re)build
tables/figures from existing results/paper1_mappo/eval_tables/.

Pipeline
--------
  1. discover  results/paper1_mappo/<campaign>/models/<net>_<algo>_<obs>_seed*/<run>/best_model.pt
  2. eval      experiment/runners/eval_compare.py  (per network; MAPPO proxy +
               IPPO + privileged when present, vs Webster/actuated/max-pressure/
               SOTL/fixed)   -> results/paper1_mappo/eval_tables/<net>_<ts>/
  3. tables    experiment/runners/make_tables.py --auto -> figures/tables/*.tex
  4. figures   experiment/plots/campaign_figures.py --all -> figures/*.pdf
  5. (opt)     experiment/runners/eval_noise_robustness.py per network proxy ckpt

Usage (from project root, after parallel_launcher.py finishes):
    python scripts/build_paper_artifacts.py                       # latest campaign
    python scripts/build_paper_artifacts.py --campaign-id campaign_2026...
    python scripts/build_paper_artifacts.py --seeds 42 123 456 789 1337 --noise
    python scripts/build_paper_artifacts.py --skip-eval           # tables+figures only
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper1_mappo"
PY = sys.executable

EVAL_COMPARE = ROOT / "experiment" / "runners" / "eval_compare.py"
MAKE_TABLES = ROOT / "experiment" / "runners" / "make_tables.py"
CAMPAIGN_FIGS = ROOT / "experiment" / "plots" / "campaign_figures.py"
NOISE_EVAL = ROOT / "experiment" / "runners" / "eval_noise_robustness.py"

BASELINE_METHODS = ["webster", "actuated", "maxpressure", "sotl", "fixed"]


def latest_campaign() -> Optional[Path]:
    cands = sorted(RESULTS.glob("campaign_*"))
    return cands[-1] if cands else None


def discover_checkpoints(campaign_dir: Path, network: str, algo: str, obs: str) -> List[Path]:
    """best_model.pt for every seed of one arm, in seed order."""
    models = campaign_dir / "models"
    found: List[Path] = []
    for job_dir in sorted(models.glob(f"{network}_{algo}_{obs}_seed*")):
        ck = sorted(job_dir.glob("**/best_model.pt"))
        if ck:
            found.append(ck[0])
    return found


def _run(cmd: List[str], label: str) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"[{label}] FAILED (exit {rc})")
        return False
    return True


def run_eval_for_network(campaign_dir: Path, network: str, args: argparse.Namespace) -> bool:
    mappo = discover_checkpoints(campaign_dir, network, "mappo", "proxy")
    ippo = discover_checkpoints(campaign_dir, network, "ippo", "proxy")
    priv = discover_checkpoints(campaign_dir, network, "mappo", "privileged")

    methods: List[str] = []
    if mappo:
        methods.append("mappo")
    if ippo:
        methods.append("ippo")
    if priv:
        methods.append("privileged")
    methods += BASELINE_METHODS

    if not (mappo or ippo or priv):
        print(f"[eval] {network}: no checkpoints found — skipping learned methods, "
              f"running baselines only")

    cmd = [PY, str(EVAL_COMPARE),
           "--network", network,
           "--split", "eval",
           "--methods", *methods,
           "--seeds", *[str(s) for s in args.seeds],
           "--reference", "mappo" if mappo else methods[0],
           "--max-steps", str(args.max_steps)]
    if args.emissions:
        cmd.append("--emissions")
    if mappo:
        cmd += ["--checkpoint", *[str(p) for p in mappo]]
    if ippo:
        cmd += ["--ippo-checkpoint", *[str(p) for p in ippo]]
    if priv:
        cmd += ["--privileged-checkpoint", *[str(p) for p in priv]]

    print(f"[eval] {network}: mappo={len(mappo)} ippo={len(ippo)} priv={len(priv)} ckpts")
    return _run(cmd, f"eval/{network}")


def run_noise_for_network(campaign_dir: Path, network: str, args: argparse.Namespace) -> bool:
    proxy = discover_checkpoints(campaign_dir, network, "mappo", "proxy")
    if not proxy:
        print(f"[noise] {network}: no proxy checkpoint — skip")
        return True
    ok = True
    for ckpt in proxy[: args.noise_max_ckpts]:
        cmd = [PY, str(NOISE_EVAL),
               "--checkpoint", str(ckpt),
               "--network", network,
               "--seeds", *[str(s) for s in args.seeds]]
        ok = _run(cmd, f"noise/{network}") and ok
    return ok


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-id", type=str, default=None,
                   help="campaign dir under results/paper1_mappo (default: latest)")
    p.add_argument("--networks", nargs="+", default=["n2_corridor", "n3_grid"],
                   choices=["n1", "n2_corridor", "n3_grid"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    p.add_argument("--max-steps", type=int, default=1080)
    p.add_argument("--emissions", action="store_true",
                   help="enable SUMO emissions device for post-hoc CO2/fuel (TABLE-10)")
    p.add_argument("--skip-eval", action="store_true",
                   help="skip evaluation; only (re)build tables + figures")
    p.add_argument("--noise", action="store_true",
                   help="also run the sensing-noise robustness sweep (proxy arm)")
    p.add_argument("--noise-max-ckpts", type=int, default=1,
                   help="how many proxy seeds to run noise on (default 1)")
    args = p.parse_args()

    campaign_dir = (RESULTS / args.campaign_id) if args.campaign_id else latest_campaign()
    if campaign_dir is None or not campaign_dir.exists():
        if not args.skip_eval:
            sys.exit("[build] no campaign dir found under results/paper1_mappo "
                     "(run scripts/parallel_launcher.py first, or use --skip-eval)")
    else:
        print(f"[build] campaign: {campaign_dir.name}")

    failures = 0

    if not args.skip_eval:
        for net in args.networks:
            if not run_eval_for_network(campaign_dir, net, args):
                failures += 1
        if args.noise:
            for net in args.networks:
                if not run_noise_for_network(campaign_dir, net, args):
                    failures += 1
    else:
        print("[build] --skip-eval: using existing eval_tables/")

    # tables (LaTeX) from the newest summary per network
    if not _run([PY, str(MAKE_TABLES), "--auto"], "tables"):
        failures += 1

    # figures (multi-seed). campaign_figures auto-detects the campaign for the
    # convergence/loss panels; comparison/restriction-cost/noise read eval_tables.
    fig_cmd = [PY, str(CAMPAIGN_FIGS), "--all"]
    if campaign_dir is not None and campaign_dir.exists():
        fig_cmd += ["--campaign-id", campaign_dir.name]
    if not _run(fig_cmd, "figures"):
        failures += 1

    print("\n" + "=" * 60)
    print("PAPER ARTIFACTS BUILD COMPLETE" if failures == 0
          else f"BUILD FINISHED WITH {failures} FAILED STEP(S)")
    print("=" * 60)
    print(f"  tables : {ROOT / 'figures' / 'tables'}")
    print(f"  figures: {ROOT / 'figures'}")
    print(f"  evals  : {RESULTS / 'eval_tables'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
