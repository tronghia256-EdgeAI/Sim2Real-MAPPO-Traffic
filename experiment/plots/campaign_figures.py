from __future__ import annotations

"""
campaign_figures.py
===================
Campaign-aware, multi-seed paper figures. Unlike the legacy single-run
generate_paper_figures.py (hard-wired to one log dir), this reads the W5-9
campaign layout produced by scripts/parallel_launcher.py:

    results/paper1_mappo/<campaign>/
        tblogs/<net>_<algo>_<obs>_seed<seed>/<run_id>/train_episodes.csv
                                                       train_updates.csv
    results/paper1_mappo/eval_tables/<scenario>_<ts>/summary.json
    results/paper1_mappo/noise_robustness/<ckpt_tag>/sweep.csv

Figures (IEEE column-width PDFs in figures/):
    convergence.pdf           FIG-3  reward vs steps, mean±std over seeds, per net
    training_losses.pdf       FIG-4  policy/value loss + entropy/KL (mean over seeds)
    comparison_<scen>.pdf     VI-A   grouped bars per metric, 95% CI
    restriction_cost_<scen>.pdf VI-C proxy vs privileged per metric
    noise_robustness.pdf      FIG-6  mean travel vs noise scale

Every figure degrades gracefully: missing data → that figure is skipped with a
message, never a crash.

Usage (from project root):
    python experiment/plots/campaign_figures.py --all                 # auto-detect
    python experiment/plots/campaign_figures.py --campaign-id campaign_2026... --all
    python experiment/plots/campaign_figures.py --convergence --losses
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
RESULTS = _ROOT / "results" / "paper1_mappo"
OUT_DIR = _ROOT / "figures"

# matplotlib configured lazily inside _setup_mpl() so --help works without it.
COL_W, PAGE_W = 3.5, 7.16
ARM_COLORS = {
    "mappo_proxy": "#1565c0",
    "mappo_privileged": "#6a1b9a",
    "ippo_proxy": "#e65100",
}
METRIC_KEYS = ("mean_travel_time_s", "mean_waiting_time_s",
               "mean_time_loss_s", "p95_waiting_time_s", "n_trips")
METRIC_LABELS = {
    "mean_travel_time_s": "Travel (s)", "mean_waiting_time_s": "Waiting (s)",
    "mean_time_loss_s": "Time loss (s)", "p95_waiting_time_s": "P95 wait (s)",
    "n_trips": "Served",
}


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "lines.linewidth": 1.2, "axes.linewidth": 0.7,
        "grid.linewidth": 0.5, "grid.alpha": 0.4, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return plt


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def latest_campaign() -> Optional[Path]:
    cands = sorted(RESULTS.glob("campaign_*"))
    return cands[-1] if cands else None


def parse_job_name(name: str) -> Optional[Tuple[str, str, str, int]]:
    """'<network>_<algo>_<obs>_seed<seed>' -> (network, algo, obs, seed)."""
    if "_seed" not in name:
        return None
    head, _, seed_s = name.rpartition("_seed")
    try:
        seed = int(seed_s)
    except ValueError:
        return None
    for obs in ("privileged", "proxy"):
        if head.endswith("_" + obs):
            head2 = head[: -(len(obs) + 1)]
            for algo in ("mappo", "ippo"):
                if head2.endswith("_" + algo):
                    return head2[: -(len(algo) + 1)], algo, obs, seed
    return None


def discover_runs(campaign_dir: Path) -> Dict[Tuple[str, str, str], List[Path]]:
    """{(network, algo, obs): [train_episodes.csv per seed]}."""
    tblogs = campaign_dir / "tblogs"
    out: Dict[Tuple[str, str, str], List[Path]] = {}
    if not tblogs.exists():
        return out
    for job_dir in sorted(tblogs.iterdir()):
        if not job_dir.is_dir():
            continue
        parsed = parse_job_name(job_dir.name)
        if parsed is None:
            continue
        net, algo, obs, _seed = parsed
        csvs = sorted(job_dir.glob("**/train_episodes.csv"))
        if csvs:
            # one entry PER SEED = the job_dir; its resume segments are stitched
            # later (see _stitch). Picking csvs[0] would keep only the first
            # pre-crash segment and truncate the curve.
            out.setdefault((net, algo, obs), []).append(job_dir)
    return out


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    import csv
    cols: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k, []).append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    # Normalise by the ACTUAL number of overlapping taps at each position so the
    # boundaries are not suppressed toward zero (plain mode="same" divides edge
    # points by the full window, halving the first/last samples and faking a
    # low-start / late-drop in the curve).
    kernel = np.ones(window)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / den


def _stitch(
    job_dir: Path, filename: str, x_key: str, y_key: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Merge all resume segments of ONE seed into a single (x, y) series.

    A crashed+resumed job leaves several <run_id>/<filename> CSVs, each covering
    a contiguous slice of global_step (with occasional overlap when a restart
    re-ran a few episodes). We iterate run dirs in timestamp order (lexical sort
    of the run_id) and map global_step -> y, so a later resume overwrites the
    earlier value on any overlap. Returns steps sorted ascending.
    """
    merged: Dict[float, float] = {}
    for p in sorted(job_dir.glob(f"**/{filename}")):
        d = _read_csv(p)
        if x_key not in d or y_key not in d:
            continue
        for xi, yi in zip(d[x_key], d[y_key]):
            if np.isfinite(xi):
                merged[float(xi)] = float(yi)
    if len(merged) < 2:
        return None
    xs = np.array(sorted(merged))
    ys = np.array([merged[x] for x in xs])
    return xs, ys


def _multiseed_band(
    job_dirs: List[Path], filename: str, x_key: str, y_key: str,
    n_grid: int = 200, smooth_w: int = 15,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stitch each seed's resume segments, interpolate onto a common grid
    -> (grid, mean, std)."""
    series = []
    x_max_common = np.inf
    for jd in job_dirs:
        st = _stitch(jd, filename, x_key, y_key)
        if st is None:
            continue
        x, y = st
        y = _smooth(y, smooth_w)
        series.append((x, y))
        x_max_common = min(x_max_common, float(x[-1]))
    if not series or not np.isfinite(x_max_common):
        return None
    x_min = min(float(s[0][0]) for s in series)
    grid = np.linspace(x_min, x_max_common, n_grid)
    mat = np.vstack([np.interp(grid, x, y) for x, y in series])
    return grid, mat.mean(axis=0), mat.std(axis=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_convergence(campaign_dir: Path) -> None:
    plt = _setup_mpl()
    runs = discover_runs(campaign_dir)
    if not runs:
        print("[fig] convergence: no tblogs found — skip")
        return
    networks = sorted({net for (net, _, _) in runs})
    fig, axes = plt.subplots(1, len(networks), figsize=(PAGE_W, 2.8), squeeze=False)
    for ax, net in zip(axes[0], networks):
        plotted = False
        for (n, algo, obs), job_dirs in sorted(runs.items()):
            if n != net:
                continue
            band = _multiseed_band(job_dirs, "train_episodes.csv",
                                   "global_step", "mean_reward")
            if band is None:
                continue
            grid, mean, std = band
            arm = f"{algo}_{obs}"
            color = ARM_COLORS.get(arm, None)
            ax.plot(grid / 1e3, mean, color=color, label=f"{arm} (n={len(job_dirs)})")
            ax.fill_between(grid / 1e3, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
            plotted = True
        ax.set_title(net)
        ax.set_xlabel("Global step (×10³)")
        ax.set_ylabel("Mean episode reward")
        ax.grid(True)
        if plotted:
            ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "convergence.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {out}")


def fig_losses(campaign_dir: Path, arm: str = "mappo_proxy") -> None:
    plt = _setup_mpl()
    tblogs = campaign_dir / "tblogs"
    if not tblogs.exists():
        print("[fig] losses: no tblogs — skip")
        return
    algo, obs = arm.split("_", 1)
    upd_jobs: List[Path] = []
    for job_dir in sorted(tblogs.iterdir()):
        parsed = parse_job_name(job_dir.name) if job_dir.is_dir() else None
        if parsed and parsed[1] == algo and parsed[2] == obs:
            if list(job_dir.glob("**/train_updates.csv")):
                upd_jobs.append(job_dir)
    if not upd_jobs:
        print(f"[fig] losses: no train_updates.csv for {arm} — skip")
        return

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.6), gridspec_kw={"wspace": 0.4})
    specs = [("policy_loss", "Policy loss", False),
             ("value_loss", "Value loss", True),
             ("entropy", "Entropy / KL", False)]
    for ax, (key, label, logy) in zip(axes, specs):
        band = _multiseed_band(upd_jobs, "train_updates.csv", "global_step", key, smooth_w=50)
        if band is None:
            continue
        grid, mean, std = band
        ax.plot(grid / 1e3, mean, color="#1565c0")
        ax.fill_between(grid / 1e3, mean - std, mean + std, color="#90caf9", alpha=0.3, linewidth=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Global step (×10³)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True)
        if key == "entropy":
            kl = _multiseed_band(upd_jobs, "train_updates.csv", "global_step", "approx_kl", smooth_w=50)
            if kl is not None:
                axk = ax.twinx()
                axk.plot(kl[0] / 1e3, kl[1], color="#e65100", linestyle="--")
                axk.axhline(0.015, color="#e65100", linestyle=":", linewidth=0.8)
                axk.set_ylabel("Approx KL", color="#e65100")
    fig.suptitle(f"Training diagnostics ({arm}, mean over {len(upd_jobs)} seeds)",
                 fontsize=11, y=0.99)
    # reserve headroom for the suptitle (subplots_adjust, not tight_layout, since
    # the entropy panel's twinx KL axis is incompatible with tight_layout)
    fig.subplots_adjust(top=0.86, wspace=0.4)
    out = OUT_DIR / "training_losses.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {out}")


def _load_summaries() -> List[dict]:
    base = RESULTS / "eval_tables"
    if not base.exists():
        return []
    by_net: Dict[str, Path] = {}
    for sp in sorted(base.glob("*/summary.json")):
        scen = json.loads(sp.read_text(encoding="utf-8")).get("scenario", sp.parent.name)
        if scen not in by_net or sp.parent.name > by_net[scen].parent.name:
            by_net[scen] = sp
    return [json.loads(p.read_text(encoding="utf-8")) for p in by_net.values()]


def fig_comparison() -> None:
    plt = _setup_mpl()
    summaries = _load_summaries()
    if not summaries:
        print("[fig] comparison: no eval_tables/*/summary.json — skip")
        return
    for summary in summaries:
        scen = summary.get("scenario", "scenario")
        aggregates = summary["aggregates"]
        methods = list(aggregates)
        # show the 4 delay metrics (skip n_trips: different units)
        metrics = [m for m in METRIC_KEYS if m != "n_trips"]
        fig, ax = plt.subplots(figsize=(PAGE_W, 3.0))
        x = np.arange(len(metrics))
        w = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            means = [aggregates[method].get(m, {}).get("mean", np.nan) for m in metrics]
            cis = [aggregates[method].get(m, {}).get("ci95", 0.0) for m in metrics]
            ax.bar(x + i * w, means, w, yerr=cis, capsize=2, label=method)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
        ax.set_ylabel("seconds (lower is better)")
        ax.set_title(f"Controller comparison — {scen}")
        ax.legend(ncol=2, fontsize=6)
        ax.grid(True, axis="y")
        out = OUT_DIR / f"comparison_{scen}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[fig] saved {out}")


def fig_restriction_cost() -> None:
    plt = _setup_mpl()
    summaries = _load_summaries()
    drawn = False
    for summary in summaries:
        agg = summary["aggregates"]
        if "mappo" not in agg or "privileged" not in agg:
            continue
        scen = summary.get("scenario", "scenario")
        metrics = [m for m in METRIC_KEYS if m != "n_trips"]
        fig, ax = plt.subplots(figsize=(COL_W, 2.8))
        x = np.arange(len(metrics))
        priv = [agg["privileged"].get(m, {}).get("mean", np.nan) for m in metrics]
        prox = [agg["mappo"].get(m, {}).get("mean", np.nan) for m in metrics]
        priv_ci = [agg["privileged"].get(m, {}).get("ci95", 0.0) for m in metrics]
        prox_ci = [agg["mappo"].get(m, {}).get("ci95", 0.0) for m in metrics]
        ax.bar(x - 0.2, priv, 0.4, yerr=priv_ci, capsize=2, label="Privileged", color="#6a1b9a")
        ax.bar(x + 0.2, prox, 0.4, yerr=prox_ci, capsize=2, label="Proxy (deployable)", color="#1565c0")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15)
        ax.set_ylabel("seconds")
        ax.set_title(f"Information-restriction cost — {scen}")
        ax.legend()
        ax.grid(True, axis="y")
        out = OUT_DIR / f"restriction_cost_{scen}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        drawn = True
        print(f"[fig] saved {out}")
    if not drawn:
        print("[fig] restriction_cost: need both 'mappo' and 'privileged' arms — skip")


def fig_noise() -> None:
    plt = _setup_mpl()
    base = RESULTS / "noise_robustness"
    if not base.exists():
        print("[fig] noise: no noise_robustness/ — skip")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    drawn = False
    for sweep in sorted(base.glob("*/sweep.csv")):
        d = _read_csv(sweep)
        labels_path = sweep.parent / "summary.json"
        # x = scale from 'scale_<v>' conditions; structural modes plotted as markers
        import csv as _csv
        scales, travel = [], []
        with open(sweep, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                cond = row["condition"]
                if cond.startswith("scale_"):
                    try:
                        scales.append(float(cond.split("_", 1)[1]))
                        travel.append(float(row["mean_travel_time_s"]))
                    except ValueError:
                        pass
        if scales:
            order = np.argsort(scales)
            sc = np.array(scales)[order]
            tv = np.array(travel)[order]
            ax.plot(sc, tv, marker="o", label=sweep.parent.name[:18])
            drawn = True
    ax.set_xlabel("Noise envelope scale (×)")
    ax.set_ylabel("Mean travel time (s)")
    ax.set_title("Robustness to sensing noise")
    ax.grid(True)
    if drawn:
        ax.legend(fontsize=6)
        out = OUT_DIR / "noise_robustness.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[fig] saved {out}")
    else:
        print("[fig] noise: no scale_* conditions found — skip")
    plt.close(fig)


def fig_feature_recovery() -> None:
    """FIG-1a: III-E per-feature recovery error (bias ± σ) from calibrate_noise."""
    plt = _setup_mpl()
    csv_path = RESULTS / "iii_e" / "feature_recovery.csv"
    if not csv_path.exists():
        print("[fig] feature_recovery: run calibrate_noise.py (--gt-csv/--est-csv) first — skip")
        return
    d = _read_csv(csv_path)
    import csv as _csv
    feats, bias, sigma = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            feats.append(row["feature"].replace("_norm", "").replace("_", " "))
            bias.append(float(row.get("bias", "nan") or "nan"))
            sigma.append(float(row.get("sigma", "nan") or "nan"))
    if not feats:
        print("[fig] feature_recovery: empty csv — skip")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    x = np.arange(len(feats))
    ax.bar(x, bias, yerr=sigma, capsize=3, color="#1565c0", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("estimate − ground truth")
    ax.set_title("Feature recovery (bias ± σ)")
    ax.grid(True, axis="y")
    out = OUT_DIR / "feature_recovery.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {out}")


def fig_demand_profile() -> None:
    """FIG-2 (left panel): the trapezoidal within-episode demand λ(t)."""
    plt = _setup_mpl()
    # knots from scripts/generate_demand.py (N1 reference scale, veh/s)
    knots = [(0, 0.40), (900, 0.40), (1500, 1.67), (3000, 1.67),
             (3600, 0.83), (4800, 0.83), (5400, 0.40)]
    t = [k[0] for k in knots]; r = [k[1] for k in knots]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.plot(t, r, color="#1565c0", marker="o", markersize=3)
    ax.fill_between(t, r, color="#90caf9", alpha=0.3)
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Demand λ (veh/s, N1 scale)")
    ax.set_title("Within-episode demand profile")
    ax.grid(True)
    out = OUT_DIR / "demand_profile.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {out}  (FIG-2: pair with a netedit/SUMO-GUI network diagram)")


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-id", type=str, default=None,
                   help="campaign dir name under results/paper1_mappo (default: latest)")
    p.add_argument("--arm", type=str, default="mappo_proxy",
                   help="arm for the loss-diagnostics figure")
    p.add_argument("--all", action="store_true")
    p.add_argument("--convergence", action="store_true")
    p.add_argument("--losses", action="store_true")
    p.add_argument("--comparison", action="store_true")
    p.add_argument("--restriction-cost", action="store_true")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--feature-recovery", action="store_true")
    p.add_argument("--demand-profile", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    campaign_dir = None
    if args.campaign_id:
        campaign_dir = RESULTS / args.campaign_id
    else:
        campaign_dir = latest_campaign()

    want = {k: getattr(args, k) for k in
            ("convergence", "losses", "comparison", "noise", "feature_recovery", "demand_profile")}
    want["restriction_cost"] = args.restriction_cost
    if args.all or not any(want.values()):
        want = {k: True for k in want}

    if want["convergence"] or want["losses"]:
        if campaign_dir is None or not campaign_dir.exists():
            print("[fig] no campaign dir found — convergence/losses skipped")
        else:
            print(f"[fig] campaign: {campaign_dir.name}")
            if want["convergence"]:
                fig_convergence(campaign_dir)
            if want["losses"]:
                fig_losses(campaign_dir, arm=args.arm)
    if want["comparison"]:
        fig_comparison()
    if want["restriction_cost"]:
        fig_restriction_cost()
    if want["noise"]:
        fig_noise()
    if want["feature_recovery"]:
        fig_feature_recovery()
    if want["demand_profile"]:
        fig_demand_profile()


if __name__ == "__main__":
    main()
