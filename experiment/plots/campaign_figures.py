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

    # Clean 2x2 grid: one metric per panel, single blue mean line + std band.
    # (Previous layout overlaid approx-KL on the entropy panel via a twinx orange
    # axis — visually cluttered; KL now gets its own panel with the target line.)
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 4.6), constrained_layout=True)
    specs = [("policy_loss", "Policy loss", False, None),
             ("value_loss", "Value loss", True, None),
             ("entropy", "Entropy", False, None),
             ("approx_kl", "Approx KL", False, 0.015)]
    for ax, (key, label, logy, hline) in zip(axes.flat, specs):
        band = _multiseed_band(upd_jobs, "train_updates.csv", "global_step", key, smooth_w=50)
        if band is None:
            ax.set_visible(False)
            continue
        grid, mean, std = band
        ax.plot(grid / 1e3, mean, color="#1565c0", linewidth=1.3)
        ax.fill_between(grid / 1e3, mean - std, mean + std, color="#90caf9", alpha=0.3, linewidth=0)
        if hline is not None:
            ax.axhline(hline, color="#c62828", linestyle="--", linewidth=0.9,
                       label=f"target {hline}")
            ax.legend(loc="upper right", fontsize=7, frameon=False)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Global step (×10³)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Training diagnostics ({arm}, mean over {len(upd_jobs)} seeds)",
                 fontsize=10)
    out = OUT_DIR / "training_losses.pdf"
    fig.savefig(out, dpi=300)
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


# Per-method palette (consistent across panels). Baselines muted greys/earth
# tones; the RL methods get saturated colours so "ours" reads at a glance.
METHOD_STYLE = {
    "fixed":       ("#bdbdbd", "Fixed-Time"),
    "webster":     ("#9e9e9e", "Webster"),
    "actuated":    ("#78909c", "Actuated"),
    "sotl":        ("#a1887f", "SOTL"),
    "maxpressure": ("#4db6ac", "MaxPressure"),
    "ippo":        ("#e65100", "IPPO"),
    "privileged":  ("#6a1b9a", "MAPPO-priv"),
    "mappo":       ("#1565c0", "MAPPO (ours)"),
}
# canonical left-to-right ordering (baselines first, ours last so it stands out)
METHOD_ORDER = ["fixed", "webster", "actuated", "sotl", "maxpressure",
                "ippo", "privileged", "mappo"]


def _ordered_methods(aggregates) -> list:
    present = [m for m in METHOD_ORDER if m in aggregates]
    present += [m for m in aggregates if m not in present]  # unknowns appended
    return present


def fig_comparison() -> None:
    plt = _setup_mpl()
    summaries = _load_summaries()
    if not summaries:
        print("[fig] comparison: no eval_tables/*/summary.json — skip")
        return
    for summary in summaries:
        scen = summary.get("scenario", "scenario")
        aggregates = summary["aggregates"]
        methods = _ordered_methods(aggregates)
        metrics = [m for m in METRIC_KEYS if m != "n_trips"]

        # one panel per metric → each gets its own y-scale so P95 (~500-800s)
        # no longer squashes Waiting (~140s).
        fig, axes = plt.subplots(1, len(metrics), figsize=(PAGE_W, 2.7))
        x = np.arange(len(methods))
        handles: dict = {}
        for ax, m in zip(axes, metrics):
            means = [aggregates[mm].get(m, {}).get("mean", np.nan) for mm in methods]
            cis = [aggregates[mm].get(m, {}).get("ci95", 0.0) for mm in methods]
            best = int(np.nanargmin(means)) if np.any(np.isfinite(means)) else -1
            for i, mm in enumerate(methods):
                color, label = METHOD_STYLE.get(mm, ("#607d8b", mm))
                is_ours = mm in ("mappo", "privileged")
                bar = ax.bar(
                    x[i], means[i], 0.78, yerr=cis[i], capsize=2,
                    color=color, edgecolor="black",
                    linewidth=1.1 if is_ours else 0.4,
                    error_kw={"elinewidth": 0.7, "capthick": 0.7},
                )
                handles.setdefault(label, bar)
                if i == best:  # mark the winner per metric
                    ax.plot(x[i], means[i] + cis[i], marker="v", ms=4,
                            color="#2e7d32", clip_on=False)
            ax.set_title(METRIC_LABELS[m])
            ax.set_xticks([])
            ax.grid(True, axis="y")
            ax.margins(y=0.15)
        axes[0].set_ylabel("seconds (lower is better)")
        fig.legend(handles.values(), handles.keys(), loc="lower center",
                   ncol=len(handles), fontsize=6.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Controller comparison — {scen}  (▼ = best, 95% CI)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        out = OUT_DIR / f"comparison_{scen}.pdf"
        fig.savefig(out, dpi=300)
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
    """FIG-6 -- sensing-noise robustness, built for an IEEE Transactions page.

    Two panels sharing one truncated y-axis:
      (a) mean travel time vs the common noise-envelope scale, with a 95%
          confidence band over the route seeds;
      (b) the two structural failure modes (1-step latency, camera dropout)
          against the clean reference, as a dot-and-whisker panel.

    Panel (b) uses point estimates with 95% CI whiskers rather than bars: the
    y-axis is shared with (a) and therefore truncated, and a bar encodes value
    as length from zero, so bars on a non-zero baseline would misstate the
    differences. Dots do not carry that implication and read honestly on a
    truncated scale.

    The strongest camera-deployable classical baseline (read from the MERGED
    main-table eval for the same scenario) is a dashed reference drawn across
    both panels; everything above it is the region where the learned policy
    would trail that baseline. Exports vector PDF + SVG + 600-dpi PNG.
    """
    _setup_mpl()
    import csv as _csv
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    base = RESULTS / "noise_robustness"
    if not base.exists():
        print("[fig] noise: no noise_robustness/ -- skip")
        return

    # -- strongest camera-deployable baseline (fallback keeps the fig honest
    #    even when run outside the campaign layout) ----------------------
    ref_val, ref_name = 321.6, "webster"
    for sj in sorted((RESULTS / "eval_tables").glob("*MERGED*/summary.json")):
        try:
            agg = json.loads(sj.read_text(encoding="utf-8"))["aggregates"]
            cands = [(agg[m]["mean_travel_time_s"]["mean"], m)
                     for m in ("webster", "maxpressure", "sotl", "fixed")
                     if m in agg]
            if cands:
                ref_val, ref_name = min(cands)
        except Exception:
            pass

    # -- read the sweep(s): scale points + the two structural conditions --
    STRUCT = {"delay_1step": ("D", "1-step\nlatency"),
              "dropout_approach0": ("s", "Camera\ndropout")}
    runs = []  # (scales, mean, ci95, {cond: (mean, ci95)})
    for sweep in sorted(base.glob("*/sweep.csv")):
        sc, mu, ci, struct = [], [], [], {}
        with open(sweep, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                cond = row["condition"]
                try:
                    m = float(row["mean_travel_time_s"])
                    sd = float(row.get("std_travel_time_s") or 0.0)
                    n = max(int(float(row.get("n_seeds") or 1)), 1)
                except (ValueError, TypeError):
                    continue
                half = 1.96 * sd / np.sqrt(n)
                if cond.startswith("scale_"):
                    try:
                        s = float(cond.split("_", 1)[1])
                    except ValueError:
                        continue
                    sc.append(s); mu.append(m); ci.append(half)
                    if s == 0.0:
                        struct["clean"] = (m, half)
                elif cond in STRUCT:
                    struct[cond] = (m, half)
        if sc:
            o = np.argsort(sc)
            runs.append((np.array(sc)[o], np.array(mu)[o],
                         np.array(ci)[o], struct))
    if not runs:
        print("[fig] noise: no scale_* conditions found -- skip")
        return

    # -- shared, nicely-rounded y-range -----------------------------------
    lo = min([float((mu - ci).min()) for _, mu, ci, _ in runs]
             + [v - h for _, _, _, st in runs for v, h in st.values()])
    hi = max([float((mu + ci).max()) for _, mu, ci, _ in runs]
             + [v + h for _, _, _, st in runs for v, h in st.values()])
    y0 = 10 * np.floor((lo - 4) / 10)
    y1 = 10 * np.ceil((hi + 12) / 10)

    # -- palette: one muted accent + neutral grays, colour-blind & B/W safe
    ACCENT, NEUTRAL, REF = "#1f5c99", "#8c9196", "#333333"
    BAND = "#cfe0f0"
    DASH = (0, (5.5, 2.5))
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "lines.solid_capstyle": "round", "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig, (axA, axB) = plt.subplots(
            1, 2, figsize=(COL_W, 2.5), sharey=True,
            gridspec_kw={"width_ratios": [1.78, 1.0], "wspace": 0.06})

        # common frame for both panels --------------------------------
        for ax in (axA, axB):
            ax.set_ylim(y0, y1)
            ax.set_axisbelow(True)
            ax.grid(axis="y", lw=0.5, color="0.9", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.axhspan(ref_val, y1, color="0.5", alpha=0.055, lw=0, zorder=0)
            ax.axhline(ref_val, ls=DASH, lw=1.2, color=REF, zorder=2)

        # ---- panel (a): the noise-scale sweep -----------------------
        for sc, mu, ci, _ in runs:
            axA.fill_between(sc, mu - ci, mu + ci, color=BAND, alpha=0.85,
                             lw=0, zorder=1)
        for sc, mu, ci, _ in runs:
            axA.plot(sc, mu, color=ACCENT, lw=1.9, marker="o", ms=4.2,
                     mfc=ACCENT, mec="white", mew=0.8, zorder=4,
                     clip_on=False)
        axA.text(0.028, 0.965, "worse than baseline", transform=axA.transAxes,
                 ha="left", va="top", fontsize=6.5, style="italic",
                 color="0.4")
        axA.set_xlim(-0.06, 2.06)
        axA.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        axA.xaxis.set_minor_locator(MultipleLocator(0.25))
        axA.set_xlabel(r"Noise-envelope scale ($\times$ calibrated)")
        axA.set_ylabel("Mean travel time (s)")
        axA.set_title(r"$\mathbf{(a)}$  Noise magnitude", loc="left", pad=5)

        # ---- panel (b): structural failures (dot + 95% CI) ----------
        order = ["clean", "delay_1step", "dropout_approach0"]
        ticks, ticklab = [], []
        nb = len(runs)
        for xi, cond in enumerate(order):
            ticks.append(xi)
            ticklab.append("Clean" if cond == "clean" else STRUCT[cond][1])
            for i, (_, _, _, struct) in enumerate(runs):
                if cond not in struct:
                    continue
                v, h = struct[cond]
                dx = 0.0 if nb == 1 else (i - (nb - 1) / 2) * 0.16
                is_clean = cond == "clean"
                mk = "o" if is_clean else STRUCT[cond][0]
                col = NEUTRAL if is_clean else ACCENT
                axB.errorbar(xi + dx, v, yerr=h, fmt=mk, ms=6.0, color=col,
                             mfc=col, mec="white", mew=0.8, ecolor="0.35",
                             elinewidth=0.9, capsize=3.0, capthick=0.9,
                             zorder=4, clip_on=False)
                if nb == 1:
                    axB.annotate(f"{v:.0f}", (xi, v + h),
                                 textcoords="offset points", xytext=(0, 4.5),
                                 ha="center", va="bottom", fontsize=7,
                                 color="0.12", zorder=6,
                                 bbox=dict(boxstyle="round,pad=0.12",
                                           fc="white", ec="none", alpha=0.9))
        axB.set_xlim(-0.55, 2.55)
        axB.set_xticks(ticks)
        axB.set_xticklabels(ticklab, fontsize=7)
        axB.set_title(r"$\mathbf{(b)}$  Structural failures", loc="left",
                      pad=5)
        axB.tick_params(axis="y", length=0)

        # ---- shared legend beneath both panels ----------------------
        handles = [
            Line2D([0], [0], color=ACCENT, lw=1.9, marker="o", ms=4.2,
                   mfc=ACCENT, mec="white", mew=0.8, label="MAPPO proxy (mean)"),
            Patch(fc=BAND, ec="none", label="95% CI (5 seeds)"),
            Line2D([0], [0], color=REF, lw=1.2, ls=DASH,
                   label=f"{ref_name.capitalize()} (best deployable)"),
        ]
        fig.subplots_adjust(bottom=0.30)
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02), columnspacing=1.6,
                   handlelength=2.1, handletextpad=0.5)

        for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
            out = OUT_DIR / f"noise_robustness.{ext}"
            fig.savefig(out, bbox_inches="tight", **kw)
            print(f"[fig] saved {out}")
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
