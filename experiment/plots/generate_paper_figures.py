"""
generate_paper_figures.py
=========================
Generates publication-quality figures for the Traffic-Guard-AI IEEE paper.

Output (LaTeX-ready PDFs):
    figures/training_curve.pdf       Fig. 4 — reward convergence
    figures/training_losses.pdf      Fig. 5 — policy/value/entropy loss
    figures/reward_distribution.pdf  Fig. 6 — early vs late reward distribution

LaTeX usage:
    \\includegraphics[width=\\columnwidth]{figures/training_curve.pdf}
    \\includegraphics[width=\\textwidth]{figures/training_losses.pdf}
    \\includegraphics[width=\\columnwidth]{figures/reward_distribution.pdf}

Usage (from project root):
    python experiment/plots/generate_paper_figures.py

Requirements: matplotlib, pandas, numpy
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parents[2]
LOG_DIR    = ROOT / "logs" / "mappo" / "20260418_215140"
OUT_DIR    = ROOT / "figures"
EPS_CSV    = LOG_DIR / "train_episodes.csv"
UPD_CSV    = LOG_DIR / "train_updates.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── IEEE two-column figure settings ──────────────────────────────────────────
# Column width = 3.5 in, full page = 7.16 in
COL_W  = 3.5   # single-column figure width [inches]
PAGE_W = 7.16  # double-column figure width [inches]

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.framealpha":  0.85,
    "lines.linewidth":    1.2,
    "axes.linewidth":     0.7,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.4,
    "pdf.fonttype":       42,   # TrueType fonts in PDF (required by IEEE)
    "ps.fonttype":        42,
})

COLORS = {
    "raw":      "#b0bec5",   # light grey for raw data
    "smooth":   "#1565c0",   # IEEE blue for rolling average
    "band":     "#90caf9",   # light blue for confidence band
    "aux1":     "#e65100",   # orange for secondary metric
    "aux2":     "#2e7d32",   # green for third metric
    "policy":   "#c62828",   # red for policy loss
    "value":    "#6a1b9a",   # purple for value loss
    "entropy":  "#00695c",   # teal for entropy
}

# ── Helper: rolling statistics ────────────────────────────────────────────────

def rolling_stats(series: pd.Series, window: int):
    """Return (mean, std) rolling series with min_periods=1."""
    rm  = series.rolling(window, min_periods=1).mean()
    rs  = series.rolling(window, min_periods=1).std().fillna(0)
    return rm, rs

# ── Load data ─────────────────────────────────────────────────────────────────

def load_episodes():
    df = pd.read_csv(EPS_CSV)
    # global_step in raw CSV; convert to ×10³ for readability
    df["step_k"] = df["global_step"] / 1_000
    return df

def load_updates():
    df = pd.read_csv(UPD_CSV)
    df["step_k"] = df["global_step"] / 1_000
    return df

# =============================================================================
# Figure 1 — Training Convergence  (single-column, 3.5 × 4.5 in)
# =============================================================================

def fig_training_curve(eps: pd.DataFrame, window: int = 50):
    """
    Two-panel figure:
      Top   : Mean episode reward (raw + rolling avg ± 1 std)
      Bottom: Throughput and mean waiting time (rolling avg)
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(COL_W, 4.5),
        sharex=True,
        gridspec_kw={"hspace": 0.35}
    )

    ep   = eps["episode"].values
    step = eps["step_k"].values

    # --- Panel A: Mean Reward ---
    ax = axes[0]
    rw_raw  = eps["mean_reward"]
    rw_avg, rw_std = rolling_stats(rw_raw, window)

    # Clip display range: focus on convergence region, suppress crash outliers.
    # Outliers below y_lo are visible as faint raw data but don't distort scale.
    q05 = float(rw_raw.quantile(0.05))
    q99 = float(rw_raw.quantile(0.99))
    y_lo = max(q05 - 10, rw_avg.min() - 20)   # show a little below avg trough
    y_hi = q99 + 5

    ax.fill_between(ep,
                    (rw_avg - rw_std).clip(lower=y_lo),
                    (rw_avg + rw_std).clip(upper=y_hi),
                    color=COLORS["band"], alpha=0.45, linewidth=0)
    ax.plot(ep, rw_raw.clip(lower=y_lo).values,
            color=COLORS["raw"], alpha=0.25, linewidth=0.5, zorder=1)
    ax.plot(ep, rw_avg.values,
            color=COLORS["smooth"], linewidth=1.4, zorder=3,
            label=f"Rolling avg ({window} ep)")

    # Mark clipped outlier episodes with small rug at y_lo
    outlier_mask = rw_raw < y_lo
    if outlier_mask.any():
        ax.scatter(ep[outlier_mask], np.full(outlier_mask.sum(), y_lo + 1),
                   marker="v", s=6, color=COLORS["raw"], alpha=0.5, zorder=2,
                   label=f"Clipped outliers ({outlier_mask.sum()})")

    # Annotate converged mean (last 20% of training)
    tail_mean = rw_avg.iloc[int(len(rw_avg) * 0.8):].mean()
    ax.axhline(tail_mean, color=COLORS["smooth"],
               linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(ep[-1] * 0.98, tail_mean + 1.5,
            f"Converged: {tail_mean:.1f}",
            ha="right", va="bottom", fontsize=6.5,
            color=COLORS["smooth"])

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("(a) Reward Convergence")
    ax.legend(loc="lower right", handlelength=1.5, fontsize=6)
    ax.grid(True, which="major")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # --- Panel B: Throughput & Waiting Time ---
    ax2 = axes[1]

    thr_avg, _ = rolling_stats(eps["throughput"],       window)
    wt_avg,  _ = rolling_stats(eps["avg_waiting_proxy"], window)

    ax2b = ax2.twinx()

    l1, = ax2.plot(ep, thr_avg.values,
                   color=COLORS["smooth"], linewidth=1.3,
                   label="Throughput (veh/ep)")
    l2, = ax2b.plot(ep, wt_avg.values,
                    color=COLORS["aux1"], linewidth=1.3, linestyle="--",
                    label="Avg waiting (s)")

    ax2.set_ylabel("Throughput [veh/ep]", color=COLORS["smooth"])
    ax2b.set_ylabel("Avg waiting [s]",    color=COLORS["aux1"])
    ax2.tick_params(axis="y", labelcolor=COLORS["smooth"])
    ax2b.tick_params(axis="y", labelcolor=COLORS["aux1"])
    ax2.set_xlabel("Episode")
    ax2.set_title("(b) Traffic Performance Metrics")
    ax2.grid(True, which="major")

    combined = [l1, l2]
    ax2.legend(combined, [l.get_label() for l in combined],
               loc="upper right", handlelength=1.5)

    # Shared x-axis formatting
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(200))
    axes[-1].set_xlim(0, ep.max())

    # Secondary x-axis: global step (×10³)
    ax_top = axes[0].twiny()
    ax_top.set_xlim(axes[0].get_xlim())
    # Map episode → step_k at regular episode ticks
    ep_ticks = np.arange(0, ep.max() + 1, 400)
    step_ticks = np.interp(ep_ticks, ep, step)
    ax_top.set_xticks(ep_ticks)
    ax_top.set_xticklabels([f"{int(s)}k" for s in step_ticks], fontsize=6)
    ax_top.set_xlabel("Global Step", fontsize=7)

    out = OUT_DIR / "training_curve.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Figure 2 — Policy Convergence  (double-column, 7.16 × 2.8 in)
# =============================================================================

def fig_training_losses(upd: pd.DataFrame, window: int = 200):
    """
    Three-panel figure:
      Left  : Policy loss (PPO clip)
      Center: Value loss (log scale)
      Right : Policy entropy + approx KL
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(PAGE_W, 2.8),
        gridspec_kw={"wspace": 0.38}
    )

    step = upd["step_k"].values

    def _draw(ax, series, color, title, ylabel, logy=False, window=window):
        avg, std = rolling_stats(series, window)
        ax.fill_between(step,
                        (avg - std).clip(lower=series.min()),
                        avg + std,
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(step, series.values,
                color=color, alpha=0.15, linewidth=0.4)
        ax.plot(step, avg.values,
                color=color, linewidth=1.3,
                label=f"Rolling avg ({window})")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Global Step (×10³)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both" if logy else "major")
        ax.legend(loc="best", handlelength=1.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(500))

    # Panel A: policy loss
    _draw(axes[0],
          upd["policy_loss"],
          COLORS["policy"],
          "(a) Policy Loss (PPO Clip)",
          "Policy Loss")
    axes[0].axhline(0, color="black", linewidth=0.6, linestyle=":")

    # Panel B: value loss (log scale)
    _draw(axes[1],
          upd["value_loss"].clip(lower=1e-5),
          COLORS["value"],
          "(b) Value Loss",
          "Value Loss",
          logy=True)

    # Panel C: entropy + approx KL
    ax = axes[2]
    ent_avg, _ = rolling_stats(upd["entropy"],    window)
    kl_avg,  _ = rolling_stats(upd["approx_kl"],  window)

    axc2 = ax.twinx()
    l1, = ax.plot(step, ent_avg.values,
                  color=COLORS["entropy"], linewidth=1.3,
                  label="Entropy H[π]")
    l2, = axc2.plot(step, kl_avg.values,
                    color=COLORS["aux1"], linewidth=1.3, linestyle="--",
                    label="Approx KL")

    # KL target line
    axc2.axhline(0.015, color=COLORS["aux1"],
                 linestyle=":", linewidth=0.8, alpha=0.7)
    axc2.text(step.max() * 0.98, 0.016, "KL target",
              ha="right", va="bottom", fontsize=5.5, color=COLORS["aux1"])

    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Entropy H[π]",    color=COLORS["entropy"])
    axc2.set_ylabel("Approx KL",     color=COLORS["aux1"])
    ax.tick_params(axis="y", labelcolor=COLORS["entropy"])
    axc2.tick_params(axis="y", labelcolor=COLORS["aux1"])
    ax.set_title("(c) Exploration & KL Divergence")
    ax.grid(True, which="major")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()],
              loc="upper right", handlelength=1.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))

    out = OUT_DIR / "training_losses.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Figure 3 — Reward Distribution (early vs late)  single-column, 3.5 × 2.8 in
# =============================================================================

def fig_reward_distribution(eps: pd.DataFrame, pct: float = 0.20):
    """
    Box + violin plot comparing reward distribution between first pct
    and last pct of training episodes.
    """
    n = len(eps)
    cutoff = int(n * pct)
    early = eps["mean_reward"].iloc[:cutoff].values
    late  = eps["mean_reward"].iloc[-cutoff:].values

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    colors_vp = [COLORS["aux1"], COLORS["smooth"]]

    # Clip display to IQR-based range — show bulk distribution, not deep tails.
    all_data  = np.concatenate([early, late])
    q10, q90  = np.percentile(all_data, [2, 98])
    margin    = (q90 - q10) * 0.15
    y_lo_d    = q10 - margin
    y_hi_d    = q90 + margin

    early_clipped = np.clip(early, y_lo_d, y_hi_d)
    late_clipped  = np.clip(late,  y_lo_d, y_hi_d)

    parts = ax.violinplot(
        [early_clipped, late_clipped],
        positions=[1, 2],
        widths=0.5,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_vp[i])
        pc.set_alpha(0.35)
        pc.set_edgecolor(colors_vp[i])

    # Overlay box plots (use original data so stats are honest)
    bp = ax.boxplot(
        [early, late],
        positions=[1, 2],
        widths=0.18,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=2, alpha=0.35,
                        markerfacecolor="grey", markeredgecolor="none"),
    )
    for patch, color in zip(bp["boxes"], colors_vp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate means (use original, unclipped means)
    for x, data, color in zip([1, 2], [early, late], colors_vp):
        mean = data.mean()
        # clamp annotation position to visible range
        y_ann = float(np.clip(mean, y_lo_d + 1, y_hi_d - 3))
        ax.hlines(mean, x - 0.28, x + 0.28,
                  colors=color, linewidths=1.2, linestyles="--")
        ax.text(x + 0.31, y_ann, f"{mean:.1f}",
                va="center", ha="left", fontsize=6.5, color=color)

    # Delta annotation
    delta = late.mean() - early.mean()
    ax.annotate(
        f"$\\Delta = {delta:+.1f}$",
        xy=(1.5, y_hi_d - margin * 0.5),
        ha="center", fontsize=8, color="black",
        fontweight="bold"
    )

    ax.set_ylim(y_lo_d, y_hi_d)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Early {int(pct*100)}\%\n(ep 1–{cutoff})",
        f"Late {int(pct*100)}\%\n(ep {n-cutoff}–{n})"
    ])
    ax.set_ylabel("Mean Reward per Episode")
    ax.set_title("Reward Distribution: Early vs.\ Late Training")
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    # Note about clipping
    n_clip_e = int(np.sum(early < y_lo_d))
    n_clip_l = int(np.sum(late  < y_lo_d))
    if n_clip_e + n_clip_l > 0:
        ax.text(0.5, 0.01,
                f"*y-axis clipped; {n_clip_e+n_clip_l} outlier ep. below {y_lo_d:.0f} not shown",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.5, color="grey", style="italic")

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["aux1"],   linewidth=4, alpha=0.6,
               label=f"Early (mean={early.mean():.1f})"),
        Line2D([0], [0], color=COLORS["smooth"], linewidth=4, alpha=0.6,
               label=f"Late  (mean={late.mean():.1f})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", handlelength=1.5)

    out = OUT_DIR / "reward_distribution.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not EPS_CSV.exists():
        sys.exit(f"[ERROR] Not found: {EPS_CSV}\nRun from the project root.")

    print("Loading episode log …")
    eps = load_episodes()
    print(f"  {len(eps)} episodes  |  "
          f"global steps: {int(eps['global_step'].min())}–{int(eps['global_step'].max())}")

    print("Loading update log …")
    upd = load_updates()
    print(f"  {len(upd)} update rows")

    print("\nGenerating figures …")
    fig_training_curve(eps,       window=50)
    fig_training_losses(upd,      window=200)
    fig_reward_distribution(eps,  pct=0.20)

    print("\nAll figures saved to:", OUT_DIR)
    print("Insert into paper:")
    print("  \\includegraphics[width=\\columnwidth]{figures/training_curve.pdf}")
    print("  \\includegraphics[width=\\textwidth]{figures/training_losses.pdf}")
    print("  \\includegraphics[width=\\columnwidth]{figures/reward_distribution.pdf}")
