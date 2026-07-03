"""Generate all Paper-1 schematic figures as vector PDFs (matplotlib + sumolib).

Reproducible, version-controlled replacements for the hand-drawn draw.io PDFs in
``docs/paper1_tsc/paper1_latex/assets/figures/``. Drop-in: same filenames, so the
``\\includegraphics`` lines in the .tex are unchanged.

Figures
-------
  fig1.pdf                 vision-proxy construction (26-dim observation)
  fig2.pdf                 two-phase signal FSM (min-green / yellow timing)
  fig3.pdf                 sensing-noise injection pipeline (III-E / VI-D)
  system_architecture.pdf  CTDE deploy/train data-flow
  n2_sumo.pdf              N2 corridor (1x5) topology   [from sumolib]
  n3_sumo.pdf              N3 grid (4x4) topology       [from sumolib]

Usage:  python experiment/plots/paper_diagrams.py [--only fig2 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "paper1_tsc" / "paper1_latex" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# draw.io-matching palette: (fill, edge)
GREEN = ("#d5e8d4", "#82b366")
BLUE = ("#dae8fc", "#6c8ebf")
ORANGE = ("#ffe6cc", "#d79b00")
GRAY = ("#f5f5f5", "#999999")
DARKGRAY = ("#666666", "#444444")


def _box(ax, cx, cy, w, h, text, color, fs=9, lw=1.3, weight="normal", tc="#1a1a1a"):
    fc, ec = color
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3,
            weight=weight, color=tc, linespacing=1.25)


def _arrow(ax, p1, p2, ls="-", color="#3a3a3a", lw=1.5):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, lw=lw,
                                color=color, linestyle=ls, shrinkA=2, shrinkB=2),
                zorder=1)


def _label(ax, x, y, text, fs=8, color=DARKGRAY, tc="white"):
    fc, ec = color
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.8))


def _save(fig, name):
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {name}")


# ----------------------------------------------------------------------------
def fig1_vision_proxy():
    fig, ax = plt.subplots(figsize=(5.2, 9.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 19); ax.axis("off")

    # camera container with 4 ROI boxes
    ax.add_patch(FancyBboxPatch((2.4, 13.3), 5.4, 5.3, boxstyle="round,pad=0.02",
                                fc="#fafafa", ec="#bbbbbb", lw=1.2, zorder=1))
    ax.text(5.1, 18.2, "Overhead camera  ·  one ROI per approach",
            ha="center", va="center", fontsize=8.3, style="italic", color="#555555")
    for i in range(4):
        _box(ax, 5.1, 17.1 - i * 1.05, 4.0, 0.78, f"ROI {i}", GREEN, fs=9)

    _arrow(ax, (5.1, 13.3), (5.1, 12.5))
    _box(ax, 5.1, 11.55, 7.8, 1.6,
         "Per-approach estimator  $\\hat{\\varphi}$  (YOLOv11 + ByteTrack)\n"
         "each ROI → 5 bounded features:\nqueue · occupancy · speed · moto-share · heavy-share",
         BLUE, fs=8.2)

    _arrow(ax, (5.1, 10.75), (5.1, 10.1))
    _box(ax, 5.1, 9.5, 4.6, 0.95, "20 approach features  (4 × 5)", BLUE, fs=8.6, weight="bold")

    # branch: queue values only -> phase-imbalance (dashed)
    _arrow(ax, (4.6, 9.02), (4.1, 7.05), ls=(0, (4, 3)))
    _label(ax, 6.5, 8.0, "queue values only", fs=7.4, color=GRAY, tc="#333333")
    _box(ax, 4.0, 6.0, 5.2, 2.0,
         "Signed phase-imbalance (1 derived dim)\n"
         "$\\mathrm{clip}(0.5\\,(1+\\bar{q}_A-\\bar{q}_B),\\,0,\\,1)$\n"
         "reuses only the 4 queue values (A / B)", ORANGE, fs=7.4)
    _box(ax, 8.5, 6.0, 2.6, 1.7,
         "Controller-internal\n(not camera):\nphase one-hot (4)\n+ green-timer (1)", ORANGE, fs=7.5)

    # 26-dim obs (bottom)
    _box(ax, 5.0, 2.5, 7.8, 1.7,
         "Local observation  $\\mathbf{o}_i$  —  26 dims\n"
         "20 approach + 4 phase + 1 timer + 1 phase-imbalance", ORANGE, fs=8.3, weight="bold")
    _arrow(ax, (4.0, 5.0), (4.4, 3.37))
    _arrow(ax, (8.5, 5.15), (6.1, 3.37))
    # 20-feat -> obs direct (20 dims), routed down the clear left margin
    ax.plot([2.8, 0.7, 0.7], [9.5, 9.5, 2.5], color="#3a3a3a", lw=1.5, zorder=1)
    _arrow(ax, (0.7, 2.5), (1.1, 2.5))
    _save(fig, "fig1.pdf")


# ----------------------------------------------------------------------------
def fig2_signal_fsm():
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")

    A = (2.7, 7.8); YAB = (9.3, 7.8); B = (9.3, 2.9); YBA = (2.7, 2.9)
    _box(ax, *A, 3.0, 1.15, "Group A green\nSUMO phase 0", GREEN, fs=9, weight="bold")
    _box(ax, *YAB, 3.0, 1.15, "Yellow A→B\nSUMO phase 1  ·  3 s", ORANGE, fs=9)
    _box(ax, *B, 3.0, 1.15, "Group B green\nSUMO phase 2", GREEN, fs=9, weight="bold")
    _box(ax, *YBA, 3.0, 1.15, "Yellow B→A\nSUMO phase 3  ·  3 s", ORANGE, fs=9)

    # clockwise cycle
    _arrow(ax, (4.2, 7.8), (7.8, 7.8))                 # A -> YAB (top)
    _arrow(ax, (9.3, 7.22), (9.3, 3.48))               # YAB -> B (right)
    _arrow(ax, (7.8, 2.9), (4.2, 2.9))                 # B -> YBA (bottom)
    _arrow(ax, (2.7, 3.48), (2.7, 7.22))               # YBA -> A (left)
    _label(ax, 6.0, 7.8, "a = B,  green ≥ 15 s  (switch)")
    _label(ax, 6.0, 2.9, "a = A,  green ≥ 15 s  (switch)")

    # self-loops (clean arcs OUTSIDE the boxes) — CORRECT min-green override condition
    ax.add_patch(FancyArrowPatch((A[0] - 0.55, 8.375), (A[0] + 0.55, 8.375),
                 connectionstyle="arc3,rad=1.6", arrowstyle="-|>",
                 mutation_scale=13, lw=1.4, color="#3a3a3a"))
    ax.add_patch(FancyArrowPatch((B[0] + 0.55, 2.325), (B[0] - 0.55, 2.325),
                 connectionstyle="arc3,rad=1.6", arrowstyle="-|>",
                 mutation_scale=13, lw=1.4, color="#3a3a3a"))
    _label(ax, 3.1, 9.6, "hold A:  a = A,  or  a = B ∧ green < 15 s  (min-green)",
           fs=7.5, color=GRAY, tc="#333333")
    _label(ax, 8.9, 1.2, "hold B:  a = B,  or  a = A ∧ green < 15 s  (min-green)",
           fs=7.5, color=GRAY, tc="#333333")
    _save(fig, "fig2.pdf")


# ----------------------------------------------------------------------------
def fig3_noise_model():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    _box(ax, 3.0, 7.7, 5.0, 1.25,
         "Clean proxy features  $\\varphi(\\mathbf{s}_t)$\nsimulator, noiseless ($\\varepsilon=0$) during training",
         GREEN, fs=8.4)
    _box(ax, 9.2, 7.7, 4.2, 1.25, "Noise scale\n$c \\in \\{0,\\ 0.5\\times,\\ 1\\times,\\ 2\\times\\}$", BLUE, fs=8.6)

    _box(ax, 6.0, 5.0, 6.6, 1.7,
         "Eval-time injection (Section VI-D only):\n"
         "$\\mathbf{o}_i = \\varphi(\\mathbf{s}_t) + c\\cdot\\varepsilon$\n"
         "$\\varepsilon$ from the per-feature modes (Section III-E)", BLUE, fs=8.6)
    _arrow(ax, (3.0, 7.07), (5.0, 5.86))
    _arrow(ax, (9.2, 7.07), (7.0, 5.86))

    _box(ax, 6.0, 2.7, 5.0, 1.05, "Perturbed observation  $\\mathbf{o}_i$  →  actor  $\\pi_\\theta$", ORANGE, fs=8.8)
    _arrow(ax, (6.0, 4.15), (6.0, 3.23))
    _box(ax, 6.0, 0.85, 5.0, 1.05, "Degradation curve:  metric vs noise scale $c$", ORANGE, fs=8.8)
    _arrow(ax, (6.0, 2.17), (6.0, 1.38))
    _save(fig, "fig3.pdf")


# ----------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(0, 28); ax.set_ylim(0, 12.5); ax.axis("off")

    # zone backdrops
    ax.add_patch(FancyBboxPatch((0.3, 7.4), 27.4, 4.7, boxstyle="round,pad=0.02",
                                fc="#fbfbfb", ec="#cccccc", lw=1.2, zorder=0))
    ax.add_patch(FancyBboxPatch((6.0, 0.3), 21.7, 5.7, boxstyle="round,pad=0.02",
                                fc="#fbfbfb", ec="#cccccc", lw=1.2, zorder=0))
    ax.text(14, 11.8, "Deployment — on-camera, no simulator  (2 of N agents shown)",
            ha="center", fontsize=8.6, style="italic", color="#555555")
    ax.text(16.8, 5.75, "Training only — SUMO (CTDE)",
            ha="center", fontsize=8.6, style="italic", color="#555555")

    # --- deployment row ---
    _box(ax, 2.6, 10.3, 3.4, 1.0, "Camera ROIs\nagent 1", BLUE, fs=8)
    _box(ax, 2.6, 8.4, 3.4, 1.0, "Camera ROIs\nagent 2", BLUE, fs=8)
    _box(ax, 5.9, 10.3, 2.4, 0.8, "YOLO +\nByteTrack", DARKGRAY, fs=7.5, tc="white")
    _box(ax, 5.9, 8.4, 2.4, 0.8, "YOLO +\nByteTrack", DARKGRAY, fs=7.5, tc="white")
    _box(ax, 8.9, 10.3, 2.6, 0.8, "Feature\nextractor $\\hat\\varphi$", BLUE, fs=8)
    _box(ax, 8.9, 8.4, 2.6, 0.8, "Feature\nextractor $\\hat\\varphi$", BLUE, fs=8)
    _box(ax, 13.0, 9.35, 3.7, 1.9,
         "VisionBuffer\nper-dim EMA ($\\alpha$=0.6,\nlast 5 frames) + clip [0,1]\n~10 Hz → 1 decision / 5 s", BLUE, fs=7.6)
    _box(ax, 17.3, 10.2, 2.7, 0.85, "$\\mathbf{o}_1$  (26 dims)", GREEN, fs=8.2)
    _box(ax, 17.3, 8.5, 2.7, 0.85, "$\\mathbf{o}_2$  (26 dims)", GREEN, fs=8.2)
    _box(ax, 21.2, 9.35, 3.1, 1.5, "Observation\nnormalization\n(Welford mean/std)", BLUE, fs=7.8)
    _box(ax, 24.9, 9.35, 2.4, 1.1, "Param-shared\nactor $\\pi_\\theta$", BLUE, fs=8)
    _box(ax, 24.9, 6.7, 2.4, 0.8, "Actions\n$a_1, a_2$", BLUE, fs=8.2, weight="bold")

    for y in (10.3, 8.4):
        _arrow(ax, (4.3, y), (4.7, y)); _arrow(ax, (7.1, y), (7.7, y))
    _arrow(ax, (10.2, 10.3), (11.15, 9.7)); _arrow(ax, (10.2, 8.4), (11.15, 9.0))
    _arrow(ax, (14.85, 9.7), (15.95, 10.2)); _arrow(ax, (14.85, 9.0), (15.95, 8.5))
    _arrow(ax, (18.65, 10.2), (19.65, 9.7)); _arrow(ax, (18.65, 8.5), (19.65, 9.0))
    _arrow(ax, (22.75, 9.35), (23.7, 9.35))
    _arrow(ax, (24.9, 8.8), (24.9, 7.1))

    # --- training row ---
    _box(ax, 8.3, 3.0, 3.2, 1.2, "Global state\n$\\mathbf{s}=[\\mathbf{o}_1;\\dots;\\mathbf{o}_N]$", ORANGE, fs=8)
    _box(ax, 12.6, 3.0, 3.0, 1.2, "State\nnormalization\n(mean/std)", ORANGE, fs=7.8)
    _box(ax, 16.9, 3.6, 3.2, 1.3, "Centralized critic $V_\\psi$\n(one value head\nper agent)", ORANGE, fs=7.8)
    _box(ax, 16.9, 1.3, 3.2, 0.9, "Per-agent rewards\n$r_1, r_2$", ORANGE, fs=8)
    _box(ax, 22.0, 2.5, 3.6, 1.5, "Per-agent GAE\n$\\hat{A}_1, \\hat{A}_2$", ORANGE, fs=8)

    _arrow(ax, (9.9, 3.0), (11.1, 3.0)); _arrow(ax, (14.1, 3.0), (15.3, 3.4))
    _arrow(ax, (18.5, 3.6), (20.2, 2.8))
    _arrow(ax, (18.5, 1.3), (20.2, 2.2))
    ax.text(19.4, 3.35, "values $V_\\psi(\\mathbf{s})_i$", fontsize=7, color="#555", ha="center")
    ax.text(19.4, 1.35, "returns $\\hat{R}_i$ → value loss", fontsize=7, color="#555", ha="center")
    # GAE -> PPO update of actor (up to deployment actor)
    _arrow(ax, (23.8, 3.25), (26.1, 6.3))
    ax.text(26.4, 4.8, "$\\hat{A}_i$ → PPO\nclipped update\nof $\\theta$", fontsize=7, color="#555", ha="center")

    # dashed train/deploy bridge: deployment obs feed the global state
    ax.annotate("", xy=(8.3, 3.6), xytext=(15.5, 7.6),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                color="#888", linestyle=(0, (5, 4))))
    _save(fig, "system_architecture.pdf")


# ----------------------------------------------------------------------------
def fig_network(net_name, out_name, figsize):
    import sumolib
    net_path = ROOT / "sumo_configs" / "networks" / net_name / "intersections.net.xml"
    net = sumolib.net.readNet(str(net_path))
    fig, ax = plt.subplots(figsize=figsize)
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        for lane in edge.getLanes():
            xs, ys = zip(*lane.getShape())
            ax.plot(xs, ys, color="#4d4d4d", lw=2.2, solid_capstyle="round", zorder=1)
    tls = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    xs = [n.getCoord()[0] for n in tls]; ys = [n.getCoord()[1] for n in tls]
    ax.scatter(xs, ys, s=90, c="#cc0000", edgecolors="white", linewidths=1.2, zorder=3,
               label=f"{len(tls)} signalized junctions")
    ax.set_aspect("equal"); ax.axis("off")
    # scale bar (100 m)
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.04 * (xmax - xmin); y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([x0, x0 + 100], [y0, y0], color="black", lw=2)
    ax.text(x0 + 50, y0 + 0.02 * (ymax - ymin), "100 m", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _save(fig, out_name)


FIGS = {
    "fig1": fig1_vision_proxy,
    "fig2": fig2_signal_fsm,
    "fig3": fig3_noise_model,
    "arch": fig_architecture,
    "n2": lambda: fig_network("n2_corridor", "n2_sumo.pdf", (8.2, 3.4)),
    "n3": lambda: fig_network("n3_grid", "n3_sumo.pdf", (6.8, 6.8)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=list(FIGS))
    args = ap.parse_args()
    print(f"writing to {OUT}")
    for k in args.only:
        FIGS[k]()


if __name__ == "__main__":
    main()
