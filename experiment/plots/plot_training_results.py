"""
plot_training_results.py
========================
Vẽ biểu đồ kết quả huấn luyện MAPPO từ log CSV.

Tạo 3 figure chất lượng cao (300 DPI) để chèn vào slide:
  - Figure 1: Reward & Traffic Performance (4 biểu đồ)
  - Figure 2: Policy Training Losses (4 biểu đồ)
  - Figure 3: Summary Chart gọn cho slide (3 biểu đồ nằm ngang)

Chạy:
    python experiment/plots/plot_training_results.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình đường dẫn
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
LOG_DIR   = ROOT / "logs" / "mappo" / "20260418_215140"
OUT_DIR   = LOG_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EP_CSV  = LOG_DIR / "train_episodes.csv"
UPD_CSV = LOG_DIR / "train_updates.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.framealpha": 0.8,
    "figure.dpi":        150,
})

# Bảng màu đồng nhất
C_RAW    = "#B0BEC5"   # xám nhạt — dữ liệu thô
C_SMOOTH = "#1565C0"   # xanh đậm — đường mượt
C_LOSS   = "#C62828"   # đỏ đậm
C_ENT    = "#558B2F"   # xanh lá
C_KL     = "#6A1B9A"   # tím
C_CLIP   = "#E65100"   # cam
C_THRU   = "#00838F"   # cyan
C_QUEUE  = "#AD1457"   # hồng đậm
C_WAIT   = "#F57F17"   # vàng đậm

ROLL_EP  = 30   # rolling window cho episodes
ROLL_UPD = 200  # rolling window cho updates

OUTLIER_REWARD_MIN = -200.0   # lọc episode SUMO crash

# ─────────────────────────────────────────────────────────────────────────────
# Load & làm sạch dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    ep  = pd.read_csv(EP_CSV)
    upd = pd.read_csv(UPD_CSV)

    # Lọc episode anomaly (SUMO crash hoặc simulation lỗi)
    ep_clean = ep[ep["mean_reward"] > OUTLIER_REWARD_MIN].copy()
    ep_clean = ep_clean.reset_index(drop=True)

    # Lọc value_loss outlier (>10 std)
    vl_mean = upd["value_loss"].mean()
    vl_std  = upd["value_loss"].std()
    upd_clean = upd[upd["value_loss"] < vl_mean + 10 * vl_std].copy()
    upd_clean = upd_clean.reset_index(drop=True)

    print(f"Episodes  : {len(ep)} total -> {len(ep_clean)} sau loc outlier")
    print(f"Updates   : {len(upd)} total -> {len(upd_clean)} sau loc value_loss")
    return ep_clean, upd_clean


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Reward & Traffic Performance
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure1(ep: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MAPPO Training — Reward & Traffic Performance", fontsize=15, fontweight="bold", y=1.01)

    x = ep["episode"]

    # ── 1. Mean Reward ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, ep["mean_reward"],        color=C_RAW,    alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, rolling(ep["mean_reward"], ROLL_EP), color=C_SMOOTH, linewidth=2.0,  label=f"Rolling avg ({ROLL_EP} ep)")
    ax.set_title("Mean Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.annotate(
        f"Best: {ep['mean_reward'].max():.1f}",
        xy=(ep["mean_reward"].idxmax(), ep["mean_reward"].max()),
        xytext=(ep["mean_reward"].idxmax() - len(ep)*0.15, ep["mean_reward"].max() + 3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray",
    )

    # ── 2. Throughput ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, ep["throughput"],              color=C_RAW,   alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, rolling(ep["throughput"], ROLL_EP), color=C_THRU, linewidth=2.0, label=f"Rolling avg ({ROLL_EP} ep)")
    ax.axhline(ep["throughput"].mean(), color=C_THRU, linestyle=":", linewidth=1.5,
               label=f"Mean = {ep['throughput'].mean():.0f}")
    ax.set_title("Throughput per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Số xe đi qua (vehicles)")
    ax.legend()

    # ── 3. Queue Total Proxy ─────────────────────────────────────────────────
    ax = axes[1, 0]
    queue_k = ep["queue_total_proxy"] / 1000
    ax.plot(x, queue_k,                           color=C_RAW,   alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, rolling(queue_k, ROLL_EP),          color=C_QUEUE, linewidth=2.0, label=f"Rolling avg ({ROLL_EP} ep)")
    ax.set_title("Queue Total Proxy (×1000)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Queue proxy (×10³)")
    ax.legend()

    # ── 4. Average Waiting Proxy ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, ep["avg_waiting_proxy"],              color=C_RAW,  alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, rolling(ep["avg_waiting_proxy"], ROLL_EP), color=C_WAIT, linewidth=2.0, label=f"Rolling avg ({ROLL_EP} ep)")
    ax.set_title("Avg Waiting Proxy per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Waiting proxy (s)")
    ax.legend()

    fig.tight_layout()
    out = OUT_DIR / "fig1_reward_traffic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Policy Training Losses
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure2(upd: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MAPPO Training — Policy Loss & Convergence", fontsize=15, fontweight="bold", y=1.01)

    x = upd["global_step"] / 1_000   # đổi sang nghìn steps

    # ── 1. Policy Loss ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, upd["policy_loss"],                    color=C_RAW,  alpha=0.2, linewidth=0.5)
    ax.plot(x, rolling(upd["policy_loss"], ROLL_UPD), color=C_LOSS, linewidth=2.0, label=f"Rolling avg ({ROLL_UPD})")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Policy Loss (PPO Clip)")
    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Loss")
    ax.legend()

    # ── 2. Value Loss ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, upd["value_loss"],                    color=C_RAW,  alpha=0.2, linewidth=0.5)
    ax.plot(x, rolling(upd["value_loss"], ROLL_UPD), color=C_LOSS, linewidth=2.0, label=f"Rolling avg ({ROLL_UPD})")
    ax.set_title("Value Loss")
    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()

    # ── 3. Entropy ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(x, upd["entropy"],                    color=C_RAW, alpha=0.2, linewidth=0.5)
    ax.plot(x, rolling(upd["entropy"], ROLL_UPD), color=C_ENT, linewidth=2.0, label=f"Rolling avg ({ROLL_UPD})")
    ax.set_title("Policy Entropy (Exploration)")
    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Entropy H(π)")
    ax.legend()

    # ── 4. Approx KL + Clip Fraction ─────────────────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(x,  rolling(upd["approx_kl"],      ROLL_UPD), color=C_KL,   linewidth=2.0, label="Approx KL")
    ax2.plot(x, rolling(upd["clip_fraction"],  ROLL_UPD), color=C_CLIP, linewidth=2.0, linestyle="--", label="Clip Fraction")
    ax.axhline(0.015, color=C_KL, linestyle=":", linewidth=1.2, alpha=0.7, label="KL target (0.015)")
    ax.set_title("Approx KL & Clip Fraction")
    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Approx KL", color=C_KL)
    ax2.set_ylabel("Clip Fraction", color=C_CLIP)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig2_training_losses.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Summary Chart (3 biểu đồ nằm ngang, compact cho slide)
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure3(ep: pd.DataFrame, upd: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MAPPO Training — Summary", fontsize=15, fontweight="bold")

    # ── 1. Reward Convergence ────────────────────────────────────────────────
    ax = axes[0]
    x_ep = ep["episode"]
    r_smooth = rolling(ep["mean_reward"], ROLL_EP)
    ax.fill_between(x_ep, ep["mean_reward"].rolling(ROLL_EP, min_periods=1).min(),
                    ep["mean_reward"].rolling(ROLL_EP, min_periods=1).max(),
                    alpha=0.1, color=C_SMOOTH, label="Min–Max band")
    ax.plot(x_ep, r_smooth, color=C_SMOOTH, linewidth=2.5, label="Rolling mean")
    ax.set_title("Reward Convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward per Episode")
    ax.legend(fontsize=9)
    # Chú thích giai đoạn hội tụ
    conv_ep = ep[ep["episode"] > len(ep) * 0.5]["mean_reward"].mean()
    early_ep = ep[ep["episode"] < len(ep) * 0.1]["mean_reward"].mean()
    ax.annotate(f"Early avg\n{early_ep:.1f}",
                xy=(len(ep) * 0.05, early_ep), xytext=(len(ep)*0.15, early_ep - 5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
    ax.annotate(f"Late avg\n{conv_ep:.1f}",
                xy=(len(ep) * 0.75, conv_ep), xytext=(len(ep)*0.65, conv_ep - 5),
                arrowprops=dict(arrowstyle="->", color=C_SMOOTH), fontsize=8, color=C_SMOOTH)

    # ── 2. Throughput & Queue ────────────────────────────────────────────────
    ax = axes[1]
    ax2 = ax.twinx()
    t_smooth = rolling(ep["throughput"], ROLL_EP)
    q_smooth = rolling(ep["queue_total_proxy"] / 1000, ROLL_EP)
    ax.plot(x_ep,  t_smooth, color=C_THRU,  linewidth=2.5, label="Throughput")
    ax2.plot(x_ep, q_smooth, color=C_QUEUE, linewidth=2.0, linestyle="--", label="Queue (×10³)")
    ax.set_title("Throughput vs Queue")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (vehicles)", color=C_THRU)
    ax2.set_ylabel("Queue proxy (×10³)", color=C_QUEUE)
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=9)

    # ── 3. Policy Loss & Entropy ─────────────────────────────────────────────
    ax = axes[2]
    ax2 = ax.twinx()
    x_upd = upd["global_step"] / 1_000
    ax.plot(x_upd,  rolling(upd["policy_loss"], ROLL_UPD), color=C_LOSS, linewidth=2.5, label="Policy Loss")
    ax2.plot(x_upd, rolling(upd["entropy"],     ROLL_UPD), color=C_ENT,  linewidth=2.0, linestyle="--", label="Entropy")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Policy Loss & Entropy")
    ax.set_xlabel("Global Step (×10³)")
    ax.set_ylabel("Policy Loss", color=C_LOSS)
    ax2.set_ylabel("Entropy H(π)", color=C_ENT)
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig3_summary_slide.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Reward Distribution (histogram so sánh early vs late training)
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure4(ep: pd.DataFrame):
    n = len(ep)
    early = ep[ep["episode"] <= n * 0.2]["mean_reward"]
    late  = ep[ep["episode"] >  n * 0.8]["mean_reward"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reward Distribution — Early vs Late Training", fontsize=14, fontweight="bold")

    bins = np.linspace(ep["mean_reward"].min() - 2, ep["mean_reward"].max() + 2, 40)

    # Histogram
    ax = axes[0]
    ax.hist(early, bins=bins, color="#EF9A9A", edgecolor="white", alpha=0.8, label=f"Early 20% (ep 1–{int(n*0.2)})")
    ax.hist(late,  bins=bins, color=C_SMOOTH,  edgecolor="white", alpha=0.8, label=f"Late 20%  (ep {int(n*0.8)}–{n})")
    ax.axvline(early.mean(), color="#C62828", linestyle="--", linewidth=1.8, label=f"Early mean = {early.mean():.1f}")
    ax.axvline(late.mean(),  color="#0D47A1", linestyle="--", linewidth=1.8, label=f"Late mean  = {late.mean():.1f}")
    ax.set_title("Reward Histogram")
    ax.set_xlabel("Mean Reward per Episode")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(
        [early.values, late.values],
        labels=["Early 20%", "Late 20%"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor("#EF9A9A")
    bp["boxes"][1].set_facecolor(C_SMOOTH)
    ax.set_title("Reward Box Plot")
    ax.set_ylabel("Mean Reward per Episode")

    # Ghi chú cải thiện
    improvement = late.mean() - early.mean()
    ax.text(1.5, (early.mean() + late.mean()) / 2,
            f"Δ = {improvement:+.1f}", ha="center", fontsize=11,
            color=C_SMOOTH if improvement > 0 else "#C62828", fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig4_reward_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Traffic Guard AI - MAPPO Training Plot Generator")
    print("=" * 55)

    ep, upd = load_data()

    print("\nVe Figure 1: Reward & Traffic Performance...")
    plot_figure1(ep)

    print("Ve Figure 2: Policy Training Losses...")
    plot_figure2(upd)

    print("Ve Figure 3: Summary Chart (danh cho slide)...")
    plot_figure3(ep, upd)

    print("Ve Figure 4: Reward Distribution (early vs late)...")
    plot_figure4(ep)

    print(f"\nTat ca bieu do da luu tai: {OUT_DIR}")
    print("\nThong ke nhanh:")
    print(f"  Total episodes (sau loc): {len(ep)}")
    print(f"  Reward early 20%: {ep[ep['episode'] <= len(ep)*0.2]['mean_reward'].mean():.2f}")
    print(f"  Reward late  20%: {ep[ep['episode'] >  len(ep)*0.8]['mean_reward'].mean():.2f}")
    print(f"  Throughput trung binh: {ep['throughput'].mean():.1f} xe/episode")
    print(f"  Throughput cao nhat:   {ep['throughput'].max():.0f} xe/episode")


if __name__ == "__main__":
    main()
