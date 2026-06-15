"""Quick convergence diagnostic for a single run (n3_grid/seed42 probe).

Plots episode-level traffic metrics + training-health signals vs global_step
with rolling-mean smoothing so we can eyeball whether the curve has flattened
(i.e. whether 500k/1M is a sufficient horizon for the campaign).
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = sys.argv[1] if len(sys.argv) > 1 else "20260613_152220"
LOG = Path("logs/rl") / RUN
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

ep = pd.read_csv(LOG / "train_episodes.csv")
up = pd.read_csv(LOG / "train_updates.csv")


def roll(s, w=15):
    return s.rolling(w, min_periods=1, center=True).mean()


def tail_delta(s, frac=0.2):
    """% change of the rolling-mean over the last `frac` of training (flatness)."""
    sm = roll(s)
    n = max(2, int(len(sm) * frac))
    a, b = sm.iloc[-n], sm.iloc[-1]
    return (b - a) / (abs(a) + 1e-9) * 100.0


panels = [
    ("mean_reward",       ep, "Episode mean reward",      False),
    ("avg_waiting_proxy", ep, "Avg waiting proxy (s)",    True),
    ("queue_total_proxy", ep, "Queue total proxy",        True),
    ("throughput",        ep, "Throughput (veh/ep)",      False),
    ("explained_variance", up, "Critic explained var",    False),
    ("approx_kl",         up, "Approx KL",                True),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, df, title, lower_better) in zip(axes.ravel(), panels):
    x = df.global_step / 1000.0
    ax.plot(x, df[col], color="0.8", lw=0.7, label="raw")
    ax.plot(x, roll(df[col]), color="C0", lw=2.0, label="rolling(15)")
    d = tail_delta(df[col])
    arrow = "↓" if d < 0 else "↑"
    ax.set_title(f"{title}\n(last-20% Δ = {d:+.1f}% {arrow})", fontsize=10)
    ax.set_xlabel("global step (k)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

fig.suptitle(f"Convergence probe — {RUN}  (n3_grid / mappo / seed 42, 500k steps)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"convergence_probe_{RUN}.{ext}", dpi=130, bbox_inches="tight")

# print flatness summary to stdout for the assessment
print("FLATNESS (rolling-mean % change over last 20% of training):")
for col, df, title, lb in panels:
    print(f"  {title:28s} {tail_delta(df[col]):+7.1f}%")
print(f"\nSaved: {OUT / f'convergence_probe_{RUN}.png'} (+ .pdf)")
