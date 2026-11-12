"""Overlay mean regret (3 seeds) of several result folders/modes on one log-log plot."""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

TAG = sys.argv[1] if len(sys.argv) > 1 else "1e6"
# series = [  # (folder, mode, label, color)
#     (f"results_{TAG}_numba", "computed", "computed epochs", "#2a78d6"),
#     (f"results_{TAG}_numba", "early", "early-stop epochs", "#eb6834"),
#     (f"results_{TAG}_fixed100", "fixed", "fixed, 100 steps", "#1baf7a"),
#     (f"results_{TAG}_fixed1000", "fixed", "fixed, 1,000 steps", "#eda100"),
#     (f"results_{TAG}_fixed10000", "fixed", "fixed, 10,000 steps", "#e87ba4"),
# ]
series = [  # (folder, mode, label, color)
    ("results/v3_10x10_M1",            "computed", "computed epochs",     "#2a78d6"),
    ("results/v3_10x10_M1",            "early",    "early-stop epochs",   "#eb6834"),
    ("results/v3_10x10_M1_fixed1000",  "fixed",    "fixed, 1,000 steps",  "#eda100"),
    ("results/v3_10x10_M1_fixed10000", "fixed",    "fixed, 10,000 steps", "#e87ba4"),
]
fig, ax = plt.subplots(figsize=(7, 4.4), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
D_u = None
for folder, mode, label, c in series:
    z = np.load(f"{folder}/regret_{mode}.npz")
    K, C = z["K"], z["regret"]
    S = C.shape[0]
    m = C.mean(0); half = student_t.ppf(0.975, S - 1) * C.std(0, ddof=1) / math.sqrt(S)
    ax.plot(K, m, color=c, lw=2, label=label)
    ax.fill_between(K, np.maximum(m - half, 1e-6), m + half, color=c, alpha=0.15, lw=0)
    D_u = json.load(open(f"{folder}/summary.json"))["D_u"]
ax.axhline(D_u, color="#52514e", lw=1, ls="--")
ax.text(1.2, D_u * 1.08, "uniform coverage", color="#52514e", fontsize=8)
for eps in (0.5 * D_u, 0.1 * D_u):
    ax.axhline(eps, color="#9a9994", lw=0.8, ls=":")
    ax.text(1.2, eps * 1.08, f"ε = {eps:.3g}", color="#52514e", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("time step K (per agent)"); ax.set_ylabel("regret  (1/K) Σ ‖ρ̂ₖ − ρ*‖₁")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"compare_{TAG}.png", dpi=160); print(f"saved compare_{TAG}.png")
