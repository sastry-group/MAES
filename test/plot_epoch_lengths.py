"""Epoch length vs epoch number (log y) from a results folder's summary.json (seed 0)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "results_1e6"
s = json.load(open(f"{folder}/summary.json"))
colors = {"computed": "#2a78d6", "early": "#eb6834"}
fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for mode, d in s["modes"].items():
    ep = d["epochs_seed0"][:-1]          # drop the last epoch, truncated by the horizon
    e = [x["e"] for x in ep]
    n = [x["steps"] for x in ep]
    ax.plot(e, n, color=colors[mode], lw=2, marker="o", ms=4,
            markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=f"{mode} epochs")
ax.set_yscale("log")
ax.set_xlabel("epoch e", color="#0b0b0b")
ax.set_ylabel("epoch length (steps per agent)", color="#0b0b0b")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{folder}/epoch_lengths.png", dpi=160)
print("saved", f"{folder}/epoch_lengths.png")
