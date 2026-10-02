#!/usr/bin/env python
"""
Plot GP refit wall-clock from the gp_timing dataset (T4(a) / sweep F).

    python3 viz/plot_gp_timing.py --parent results/gp_studies_H

gp_timing columns: agent, t, n_fit, fit_s, predict_s
One marker/colour per run folder, labelled by learn mode and kernel from the stem.
"""
import argparse, glob, os, re
import h5py
import numpy as np
import matplotlib.pyplot as plt


FULL = 50000   # gp_obs_hist >= FULL means "full history" (BIG in run_batch.py)

def label(name):
    g = re.search(r"_gpm(exact|sgpr)_M(\d+)_bin(\d+)_fitc(\d)", name)
    B = re.search(r"_GPobs(\d+)", name)
    if not g:
        return "legacy"
    m, M, b, f = g.groups()
    s = f"Sparse M={M}" if m == "sgpr" else (f"Fused Δ={b}" if b != "0" else "Exact")
    mem = "full history" if B and int(B.group(1)) >= FULL else (f"B={B.group(1)}" if B else "")
    return s + (" FITC" if f == "1" else "") + (f", {mem}" if mem else "")

def band(ax, t, v, color, label=None, logy=True):
    ts = np.unique(t)
    q = np.array([np.percentile(v[t == s], [25, 50, 75]) for s in ts])
    ax.plot(ts, q[:, 1], color=color, lw=1.3, label=label)
    ax.fill_between(ts, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)

def per_agent(ax, ag, t, v, color, label=None):
    """Every agent at every refit, one dot each."""
    ax.scatter(t, v, s=4, alpha=0.3, color=color, lw=0, rasterized=True, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agg", choices=["median", "agents"], default="median",
                    help="median: median+IQR across agents and runs; agents: one line per agent")
    ap.add_argument("--agent", type=int, default=None, help="plot only this agent id")
    ap.add_argument("--x", choices=["n", "t"], default="t")
    a = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.parent, "*", "run.h5"))):
        with h5py.File(p, "r") as f:
            if "gp_timing" not in f:
                print("skip (no gp_timing):", os.path.basename(os.path.dirname(p)))
                continue
            g = f["gp_timing"][()]
        if len(g):
            runs.setdefault(label(os.path.basename(os.path.dirname(p))), []).append(g)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ci, lab in enumerate(sorted(runs)):
        g = np.vstack(runs[lab])
        g = g[(g[:, 2] >= 2) & (g[:, 4] > 0)]            # drop warm-up rows
        if a.agent is not None:
            g = g[g[:, 0] == a.agent]
            if not len(g):
                continue
        ag, t, n, fit, pred = g[:, 0], g[:, 1], g[:, 2], 1e3 * g[:, 3], 1e3 * g[:, 4]
        c = f"C{ci}"
        draw = band if a.agg == "median" else (lambda ax_, t_, v_, c_, label=None:
                                               per_agent(ax_, ag, t_, v_, c_, label))
        draw(ax[0], t, n, c, label=lab)
        draw(ax[1], t, fit, c)
        draw(ax[2], t, pred, c)
        print(f"{lab:35s} runs={len(runs[lab])}  N_max={n.max():.0f}  "
              f"fit={np.median(fit):8.1f} ms (median)  predict={np.median(pred):6.1f} ms")

    titles = ["Points in GP solve, N", "GP fit [ms]", "GP predict, all regions [ms]"]
    for i, (axi, tt) in enumerate(zip(ax, titles)):
        if i > 0:
            axi.set_yscale("log")
        axi.set_title(tt); axi.set_xlabel("time step t"); axi.grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, markerscale=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = a.out or os.path.join(a.parent, "analysis", "gp_timing.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()