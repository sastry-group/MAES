#!/usr/bin/env python
"""
Plot GP refit wall-clock from the gp_timing dataset (T4(a) / sweep F).

    python3 viz/plot_gp_timing.py --parent results/gp_studies_F

gp_timing columns: agent, t, n_fit, fit_s, predict_s
One marker/colour per run folder, labelled by learn mode and kernel from the stem.
"""
import argparse, glob, os, re
import h5py
import numpy as np
import matplotlib.pyplot as plt


def label(name):
    g = re.search(r"_gpm(exact|sgpr)_M(\d+)_bin(\d+)_fitc(\d)", name)
    B = re.search(r"_GPobs(\d+)", name)
    if not g:
        return "legacy"
    m, M, b, f = g.groups()
    s = f"sgpr M={M}" if m == "sgpr" else ("bin" + b if b != "0" else "exact")
    return s + (" fitc" if f == "1" else "") + (f", B={B.group(1)}" if B else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", default=None)
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

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lab, gs in runs.items():
        g = np.vstack(gs)
        n, fit, pred = g[:, 2], 1e3 * g[:, 3], 1e3 * g[:, 4]
        sc = ax[0].scatter(n, fit, s=6, alpha=0.4, label=lab)
        ax[1].scatter(n, pred, s=6, alpha=0.4, color=sc.get_facecolor()[0])
        print(f"{lab:35s} runs={len(gs)}  N_max={n.max():.0f}  "
              f"fit={np.median(fit):8.1f} ms (median)  predict={np.median(pred):6.1f} ms")

    for axi, t in zip(ax, ["GP fit", "GP predict (all cells)"]):
        axi.set_yscale("log"); axi.set_xlabel("observations in GP, N (≤ B)")
        axi.set_ylabel("wall-clock per refit [ms]"); axi.set_title(t); axi.grid(alpha=0.3)
    ax[0].legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    out = a.out or os.path.join(a.parent, "analysis", "gp_timing.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()