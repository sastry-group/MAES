#!/usr/bin/env python
"""
Plot GP refit wall-clock from the gp_timing dataset (T4(a) / sweep F).

    python3 viz/plot_gp_timing.py --parent results/gp_study_H

gp_timing columns: agent, t, n_fit, fit_s, predict_s
One marker/colour per run folder, labelled by learn mode and kernel from the stem.
"""
import argparse, glob, os, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FULL = 50000   # gp_obs_hist >= FULL means "full history" (BIG in run_batch.py)


GP_KEYS = [("exact", "100000", "0",  "200"), ("exact", "1000", "0",  "200"),
           ("exact", "100000", "25", "200"), ("exact", "1000", "25", "200"),
           ("sgpr",  "100000", "0",  "500"), ("sgpr",  "1000", "0",  "500"),
           ("sgpr",  "100000", "0",  "1000")]                   # legend order
GP_LABELS = {("exact", "100000", "0",  "200"):  "Exact",
             ("exact", "1000",   "0",  "200"):  r"Exact, $B=1000$",
             ("exact", "100000", "25", "200"):  "Fused (25)",
             ("exact", "1000",   "25", "200"):  r"Fused (25), $B=1000$",
             ("sgpr",  "100000", "0",  "500"):  r"SGPR, $M=500$",
             ("sgpr",  "1000",   "0",  "500"):  r"SGPR, $M=500$, $B=1000$",
             ("sgpr",  "100000", "0",  "1000"): r"SGPR, $M=1000$"}
GP_STYLE = {("exact", "100000", "0",  "200"):  ("#0072B2", "-"),    # full history: Okabe-Ito
            ("exact", "1000",   "0",  "200"):  ("#6BAED6", "--"),   # B=1000: darker shade
            ("exact", "100000", "25", "200"):  ("#E36C0A", "-"),
            ("exact", "1000",   "25", "200"):  ("#F5A65B", "--"),
            ("sgpr",  "100000", "0",  "500"):  ("#00838F", "-"),
            ("sgpr",  "1000",   "0",  "500"):  ("#5CC0C7", "--"),
            ("sgpr",  "100000", "0",  "1000"): ("#CC79A7", "-")}
 
# panel -> (gp_timing column, scale, y label, log y)
PANELS = {"n":    (2, 1.0, r"Observations $N$", False),
          "fit":  (3, 1e3, "GP fit [ms]", True),
          "pred": (4, 1e3, "GP predict [ms]", True)}
 
COL_W, PAGE_W = 3.45, 7.16   # IEEE single / double column, inches
 

def set_style():   # match with  plot_for_paper.py
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "grid.linewidth": 0.4, "grid.alpha": 0.3,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def label(name):   # kept: plot_for_paper.py still imports this
    g = re.search(r"_gpm(exact|sgpr)_M(\d+)_bin(\d+)_fitc(\d)", name)
    B = re.search(r"_GPobs(\d+)", name)
    if not g:
        return "legacy"
    m, M, b, f = g.groups()
    s = f"Sparse M={M}" if m == "sgpr" else (f"Fused Δ={b}" if b != "0" else "Exact")
    mem = "full history" if B and int(B.group(1)) >= FULL else (f"B={B.group(1)}" if B else "")
    return s + (" FITC" if f == "1" else "") + (f", {mem}" if mem else "")
 
 
def gp_key(name):
    g = re.search(r"_gpm(exact|sgpr)_M(\d+)_bin(\d+)_fitc\d", name)
    B = re.search(r"_GPobs(\d+)", name)
    return (g.group(1), B.group(1), g.group(3), g.group(2)) if g and B else None
 
 
def band(ax, t, v, color, label=None, logy=True, ls="-"):
    """Median + IQR across agents and runs at each t. Returns the end point."""
    ts = np.unique(t)
    q = np.array([np.percentile(v[t == s], [25, 50, 75]) for s in ts])
    ax.plot(ts, q[:, 1], color=color, ls=ls, label=label)
    ax.fill_between(ts, q[:, 0], q[:, 2], color=color, alpha=0.15, lw=0)
    return ts[-1], q[-1, 1]
 
 
def per_agent(ax, ag, t, v, color, label=None, ls="-"):
    """Every agent at every refit, one dot each. Returns the end point (median)."""
    ax.scatter(t, v, s=1.5, alpha=0.3, color=color, lw=0, rasterized=True, label=label)
    last = t.max()
    return last, np.median(v[t == last])
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", default=None, help="output stem (no extension)")
    ap.add_argument("--fmt", nargs="+", default=["pdf", "png"])
    ap.add_argument("--panels", nargs="+", choices=list(PANELS), default=["fit", "pred"],
                    help="which panels, in order (max 3)")
    ap.add_argument("--n_annot", choices=["end", "legend", "none"], default="none",
                    help="where to show N at the last refit")
    ap.add_argument("--agg", choices=["median", "agents"], default="agents",
                    help="median: median+IQR across agents and runs; agents: one dot per agent")
    ap.add_argument("--agent", type=int, default=None, help="plot only this agent id")
    ap.add_argument("--x", choices=["n", "t"], default="t")
    a = ap.parse_args()
    if len(a.panels) > 3:
        raise SystemExit("at most 3 panels")
    annot = None if (a.n_annot == "none" or a.x == "n") else a.n_annot
 
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.parent, "*", "run.h5"))):
        name = os.path.basename(os.path.dirname(p))
        k = gp_key(name)
        if k not in GP_LABELS:
            print("skip (variant not in GP_LABELS):", name)
            continue
        with h5py.File(p, "r") as f:
            if "gp_timing" not in f:
                print("skip (no gp_timing):", name)
                continue
            g = f["gp_timing"][()]
        if len(g):
            runs.setdefault(k, []).append(g)
    if not runs:
        raise SystemExit(f"no gp_timing data under {a.parent}")
 
    set_style()
    npan = len(a.panels)
    fig, axes = plt.subplots(1, npan, figsize=(COL_W if npan <= 2 else PAGE_W, 2.3),
                             sharex=True, layout="constrained", squeeze=False)
    axes = axes[0]
    xmax = 0
    for k in [k for k in GP_KEYS if k in runs]:
        g = np.vstack(runs[k])
        g = g[(g[:, 2] >= 2) & (g[:, 4] > 0)]            # drop warm-up rows
        if a.agent is not None:
            g = g[g[:, 0] == a.agent]
            if not len(g):
                continue
        ag, t, n = g[:, 0], g[:, 1], g[:, 2]
        x = n if a.x == "n" else t
        xmax = max(xmax, x.max())
        c, ls = GP_STYLE[k]
        nN = np.median(n[t == t.max()])                  # N at the last refit
        lbl = GP_LABELS[k] + (f" ({nN / 1e3:.1f}k obs.)" if annot == "legend" else "")
        for i, p in enumerate(a.panels):
            col, scale, _, _ = PANELS[p]
            v = g[:, col] * scale
            if a.agg == "median":
                xe, ye = band(axes[i], x, v, c, lbl if i == 0 else None, ls=ls)
            else:
                xe, ye = per_agent(axes[i], ag, x, v, c, lbl if i == 0 else None, ls=ls)
            if annot == "end" and i == npan - 1:
                axes[i].annotate(f"{nN / 1e3:.1f}k", (xe, ye), xytext=(2, 0),
                                 textcoords="offset points", ha="left", va="center",
                                 fontsize=5, color=c)
        print(f"{GP_LABELS[k]:28s} runs={len(runs[k])}  N_end={nN:.0f}  "
              f"fit={1e3 * np.median(g[:, 3]):8.1f} ms (median)  "
              f"predict={1e3 * np.median(g[:, 4]):6.1f} ms")
 
    for i, (ax, p) in enumerate(zip(axes, a.panels)):
        _, _, yl, logy = PANELS[p]
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(yl)
        ax.set_xlabel("Time step" if a.x == "t" else r"Observations $N$")
        ax.set_title(f"({chr(97 + i)})", loc="left", pad=2, fontweight="bold")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    if annot == "end":
        axes[-1].set_xlim(right=1.15 * xmax)             # room for the N labels
 
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="outside lower center", ncol=2 if npan <= 2 else 4,
                     markerscale=4 if a.agg == "agents" else 1)
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1)                                  # solid swatches for faint scatter
 
    stem = a.out or os.path.join(a.parent, "analysis", "gp_timing")
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in a.fmt:
        fig.savefig(f"{stem}.{ext}")
        print("saved", f"{stem}.{ext}")
 
 
if __name__ == "__main__":
    main()