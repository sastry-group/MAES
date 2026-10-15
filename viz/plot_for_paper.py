#!/usr/bin/env python3
# viz/plot_for_paper.py
"""
Paper-ready versions of plot_sweep / plot_gp_timing (those stay as your scratch tools).

    python3 viz/plot_for_paper.py --list
    python3 viz/plot_for_paper.py --fig kernel
    python3 viz/plot_for_paper.py --fig kernel timing
    python3 viz/plot_for_paper.py --all
    python3 viz/plot_for_paper.py --fig kernel --parent results/gp_studies_D_10 --refresh
    python3 viz/plot_for_paper.py --fig gp_process --parent results/gp_study_H --refresh
    python3 viz/plot_for_paper.py --fig planner --parent results/PLANNER --refresh
Edit the FIGURES dict below to choose runs, panels and order. Each entry is one
half-column (3.45 in) figure, saved as vector PDF + 300-dpi PNG.
Needs matplotlib >= 3.7 (constrained layout + "outside" legend).
"""
import os
import sys
import glob
import argparse
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from viz import plot_sweep as ps
from viz import plot_gp_timing as pt

# ── style ────────────────────────────────────────────────────────────────────
COL_W = 3.45            # IEEE single-column width, inches
FIG_H = 3.05            # 2x2 panels + legend row
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 5.5,
        "axes.linewidth": 0.6, "lines.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "grid.linewidth": 0.4, "grid.alpha": 0.3,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0, "legend.handletextpad": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType (IEEE-safe)
        "savefig.dpi": 300,
    })


# ── legend labels: key -> value-string -> pretty label ───────────────────────
NAMES = {
    "kernel": {"matern12": r"Mat\'ern-1/2", "matern32": r"Mat\'ern-3/2", "matern52": r"Mat\'ern-5/2", "rbf": "RBF", "prod_matern_exp":r"Separable Exp. Mat\'ern"},
    # "planner": {"ergodic": "Ours", "mac_dt": "MAC-DT"},        # EDIT to match the paper
    # "learn": {"none": "fixed", "once": "learn once", "every": "learn every step"},
}
NAMES["kernel"] = {"matern12": "Matérn-1/2", "matern32": "Matérn-3/2", "matern52": "Matérn-5/2", "rbf": "RBF"}
NAMES["kernel"]["prod_matern_exp"] = "Matérn-3/2xExp"
KCOLOR = {("prod_matern_exp", "10000"): "#0072B2",   # blue   (chosen)
          ("prod_matern_exp", "1e+07"): "#56B4E9",   # light blue
          ("matern32", "10000"):        "#009E73",   # green
          ("matern32", "200"):          "#CC79A7",   # pink
          ("rbf", "10000"):             "#D52700"}   # vermillion
LTSTYLE = {"10000": "-", "200": "--", "1e+07": ":"}
FORMATS = {                                                      # key -> f-string
    "tau": r"$\tau_P={}$", "gp_obs_hist": r"$B={}$", "gp_M": r"$M={}$",
    "gp_bin_dt": r"bin {}", "noise_rel": r"$\sigma_{{rel}}={}$",
}


def pretty(key, val):
    if key == "l_time":
        return r"$\ell_t=\infty$" if float(val) >= 1e6 else rf"$\ell_t={float(val):g}$"
    if key in NAMES and val in NAMES[key]:
        return NAMES[key][val]
    if key in FORMATS:
        return FORMATS[key].format(val)
    return val


# ── what to plot (EDIT the parents / fix / compare) ──────────────────────────
FOUR = ["regret", "erg_err", "rmse", "msll"]   # regret, , belief L1, RMSE, MSLL, erg_err 
TIMING = {"gp_fit": ([3], "GP fit [ms]"), "gp_predict": ([4], "GP predict [ms]"),
          "gp_update": ([3, 4], "GP update [ms]")}   # gp_timing columns (summed), ylabel
FIGURES = {
    "kernel": dict(
        parent="results/gp_study_D", fix=["noise_rel=0.05"],
        compare=["kernel", "l_time"], average=["start", "map"],
        keep=[("prod_matern_exp", "10000"), ("prod_matern_exp", "200"),
              ("prod_matern_exp", "1e+07"), ("matern32", "10000"), ("matern32", "200"), ("rbf", "10000")],
        panels=["regret", "L_belief", "rmse","msll"], # ylim={"msll": (-5, 65)},
        zoom_panels=["regret"], vlines=True, logy=[],height=2.3, legend_ncol=3),
    "gp": dict(
        parent="results/gp_study_H",          # EDIT: folder with the T4000 ... gpm* runs
        compare=["gp_method", "gp_obs_hist", "gp_bin_dt", "gp_M"],
        average=["start", "map"],
        keep=[("exact", "100000", "0",  "200"),
              ("exact", "1000",   "0",  "200"),   
              ("exact", "100000", "25", "200"),
              ("exact", "1000",   "25", "200"),  
              ("sgpr",  "100000", "0",  "500"),
              ("sgpr",  "100000", "0",  "1000"),
              ("sgpr",  "1000",   "0",  "500")],
        labels={("exact", "100000", "0",  "200"):  "Exact",
                ("sgpr",  "100000", "0",  "500"):  r"SGPR, $M=500$",
                ("sgpr",  "100000", "0",  "1000"): r"SGPR, $M=1000$",
                ("exact", "100000", "25", "200"):  "Fused (25)",
                ("exact", "1000",   "0",  "200"):  r"Exact, $B=1000$",
                ("exact", "1000",   "25", "200"):  r"Fused (25), $B=1000$",
                ("sgpr",  "1000",   "0",  "500"):  r"SGPR, $M=500$, $B=1000$"},
        # colour = method, line style = memory (solid full history, dashed B=1000)
        style={("exact", "100000", "0",  "200"):  ("#0072B2", "-"),    # blue
               ("exact", "1000",   "0",  "200"):  ("#6BAED6", "--"),   # light blue
               ("exact", "100000", "25", "200"):  ("#E36C0A", "-"),    # orange (not yellow)
               ("exact", "1000",   "25", "200"):  ("#F5A65B", "--"),   # light orange
               ("sgpr",  "100000", "0",  "500"):  ("#00838F", "-"),    # dark cyan / teal
               ("sgpr",  "1000",   "0",  "500"):  ("#5CC0C7", "--"),   # light teal
               ("sgpr",  "100000", "0",  "1000"): ("#CC79A7", "-")},   # pink
        panels=FOUR, zoom_panels=["regret", "rmse", "msll", "erg_err"], vlines=False, logy=[]),
    "tau": dict(
        parent="results/TAU2",
        # fix=["kernel=prod_matern_exp", "l_time=10000"],

        # fix=["l_time=10000"],
        compare=["tau"], average=["start", "map"],
        keep=[("10",), ("25",), ("100",), ("500",)],
        style={
            # ("5",):   ("#56B4E9", "-"),
               ("10",):  ("#009E73", "-"),
               ("25",):  ("#0072B2", "-"),     # default tau_P, same blue as the kernel fig
               ("100",):  ("#E69F00", "-"),
               ("500",): ("#D55E00", "-")},
        panels=["regret", "rmse"], zoom_panels=["regret","erg_err"], vlines=True,legend_ncol=4, logy=[]),
    "timing": dict(kind="timing", parent="results/gp_study_D"),
    "planner": dict(
        parent="results/PLANNER_TEST3",
        
        compare=["planner", "n_agents"], average=["start", "map"],
        # keep=[("ergodic", "6"), ("ergodic", "10"), ("mac_dt", "6"), ("mac_dt", "10"),("hedac", "10"),("hedac", "6")],

        # keep=[("ergodic", "6"), ("ergodic", "10"), ("mac_dt", "6"), ("mac_dt", "10"),("hedac", "10"),("hedac", "6")],
        labels={("ergodic", "20"): "Ours, 20 agents",
                ("ergodic", "10"): "Ours, 10 agents",
                ("mac_dt", "20"):  "MAC-DT, 20 agents",
                ("mac_dt", "10"):  "MAC-DT, 10 agents",
                ("hedac", "20"):  "HEDAC, 20 agents",
                ("hedac", "6"):  "HEDAC, 6 agents",
                },
       
        style={("ergodic", "20"): ("#0072B2", "-"),
               ("ergodic", "10"): ("#6BAED6", "--"),
               ("mac_dt", "20"):  ("#444444", "-"),
               ("mac_dt", "10"):  ("#9E9E9E", "--"),
               ("hedac", "10"):  ("#ED64C4", "--"),
               ("hedac", "20"):  ("#F019B0", "-")
               },
        panels=["regret", "L_belief", "rmse", "msll"], zoom_panels=[],height=2.3, vlines=True,  legend_ncol=3, logy=[]),
    "comms": dict(
        parent="results/ROBUST",
        compare=["comm_radius", "comm_delay", "attrition"], average=["start", "map"],
        keep=[("0", "0", "0"), ("5", "0", "0"), ("15", "0", "0"),
              ("15", "10", "0")],
        labels={("0", "0", "0"):   r"$R_{comms}=None$",
                ("5", "0", "0"):   r"$R_{comms}=Mid$",
                ("15", "0", "0"):  r"$R_{comms}=Global$",
                ("15", "10", "0"): r"$R_{comms}=Global$, delay ~ 10 steps"},
        style={("0", "0", "0"):   ("#9ECAE1", "-"),
               ("5", "0", "0"):   ("#009E73", "-"),
               ("15", "0", "0"):  ("#08519C", "-"),
               ("15", "10", "0"): ("#08519C", "--")},
        panels=["regret", "L_belief", "rmse", "msll"], zoom_panels=["regret"],
        vlines=True, logy=[]),
    "attrition": dict(
        parent="results/ROBUST",
        compare=["comm_radius", "comm_delay", "attrition"], average=["start", "map"],
        keep=[("15", "0", "0"), ("15", "0", "2"), ("15", "0", "4"),
              ("0", "0", "0"),  ("0", "0", "2"),  ("0", "0", "4")],
        labels={("15", "0", "0"): r"$R_{comms}=Global$",
                ("15", "0", "2"): r"$R_{comms}=Global$, 2 agents lost",
                ("15", "0", "4"): r"$R_{comms}=Global$, 4 agents lost",
                ("0", "0", "0"):  r"$R_{comms}=None$",
                ("0", "0", "2"):  r"$R_{comms}=None$, 2 agents lost",
                ("0", "0", "4"):  r"$R_{comms}=None$, 4 agents lost"},
        style={("15", "0", "0"): ("#0072B2", "-"),
               ("15", "0", "2"): ("#E69F00", "-"),
               ("15", "0", "4"): ("#D55E00", "-"),
               ("0", "0", "0"):  ("#0072B2", "--"),
               ("0", "0", "2"):  ("#E69F00", "--"),
               ("0", "0", "4"):  ("#D55E00", "--")},
        panels=["regret", "L_belief", "rmse", "msll"], zoom_panels=["regret"],
        vlines=True, logy=[]),
}
FIGURES["gp_paper"] = dict(
    FIGURES["gp"], keep=[k for k in FIGURES["gp"]["keep"] if k != ("sgpr", "100000", "0", "1000")],
    panels=["regret", "gp_update"], zoom_panels=["regret"], height=1.6, legend_ncol=3, timing_dots=True)


# for the box zoom
ZOOM_LAST = 500                       # last N steps shown in the inset; 0 = off
ZOOM_RECT = {"default": [0.42, 0.42, 0.55, 0.50]}   # axes-fraction [x, y, w, h]
# per-panel override if it hits a curve, e.g. ZOOM_RECT["msll"] = [0.42, 0.08, 0.55, 0.45]
ZOOM_WIN = {"regret": (1500, 2000)}   # metric -> (x0, x1); others: last ZOOM_LAST steps
ZOOM_RECT["rmse"] = [0.45, 0.55, 0.52, 0.40]   # upper right, above the curves
ZOOM_RECT["msll"] = [0.45, 0.55, 0.52, 0.40]   # upper right is empty (curves go down)
ZOOM_RECT["erg_err"] = [0.45, 0.55, 0.52, 0.40]   # upper right is empty (curves go down)
ZOOM_PANELS = {"regret", "erg_err"}              # panels that get an inset; e.g. {"regret", "erg_err"}

# ── sweep-style figure ───────────────────────────────────────────────────────
def collect(spec, refresh):
    runs = [os.path.normpath(d)
            for d in sorted(glob.glob(os.path.join(spec["parent"], "*")))
            if os.path.isdir(d) and (glob.glob(os.path.join(d, "*.h5"))
                                     or glob.glob(os.path.join(d, "*.npz")))]
    if not runs:
        raise SystemExit(f"no runs under {spec['parent']}")
    args = SimpleNamespace(parent=spec["parent"], refresh=refresh, emp_window=500,
                           msll_sigma_eps=None, stop_on_error=False)
    loaded = ps.load_all(runs, args)
    fixes = dict(f.split("=", 1) for f in spec.get("fix", []))
    loaded = ps.apply_fix(loaded, fixes)
    if not loaded:
        raise SystemExit(f"no runs left after fix {fixes}")
    grouped = defaultdict(list)
    for keys, ser in loaded:
        grouped[tuple(ps.as_str(keys.get(k)) for k in spec["compare"])].append(ser)
    return grouped, runs

def timing_by_group(spec):
    """group -> stacked gp_timing rows (agent, t, n_fit, fit_s, predict_s) over its runs."""
    fixes = dict(f.split("=", 1) for f in spec.get("fix", []))
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(spec["parent"], "*", "run.h5"))):
        with h5py.File(p, "r") as f:
            g = f["gp_timing"][()] if "gp_timing" in f else []
        if not len(g):
            continue
        d = os.path.dirname(p)
        data, _ = ps.run_io.load_run(d)
        k = ps.run_keys(d, data)
        if all(ps.as_str(k.get(fk)) == v for fk, v in fixes.items()):
            out[tuple(ps.as_str(k.get(c)) for c in spec["compare"])].append(g)
    return {grp: np.vstack(gs) for grp, gs in out.items()}


def curve_style(spec, g, ci):
    sty = spec.get("style", {}).get(g)
    c = sty[0] if sty else KCOLOR.get(g, OKABE_ITO[ci % len(OKABE_ITO)])
    ls = sty[1] if sty else (LTSTYLE.get(g[1], "-") if len(g) > 1
                             else LINESTYLES[ci % len(LINESTYLES)])
    lbl = spec.get("labels", {}).get(g) or \
          ", ".join(pretty(k, v) for k, v in zip(spec["compare"], g))
    return c, ls, lbl

def change_times(runs):
    per = []
    for d in runs:
        p = os.path.join(d, "run.h5")
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            if "true_phi_change_times" in f:
                per.append(tuple(sorted(int(t) for t in f["true_phi_change_times"][()] if t > 0)))
    if not per:
        print("no true_phi_change_times found; no vertical lines")
        return []
    if len(set(per)) > 1:
        print(f"!! change times differ across runs ({len(set(per))} patterns); drawing the union")
    return sorted(set().union(*per))


def dry_report(name, spec):
    """Print which runs go into each curve. Reads h5 keys directly (no metrics, no cache)."""
    runs = [os.path.normpath(d)
            for d in sorted(glob.glob(os.path.join(spec["parent"], "*")))
            if os.path.isdir(d) and (glob.glob(os.path.join(d, "*.h5"))
                                     or glob.glob(os.path.join(d, "*.npz")))]
    fixes = dict(f.split("=", 1) for f in spec.get("fix", []))
    rows = []
    for d in runs:
        data, _ = ps.run_io.load_run(d)
        k = ps.run_keys(d, data)
        if all(ps.as_str(k.get(f)) == v for f, v in fixes.items()):
            rows.append((os.path.basename(d), k))
    cmp_, avg = spec["compare"], spec.get("average", [])
    print(f"\n== {name}: {spec['parent']}  fix={fixes}  -> {len(rows)}/{len(runs)} runs")

    assigned = set(fixes) | set(cmp_) | set(avg)
    for key in ps.KEYS:
        vals = sorted({ps.as_str(k.get(key)) for _, k in rows})
        if key not in assigned and len(vals) > 1:
            print(f"!! {key} still varies {vals} -- would be averaged into the band")

    grouped = defaultdict(list)
    for run, k in rows:
        grouped[tuple(ps.as_str(k.get(c)) for c in cmp_)].append((run, k))
    keep = [tuple(g) for g in spec.get("keep", [])] or sorted(grouped)
    for g in keep:
        members = grouped.get(g, [])
        combos = [tuple(ps.as_str(k.get(a)) for a in avg) for _, k in members]
        dup = sorted({c for c in combos if combos.count(c) > 1})
        tag = "!! MISSING" if not members else (f"!! DUPLICATE {avg}={dup}" if dup else "")
        print(f"  {g}  n={len(members)}  {tag}")
        for run, _ in sorted(members):
            print("      " + run)
    unused = sorted(set(grouped) - set(keep))
    if unused:
        print(f"  (in folder, not in keep: {unused})")
    ns = sorted({len(grouped.get(g, [])) for g in keep})
    if len(ns) > 1:
        print(f"!! unequal n across curves: {ns}")



def add_zoom(ax, drawn, metric, last, logy=False):
    """drawn: list of (t, mean, color, ls). Inset on the last `last` steps."""
    n = max(len(t) for t, *_ in drawn)
    if last <= 0 or n <= last:
        return
    x0, x1 = n - last, n - 1
    x0, x1 = ZOOM_WIN.get(metric, (x0, x1))
    wins = [m[x0:x1 + 1] for t, m, *_ in drawn if len(m) > x0]
    lo, hi = min(w.min() for w in wins), max(w.max() for w in wins)
    pad = 0.2 * (hi - lo + 1e-10)
    rect = ZOOM_RECT.get(metric, ZOOM_RECT["default"])
    axin = ax.inset_axes(rect)
    for t, m, c, ls in drawn:
        axin.plot(t, m, color=c, ls=ls, lw=0.9)
    axin.set_xlim(x0, x1)
    axin.set_ylim(lo - pad, hi + pad)
    if logy:
        axin.set_yscale("log")
    axin.xaxis.set_major_locator(MaxNLocator(2))
    axin.yaxis.set_major_locator(MaxNLocator(3))
    axin.tick_params(labelsize=5, length=1.5, pad=1)
    axin.grid(True, alpha=0.3)
    for s in axin.spines.values():
        s.set_linewidth(0.5)
    ax.add_patch(Rectangle((x0, lo - pad), x1 - x0, hi - lo + 2 * pad,
                           fill=False, ec="0.15", lw=0.8, ls="--"))
    # ax.indicate_inset_zoom(axin, edgecolor="0.15", lw=0.8, alpha=1)    

def plot_sweep_fig(name, spec, outdir, refresh, fmts, zoom=ZOOM_LAST):
    grouped, runs = collect(spec, refresh)
    keep = spec.get("keep")
    if keep:
        groups = [tuple(k) for k in keep if tuple(k) in grouped]
    else:
        order = spec.get("order")
        groups = sorted(grouped, key=lambda g: (order.index(g[0]) if order and g[0] in order
                                                else len(order or []), g))
    # panels = [p for p in spec["panels"] if any(p in s for g in groups for s in grouped[g])]
    panels = [p for p in spec["panels"]
              if p in TIMING or any(p in s for g in groups for s in grouped[g])]
    timing = timing_by_group(spec) if any(p in TIMING for p in panels) else {}
    tchg = change_times(runs) if spec.get("vlines") else []
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_W, spec.get("height", FIG_H * nrows / 2)),
                             sharex=True, layout="constrained", squeeze=False)
    for i, metric in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        drawn = []
        for ci, g in enumerate(groups):
            if metric in TIMING:                         # median + IQR over agents and runs, per refit
                G = timing.get(g)
                if G is None:
                    continue
                G = G[(G[:, 2] >= 2) & (G[:, 4] > 0)]    # drop warm-up rows
                ts = np.unique(G[:, 1])
                # q = np.array([np.percentile(1e3 * G[G[:, 1] == s, TIMING[metric][0]], [25, 50, 75])
                #               for s in ts])
                v = 1e3 * G[:, TIMING[metric][0]].sum(axis=1)   # per agent, per refit
                q = np.array([np.percentile(v[G[:, 1] == s], [25, 50, 75]) for s in ts])
                # c, ls, lbl = curve_style(spec, g, ci)
                # ax.plot(ts, q[:, 1], color=c, ls=ls, label=lbl)
                # ax.fill_between(ts, q[:, 0], q[:, 2], color=c, alpha=0.15, lw=0)
                c, ls, lbl = curve_style(spec, g, ci)
                if spec.get("timing_dots"):              # every agent, every refit, all runs
                    ax.scatter(G[:, 1], v, s=0.8, alpha=0.25, color=c, lw=0,
                               rasterized=True, zorder=1)
                else:
                    ax.fill_between(ts, q[:, 0], q[:, 2], color=c, alpha=0.15, lw=0)
                ax.plot(ts, q[:, 1], color=c, ls=ls, label=lbl, zorder=2)
                continue
            m, sd, n = ps.band(grouped[g], metric)
            if m is None:
                continue
            t = np.arange(len(m))
            # c = KCOLOR.get(g[0], OKABE_ITO[ci % len(OKABE_ITO)])
            # c = KCOLOR.get(g, OKABE_ITO[ci % len(OKABE_ITO)])
            # ls = LTSTYLE.get(g[1], "-") if len(g) > 1 else LINESTYLES[ci % len(LINESTYLES)]
            sty = spec.get("style", {}).get(g)
            c = sty[0] if sty else KCOLOR.get(g, OKABE_ITO[ci % len(OKABE_ITO)])
            ls = sty[1] if sty else (LTSTYLE.get(g[1], "-") if len(g) > 1
                                     else LINESTYLES[ci % len(LINESTYLES)])
            # lbl = ", ".join(pretty(k, v) for k, v in zip(spec["compare"], g))
            lbl = spec.get("labels", {}).get(g) or \
                  ", ".join(pretty(k, v) for k, v in zip(spec["compare"], g))
            ax.plot(t, m, color=c, ls=ls, label=lbl)
            drawn.append((t, m, c, ls))
            if n > 1 and spec.get("bands", True):
                ax.fill_between(t, m - sd, m + sd, color=c, alpha=0.08, lw=0)
        for tc in tchg:
            ax.axvline(tc, color="0.55", lw=0.5, ls=":", zorder=0)
        # ax.set_ylabel(ps.LABELS[metric])
        ax.set_ylabel(TIMING[metric][1] if metric in TIMING else ps.LABELS[metric])
        ax.set_title(f"({chr(97 + i)})", loc="left", pad=2, fontweight="bold")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if metric in spec.get("ylim", {}):
            ax.set_ylim(*spec["ylim"][metric])
        # if metric in spec.get("logy", []):
        if metric in spec.get("logy", []) or metric in TIMING:
            ax.set_yscale("log")
        if metric == "msll":
            ax.axhline(0, color="k", lw=0.6)
        if i // ncols == nrows - 1 or i + ncols >= len(panels):
            ax.set_xlabel("Time step")
        if drawn and metric in spec.get("zoom_panels", ZOOM_PANELS):
            add_zoom(ax, drawn, metric, zoom, logy=metric in spec.get("logy", []))
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    if len(panels) < nrows * ncols:                      # legend in the blank slot
        axes[-1][-1].legend(h, l, loc="center", frameon=False)
    else:
        # fig.legend(h, l, loc="outside lower center", ncol=min(len(l), 2))
        fig.legend(h, l, loc="outside lower center", ncol=spec.get("legend_ncol", min(len(l), 2)))
    save(fig, name, outdir, fmts)


# ── timing figure (reads run.h5 gp_timing like plot_gp_timing.py) ────────────
def plot_timing_fig(name, spec, outdir, refresh, fmts):
    runs = {}
    for p in sorted(glob.glob(os.path.join(spec["parent"], "*", "run.h5"))):
        with h5py.File(p, "r") as f:
            if "gp_timing" not in f:
                continue
            g = f["gp_timing"][()]
        if len(g):
            runs.setdefault(pt.label(os.path.basename(os.path.dirname(p))), []).append(g)
    if not runs:
        raise SystemExit(f"no gp_timing data under {spec['parent']}")

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharex=True,
                             layout="constrained")
    for ci, (lab, gs) in enumerate(runs.items()):
        g = np.vstack(gs)
        n, fit, pred = g[:, 2], 1e3 * g[:, 3], 1e3 * g[:, 4]
        c = OKABE_ITO[ci % len(OKABE_ITO)]
        axes[0].scatter(n, fit, s=1.5, alpha=0.3, color=c, lw=0, rasterized=True, label=lab)
        axes[1].scatter(n, pred, s=1.5, alpha=0.3, color=c, lw=0, rasterized=True)
    for i, (ax, yl) in enumerate(zip(axes, ["GP fit [ms]", "GP predict [ms]"])):
        ax.set_yscale("log")
        ax.set_xlabel(r"Observations $N\,(\leq B)$")
        ax.set_ylabel(yl)
        ax.set_title(f"({chr(97 + i)})", loc="left", pad=2, fontweight="bold")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(len(l), 3), markerscale=4)
    save(fig, name, outdir, fmts)


def save(fig, name, outdir, fmts):
    os.makedirs(outdir, exist_ok=True)
    for ext in fmts:
        out = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(out)
        print("saved", out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", choices=list(FIGURES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parent", default=None, help="override parent (single --fig only)")
    ap.add_argument("--outdir", default="results/paper_figs")
    ap.add_argument("--fmt", nargs="+", default=["pdf", "png"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--zoom", type=int, default=ZOOM_LAST, help="inset window (steps), 0=off")
    ap.add_argument("--dry", action="store_true", help="print curve membership and exit")
    a = ap.parse_args()

    if a.list:
        for k, v in FIGURES.items():
            print(f"{k:8s} {v['parent']}  {v.get('kind', 'sweep')}")
        return
    names = list(FIGURES) if a.all else (a.fig or [])
    if not names:
        raise SystemExit("give --fig NAME... or --all (see --list)")
    if a.parent and len(names) != 1:
        raise SystemExit("--parent only works with a single --fig")

    set_style()
    for n in names:
        spec = dict(FIGURES[n])
        if a.parent:
            spec["parent"] = a.parent
        if a.dry:
            if spec.get("kind") != "timing":
                dry_report(n, spec)
            continue
        if spec.get("kind") == "timing":
            plot_timing_fig(n, spec, a.outdir, a.refresh, a.fmt)
        else:
            plot_sweep_fig(n, spec, a.outdir, a.refresh, a.fmt, zoom=a.zoom)


if __name__ == "__main__":
    main()