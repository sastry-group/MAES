#!/usr/bin/env python
"""
Reproduce the 3-panel update-frequency figure (regret / empirical-vs-true /
belief-vs-true) from HDF5 run folders.

Adapted from the old npz-based notebook plotting:
  - uses an h5-aware loader (run.h5) instead of run_*.npz
  - reconstructs the dense belief_phi_map from the sparse forward-fill
    snapshots saved by the new finalize() (d_belief_phi_t index)
The error computation (convergence_errors_from_run / empirical_full) is
unchanged and imported from metrics.py.
"""

import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# repo on path so `metrics` and `env.grid` import the same as the notebook
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics import convergence_errors_from_run, empirical_full, empirical_window, empirical_soft_forget
from env.grid import make_env_from_json


# ── CONFIG ──────────────────────────────────────────────────────────────────
BATCH_DIR = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/"
    "results/P_GP_update_6K"
)
OUTPUT_PNG = "P_GP_update_separate_3plots_uniform_window4000_6K_1tau_fix.png"
OUTPUT_PDF = "P_GP_update_separate_3plots_uniform_window4000_6K_1tau_fix.pdf"

ZOOM_LAST = 200   # window (in steps) shown in each zoomed inset
INSET_RECT = [0.70, 0.42, 0.35, 0.35] # for the zooom fox x,y,w,h

 # ── LOADER (h5 + legacy npz) ─────────────────────────────────────────────────
def load_run(run_dir: Path):
    npz_path = next(Path(run_dir).glob("run_*.npz"), None)
    if npz_path is not None:
        z = np.load(npz_path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        z.close()
        return data
 
    h5_path = Path(run_dir) / "run.h5"
    if h5_path.exists():
        data = {}
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                data[key] = f[key][()]
            for key, val in f.attrs.items():
                data[key] = val
        return data
 
    raise FileNotFoundError(f"No run_*.npz or run.h5 in {run_dir}")
 
 
def _item(v):
    return v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v
 

def densify(snaps, times, T):
    #this is a helper to patch the env changes without having to save every timestep of unchanged data in ergodic sim
    times = np.asarray(times)
    valid = times >= 0
    snaps = np.asarray(snaps)[valid]
    times = times[valid]
    rows = np.searchsorted(times, np.arange(T), side="right") - 1
    rows = np.clip(rows, 0, len(snaps) - 1)
    return snaps[rows] 
 
# ── belief -> dense (T, nstates), averaged over agents ───────────────────────
def _orient_belief(arr, nstates, n_agents):
    """Coerce a belief array to (time, nstates), averaging over agents if 3-D."""
    arr = np.asarray(arr)
 
    if arr.ndim == 3:
        # average out the agent axis (the one matching n_agents)
        if arr.shape[0] == n_agents:
            arr = arr.mean(axis=0)
        elif arr.shape[1] == n_agents:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=0)  # fallback: assume axis 0 is agents
 
    if arr.ndim != 2:
        return None
    # put nstates on the last axis
    if arr.shape[1] == nstates:
        return arr
    if arr.shape[0] == nstates:
        return arr.T
    return None
 
 
def get_belief_phi_map(data, nstates, n_agents):
    """
    Return the network-averaged belief as (T, nstates).
    Prefers belief_phi_map_avg; falls back to averaging per-agent belief_phi_map.
    """
    for key in ("belief_phi_map_avg", "belief_phi_map"):
        if key in data:
            oriented = _orient_belief(data[key], nstates, n_agents)
            if oriented is not None:
                return oriented
    raise KeyError(
        f"No usable belief array (nstates={nstates}). "
        f"Available keys: {sorted(data.keys())}"
    )
 
 
# ── build curves ─────────────────────────────────────────────────────────────
class DummyEnv:
    def __init__(self, n):
        self.nstates = n
 
 
def build_curves(batch_dir: Path):
    curves = defaultdict(lambda: {
        "emp_bel": [], "bel_true": [], "emp_true": [], "regret": []
    })
 
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
    print(f"{len(run_dirs)} run folders in {batch_dir.name}")
 
    for i, run_dir in enumerate(run_dirs):
        data = load_run(run_dir)
 
        nstates = int(_item(data["nstates"]))
        n_agents = int(_item(data["n_agents"]))
        traj_states = np.asarray(data["traj_states"])
        T = traj_states.shape[0]
        # belief
        if "belief_update_times" in data:
            snaps = np.asarray(data["belief_phi_map"])            # (n_upd, N, nstates)
            belief_phi_map = densify(np.nanmean(snaps, axis=1),
                                     data["belief_update_times"], T)
        else:
            belief_phi_map = get_belief_phi_map(data, nstates, n_agents)

        # true-phi
        if "true_phi_change_times" in data:                       # new (static or dynamic)
            true_phi_map_time = densify(data["true_phi_map_time"],
                                        data["true_phi_change_times"], T)
        else:                                                     # old files
            tp = np.asarray(data["true_phi_map_time"])
            true_phi_map_time = tp if tp.shape[0] == T else np.repeat(tp[:1], T, axis=0)


        # true_phi_map_time = np.asarray(data["true_phi_map_time"])
        # belief_phi_map = get_belief_phi_map(data, nstates, n_agents)
 
        if i == 0:
            print("first-run shapes:",
                  f"traj={traj_states.shape}, belief={belief_phi_map.shape}, "
                  f"true_phi={true_phi_map_time.shape}, nstates={nstates}")
 
        err_emp_bel, err_bel_true, err_emp_true = convergence_errors_from_run(
            traj_states=traj_states,
            belief_phi_map=belief_phi_map,
            true_phi_map_time=true_phi_map_time,
            env=DummyEnv(nstates),
            empirical_fn=empirical_window,
        )
 
        key = (
            n_agents,
            int(_item(data["gp_update"])),
            int(_item(data["p_update"])),
        )
        curves[key]["emp_bel"].append(err_emp_bel)
        curves[key]["bel_true"].append(err_bel_true)
        curves[key]["emp_true"].append(err_emp_true)
        regret = np.cumsum(err_emp_true) / (np.arange(len(err_emp_true)) + 1)
        curves[key]["regret"].append(regret)
 
    return curves
 
 
# ── plot ─────────────────────────────────────────────────────────────────────
def plot_three_panel(curves):
    plt.rcParams.update({
        "font.size": 19, "axes.labelsize": 20, "axes.titlesize": 20,
        "legend.fontsize": 19, "lines.linewidth": 2.2,
    })
 
    # colorblind_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    # linestyles = ["-", "--", "-.", ":"]
 
    colorblind_colors = ["#0072B2", "#D55E00", "#009E28", "#EB54A7", "#F71D00","#474342" ]
    linestyles = ["-", "-", "-", "-", "-", "-"]
  

    update_pairs = sorted({(gp, p) for (_, gp, p) in curves})
    if len(update_pairs) > len(colorblind_colors):
        raise ValueError(
            f"{len(update_pairs)} update pairs but only "
            f"{len(colorblind_colors)} colors. Add more."
        )
    color_map = {pair: colorblind_colors[i] for i, pair in enumerate(update_pairs)}
    ls_map = {pair: linestyles[i % len(linestyles)] for i, pair in enumerate(update_pairs)}
 
    target_n = sorted({k[0] for k in curves})[0]
    items = [(k, v) for k, v in curves.items() if k[0] == target_n]
 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True)
 
    def _stack(d, metric, m=None):
        Y = d[metric]
        mm = min(len(y) for y in Y) if m is None else min(m, min(len(y) for y in Y))
        Y = np.stack([y[:mm] for y in Y])
        return Y.mean(0), Y.std(0), np.arange(mm), mm
 
    def draw(ax, metric, zoom_last=ZOOM_LAST):
        means = {}
        m_common = None
        for (n, gp, p), d in items:
            c, ls = color_map[(gp, p)], ls_map[(gp, p)]
            mean, std, x, m = _stack(d, metric)
            m_common = m if m_common is None else min(m_common, m)
            means[(gp, p)] = (mean, std, x)
            ax.plot(x, mean, color=c, ls=ls)
            ax.fill_between(x, mean - std, mean + std, color=c,
                            alpha=0.1, linewidth=0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---- zoomed inset on the last `zoom_last` steps ----
        if zoom_last and m_common > zoom_last:
            x0, x1 = m_common - zoom_last, m_common - 1
            axin = ax.inset_axes(INSET_RECT)
            ymins, ymaxs = [], []
            for (gp, p), (mean, std, x) in means.items():
                c, ls = color_map[(gp, p)], ls_map[(gp, p)]
                axin.plot(x, mean, color=c, ls=ls, lw=1.6)
                w = mean[x0:x1 + 1]
                ymins.append(w.min())
                ymaxs.append(w.max())
            pad = 0.1 * (max(ymaxs) - min(ymins) + 1e-9)
            axin.set_xlim(x0, x1)
            # axin.set_ylim(min(ymins) - pad, max(ymaxs) + pad)
            if metric == "regret":
                axin.set_ylim(0, 0.6)
            elif metric == "emp_true":
                axin.set_ylim(0, 0.15)
            else:
                axin.set_ylim(0, 0.05)


            axin.set_box_aspect(0.5)                       # square
            axin.xaxis.set_major_locator(MaxNLocator(3))
            axin.yaxis.set_major_locator(MaxNLocator(4))
            axin.tick_params(labelsize=12)
            axin.grid(True, axis="y", alpha=0.25)
            for s in ("top", "right"):
                axin.spines[s].set_visible(False)
            # thin dashed gray selection box + connectors (less visual clutter)
            rect, conns = ax.indicate_inset_zoom(
                axin, edgecolor="0.45", linewidth=1.3,
                linestyle=(0, (4, 3)), alpha=0.9,
            )
            for ln in conns:
                ln.set_linestyle((0, (4, 3)))
                ln.set_linewidth(0.9)
                ln.set_color("0.6")
                ln.set_alpha(0.7)

    draw(ax1, "regret")
    ax1.set_ylabel(r"$\mathrm{Regret}_k$")
    ax1.set_xlim(0, 6000)

    draw(ax2, "emp_true")
    ax2.set_ylabel(r"$\|\hat{\rho}_k - \rho_k^\star\|_1$")
    ax2.set_xlim(0, 6000)

    draw(ax3, "bel_true")
    ax3.set_ylabel(r"$\mathcal{L}_{belief, k}$")
    ax3.set_xlabel(r"Time step $k$")
    ax3.set_ylim(0, 1.0) 
    ax3.set_xlim(0, 6000)

    legend_handles = [
        Line2D([0], [0], color=color_map[pair], lw=2.6, ls=ls_map[pair],
            #    label=rf"$\tau_{{GP}}={pair[0]},\ \tau_{{P}}={pair[1]}$")
                label=rf"$\tau_{{P}}={pair[1]}$")
        for pair in update_pairs
    ]
    ncol = min(3, len(legend_handles))
    leg = fig.legend(handles=legend_handles, title="Update frequency",
                        loc="lower center", ncol=ncol, frameon=True,
                        fontsize=18, title_fontsize=16,
                        bbox_to_anchor=(0.5, 0.005), borderpad=0.6,
                        handlelength=2.2, columnspacing=1.4, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor("0.6")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"saved {OUTPUT_PNG} / {OUTPUT_PDF}")
    plt.show()
 
 
if __name__ == "__main__":
    curves = build_curves(BATCH_DIR)
    plot_three_panel(curves)