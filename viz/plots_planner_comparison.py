#!/usr/bin/env python
"""
plots_planner_comparison.py

Three figures (one per metric) comparing planners (ergodic vs mac_dt) across maps:
  - regret_compare.png    : running-average empirical-vs-true ("regret_k")
  - emp_true_compare.png  : || rho_hat_k - rho*_k ||_1   (ergodic/coverage metric)
  - bel_true_compare.png  : || rho_bar_k - rho*_k ||_1   (belief metric)

Each figure is a grid of subplots, one per map, with one line per planner
(mean over replicate runs, +/- std band).

Reuses convergence_errors_from_run / empirical_window / densify, matching the
reference 3-panel script. No update-frequency dimension here — this compares
planners only.

Requires: ergodic finalize() must write attrs["planner"]="ergodic".
"""

import os
import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics import convergence_errors_from_run, empirical_window


# ── CONFIG ───────────────────────────────────────────────────────────────────
BATCH_DIR = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/"
    "results/MAC-DT_Ergodic_dynamic_6K_3a"
)
OUT_DIR = BATCH_DIR / "analysis"

# line style encodes planner; color encodes metric
PLANNER_STYLE = {
    "ergodic": dict(ls="-"),
    "mac_dt":  dict(ls="--"),
}
PLANNER_LABEL = {"ergodic": "Ergodic", "mac_dt": "MAC-DT"}

# regret = blue, belief = magenta (axes are colored to match)
METRIC_STYLE = {
    "regret":   dict(color="#06021C", label=r"$\mathrm{Regret}_k$"),
    "bel_true": dict(color="#D50019", label=r"$\mathcal{L}_{belief, k}$"),
}


# ── loaders / helpers (shared shape with reference) ──────────────────────────
def load_run(run_dir: Path):
    h5_path = Path(run_dir) / "run.h5"
    if not h5_path.exists():
        return None
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
        for key, val in f.attrs.items():
            data[key] = val
    return data


def _item(v):
    return v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v


def _canon_planner(s):
    s = str(s).lower()
    if "mac" in s:
        return "mac_dt"
    if any(k in s for k in ("ergodic", "mh")):
        return "ergodic"
    return s


def _planner(data):
    p = data.get("planner", data.get("method", b"unknown"))
    p = p.decode() if isinstance(p, bytes) else str(p)
    return _canon_planner(p)


def _map_id(data, run_dir):
    src = data.get("env_source", None)
    if src is not None:
        src = src.decode() if isinstance(src, bytes) else str(src)
        name = os.path.basename(src.rstrip("/")).replace(".json", "")
        if name and name != "NA":
            return name
    return Path(run_dir).name


def densify(snaps, times, T):
    times = np.asarray(times)
    valid = times >= 0
    snaps = np.asarray(snaps)[valid]
    times = times[valid]
    rows = np.searchsorted(times, np.arange(T), side="right") - 1
    rows = np.clip(rows, 0, len(snaps) - 1)
    return snaps[rows]


def _orient_belief(arr, nstates, n_agents):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == n_agents:
            arr = arr.mean(axis=0)
        elif arr.shape[1] == n_agents:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=0)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == nstates:
        return arr
    if arr.shape[0] == nstates:
        return arr.T
    return None


def get_belief_phi_map(data, nstates, n_agents, T):
    if "belief_update_times" in data:
        snaps = np.asarray(data["belief_phi_map"])           # (n_upd, N, nstates)
        return densify(np.nanmean(snaps, axis=1), data["belief_update_times"], T)
    for key in ("belief_phi_map_avg", "belief_phi_map"):
        if key in data:
            oriented = _orient_belief(data[key], nstates, n_agents)
            if oriented is not None:
                return oriented
    raise KeyError(f"No usable belief array (nstates={nstates}).")


def get_true_phi(data, T):
    if "true_phi_change_times" in data:
        return densify(data["true_phi_map_time"], data["true_phi_change_times"], T)
    tp = np.asarray(data["true_phi_map_time"])
    return tp if tp.shape[0] == T else np.repeat(tp[:1], T, axis=0)


class DummyEnv:
    def __init__(self, n):
        self.nstates = n


# ── build curves keyed by (map, planner) ─────────────────────────────────────
def build_curves(batch_dir: Path):
    # curves[map][planner][metric] = list of per-run arrays
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name != "analysis"]
    print(f"{len(run_dirs)} run folders in {batch_dir}")

    for i, run_dir in enumerate(run_dirs):
        data = load_run(run_dir)
        if data is None:
            continue
        planner = _planner(data)
        map_id  = _map_id(data, run_dir)
        nstates = int(_item(data["nstates"]))
        n_agents = int(_item(data["n_agents"]))
        traj_states = np.asarray(data["traj_states"])
        T = traj_states.shape[0]

        belief = get_belief_phi_map(data, nstates, n_agents, T)
        true_phi = get_true_phi(data, T)

        if i == 0:
            print("first-run shapes:",
                  f"traj={traj_states.shape}, belief={belief.shape}, "
                  f"true={true_phi.shape}, nstates={nstates}")

        err_emp_bel, err_bel_true, err_emp_true = convergence_errors_from_run(
            traj_states=traj_states,
            belief_phi_map=belief,
            true_phi_map_time=true_phi,
            env=DummyEnv(nstates),
            empirical_fn=empirical_window,
        )
        regret = np.cumsum(err_emp_true) / (np.arange(len(err_emp_true)) + 1)

        curves[map_id][planner]["emp_true"].append(err_emp_true)
        curves[map_id][planner]["bel_true"].append(err_bel_true)
        curves[map_id][planner]["regret"].append(regret)

    return curves


def _stack(arrs):
    m = min(len(a) for a in arrs)
    A = np.stack([a[:m] for a in arrs])
    return A.mean(0), A.std(0), np.arange(m)


# ── single axes: pooled over ALL maps + runs; color = planner, style = metric ─
def plot_combined(curves, out_path):
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
        "legend.fontsize": 13, "lines.linewidth": 2.2,
    })

    # pool every (map, run) replicate together -> pooled[planner][metric] = [arrays]
    pooled = defaultdict(lambda: defaultdict(list))
    for map_id in curves:
        for planner in curves[map_id]:
            for metric in METRIC_STYLE:
                pooled[planner][metric].extend(curves[map_id][planner].get(metric, []))

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax2 = ax.twinx()                                    # right axis = belief
    metric_axis = {"regret": ax, "bel_true": ax2}       # left=regret, right=belief
    for planner in PLANNER_STYLE:                       # fixed order
        if planner not in pooled:
            continue
        cstyle = PLANNER_STYLE[planner]
        for metric, mstyle in METRIC_STYLE.items():
            runs = pooled[planner].get(metric)
            if not runs:
                continue
            mean, std, x = _stack(runs)
            tgt = metric_axis[metric]
            tgt.plot(x, mean, color=mstyle["color"], ls=cstyle["ls"])
            tgt.fill_between(x, mean - std, mean + std,
                            color=mstyle["color"], alpha=0.12, linewidth=0)
        print(f"{PLANNER_LABEL.get(planner, planner)}: "
              f"{len(pooled[planner].get('regret', []))} replicates pooled")

    reg_c = METRIC_STYLE["regret"]["color"]
    bel_c = METRIC_STYLE["bel_true"]["color"]

    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel(r"Time step $k$")

    # left axis = regret (blue)
    ax.set_ylabel(r"$\mathrm{Regret}_k$", color=reg_c)
    ax.tick_params(axis="y", colors=reg_c)
    ax.spines["left"].set_color(reg_c)
    ax2.spines["left"].set_color(reg_c)     # overlaps ax's left spine — color to match

    # right axis = belief (magenta)
    ax2.set_ylabel(r"$\mathcal{L}_{belief,\,k}$", color=bel_c)
    ax2.tick_params(axis="y", colors=bel_c)
    ax2.spines["right"].set_color(bel_c)

    # combined legend: color = planner, line style = metric
    # color_handles = [Line2D([0], [0], color=PLANNER_STYLE[p]["color"], lw=2.2,
    #                         label=PLANNER_LABEL[p]) for p in PLANNER_STYLE]
    style_handles = [Line2D([0], [0], color="black", lw=2.2, ls=PLANNER_STYLE[p]["ls"],
                            label=PLANNER_LABEL[p]) for p in PLANNER_STYLE]
    ax2.legend(handles=style_handles, frameon=True, loc="upper right", ncol=2)
    # ax.legend(handles=color_handles + style_handles, frameon=True, ncol=2,
    #           loc="upper right")


    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = build_curves(BATCH_DIR)
    if not curves:
        print("No runs found — check BATCH_DIR.")
        return
    plot_combined(curves, OUT_DIR / "regret_belief_compare.png")


if __name__ == "__main__":
    main()