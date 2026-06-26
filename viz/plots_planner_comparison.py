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
    "results/planner_comparison"
)
OUT_DIR = BATCH_DIR / "analysis"

# stable color per planner
PLANNER_STYLE = {
    "ergodic": dict(color="#0072B2", ls="-"),
    "mac_dt":  dict(color="#D55E00", ls="-"),
}
METRICS = {
    "regret":   r"$\mathrm{Regret}_k$",
    "emp_true": r"$\|\hat{\rho}_k - \rho_k^\star\|_1$",
    "bel_true": r"$\mathcal{L}_{\mathrm{belief},k}$",
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


def _planner(data):
    p = data.get("planner", data.get("method", b"unknown"))
    return p.decode() if isinstance(p, bytes) else str(p)


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


# ── plot one figure per metric, subplot per map ──────────────────────────────
def plot_metric(curves, metric, ylabel, out_path):
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
        "legend.fontsize": 13, "lines.linewidth": 2.0,
    })
    maps = sorted(curves.keys())
    ncols = min(3, len(maps))
    nrows = math.ceil(len(maps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows),
                             squeeze=False)

    planners_seen = set()
    for idx, map_id in enumerate(maps):
        ax = axes[idx // ncols][idx % ncols]
        for planner, by_metric in sorted(curves[map_id].items()):
            if metric not in by_metric or not by_metric[metric]:
                continue
            style = PLANNER_STYLE.get(planner, dict(color="gray", ls="-"))
            mean, std, x = _stack(by_metric[metric])
            ax.plot(x, mean, label=planner, **style)
            ax.fill_between(x, mean - std, mean + std, color=style["color"],
                            alpha=0.12, linewidth=0)
            planners_seen.add(planner)
        ax.set_title(map_id)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"Time step $k$")
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)

    # hide any unused axes
    for j in range(len(maps), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [Line2D([0], [0], label=p, **PLANNER_STYLE.get(p, dict(color="gray", ls="-")))
               for p in sorted(planners_seen)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
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
    for metric, ylabel in METRICS.items():
        plot_metric(curves, metric, ylabel, OUT_DIR / f"{metric}_compare.png")


if __name__ == "__main__":
    main()
