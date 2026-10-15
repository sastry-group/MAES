#!/usr/bin/env python
"""
plot_noise_sweep.py

RMSE / MSLL / E_k / L_belief vs time, and vs noise_rel, across a noise sweep.

Prefers results/index.csv (adjust ID_COL / NOISE_COL below to match your
header once it's regenerated); falls back to scanning RESULTS_DIR for
run.h5 files and parsing noise_rel from the folder name (matches the old
"noise0p5"-style convention).
"""
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics import (
    convergence_errors_from_run, empirical_window,
    rmse_series, msll_series,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
INDEX_CSV = RESULTS_DIR / "index.csv"
OUT_DIR = RESULTS_DIR / "analysis" / "noise_sweep"
ID_COL_CANDIDATES = ("run_dir", "stem", "path", "run")
NOISE_COL_HINT = "noise"          # any column containing this substring
TAIL_FRAC = 0.1                    # last 10% of steps -> "converged" summary value
TAIL_MIN = 50


# ── run discovery ────────────────────────────────────────────────────────────
def discover_runs():
    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        id_col = next((c for c in df.columns if c.lower() in ID_COL_CANDIDATES), None)
        noise_col = next((c for c in df.columns if NOISE_COL_HINT in c.lower()), None)
        if id_col is not None:
            runs = []
            for _, row in df.iterrows():
                run_dir = Path(row[id_col])
                if not run_dir.is_absolute():
                    run_dir = RESULTS_DIR / run_dir
                nr = float(row[noise_col]) if noise_col and pd.notna(row[noise_col]) else None
                runs.append((run_dir, nr))
            return runs
        print(f"[warn] {INDEX_CSV} has no recognized run-dir column "
              f"(looked for {ID_COL_CANDIDATES} among {list(df.columns)}); "
              "falling back to directory scan.")

    runs = []
    for d in sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []:
        if d.is_dir() and (d / "run.h5").exists():
            runs.append((d, _noise_from_name(d.name)))
    return runs


def _noise_from_name(name):
    m = re.search(r"noise([\d]+p?[\d]*)", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


# ── loading / alignment helpers (same conventions as the other h5 scripts) ──
def load_run(run_dir: Path):
    h5_path = run_dir / "run.h5"
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


def densify(snaps, times, T):
    times = np.asarray(times)
    valid = times >= 0
    snaps = np.asarray(snaps)[valid]
    times = times[valid]
    rows = np.searchsorted(times, np.arange(T), side="right") - 1
    rows = np.clip(rows, 0, len(snaps) - 1)
    return snaps[rows]


def dense_field(data, key, T, times_key="belief_update_times"):
    """(T, N, nstates), handling both sparse-snapshot (ergodic) and
    already-dense-per-step (mac_dt) storage."""
    arr = np.asarray(data[key])
    if times_key in data:
        return densify(arr, data[times_key], T)
    return arr


def get_true_phi(data, T):
    if "true_phi_change_times" in data:
        return densify(data["true_phi_map_time"], data["true_phi_change_times"], T)
    tp = np.asarray(data["true_phi_map_time"])
    return tp if tp.shape[0] == T else np.repeat(tp[:1], T, axis=0)


class DummyEnv:
    def __init__(self, n):
        self.nstates = n


# ── per-run metric computation ───────────────────────────────────────────────
def compute_run_metrics(data):
    nstates = int(_item(data["nstates"]))
    traj_states = np.asarray(data["traj_states"])
    T = traj_states.shape[0]

    belief_phi = dense_field(data, "belief_phi_map", T)      # (T, N, nstates)
    mu_time    = dense_field(data, "belief_mu_map", T)
    std_time   = dense_field(data, "belief_sigma_map", T)
    true_phi   = get_true_phi(data, T)                       # (T, nstates)
    sigma_eps  = float(data.get("sigma_eps", 0.05))

    rmse_team = rmse_series(mu_time, true_phi).mean(axis=1)             # (T,)
    msll_team = msll_series(mu_time, std_time, true_phi, sigma_eps).mean(axis=1)

    belief_team = belief_phi.mean(axis=1) if belief_phi.ndim == 3 else belief_phi
    _, err_bel_true, err_emp_true = convergence_errors_from_run(
        traj_states=traj_states, belief_phi_map=belief_team,
        true_phi_map_time=true_phi, env=DummyEnv(nstates),
        empirical_fn=empirical_window,
    )
    E_k = np.cumsum(err_emp_true) / (np.arange(len(err_emp_true)) + 1)
    L_belief = err_bel_true

    return {"rmse": rmse_team, "msll": msll_team, "E_k": E_k, "L_belief": L_belief}


def tail_mean(x):
    w = max(int(len(x) * TAIL_FRAC), TAIL_MIN)
    return float(np.mean(x[-w:]))


# ── plotting ─────────────────────────────────────────────────────────────────
METRICS = ("rmse", "msll", "E_k", "L_belief")
LABELS = {"rmse": "RMSE", "msll": "MSLL", "E_k": r"$\mathrm{Regret}_k$",
          "L_belief": r"$\mathcal{L}_{belief,k}$"}


def plot_vs_time(by_noise, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(by_noise)))
    for ax, metric in zip(axes.flat, METRICS):
        for (nr, runs), c in zip(sorted(by_noise.items(), key=lambda kv: (kv[0] is None, kv[0])), colors):
            arrs = [r[metric] for r in runs]
            m = min(len(a) for a in arrs)
            Y = np.stack([a[:m] for a in arrs])
            mean, std = Y.mean(0), Y.std(0)
            label = f"noise_rel={nr:g}" if nr is not None else "unknown noise_rel"
            ax.plot(mean, color=c, label=label)
            ax.fill_between(np.arange(m), mean - std, mean + std, color=c, alpha=0.15, linewidth=0)
        ax.set_ylabel(LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
    axes[1, 0].set_xlabel("Time step $k$")
    axes[1, 1].set_xlabel("Time step $k$")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {out_path}")


def plot_vs_noise(by_noise, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    items = sorted((nr, runs) for nr, runs in by_noise.items() if nr is not None)
    xs = [nr for nr, _ in items]
    for ax, metric in zip(axes.flat, METRICS):
        means = [np.mean([tail_mean(r[metric]) for r in runs]) for _, runs in items]
        stds  = [np.std([tail_mean(r[metric]) for r in runs]) for _, runs in items]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3)
        ax.set_ylabel(LABELS[metric])
        ax.set_xlabel("noise_rel")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    print(f"{len(runs)} runs discovered")

    by_noise = defaultdict(list)
    for run_dir, noise_rel in runs:
        data = load_run(run_dir)
        if data is None:
            print(f"[skip] no run.h5 in {run_dir}")
            continue
        try:
            metrics = compute_run_metrics(data)
        except Exception as e:
            print(f"[skip] {run_dir}: {e}")
            continue
        if noise_rel is None:
            noise_rel = data.get("noise_rel", None)
        by_noise[noise_rel].append(metrics)

    if not by_noise:
        print("No usable runs found — check RESULTS_DIR / index.csv.")
        return

    plot_vs_time(by_noise, OUT_DIR / "metrics_vs_time.png")
    if any(nr is not None for nr in by_noise):
        plot_vs_noise(by_noise, OUT_DIR / "metrics_vs_noise_rel.png")
    else:
        print("[warn] no run had a resolvable noise_rel — skipping vs-noise_rel figure.")


if __name__ == "__main__":
    main()