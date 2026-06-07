"""
Scalability analysis: 2-panel figure
  Panel A: Full-map coverage time vs N agents, grouped by grid size
  Panel B: Mean per-agent GP fit time vs N agents, grouped by grid size

Demonstrates:
  - Coverage scales well with team size (decreasing time)
  - Per-agent compute is roughly constant (true decentralization)
"""

import os
import sys
import re
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from viz.metrics import full_map_cover_time


# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCALABILITY_DIR = Path("results/scalability_runs")
OUTPUT_PNG = "fig_scalability.png"
OUTPUT_PDF = "fig_scalability.pdf"

# expected N values and grid sizes you ran
N_VALUES = [1, 3, 6, 10, 25, 50]
GRID_LABELS = {100: "10×10", 625: "25×25", 2500: "50×50"}
GRID_COLORS = {100: "#4C72B0", 625: "#DD8452", 2500: "#C44E52"}
GRID_MARKERS = {100: "o", 625: "s", 2500: "^"}


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _s(val):
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, bytes):
        val = val.decode()
    return val


def load_h5_safe(run_dir: Path):
    h5 = run_dir / "run.h5"
    if not h5.exists():
        return None
    try:
        data = {}
        with h5py.File(h5, "r") as f:
            for k in f.keys():
                data[k] = f[k][()]
            for k, v in f.attrs.items():
                data[k] = v
        return data
    except OSError as e:
        print(f"[WARN] corrupted: {h5} ({e})")
        return None


def get_cover_time(data):
    ts = data.get("traj_states", None)
    nstates = data.get("nstates", None)
    if ts is None or nstates is None:
        return np.nan
    result = full_map_cover_time(ts, int(_s(nstates)))
    return float(result) if not np.isnan(result) else np.nan


def get_mean_gp_time(data):
    """
    compute_log entries: (agent_id, t, gp_dt, p_dt)
    Returns mean per-agent per-call GP fit time in seconds.
    """
    log = data.get("compute_log", None)
    if log is None or len(log) == 0:
        return np.nan
    log = np.asarray(log)
    if log.ndim != 2 or log.shape[1] < 4:
        return np.nan
    gp_times = log[:, 2].astype(float)
    gp_times = gp_times[gp_times > 0]  # skip P-update rows
    return float(np.mean(gp_times)) if len(gp_times) else np.nan


# ── DATA COLLECTION ────────────────────────────────────────────────────────────

def collect_scalability() -> pd.DataFrame:
    rows = []
    if not SCALABILITY_DIR.exists():
        print(f"[ERROR] {SCALABILITY_DIR} does not exist")
        return pd.DataFrame()

    for run_dir in sorted(SCALABILITY_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_h5_safe(run_dir)
        if data is None:
            continue

        nstates  = int(_s(data.get("nstates", 0)))
        n_agents = int(_s(data.get("n_agents", 0)))
        if nstates == 0 or n_agents == 0:
            continue

        rows.append({
            "run":         run_dir.name,
            "n_agents":    n_agents,
            "nstates":     nstates,
            "grid_label":  GRID_LABELS.get(nstates, f"{nstates}"),
            "cover_time":  get_cover_time(data),
            "gp_time":     get_mean_gp_time(data),
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs")
    if not df.empty:
        print(df.groupby(["grid_label", "n_agents"]).size())
    return df


# ── PLOTTING ───────────────────────────────────────────────────────────────────

def plot_scalability(df, save=True):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    grids_in_data = sorted(df["nstates"].unique())

    # ── Panel A: Coverage time vs N ──
    for nstates in grids_in_data:
        sub = df[df["nstates"] == nstates].dropna(subset=["cover_time"])
        if sub.empty:
            continue
        agg = sub.groupby("n_agents")["cover_time"].agg(["mean", "std", "count"]).reset_index()
        ax1.errorbar(
            agg["n_agents"], agg["mean"],
            yerr=agg["std"],
            color=GRID_COLORS.get(nstates, "gray"),
            marker=GRID_MARKERS.get(nstates, "o"),
            markersize=7, linewidth=1.8, capsize=3,
            label=GRID_LABELS.get(nstates, str(nstates)),
        )

    ax1.set_xlabel("Team size, $N$ agents", fontsize=11)
    ax1.set_ylabel("Full-map coverage time (steps)", fontsize=11)
    ax1.set_title("(a) Coverage performance scales with team size", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(N_VALUES)
    ax1.set_xticklabels([str(n) for n in N_VALUES])
    ax1.legend(title="Grid", fontsize=9, framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.3, linewidth=0.6)

    # ── Panel B: Per-agent compute time vs N ──
    for nstates in grids_in_data:
        sub = df[df["nstates"] == nstates].dropna(subset=["gp_time"])
        if sub.empty:
            continue
        agg = sub.groupby("n_agents")["gp_time"].agg(["mean", "std"]).reset_index()
        ax2.errorbar(
            agg["n_agents"], agg["mean"] * 1000,  # ms
            yerr=agg["std"] * 1000,
            color=GRID_COLORS.get(nstates, "gray"),
            marker=GRID_MARKERS.get(nstates, "o"),
            markersize=7, linewidth=1.8, capsize=3,
            label=GRID_LABELS.get(nstates, str(nstates)),
        )

    ax2.set_xlabel("Team size, $N$ agents", fontsize=11)
    ax2.set_ylabel("Mean per-agent GP fit time (ms)", fontsize=11)
    ax2.set_title("(b) Per-agent compute is approximately constant", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_xticks(N_VALUES)
    ax2.set_xticklabels([str(n) for n in N_VALUES])
    ax2.legend(title="Grid", fontsize=9, framealpha=0.9)
    ax2.grid(True, which="both", alpha=0.3, linewidth=0.6)

    for ax in (ax1, ax2):
        sns.despine(ax=ax)
        ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
        plt.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
        print(f"Saved → {OUTPUT_PNG}")
        print(f"Saved → {OUTPUT_PDF}")

    plt.show()
    return fig


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = collect_scalability()
    if df.empty:
        print("No data — make sure your scalability runs are in", SCALABILITY_DIR)
        sys.exit(1)
    df.to_csv("fig_scalability_data.csv", index=False)
    plot_scalability(df)