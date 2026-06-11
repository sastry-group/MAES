"""
plot_three_approaches.py
------------------------
Compare three planner configurations on the SAME static map, in the style of
Figure 4 Column 1 (grouped boxes, x-axis = configuration, N-agents as colors).

  Row 1: ROI first-visit time
  Row 2: Full map coverage time

The three configurations differ only in three parameters:

    Approach            l_time   gp_obs_hist   share_history
    -----------------   ------   -----------   -------------
    A1  (full memory)    1000        1000           1000
    A2  (short l_time)     50        1000           1000
    A3  (short memory)   1000          50             10

IMPORTANT — these three parameters are NOT yet stored as HDF5 attributes.
They are recovered by parsing the run-folder name, e.g.:

  A1: ...commModelocal_comm_radius0_concenIter5
        (no ltime / GPobs / shareHis suffix  -> all default to 1000)
  A2: ...commModelocal_comm_radius1_concenIter5_ltime50.0
        (ltime50.0; GPobs & shareHis absent  -> default 1000)
  A3: ...commModelocal_comm_radius1_concenIter5_ltime1000.0_shareHis10_GPobs50

Once you start writing l_time / gp_obs_hist / share_history into run.h5 in
finalize(), set USE_H5_ATTRS = True and the parser becomes a fallback only.

Run from MAES repo root:
    python viz/plot_three_approaches.py
"""

import sys, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import csv

# ── repo imports ──────────────────────────────────────────────────────────────
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Reuse the loading + metric helpers from the existing Figure 4 script.
# NOTE: adjust this module name if your file is named differently
# (the uploaded file was `figure4_plot.py`; memory also refers to
#  `plot_fig4_extended.py`). Whatever it's called, it must be importable.
from figure4_plot import (
    _s,
    load_h5,
    get_roi_time,
    get_cover_time,
)

# ── CONFIG — edit here ────────────────────────────────────────────────────────

# All three approaches' runs are assumed to live under this one directory.
# If they're split across multiple dirs, see APPROACH_DIRS below instead.
BASE_DIR = Path("results/Test_global_forget")

# Optional: if each approach lives in its own directory, fill these in and the
# script will load from them instead of BASE_DIR. Leave as None to use BASE_DIR.
APPROACH_DIRS = {
    "A1": None,
    "A2": None,
    "A3": None,
}

# Once finalize() saves these as h5 attrs, flip to True (parser stays as fallback)
USE_H5_ATTRS = False

# Defaults applied when a suffix is absent from the folder name
DEFAULT_LTIME      = 1000.0
DEFAULT_GP_OBS     = 1000
DEFAULT_SHARE_HIST = 1000

# Each approach = a target (l_time, gp_obs_hist, share_history) tuple.
# A run is assigned to the approach whose tuple it matches (within tolerance).
APPROACH_TARGETS = {
    "A1": dict(l_time=1000.0, gp_obs=1000, share_hist=1000),
    "A2": dict(l_time=50.0,   gp_obs=1000, share_hist=1000),
    "A3": dict(l_time=1000.0, gp_obs=50,   share_hist=10),
}

# Plot order + short legend labels (one color per method)
APPROACH_ORDER  = ["A1", "A2", "A3"]
APPROACH_LABELS = {
    "A1": "Full memory ($\\ell_t$=1000, GPobs=1000, share=1000)",
    "A2": "Short $\\ell_t$ ($\\ell_t$=50, GPobs=1000, share=1000)",
    "A3": "Short memory ($\\ell_t$=1000, GPobs=50, share=10)",
}

# One color per method — these are the grouped boxes along each comm_radius tick
APPROACH_COLORS = {
    "A1": "#4C72B0",   # blue
    "A2": "#DD8452",   # orange
    "A3": "#55A868",   # green
}

# Semantic x-axis labels for comm_radius values (10x10 grid, diagonal ~14.1)
RADIUS_LABELS = {
    0.0:  "Local\n($R=0$)",
    1.0:  "Local\n($R=1$)",
    5.0:  "Mid\n($R=5$)",
    30.0: "Global\n($R=30$)",
}

# Team sizes to show — one column per value. Trim this list to split the figure
# (e.g. [3] for a single-N plot, or [3, 6] for two columns).
N_COLS = [3, 6, 10]

OUTPUT_PNG = "three_approaches.png"
OUTPUT_PDF = "three_approaches.pdf"
OUTPUT_CSV = "three_approaches.csv"


# ── PARAMETER PARSING ─────────────────────────────────────────────────────────

def parse_params_from_name(name: str) -> dict:
    """
    Recover (l_time, gp_obs_hist, share_history) from a run-folder name.
    Missing suffixes fall back to the DEFAULT_* values.

    Examples:
      ..._comm_radius0_concenIter5
          -> (1000, 1000, 1000)
      ..._ltime50.0
          -> (50, 1000, 1000)
      ..._ltime1000.0_shareHis10_GPobs50
          -> (1000, 50, 10)
    """
    m_lt  = re.search(r"_ltime(\d+(?:\.\d+)?)", name)
    m_gp  = re.search(r"_GPobs(\d+)",            name)
    m_sh  = re.search(r"_shareHis(\d+)",         name)

    l_time     = float(m_lt.group(1)) if m_lt else DEFAULT_LTIME
    gp_obs     = int(m_gp.group(1))   if m_gp else DEFAULT_GP_OBS
    share_hist = int(m_sh.group(1))   if m_sh else DEFAULT_SHARE_HIST
    return dict(l_time=l_time, gp_obs=gp_obs, share_hist=share_hist)


def get_params(data: dict, run_dir: Path) -> dict:
    """
    Prefer h5 attrs if USE_H5_ATTRS and present; otherwise parse folder name.
    """
    if USE_H5_ATTRS:
        l_time     = data.get("l_time", None)
        gp_obs     = data.get("gp_obs_hist", data.get("gp_observation_hist", None))
        share_hist = data.get("share_history", data.get("history_window", None))
        if l_time is not None and gp_obs is not None and share_hist is not None:
            return dict(
                l_time=float(_s(l_time)),
                gp_obs=int(_s(gp_obs)),
                share_hist=int(_s(share_hist)),
            )
    return parse_params_from_name(run_dir.name)


def classify_approach(params: dict, l_tol=1e-6) -> str | None:
    """Return the approach key whose target tuple matches params, else None."""
    for key, tgt in APPROACH_TARGETS.items():
        if (abs(params["l_time"] - tgt["l_time"]) <= l_tol
                and params["gp_obs"]     == tgt["gp_obs"]
                and params["share_hist"] == tgt["share_hist"]):
            return key
    return None


# ── DATA COLLECTION ───────────────────────────────────────────────────────────

def _iter_run_dirs():
    """
    Yield (run_dir, forced_approach_or_None).
    If APPROACH_DIRS are set, each run is forced to that approach AND still
    classified for a sanity check. Otherwise scan BASE_DIR and classify.
    """
    use_split = any(v is not None for v in APPROACH_DIRS.values())
    if use_split:
        for key, d in APPROACH_DIRS.items():
            if d is None:
                continue
            d = Path(d)
            if not d.exists():
                print(f"[WARN] {d} not found (approach {key})")
                continue
            for run_dir in sorted(d.iterdir()):
                if run_dir.is_dir():
                    yield run_dir, key
    else:
        if not BASE_DIR.exists():
            print(f"[WARN] {BASE_DIR} not found")
            return
        for run_dir in sorted(BASE_DIR.iterdir()):
            if run_dir.is_dir():
                yield run_dir, None


def collect() -> pd.DataFrame:
    rows = []
    n_unclassified = 0
    for run_dir, forced in _iter_run_dirs():
        data = load_h5(run_dir)
        if data is None:
            continue

        params   = get_params(data, run_dir)
        approach = classify_approach(params)

        # If dirs are split by approach, trust the folder but warn on mismatch
        if forced is not None:
            if approach is not None and approach != forced:
                print(f"[WARN] {run_dir.name}: parsed {approach} "
                      f"but lives in {forced} dir — using {forced}")
            approach = forced

        if approach is None:
            n_unclassified += 1
            continue

        rows.append({
            "run":         run_dir.name,
            "approach":    approach,
            "n_agents":    int(_s(data.get("n_agents", 1))),
            "comm_radius": float(_s(data.get("comm_radius", 0))),
            "l_time":      params["l_time"],
            "gp_obs":      params["gp_obs"],
            "share_hist":  params["share_hist"],
            "roi_time":    get_roi_time(data),
            "cover_time":  get_cover_time(data),
        })

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} runs ({n_unclassified} unclassified, skipped)")
    if len(df):
        for key in APPROACH_ORDER:
            sub = df[df.approach == key]
            radii = sorted(sub.comm_radius.unique()) if len(sub) else []
            print(f"  {key}: {len(sub):3d} runs | R_comm={radii} | "
                  f"roi non-nan={sub.roi_time.notna().sum()} "
                  f"cover non-nan={sub.cover_time.notna().sum()}")
    return df


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def _draw_boxes(ax, data_list, positions, width, color, alpha=0.80, zorder=2):
    ax.boxplot(
        data_list,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6, zorder=zorder + 1),
        whiskerprops=dict(linewidth=1.2, zorder=zorder),
        capprops=dict(linewidth=1.2, zorder=zorder),
        boxprops=dict(facecolor=color, alpha=alpha, linewidth=1.2, zorder=zorder),
        zorder=zorder,
    )


def plot_approaches(ax, df, metric, show_legend=False):
    """
    x-axis = comm_radius sweep (sorted unique values).
    Grouped colors  = the three methods (A1, A2, A3).
    df should already be filtered to a single team size.
    """
    x_vals  = sorted(df["comm_radius"].unique())
    methods = [k for k in APPROACH_ORDER if (df["approach"] == k).any()]

    n_grps  = len(methods)
    w       = 0.22
    gap     = 0.06
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        max(n_grps, 1),
    )

    patches = []
    for gi, key in enumerate(methods):
        color = APPROACH_COLORS[key]
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85,
                                      label=APPROACH_LABELS[key]))
        data_list, positions = [], []
        for xi, xv in enumerate(x_vals):
            mask = (df["approach"] == key) & (df["comm_radius"] == xv)
            vals = df.loc[mask, metric].dropna().values
            data_list.append(vals if len(vals) else np.array([np.nan]))
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([RADIUS_LABELS.get(v, f"$R={v:g}$") for v in x_vals], fontsize=9)
    ax.set_xlabel("Communication range, $R_{comms}$", fontsize=10)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    if show_legend and patches:
        ax.legend(handles=patches, fontsize=8.5, loc="upper right",
                  framealpha=0.7, handlelength=1.0)


def decorate(ax, metric, ylabel, title_text):
    if ylabel:
        ax.set_ylabel({
            "roi_time":   "ROI coverage time",
            "cover_time": "Full map coverage time",
        }[metric], fontsize=10)
    if title_text:
        ax.set_title(title_text, fontsize=11, pad=6)
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)


def plot_no_data(ax, metric, ylabel, title_text):
    ax.text(0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=11)
    if ylabel:
        ax.set_ylabel({
            "roi_time":   "ROI coverage time",
            "cover_time": "Full map coverage time",
        }[metric], fontsize=10)
    if title_text:
        ax.set_title(title_text, fontsize=11, pad=6)
    sns.despine(ax=ax)


# ── CSV EXPORT ────────────────────────────────────────────────────────────────

def export_csv(df, path=OUTPUT_CSV):
    if df.empty:
        print("No data to export")
        return
    cols = ["approach", "run", "n_agents", "l_time", "gp_obs",
            "share_hist", "roi_time", "cover_time"]
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"CSV saved -> {path}  ({len(df)} rows)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(df=None):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    if df is None:
        df = collect()

    # Only keep requested team sizes that actually have data, preserving order
    if df.empty:
        n_cols = list(N_COLS)
    else:
        present = set(df["n_agents"].unique())
        n_cols  = [n for n in N_COLS if n in present]
        if not n_cols:
            print(f"[WARN] none of N_COLS={N_COLS} present in data "
                  f"(have {sorted(present)})")
            n_cols = sorted(present)

    ncol = max(len(n_cols), 1)
    fig  = plt.figure(figsize=(4.6 * ncol, 6.5))
    gs   = gridspec.GridSpec(
        2, ncol, figure=fig,
        hspace=0.42, wspace=0.28,
        left=0.07, right=0.98,
        top=0.91,  bottom=0.13,
    )

    metrics = ["roi_time", "cover_time"]
    for row_i, metric in enumerate(metrics):
        for col_i, n in enumerate(n_cols):
            ax     = fig.add_subplot(gs[row_i, col_i])
            sub    = df[df["n_agents"] == n] if not df.empty else df
            ylabel = (col_i == 0)
            ttl    = f"$N = {n}$" if row_i == 0 else ""
            legend = (row_i == 0 and col_i == ncol - 1)  # once, top-right panel

            if sub.empty:
                plot_no_data(ax, metric, ylabel=ylabel, title_text=ttl)
            else:
                plot_approaches(ax, sub, metric, show_legend=legend)
                decorate(ax, metric, ylabel=ylabel, title_text=ttl)

    fig.suptitle("Three-configuration comparison (static map)", fontsize=12, y=0.97)

    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {OUTPUT_PNG}")
    print(f"Saved -> {OUTPUT_PDF}")
    plt.show()


if __name__ == "__main__":
    df = collect()
    export_csv(df)
    main(df=df)