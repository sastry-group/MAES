"""
plot_fig4_extended.py
---------------------
Extended Figure 4 — 2x4 layout matching paper style.
Loads from HDF5 .

  Row 1: ROI first-visit time
  Row 2: Full map coverage time

  Col 1: Static, comm_radius in {1,5,30}, N in {3,6,10}  (baseline)
  Col 2: Static, radius=5, delay=5, N in {3,6,10}         (delay robustness)
  Col 3: Dynamic, radius=1, N in {3,6,10}                 (dynamic robustness)
  Col 4: Static, radius=5, N=6, attrition                 (attrition robustness)

Set missing COLUMN_DIRS entries to None — those panels show "No data".

Run from MAES repo root:
    python viz/figure4_plot.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import h5py
from pathlib import Path

# ── repo imports ──────────────────────────────────────────────────────────────
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from viz.metrics import full_map_cover_time, roi_first_visit_static

# ── CONFIG — edit paths here ──────────────────────────────────────────────────
COLUMN_DIRS = {
    "col1": Path("results/Comms_analysis_static_baseline"),
    "col2": Path("results/Comms_analysis_delay5"),
    "col3": None,
    "col4": Path("results/Attrition_static_10x10"),
}

COL_FILTERS = {
    "col1": dict(env_dyn="static",  comm_delay=0),
    "col2": dict(env_dyn="static",  comm_delay=5,  comm_radius=5.0),
    "col3": dict(env_dyn="dynamic", comm_delay=0,  comm_radius=1.0),
    "col4": dict(env_dyn="static",  comm_delay=0,  comm_radius=5.0, n_agents=6),
}

COL_TITLES = {
    "col1": "Static\n($R_{comms}$ sweep)",
    "col2": "Static + Delay\n($\\tau_{delay}=5$, $R_{comms}=5$)",
    "col3": "Dynamic\n($R_{comms}=5$, no delay)",
    "col4": "Attrition\n($N=6$, $R_{comms}=5$)",
}

# col1: x=comm_radius grouped by N; cols 2-4: x=n_agents
N_LIST   = [3, 6, 10]
N_COLORS = {3: "#a8c8f0", 6: "#4C72B0", 10: "#1a3a6b"}

COL_COLOR = {
    "col1": "#4C72B0",
    "col2": "#DD8452",
    "col3": "#55A868",
    "col4": "#C44E52",
}


ATTRITION_ORDER = ["none", "mild (1 lost)", "moderate (3 lost)", "severe (4 lost)"]
ATTRITION_COLORS = {
    "none":               "#4C72B0",
    "mild (1 lost)":      "#55A868",
    "moderate (3 lost)":  "#DD8452",
    "severe (4 lost)":    "#C44E52",
}


OUTPUT_PNG = "fig4_extended.png"
OUTPUT_PDF = "fig4_extended.pdf"

# ── LOADING ───────────────────────────────────────────────────────────────────

def _s(val):
    """Unwrap numpy scalar or bytes."""
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, bytes):
        val = val.decode()
    return val


def get_comm_delay(data, run_dir):
    """Read comm_delay from h5 attrs, fall back to parsing folder name."""
    raw = data.get("comm_delay", data.get("comms_delay", None))
    if raw is not None:
        try:
            return int(_s(raw))
        except (TypeError, ValueError):
            pass
    # fallback: parse from folder name e.g. _comm_delay5
    import re
    m = re.search(r"_comm_delay(\d+)", run_dir.name)
    return int(m.group(1)) if m else 0

def load_h5(run_dir: Path):
    h5 = run_dir / "run.h5"
    if not h5.exists():
        return None
    data = {}
    with h5py.File(h5, "r") as f:
        for k in f.keys():
            data[k] = f[k][()]
        for k, v in f.attrs.items():
            data[k] = v
    return data



def passes_filter(data, filt, run_dir=None):
    for key, expected in filt.items():
        if key == "comm_delay":
            val = get_comm_delay(data, run_dir) if run_dir is not None else 0
            if int(expected) != val:
                return False
            continue
        raw = data.get(key, None)
        if raw is None:
            return False
        val = _s(raw)
        try:
            if isinstance(expected, float):
                if abs(float(val) - expected) > 1e-6:
                    return False
            elif isinstance(expected, int):
                if int(val) != expected:
                    return False
            elif isinstance(expected, str):
                if str(val) != expected:
                    return False
        except (TypeError, ValueError):
            return False
    return True

def get_attrition_level(data) -> str:
    att_str = _s(data.get("attrition_steps", ""))
    if not att_str or att_str in ("", "None", "nan"):
        return "none"
    import re
    kills = re.findall(r":\d+", str(att_str))
    n_kills = len(kills)
    if n_kills == 0:
        return "none"
    return f"{n_kills} lost"

# ── METRICS ───────────────────────────────────────────────────────────────────

def get_roi_time(data):
    """First timestep any agent visits any ROI cell."""
    traj_rc = data.get("traj_rc_agents_all", None)
    roi_rc  = data.get("roi_rc", None)
    roi_name = data.get("roi_name", None)
    if traj_rc is None or roi_rc is None or len(roi_rc) == 0:
        return np.nan

    # union of all ROI cells across all ROIs
    roi_cellset = set((int(r), int(c)) for r, c in roi_rc)
    return roi_first_visit_static(np.asarray(traj_rc), roi_cellset)


def get_cover_time(data):
    """First timestep all states visited."""
    ts      = data.get("traj_states", None)
    nstates = data.get("nstates", None)
    if ts is None or nstates is None:
        return np.nan
    result = full_map_cover_time(ts, int(_s(nstates)))
    return float(result) if not np.isnan(result) else np.nan

# ── DATA COLLECTION ───────────────────────────────────────────────────────────

def collect_column(col_key) -> pd.DataFrame:
    base_dir = COLUMN_DIRS[col_key]
    if base_dir is None:
        print(f"[SKIP] {col_key} — set to None")
        return pd.DataFrame()
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"[WARN] {base_dir} not found, skipping {col_key}")
        return pd.DataFrame()

    filt = COL_FILTERS[col_key]
    rows = []

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_h5(run_dir)
        if data is None:
            continue
        if not passes_filter(data, filt, run_dir=run_dir):
            continue

        rows.append({
            "run":         run_dir.name,
            "n_agents":    int(_s(data.get("n_agents", 1))),
            "comm_radius": float(_s(data.get("comm_radius", 0))),
            "comm_delay":  get_comm_delay(data, run_dir),
            "env_dyn":     str(_s(data.get("env_dyn", "static"))),
            "attrition_level": get_attrition_level(data),
            "roi_time":    get_roi_time(data),
            "cover_time":  get_cover_time(data),
        })

    df = pd.DataFrame(rows)
    print(f"{col_key}: {len(df)} runs loaded")
    if len(df):
        print(f"  comm_radius: {sorted(df.comm_radius.unique())}")
        print(f"  n_agents:    {sorted(df.n_agents.unique())}")
        print(f"  roi_time non-nan:   {df.roi_time.notna().sum()}")
        print(f"  cover_time non-nan: {df.cover_time.notna().sum()}")
    return df

# ── PLOTTING ──────────────────────────────────────────────────────────────────

def _draw_boxes(ax, data_list, positions, width, color):
    """Core boxplot call — matches notebook style exactly."""
    ax.boxplot(
        data_list,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.80, linewidth=1.2),
    )


def plot_col1(ax, df, metric):
    """
    Col 1: grouped boxes.
    x-axis = comm_radius {1, 5, 30}
    side-by-side groups = N {3, 6, 10}
    Matches original Fig 4 layout.
    """
    x_vals  = sorted(df["comm_radius"].unique())
    n_grps  = len(N_LIST)
    w       = 0.18
    gap     = 0.06
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        n_grps,
    )

    patches = []
    for gi, n in enumerate(N_LIST):
        color = N_COLORS[n]
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85,
                                      label=f"$N={n}$"))
        data_list = []
        positions  = []
        for xi, xv in enumerate(x_vals):
            mask = (df["comm_radius"] == xv) & (df["n_agents"] == n)
            vals = df.loc[mask, metric].dropna().values
            data_list.append(vals if len(vals) else np.array([np.nan]))
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(int(v)) if v == int(v) else str(v)
                        for v in x_vals], fontsize=9)
    ax.set_xlabel("Communication radius, $R_{comms}$", fontsize=9)
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              framealpha=0.7, handlelength=1.0)


def plot_col_simple(ax, df, metric, col_key):
    """Cols 2-4: single-color boxes, x = n_agents."""
    color  = COL_COLOR[col_key]
    x_vals = sorted(df["n_agents"].unique())
    data_list = [df.loc[df["n_agents"] == v, metric].dropna().values
                 for v in x_vals]
    _draw_boxes(ax, data_list, list(range(len(x_vals))), 0.35, color)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"$N={v}$" for v in x_vals], fontsize=9)
    ax.set_xlabel("Team size, N", fontsize=9)


def decorate(ax, metric, col_key, ylabel, title):
    if ylabel:
        ax.set_ylabel({
            "roi_time":   "ROI coverage time",
            "cover_time": "Full map coverage time",
        }[metric], fontsize=9)
    else:
        ax.set_ylabel("")
    if title:
        ax.set_title(COL_TITLES[col_key], fontsize=9, pad=5)
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)


def plot_no_data(ax, col_key, title):
    ax.text(0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=9)
    if title:
        ax.set_title(COL_TITLES[col_key], fontsize=9, pad=5)
    sns.despine(ax=ax)


def plot_col4(ax, df, metric):
    """Col 4: x=attrition level, sorted by severity."""
    
    # sort: "none" first, then by number of kills ascending
    def sort_key(level):
        if level == "none":
            return 0
        import re
        m = re.search(r"(\d+)", level)
        return int(m.group(1)) if m else 99

    x_vals = sorted(df["attrition_level"].unique(), key=sort_key)

    # color ramp from blue (none) to red (severe)
    palette = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(x_vals)))

    for i, level in enumerate(x_vals):
        vals  = df.loc[df["attrition_level"] == level, metric].dropna().values
        color = palette[i]
        ax.boxplot(
            [vals] if len(vals) else [[]],
            positions=[i],
            widths=0.4,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            boxprops=dict(facecolor=color, alpha=0.85, linewidth=1.2),
        )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, fontsize=8, rotation=15, ha="right")
    ax.set_xlabel("Attrition level", fontsize=9)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    col_keys = ["col1", "col2", "col3", "col4"]
    dfs = {k: collect_column(k) for k in col_keys}

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.52, wspace=0.36,
        left=0.07, right=0.98,
        top=0.91,  bottom=0.12,
    )

    for row_i, metric in enumerate(["roi_time", "cover_time"]):
        for col_i, col_key in enumerate(col_keys):
            ax  = fig.add_subplot(gs[row_i, col_i])
            df  = dfs[col_key]
            ttl = (row_i == 0)
            yl  = (col_i == 0)

            if df.empty:
                plot_no_data(ax, col_key, ttl)
                if yl:
                    ax.set_ylabel({
                        "roi_time":   "ROI coverage time",
                        "cover_time": "Full map coverage time",
                    }[metric], fontsize=9)
                continue



            if col_key == "col1":
                plot_col1(ax, df, metric)
            elif col_key == "col4":
                plot_col4(ax, df, metric)
            else:
                plot_col_simple(ax, df, metric, col_key)

            decorate(ax, metric, col_key, ylabel=yl, title=ttl)

    # fig.suptitle(
    #     "Map Exploration Performance under Varying Conditions",
    #     fontsize=11, fontweight="semibold", y=0.98,
    # )

    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PNG}")
    print(f"Saved → {OUTPUT_PDF}")
    plt.show()


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # base_dir = Path("/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/results/Comms_analysis_static_baseline")
    # for run_dir in sorted(base_dir.iterdir())[:3]:  # just first 3 runs
    #     if not run_dir.is_dir():
    #         continue
    #     data = load_h5(run_dir)
    #     if data is None:
    #         print(f"{run_dir.name}: NO H5 FILE")
    #         continue
    #     print(f"\n{run_dir.name}")
    #     print(f"  env_dyn:     {_s(data.get('env_dyn'))}")
    #     print(f"  comm_delay:  {_s(data.get('comm_delay', data.get('comms_delay', 'KEY MISSING')))}")
    #     print(f"  comm_radius: {_s(data.get('comm_radius'))}")
    #     print(f"  n_agents:    {_s(data.get('n_agents'))}")
    #     print(f"  nstates:     {_s(data.get('nstates'))}")
    #     print(f"  roi_rc shape:{data['roi_rc'].shape if 'roi_rc' in data else 'MISSING'}")
    #     # test filter manually
    #     filt = COL_FILTERS["col1"]
    #     print(f"  passes col1 filter: {passes_filter(data, filt)}")
        main()