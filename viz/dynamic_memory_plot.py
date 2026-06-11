"""
---------------------------
  Row 1: ROI first-visit time, RELATIVE to each env change
  Row 2: Full-map coverage time, RELATIVE to each env change

  x-axis  = environment-change segments (K = change timestep)
  colors  = each DISTINCT (l_time, gp_obs, share_hist) configuration found
  columns = team size N (trim N_COLS to split / declutter)

Configurations are NOT hard-coded. Each run declares its own parameters:
  - gp_obs / share_hist  <- h5 attrs `gp_obs_history` / `share_obs_history`
  - l_time               <- folder-name suffix `_ltime<value>` (not saved to h5)
Runs sharing the same (l_time, gp_obs, share_hist) tuple are grouped into one
configuration; colors and legend labels are generated automatically. Add or
change configs upstream and they appear here with no edits.


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

# repo imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Reuse loaders + per-segment metric helpers from the Figure 4 script.
# Adjust the module name if your file isn't `figure4_plot.py`.
from figure4_plot import (
    _s,
    load_h5,
    get_roi_time_per_cfg,
    get_cover_time_per_cfg,
)

# CONFIG - edit here

# Directory holding the dynamic memory-test runs.
BASE_DIR = Path("results/Test_memory_dynamic")

# Default l_time when a run's folder name has no `_ltime` suffix.
DEFAULT_LTIME = 1000.0

# Hold communication range fixed so the only thing varying between boxes is the
# config. Set to the radius these runs used (e.g. 5.0); None pools all radii
# (with a warning if more than one is present).
COMM_RADIUS_FILTER = None

# Team sizes -> columns. Trim to split the figure (e.g. [6] or [6, 10]).
N_COLS = [3, 6, 10]

# Optional: restrict to specific configs. None = show every config discovered.
# Otherwise a list of (l_time, gp_obs, share_hist) tuples to keep, e.g.
#   CONFIG_INCLUDE = [(1000.0, 1000, 1000), (50.0, 1000, 1000), (1000.0, 50, 10)]
CONFIG_INCLUDE = None

# Optional cosmetic overrides keyed by (l_time, gp_obs, share_hist).
# Anything not listed gets an auto label / auto color.
CONFIG_LABEL_OVERRIDES = {
    # (1000.0, 1000, 1000): "Full memory",
    # (50.0,   1000, 1000): "Short l_t",
    # (1000.0,   50,   10): "Short memory",
}
CONFIG_COLOR_OVERRIDES = {
    # (1000.0, 1000, 1000): "#4C72B0",
}

# Palette auto-assigned to configs in display order (full-memory configs first).
CONFIG_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

OUTPUT_PNG = "memory_test_dynamic.png"
OUTPUT_PDF = "memory_test_dynamic.pdf"
OUTPUT_CSV = "memory_test_dynamic.csv"


# PARAMETER EXTRACTION

def parse_ltime_from_name(name: str) -> float:
    """Recover l_time from folder-name suffix `_ltime<value>` (default if absent)."""
    m = re.search(r"_ltime(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else DEFAULT_LTIME


def _to_int(x):
    if x is None:
        return None
    try:
        return int(round(float(_s(x))))
    except (TypeError, ValueError):
        return None


def get_params(data: dict, run_dir: Path):
    """
    Pull this run's actual configuration.
      l_time     <- folder name (`_ltime...`)
      gp_obs     <- h5 attr `gp_obs_history`  (folder `_GPobs` fallback)
      share_hist <- h5 attr `share_obs_history` (folder `_shareHis` fallback)
    Returns None if gp_obs / share_hist can't be determined at all.
    """
    l_time = parse_ltime_from_name(run_dir.name)

    gp_obs = _to_int(data.get("gp_obs_history", None))
    if gp_obs is None:
        m = re.search(r"_GPobs(\d+)", run_dir.name)
        gp_obs = int(m.group(1)) if m else None

    share_hist = _to_int(data.get("share_obs_history", None))
    if share_hist is None:
        m = re.search(r"_shareHis(\d+)", run_dir.name)
        share_hist = int(m.group(1)) if m else None

    if gp_obs is None or share_hist is None:
        return None
    return dict(l_time=l_time, gp_obs=gp_obs, share_hist=share_hist)


def config_tuple(params: dict) -> tuple:
    return (round(float(params["l_time"]), 6),
            int(params["gp_obs"]), int(params["share_hist"]))


def config_id(params: dict) -> str:
    lt, gp, sh = config_tuple(params)
    return f"lt{lt:g}_gp{gp}_sh{sh}"


def auto_label(key: tuple) -> str:
    if key in CONFIG_LABEL_OVERRIDES:
        return CONFIG_LABEL_OVERRIDES[key]
    lt, gp, sh = key
    return f"$\\ell_t$={lt:g}, GPobs={gp}, share={sh}"


# CONFIG REGISTRY (discovered from data)

def build_registry(df: pd.DataFrame) -> dict:
    """
    Map config_id -> {label, color, order, l_time, gp_obs, share_hist}.
    Display order = 'fullest memory first' (l_time, gp_obs, share_hist desc).
    """
    if df.empty:
        return {}
    cfg = (df[["config_id", "l_time", "gp_obs", "share_hist"]]
           .drop_duplicates("config_id")
           .sort_values(["l_time", "gp_obs", "share_hist"],
                        ascending=[False, False, False])
           .reset_index(drop=True))

    registry = {}
    for i, row in cfg.iterrows():
        cid = row["config_id"]
        key = (round(float(row["l_time"]), 6),
               int(row["gp_obs"]), int(row["share_hist"]))
        color = CONFIG_COLOR_OVERRIDES.get(key, CONFIG_PALETTE[i % len(CONFIG_PALETTE)])
        registry[cid] = dict(
            label=auto_label(key),
            color=color,
            order=i,
            l_time=key[0], gp_obs=key[1], share_hist=key[2],
        )
    return registry


def ordered_config_ids(registry: dict) -> list:
    return sorted(registry, key=lambda c: registry[c]["order"])


# DATA COLLECTION

def collect() -> pd.DataFrame:
    rows = []
    n_skipped_static = 0
    n_skipped_params = 0

    if not BASE_DIR.exists():
        print(f"[WARN] {BASE_DIR} not found")
        return pd.DataFrame()

    for run_dir in sorted(BASE_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_h5(run_dir)
        if data is None:
            continue

        if str(_s(data.get("env_dyn", "static"))) != "dynamic":
            n_skipped_static += 1
            continue

        params = get_params(data, run_dir)
        if params is None:
            n_skipped_params += 1
            continue

        key = config_tuple(params)
        if CONFIG_INCLUDE is not None and key not in {tuple(c) for c in CONFIG_INCLUDE}:
            continue

        comm_radius = float(_s(data.get("comm_radius", 0)))
        if COMM_RADIUS_FILTER is not None and comm_radius != COMM_RADIUS_FILTER:
            continue

        row = {
            "run":         run_dir.name,
            "config_id":   config_id(params),
            "n_agents":    int(_s(data.get("n_agents", 1))),
            "comm_radius": comm_radius,
            "l_time":      params["l_time"],
            "gp_obs":      params["gp_obs"],
            "share_hist":  params["share_hist"],
        }
        for k, v in get_roi_time_per_cfg(data).items():
            row[f"roi_time_{k}"] = v
        for k, v in get_cover_time_per_cfg(data).items():
            row[f"cover_time_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} dynamic runs "
          f"({n_skipped_static} non-dynamic, {n_skipped_params} missing params)")
    if len(df):
        radii = sorted(df["comm_radius"].unique())
        if COMM_RADIUS_FILTER is None and len(radii) > 1:
            print(f"[WARN] multiple comm radii present {radii} - boxes mix radii. "
                  f"Set COMM_RADIUS_FILTER to isolate one.")
        print("Discovered configurations:")
        reg = build_registry(df)
        for cid in ordered_config_ids(reg):
            sub = df[df.config_id == cid]
            r   = reg[cid]
            ns  = sorted(sub.n_agents.unique())
            print(f"  [{r['color']}] {r['label']:<48s} "
                  f"({len(sub):3d} runs | N={ns})")
    return df


# PLOTTING

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


def _cfg_columns(df, metric):
    prefix = f"{metric}_cfg"
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(re.search(r"cfg(\d+)", c).group(1)))


def plot_dynamic_methods(ax, df, registry, metric, show_legend=False):
    """
    x-axis = env-change segments; grouped colors = discovered configs.
    df should already be filtered to a single team size.
    """
    cfg_cols = _cfg_columns(df, metric)
    if not cfg_cols:
        ax.text(0.5, 0.5, "No per-segment data", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        return

    x_labels = []
    for col in cfg_cols:
        m = re.search(r"_t(\d+)$", col)
        x_labels.append(f"$\\mathcal{{K}}={m.group(1)}$" if m else col)

    # only configs actually present in this panel, in registry order
    cids = [c for c in ordered_config_ids(registry) if (df["config_id"] == c).any()]

    n_grps  = max(len(cids), 1)
    w       = 0.8 / n_grps
    gap     = 0.02
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        n_grps,
    )

    patches = []
    for gi, cid in enumerate(cids):
        color = registry[cid]["color"]
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85,
                                      label=registry[cid]["label"]))
        data_list, positions = [], []
        for xi, col in enumerate(cfg_cols):
            vals = df.loc[df["config_id"] == cid, col].dropna().values
            data_list.append(vals if len(vals) else np.array([np.nan]))
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color)

    ax.set_xticks(range(len(cfg_cols)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel(r"Env change at $\mathcal{K}$", fontsize=10)
    ax.set_xlim(-0.5, len(cfg_cols) - 0.5)
    if show_legend and patches:
        ax.legend(handles=patches, fontsize=8.5, loc="upper right",
                  framealpha=0.7, handlelength=1.0)


def decorate(ax, metric, ylabel, title_text):
    if ylabel:
        ax.set_ylabel({
            "roi_time":   "ROI time after change",
            "cover_time": "Coverage time after change",
        }[metric], fontsize=10)
    if title_text:
        ax.set_title(title_text, fontsize=11, pad=6)
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)


# CSV EXPORT (long form)

def export_csv(df, path=OUTPUT_CSV):
    if df.empty:
        print("No data to export")
        return
    id_cols  = ["config_id", "run", "n_agents", "comm_radius",
                "l_time", "gp_obs", "share_hist"]
    seg_cols = [c for c in df.columns
                if c.startswith("roi_time_cfg") or c.startswith("cover_time_cfg")]
    long = df.melt(id_vars=id_cols, value_vars=seg_cols,
                   var_name="metric_cfg", value_name="time")
    long["metric"]      = long["metric_cfg"].str.extract(r"^(roi_time|cover_time)")
    long["cfg_index"]   = long["metric_cfg"].str.extract(r"cfg(\d+)").astype(int)
    long["change_time"] = long["metric_cfg"].str.extract(r"_t(\d+)$").astype(int)
    long = long.drop(columns=["metric_cfg"])
    long.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"CSV saved -> {path}  ({len(long)} rows, long form)")


# MAIN

def main(df=None):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    if df is None:
        df = collect()

    registry = build_registry(df)

    if df.empty:
        n_cols = list(N_COLS)
    else:
        present = set(df["n_agents"].unique())
        n_cols  = [n for n in N_COLS if n in present]
        if not n_cols:
            print(f"[WARN] none of N_COLS={N_COLS} present (have {sorted(present)})")
            n_cols = sorted(present)

    ncol = max(len(n_cols), 1)
    fig  = plt.figure(figsize=(4.6 * ncol, 6.5))
    gs   = gridspec.GridSpec(
        2, ncol, figure=fig,
        hspace=0.42, wspace=0.28,
        left=0.07, right=0.98,
        top=0.91,  bottom=0.13,
    )

    for row_i, metric in enumerate(["roi_time", "cover_time"]):
        for col_i, n in enumerate(n_cols):
            ax     = fig.add_subplot(gs[row_i, col_i])
            sub    = df[df["n_agents"] == n] if not df.empty else df
            ylabel = (col_i == 0)
            ttl    = f"$N = {n}$" if row_i == 0 else ""
            legend = (row_i == 0 and col_i == ncol - 1)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=11)
                decorate(ax, metric, ylabel=ylabel, title_text=ttl)
            else:
                plot_dynamic_methods(ax, sub, registry, metric, show_legend=legend)
                decorate(ax, metric, ylabel=ylabel, title_text=ttl)

    fig.suptitle("Configuration comparison (dynamic map)", fontsize=12, y=0.97)

    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {OUTPUT_PNG}")
    print(f"Saved -> {OUTPUT_PDF}")
    plt.show()


if __name__ == "__main__":
    df = collect()
    export_csv(df)
    main(df=df)