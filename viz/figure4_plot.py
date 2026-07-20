"""
plot_fig4_extended.py
---------------------
Extended Figure 4 — 2x4 layout matching paper style.
Loads from HDF5 .

  Row 1: ROI first-visit time
  Row 2: Full map coverage time

  Col 1: Static, x axis: comm_radius sweep, N-agents different colors (baseline)
  Col 2: Static, delays different colors, N-agents the sweep       (delay robustness)
  Col 3: Dynamic,                 (dynamic robustness)
  Col 4: Static, radius=5, N=6, attrition                 (attrition robustness)

Set missing COLUMN_DIRS entries to None — those panels show "No data".

Run from MAES repo root:
    python viz/figure4_plot.py
"""


import sys, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import h5py
from pathlib import Path
import csv
 
 
# ── repo imports ──────────────────────────────────────────────────────────────
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
 
from viz.metrics import full_map_cover_time, roi_first_visit_static, roi_first_visit_per_cfg
 
# ── CONFIG — edit paths here ──────────────────────────────────────────────────
COLUMN_DIRS = {
    "col1": Path("results/Comm_analysis_static_baseline_mh"),
    "col2": Path("results/Comms_analysis_delay5_mh"),   # contains delay=5 AND delay=10 runs
    "col3": Path("results/Dynamic_analsys_fig40510"),
    "col4": Path("results/Attrition_20x20_4k"),
}
 
COL_FILTERS = {
    "col1": dict(env_dyn="static",  comm_delay=0),
    "col2": dict(env_dyn="static",  comm_radius=5.0),   # no delay filter — load all, split by delay in plot
    "col3": dict(env_dyn="dynamic", comm_delay=0,  comm_radius=5.0),
    "col4": dict(env_dyn="static",  comm_delay=0,  comm_radius=5.0, n_agents=6),
}
 
# col2: which delay values to show, in order. No-delay (0) comes from col1 dir.
# Edit here to add/remove delay conditions.
COL2_DELAYS = [0, 5, 10]
 
# Baseline filter for col2 overlay: no-delay, radius=5, from col1 dir
COL2_BASELINE_FILTER = dict(env_dyn="static", comm_delay=0, comm_radius=5.0)
 
COL_TITLES = {
    "col1": "Static\n($R_{comms}$ sweep)",
    "col2": "Comm. Delay\n($R_{comms}=5$)",
    "col3": "Dynamic\n($R_{comms}=5$, no delay)",
    "col4": "Attrition\n($N=6$, $R_{comms}=5$)",
}
 
# col1: x=comm_radius grouped by N; cols 2–4: x=n_agents
N_LIST      = [3, 6, 10]   # col1: all N shown
N_LIST_COL3 = [6, 10]      # col3: subset to avoid clutter — edit here
N_COLORS = {3: "#a8c8f0", 6: "#4C72B0", 10: "#1a3a6b"}
 
# Col 2: one color per delay value — Okabe-Ito friendly, index matches COL2_DELAYS order
COL2_DELAY_COLORS = {
    0:  "#4C72B0",   # blue   — no delay
    5:  "#DD8452",   # orange — delay=5
    10: "#55A868",   # green  — delay=10
    # add more here if you add more delays to COL2_DELAYS
}
 
COL_COLOR = {
    "col1": "#4C72B0",
    "col2": "#DD8452",
    "col3": "#55A868",
    "col4": "#C44E52",
}
 
# Col 1: semantic labels for comm_radius values on a 10x10 grid (diagonal ≈ 14.1)
RADIUS_LABELS = {
    1.0:  "Local",
    5.0:  "Mid",
    30.0: "Global",
    # 1.0:  "Local\n($R=1$)",   # uncomment to show raw values
    # 5.0:  "Mid\n($R=5$)",
    # 30.0: "Global\n($R=30$)",
}

RADIUS_COLORS = {
    1.0:  "#DD8452",   # orange — Local
    5.0:  "#4C72B0",   # blue   — Mid
    30.0: "#55A868",   # green  — Global
}
 
# N=6, so percentage labels: 0/6, 1/6, 3/6, 4/6
ATTRITION_N = 6
ATTRITION_PCT_LABELS = {
    "none":   "0%",
    "1 lost": "17%\n(1 of 6)",
    "3 lost": "50%\n(3 of 6)",
    "4 lost": "67%\n(4 of 6)",
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
    m = re.search(r"_comm_delay(\d+)", run_dir.name)
    return int(m.group(1)) if m else 0
 
 
def load_h5(run_dir: Path):
    h5 = run_dir / "run.h5"
    if not h5.exists():
        return None
    data = {}
    try:
        with h5py.File(h5, "r") as f:
            for k in f.keys():
                data[k] = f[k][()]
            for k, v in f.attrs.items():
                data[k] = v
    except OSError as e:
        print(f"[WARN] Corrupted h5: {h5} — {e}")
        return None
    return data
 
 
def detect_cfg_starts(true_phi_map_time, atol=1e-12):
    T = true_phi_map_time.shape[0]
    starts = [0]
    for t in range(1, T):
        if not np.allclose(true_phi_map_time[t], true_phi_map_time[t - 1], atol=atol):
            starts.append(t)
    return np.array(starts, dtype=int)
 
 
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
    kills = re.findall(r":\d+", str(att_str))
    n_kills = len(kills)
    if n_kills == 0:
        return "none"
    return f"{n_kills} lost"
 
 
def _attrition_sort_key(level: str) -> int:
    if level == "none":
        return 0
    m = re.search(r"(\d+)", level)
    return int(m.group(1)) if m else 99
 
 
# ── METRICS ───────────────────────────────────────────────────────────────────
 
def get_roi_time(data):
    traj_rc  = data.get("traj_rc_agents_all", None)
    roi_rc   = data.get("roi_rc", None)
    roi_name = data.get("roi_name", None)
    if traj_rc is None or roi_rc is None or len(roi_rc) == 0:
        return np.nan
 
    if roi_name is not None:
        roi_name = [n.decode() if isinstance(n, bytes) else str(n) for n in roi_name]
 
    roi_sets = {}
    for (r, c), name in zip(roi_rc, roi_name):
        roi_sets.setdefault(name, set()).add((int(r), int(c)))
 
    traj = np.asarray(traj_rc).copy()
    active_mask = data.get("active_mask", None)
    if active_mask is not None:
        traj[~active_mask] = -1
 
    env_dyn = _s(data.get("env_dyn", "static"))
 
    if env_dyn == "dynamic":
        if "env_change_steps" in data:
            cfg_starts = np.concatenate([[0], np.array(data["env_change_steps"], dtype=int)])
        else:
            cfg_starts = detect_cfg_starts(data["true_phi_map_time"])
 
        all_times = []
        for name, cellset in roi_sets.items():
            times = roi_first_visit_per_cfg(traj_rc=traj, roi_cellset=cellset, cfg_starts=cfg_starts)
            for t_rel in times:
                if not np.isnan(t_rel):
                    all_times.append(float(t_rel))
        return float(np.mean(all_times)) if all_times else np.nan
 
    else:
        times = []
        for name, cellset in roi_sets.items():
            t = roi_first_visit_static(traj, cellset)
            if not np.isnan(t):
                times.append(float(t))
        return float(np.mean(times)) if times else np.nan
 
 
def get_cover_time(data):
    ts      = data.get("traj_states", None)
    nstates = data.get("nstates", None)
    if ts is None or nstates is None:
        return np.nan
    result = full_map_cover_time(ts, int(_s(nstates)))
    return float(result) if not np.isnan(result) else np.nan
 
 
def _get_cfg_starts(data):
    """Return absolute cfg start times array."""
    if "env_change_steps" in data:
        raw = np.array(data["env_change_steps"], dtype=int)
        if raw[0] != 0:
            raw = np.concatenate([[0], raw])
        return raw
    return detect_cfg_starts(data["true_phi_map_time"])
 
 
def get_roi_time_per_cfg(data):
    """
    For dynamic runs: returns dict {cfg_key: mean_first_visit_time}
    where cfg_key = "cfg{i}_t{abs_t}" e.g. "cfg0_t0", "cfg1_t500".
    Times are relative to each segment start.
    """
    traj_rc  = data.get("traj_rc_agents_all", None)
    roi_rc   = data.get("roi_rc", None)
    roi_name = data.get("roi_name", None)
    if traj_rc is None or roi_rc is None or len(roi_rc) == 0:
        return {}
 
    roi_name_dec = [n.decode() if isinstance(n, bytes) else str(n) for n in roi_name]
    roi_sets = {}
    for (r, c), name in zip(roi_rc, roi_name_dec):
        roi_sets.setdefault(name, set()).add((int(r), int(c)))
 
    traj = np.asarray(traj_rc).copy()
    active_mask = data.get("active_mask", None)
    if active_mask is not None:
        traj[~np.asarray(active_mask)] = -1
 
    cfg_starts = _get_cfg_starts(data)
    cfg_keys   = [f"cfg{i}_t{int(cfg_starts[i])}" for i in range(len(cfg_starts))]
 
    result = {}
    for i, key in enumerate(cfg_keys):
        seg_times = []
        for name, cellset in roi_sets.items():
            t_rel_vec = roi_first_visit_per_cfg(
                traj_rc=traj, roi_cellset=cellset, cfg_starts=cfg_starts
            )
            if i < len(t_rel_vec) and not np.isnan(t_rel_vec[i]):
                seg_times.append(float(t_rel_vec[i]))
        result[key] = float(np.mean(seg_times)) if seg_times else np.nan
    return result
 
 
def _cover_time_segment(traj_states, nstates, t_start, t_end):
    """First relative timestep within [t_start, t_end) at which all states visited."""
    seg = np.asarray(traj_states[t_start:t_end])
    if seg.ndim == 1:
        seg = seg[:, None]
    visited = set()
    for rel_t in range(len(seg)):
        for s in seg[rel_t]:
            if int(s) >= 0:
                visited.add(int(s))
        if len(visited) >= nstates:
            return float(rel_t)
    return np.nan
 
 
def get_cover_time_per_cfg(data):
    """
    For dynamic runs: returns dict {cfg_key: cover_time_relative_to_segment_start}.
    """
    ts      = data.get("traj_states", None)
    nstates = data.get("nstates", None)
    if ts is None or nstates is None:
        return {}
 
    nstates    = int(_s(nstates))
    cfg_starts = _get_cfg_starts(data)
    T_total    = np.asarray(ts).shape[0]
    cfg_keys   = [f"cfg{i}_t{int(cfg_starts[i])}" for i in range(len(cfg_starts))]
 
    result = {}
    for i, key in enumerate(cfg_keys):
        t0 = int(cfg_starts[i])
        t1 = int(cfg_starts[i + 1]) if i + 1 < len(cfg_starts) else T_total
        result[key] = _cover_time_segment(ts, nstates, t0, t1)
    return result
 
 
# ── DATA COLLECTION ───────────────────────────────────────────────────────────
 
def _collect_from_dir(base_dir: Path, filt: dict) -> pd.DataFrame:
    """Load all runs from base_dir that pass filt."""
    if not base_dir.exists():
        print(f"[WARN] {base_dir} not found")
        return pd.DataFrame()
 
    rows = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_h5(run_dir)
        if data is None:
            continue
        if not passes_filter(data, filt, run_dir=run_dir):
            continue
        env_dyn = str(_s(data.get("env_dyn", "static")))
        row = {
            "run":             run_dir.name,
            "n_agents":        int(_s(data.get("n_agents", 1))),
            "comm_radius":     float(_s(data.get("comm_radius", 0))),
            "comm_delay":      get_comm_delay(data, run_dir),
            "env_dyn":         env_dyn,
            "attrition_level": get_attrition_level(data),
            "roi_time":        get_roi_time(data),
            "cover_time":      get_cover_time(data),
        }
        # for dynamic runs, also store per-segment metrics
        if env_dyn == "dynamic":
            for k, v in get_roi_time_per_cfg(data).items():
                row[f"roi_time_{k}"] = v
            for k, v in get_cover_time_per_cfg(data).items():
                row[f"cover_time_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def collect_column(col_key) -> pd.DataFrame:
    base_dir = COLUMN_DIRS.get(col_key)
    if base_dir is None:
        print(f"[SKIP] {col_key} — set to None")
        return pd.DataFrame()
    base_dir = Path(base_dir)
    filt = COL_FILTERS[col_key]
    df = _collect_from_dir(base_dir, filt)
    print(f"{col_key}: {len(df)} runs loaded")
    if len(df):
        print(f"  comm_radius: {sorted(df.comm_radius.unique())}")
        print(f"  n_agents:    {sorted(df.n_agents.unique())}")
        print(f"  roi_time non-nan:   {df.roi_time.notna().sum()}")
        print(f"  cover_time non-nan: {df.cover_time.notna().sum()}")
    return df
 
 
def collect_col2_by_delay() -> dict:
    """
    Returns a dict {delay_val: df} for each value in COL2_DELAYS.
    delay=0 comes from col1's directory; others from col2's directory.
    """
    dfs_by_delay = {}
    for d in COL2_DELAYS:
        if d == 0:
            base_dir = COLUMN_DIRS.get("col1")
        else:
            base_dir = COLUMN_DIRS.get("col2")
        if base_dir is None:
            dfs_by_delay[d] = pd.DataFrame()
            continue
        filt = dict(env_dyn="static", comm_delay=d, comm_radius=5.0)
        df = _collect_from_dir(Path(base_dir), filt)
        print(f"col2 delay={d}: {len(df)} runs")
        dfs_by_delay[d] = df
    return dfs_by_delay
 
 
# ── PLOTTING ──────────────────────────────────────────────────────────────────
 
def _draw_boxes(ax, data_list, positions, width, color, alpha=0.80, zorder=2):
    """Core boxplot call."""
    bp = ax.boxplot(
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
    return bp
 
 
# def plot_col1(ax, df, metric):
#     """
#     Col 1: grouped boxes.
#     x-axis = comm_radius {1, 5, 30}
#     side-by-side groups = N {3, 6, 10}
#     """
#     x_vals  = sorted(df["comm_radius"].unique())
#     n_grps  = len(N_LIST)
#     w       = 0.18
#     gap     = 0.06
#     offsets = np.linspace(
#         -(n_grps - 1) / 2 * (w + gap),
#          (n_grps - 1) / 2 * (w + gap),
#         n_grps,
#     )
 
#     patches = []
#     for gi, n in enumerate(N_LIST):
#         color = N_COLORS[n]
#         patches.append(mpatches.Patch(facecolor=color, alpha=0.85, label=f"$N={n}$"))
#         data_list = []
#         positions  = []
#         for xi, xv in enumerate(x_vals):
#             mask = (df["comm_radius"] == xv) & (df["n_agents"] == n)
#             vals = df.loc[mask, metric].dropna().values
#             data_list.append(vals if len(vals) else np.array([np.nan]))
#             positions.append(xi + offsets[gi])
#         _draw_boxes(ax, data_list, positions, w, color)
 
#     ax.set_xticks(range(len(x_vals)))
#     ax.set_xticklabels([RADIUS_LABELS.get(v, str(v)) for v in x_vals], fontsize=9)
#     ax.set_xlabel("Communication range", fontsize=10)
#     ax.legend(handles=patches, fontsize=10, loc="upper right", framealpha=0.7, handlelength=1.0)
 
def plot_col1(ax, df, metric):
    """
    Col 1: grouped boxes.
    x-axis = N {3, 6, 10}
    side-by-side groups = comm_radius {Local, Mid, Global}, colored by range.
    """
    x_vals  = sorted(df["n_agents"].unique())     # x is now team size
    r_vals  = sorted(df["comm_radius"].unique())  # groups = the comm sweep
    n_grps  = len(r_vals)
    w       = 0.18
    gap     = 0.06
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        n_grps,
    )

    patches = []
    for gi, rv in enumerate(r_vals):
        color = RADIUS_COLORS.get(rv, "#888888")
        label = RADIUS_LABELS.get(rv, str(rv))
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85, label=label))
        data_list = []
        positions = []
        for xi, n in enumerate(x_vals):
            mask = (df["comm_radius"] == rv) & (df["n_agents"] == n)
            vals = df.loc[mask, metric].dropna().values
            data_list.append(vals if len(vals) else np.array([np.nan]))
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"$N={v}$" for v in x_vals], fontsize=10)
    ax.set_xlabel("Team size, N", fontsize=10)
    ax.legend(handles=patches, fontsize=10, loc="upper right",
              framealpha=0.7, handlelength=1.0)
 
def plot_col2_with_baseline(ax, dfs_by_delay, metric):
    """
    Col 2: x = n_agents, side-by-side boxes per delay value.
    dfs_by_delay: dict {delay_val: df}, keyed by COL2_DELAYS order.
    Colors come from COL2_DELAY_COLORS.
    """
    all_n = sorted(set(
        n for df in dfs_by_delay.values() if not df.empty
        for n in df["n_agents"].unique()
    ))
 
    n_grps  = len(COL2_DELAYS)
    w       = 0.18
    gap     = 0.05
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        n_grps,
    )
 
    patches = []
    for gi, d in enumerate(COL2_DELAYS):
        df    = dfs_by_delay.get(d, pd.DataFrame())
        color = COL2_DELAY_COLORS.get(d, "#888888")
        label = "No delay" if d == 0 else f"$\\tau_{{delay}}={d}$"
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85, label=label))
 
        data_list = []
        positions  = []
        for xi, n in enumerate(all_n):
            if df.empty:
                vals = np.array([np.nan])
            else:
                vals = df.loc[df["n_agents"] == n, metric].dropna().values
                vals = vals if len(vals) else np.array([np.nan])
            data_list.append(vals)
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color, alpha=0.82, zorder=2 + gi)
 
    ax.set_xticks(range(len(all_n)))
    ax.set_xticklabels([f"$N={v}$" for v in all_n], fontsize=10)
    ax.set_xlabel("Team size, N", fontsize=10)
    ax.legend(handles=patches, fontsize=10, loc="upper right", framealpha=0.7, handlelength=1.0)
 
 
def plot_col_simple(ax, df, metric, col_key):
    """Cols 3: single-color boxes, x = n_agents."""
    color  = COL_COLOR[col_key]
    x_vals = sorted(df["n_agents"].unique())
    data_list = [df.loc[df["n_agents"] == v, metric].dropna().values for v in x_vals]
    _draw_boxes(ax, data_list, list(range(len(x_vals))), 0.35, color)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"$N={v}$" for v in x_vals], fontsize=10)
    ax.set_xlabel("Team size, N", fontsize=10)
 
 
def plot_col3(ax, df, metric):
    """
    Col 3 (dynamic): grouped boxes — x = cfg segment (with actual change time as label),
    side-by-side groups = N from N_LIST_COL3 (edit at top of file).
    Each box shows the per-segment metric (time relative to that env change).
    """
    # find all per-cfg columns for this metric, e.g. roi_time_cfg0_t0, roi_time_cfg1_t500
    prefix = f"{metric}_cfg"
    cfg_cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(re.search(r"cfg(\d+)", c).group(1))
    )
 
    if not cfg_cols:
        # fallback: no per-cfg data, just show single value
        plot_col_simple(ax, df, metric, "col3")
        return
 
    # x-tick labels: extract the absolute time from column name e.g. "cfg1_t500" -> "t=500"
    # x_labels = []
    # for col in cfg_cols:
    #     m = re.search(r"_t(\d+)$", col)
    #     x_labels.append(f"$\\mathcal{{K}}={m.group(1)}$" if m else col)
 
    # n_segs  = len(cfg_cols)
    # n_grps  = len(N_LIST_COL3)
    # w       = 0.18
    # gap     = 0.06
    # offsets = np.linspace(
    #     -(n_grps - 1) / 2 * (w + gap),
    #      (n_grps - 1) / 2 * (w + gap),
    #     n_grps,
    # )
 
    change_times = []
    for col in cfg_cols:
        m = re.search(r"_t(\d+)$", col)
        change_times.append(int(m.group(1)) if m else None)

    x_labels = []
    for i, t in enumerate(change_times):
        if t is None:
            x_labels.append(cfg_cols[i])
        elif i + 1 < len(change_times) and change_times[i + 1] is not None:
            x_labels.append(f"$[{t},\\,{change_times[i+1]})$")
        else:
            x_labels.append(f"$[{t},\\,T)$") 
    n_segs  = len(cfg_cols)
    n_grps  = len(N_LIST_COL3)
    w       = 0.18
    gap     = 0.06
    offsets = np.linspace(
        -(n_grps - 1) / 2 * (w + gap),
         (n_grps - 1) / 2 * (w + gap),
        n_grps,
    )

    patches = []
    for gi, n in enumerate(N_LIST_COL3):
        color = N_COLORS[n]
        patches.append(mpatches.Patch(facecolor=color, alpha=0.85, label=f"$N={n}$"))
        data_list = []
        positions  = []
        for xi, col in enumerate(cfg_cols):
            mask = df["n_agents"] == n
            vals = df.loc[mask, col].dropna().values
            data_list.append(vals if len(vals) else np.array([np.nan]))
            positions.append(xi + offsets[gi])
        _draw_boxes(ax, data_list, positions, w, color)
 
    ax.set_xticks(range(n_segs))
    # ax.set_xticklabels(x_labels, fontsize=10)
    # ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=20)
    ax.set_xlabel(r"Time interval after each change", fontsize=10)
    
    # ax.set_xlabel(r"Env change at $\mathcal{K}$", fontsize=10)
    ax.legend(handles=patches, fontsize=10, loc="upper right", framealpha=0.7, handlelength=1.0)
 
 
def plot_col4(ax, df, metric):
    """
    Col 4: x = attrition level (sorted by severity), labeled as % of team lost.
    """
    x_vals = sorted(df["attrition_level"].unique(), key=_attrition_sort_key)
 
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
 
    pct_labels = [ATTRITION_PCT_LABELS.get(v, v) for v in x_vals]
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(pct_labels, fontsize=10, rotation=15, ha="right")
    ax.set_xlabel("Agents lost (% of team)", fontsize=10)
 
 
def decorate(ax, metric, col_key, ylabel, title):
    if ylabel:
        ax.set_ylabel({
            "roi_time":   "ROI discovery time",
            "cover_time": "Full map coverage time",
        }[metric], fontsize=10)
    else:
        ax.set_ylabel("")
    if title:
        ax.set_title(COL_TITLES[col_key], fontsize=10, pad=5)
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)
 
 
def plot_no_data(ax, col_key, title, metric=None):
    ax.text(0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=10)
    if title:
        ax.set_title(COL_TITLES[col_key], fontsize=10, pad=5)
    if metric is not None:
        ax.set_ylabel({
            "roi_time":   "ROI coverage time",
            "cover_time": "Full map coverage time",
        }[metric], fontsize=10)
    sns.despine(ax=ax)
 
 
# ── CSV EXPORT ────────────────────────────────────────────────────────────────
 
def export_csv(dfs, output_path="fig4_data.csv", cols=None):
    rows = []
    for col_key, df in dfs.items():
        if cols is not None and col_key not in cols:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                "col":             col_key,
                "run":             row["run"],
                "n_agents":        row["n_agents"],
                "comm_radius":     row["comm_radius"],
                "comm_delay":      row["comm_delay"],
                "env_dyn":         row["env_dyn"],
                "attrition_level": row.get("attrition_level", "none"),
                "roi_time":        row["roi_time"],
                "cover_time":      row["cover_time"],
            })
    if not rows:
        print("No data to export")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {output_path}  ({len(rows)} rows)")
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main(dfs=None, dfs_col2_by_delay=None):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    col_keys = ["col1", "col2", "col3", "col4"]
 
    if dfs is None:
        dfs = {k: collect_column(k) for k in col_keys}
 
    if dfs_col2_by_delay is None:
        dfs_col2_by_delay = collect_col2_by_delay()
 
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
 
            if df.empty and col_key != "col2":
                plot_no_data(ax, col_key, ttl, metric=metric if yl else None)
                continue
 
            if col_key == "col1":
                plot_col1(ax, df, metric)
            elif col_key == "col2":
                plot_col2_with_baseline(ax, dfs_col2_by_delay, metric)
            elif col_key == "col3":
                plot_col3(ax, df, metric)
            elif col_key == "col4":
                plot_col4(ax, df, metric)
            else:
                plot_col_simple(ax, df, metric, col_key)
 
            decorate(ax, metric, col_key, ylabel=yl, title=ttl)
 
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PNG}")
    print(f"Saved → {OUTPUT_PDF}")
    plt.show()
 
 
if __name__ == "__main__":
    col_keys = ["col1", "col2", "col3", "col4"]
    dfs = {k: collect_column(k) for k in col_keys}
    dfs_col2_by_delay = collect_col2_by_delay()
    export_csv(dfs, "fig4_col4_attrition.csv", cols=["col4"])
    main(dfs=dfs, dfs_col2_by_delay=dfs_col2_by_delay)