"""
export_dynamic_metrics.py
-------------------------
For dynamic-environment runs, exports two CSVs:

  1. roi_metrics_per_cfg.csv
     Columns: filename, n_agents, comm_radius, n_agents, ...
       then for each cfg segment and each ROI:
         roi_<name>_cfg<i>_t<t0>  — time to first visit that ROI after env change at t0
         (NaN if not visited within the segment)

  2. cover_metrics_per_cfg.csv
     Columns: filename, n_agents, comm_radius, ...
       then for each cfg segment:
         cover_cfg<i>_t<t0>  — time to cover full map after env change at t0
         (NaN if not covered before next change)

Both times are *relative* to the segment start (i.e. steps after the env change),
matching the existing roi_first_visit_per_cfg convention.

Usage (from MAES repo root):
    python viz/export_dynamic_metrics.py
"""

import sys, os, re, csv
import numpy as np
import h5py
from pathlib import Path
from collections import OrderedDict

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from viz.metrics import roi_first_visit_per_cfg, roi_first_visit_static

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_DIR = Path("results/Dynamic_analsys_fig40510")   # <-- edit as needed

ROI_OUTPUT  = "roi_metrics_per_cfg.csv"
COVER_OUTPUT = "cover_metrics_per_cfg.csv"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _s(val):
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, bytes):
        val = val.decode()
    return val


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
        print(f"[WARN] {h5}: {e}")
        return None
    return data


def detect_cfg_starts(true_phi_map_time, atol=1e-12):
    """Return absolute timesteps where the env map changed."""
    T = true_phi_map_time.shape[0]
    starts = [0]
    for t in range(1, T):
        if not np.allclose(true_phi_map_time[t], true_phi_map_time[t - 1], atol=atol):
            starts.append(t)
    return np.array(starts, dtype=int)


def get_cfg_starts(data):
    """Get cfg start times, preferring saved env_change_steps attr."""
    if "env_change_steps" in data:
        raw = np.array(data["env_change_steps"], dtype=int)
        # prepend 0 if not already there
        if raw[0] != 0:
            raw = np.concatenate([[0], raw])
        return raw
    return detect_cfg_starts(data["true_phi_map_time"])


def full_map_cover_time_segment(traj_states, nstates, t_start, t_end):
    """
    First timestep (relative to t_start) at which all nstates have been
    visited within traj_states[t_start:t_end].
    Returns NaN if not all states visited within the segment.

    traj_states: (T, n_agents) int array of state indices
    """
    seg = np.asarray(traj_states[t_start:t_end])   # (seg_len, n_agents)
    if seg.ndim == 1:
        seg = seg[:, None]

    visited = set()
    for rel_t in range(len(seg)):
        for agent_states in seg[rel_t]:
            s = int(agent_states)
            if s >= 0:
                visited.add(s)
        if len(visited) >= nstates:
            return float(rel_t)
    return np.nan


# ── MAIN ──────────────────────────────────────────────────────────────────────

def collect_all(batch_dir: Path):
    """
    Returns:
        roi_rows  : list of dicts (one per run)
        cover_rows: list of dicts (one per run)
        all_roi_names : sorted list of all ROI names seen across all runs
        all_cfg_cols  : sorted list of cfg column keys (by cfg index)
    """
    roi_rows   = []
    cover_rows = []

    all_roi_names = set()
    all_cfg_keys  = set()   # e.g. "cfg0_t0", "cfg1_t500"

    run_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
    print(f"Scanning {len(run_dirs)} runs in {batch_dir.name} ...")

    for run_dir in run_dirs:
        data = load_h5(run_dir)
        if data is None:
            continue

        env_dyn = _s(data.get("env_dyn", "static"))
        # only process dynamic runs; skip static silently
        if env_dyn != "dynamic":
            continue

        n_agents   = int(_s(data.get("n_agents", 1)))
        comm_radius = float(_s(data.get("comm_radius", 0)))
        comm_delay  = int(_s(data.get("comm_delay", data.get("comms_delay", 0))))
        nstates    = int(_s(data.get("nstates", 0)))

        traj_rc    = np.asarray(data.get("traj_rc_agents_all"))   # (T, N, 2)
        traj_states = np.asarray(data.get("traj_states"))          # (T, N) or (T,)
        true_phi   = np.asarray(data.get("true_phi_map_time"))    # (T, nstates)

        # apply active_mask if present (dead agents → -1)
        active_mask = data.get("active_mask", None)
        if active_mask is not None:
            traj_rc = traj_rc.copy()
            traj_rc[~np.asarray(active_mask)] = -1

        cfg_starts = get_cfg_starts(data)
        T_total    = traj_rc.shape[0]

        # build cfg column keys using actual absolute change times
        cfg_keys = [f"cfg{i}_t{int(cfg_starts[i])}" for i in range(len(cfg_starts))]
        for k in cfg_keys:
            all_cfg_keys.add(k)

        # ── ROI first-visit per cfg ────────────────────────────────────────
        roi_rc   = data.get("roi_rc",   None)
        roi_name = data.get("roi_name", None)

        roi_base = OrderedDict([
            ("filename",    run_dir.name),
            ("n_agents",    n_agents),
            ("comm_radius", comm_radius),
            ("comm_delay",  comm_delay),
        ])

        if roi_rc is not None and len(roi_rc) > 0:
            roi_name_dec = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in roi_name
            ]
            # build {name: set of (r,c)}
            roi_sets = {}
            for (r, c), name in zip(roi_rc, roi_name_dec):
                roi_sets.setdefault(name, set()).add((int(r), int(c)))

            for rname, cellset in sorted(roi_sets.items()):
                all_roi_names.add(rname)
                t_rel_vec = roi_first_visit_per_cfg(
                    traj_rc=traj_rc,
                    roi_cellset=cellset,
                    cfg_starts=cfg_starts,
                )
                row = dict(roi_base)
                row["roi"] = rname
                for i, key in enumerate(cfg_keys):
                    row[key] = float(t_rel_vec[i]) if i < len(t_rel_vec) else np.nan
                roi_rows.append(row)
        else:
            # no ROI data — still emit a placeholder row so the run shows up
            row = dict(roi_base)
            row["roi"] = "N/A"
            for key in cfg_keys:
                row[key] = np.nan
            roi_rows.append(row)

        # ── Full-map coverage per cfg ──────────────────────────────────────
        cover_row = OrderedDict([
            ("filename",    run_dir.name),
            ("n_agents",    n_agents),
            ("comm_radius", comm_radius),
            ("comm_delay",  comm_delay),
        ])
        for i, key in enumerate(cfg_keys):
            t0 = int(cfg_starts[i])
            t1 = int(cfg_starts[i + 1]) if i + 1 < len(cfg_starts) else T_total
            cover_row[key] = full_map_cover_time_segment(
                traj_states, nstates, t0, t1
            )
        cover_rows.append(cover_row)

    all_cfg_keys_sorted = sorted(
        all_cfg_keys,
        key=lambda k: int(re.search(r"cfg(\d+)", k).group(1))
    )
    all_roi_names_sorted = sorted(all_roi_names)

    print(f"  {len(roi_rows)} ROI rows, {len(cover_rows)} cover rows")
    print(f"  cfg segments found: {all_cfg_keys_sorted}")
    print(f"  ROI names found:    {all_roi_names_sorted}")

    return roi_rows, cover_rows, all_roi_names_sorted, all_cfg_keys_sorted


def write_roi_csv(roi_rows, all_roi_names, all_cfg_keys, output_path):
    """
    Wide format: one row per (run, roi).
    Columns: filename, n_agents, comm_radius, comm_delay, roi,
             cfg0_t0, cfg1_t500, ...
    """
    if not roi_rows:
        print("[SKIP] No ROI rows to write.")
        return

    fieldnames = ["filename", "n_agents", "comm_radius", "comm_delay", "roi"] + all_cfg_keys

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in roi_rows:
            # fill missing cfg cols with empty string
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    print(f"ROI CSV saved → {output_path}  ({len(roi_rows)} rows)")


def write_cover_csv(cover_rows, all_cfg_keys, output_path):
    """
    Wide format: one row per run.
    Columns: filename, n_agents, comm_radius, comm_delay,
             cfg0_t0, cfg1_t500, ...
    """
    if not cover_rows:
        print("[SKIP] No cover rows to write.")
        return

    fieldnames = ["filename", "n_agents", "comm_radius", "comm_delay"] + all_cfg_keys

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in cover_rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    print(f"Cover CSV saved → {output_path}  ({len(cover_rows)} rows)")


if __name__ == "__main__":
    roi_rows, cover_rows, all_roi_names, all_cfg_keys = collect_all(BATCH_DIR)
    write_roi_csv(roi_rows, all_roi_names, all_cfg_keys, ROI_OUTPUT)
    write_cover_csv(cover_rows, all_cfg_keys, COVER_OUTPUT)