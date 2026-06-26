#!/usr/bin/env python
"""
tables_planner_comparison.py

Summarize a planner-comparison batch (ergodic vs mac_dt) from HDF5 run folders.

Outputs four CSVs:
  1. roi_first_visit_per_run.csv   - per run, per ROI: first-visit time
  2. roi_first_visit_avg.csv       - averaged over runs, grouped by (planner, map)
  3. coverage_per_run.csv          - per run: full-map discovery time
  4. coverage_avg.csv              - averaged over runs, grouped by (planner, map)

Static envs: one first-visit / coverage time per run.
Dynamic envs: one value per environment segment (after each env change), where
the visited-set is reset at each change (time to (re)reach / (re)cover from the
change onward).

Everything is read straight from run.h5 (roi_rc/roi_name, traj_rc, traj_states,
env_change_steps, nstates) - no env reconstruction needed.

Requires: ergodic finalize() must write attrs["planner"]="ergodic"
(mac_dt already writes attrs["planner"]="mac_dt").
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py

# repo on path (mirrors the reference script)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── CONFIG ───────────────────────────────────────────────────────────────────
BATCH_DIR = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/"
    "results/MAC_DT-Ergodic_2Ksteps_0625"          
)
OUT_DIR = BATCH_DIR / "analysis"


# ── loaders / helpers ────────────────────────────────────────────────────────
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
    """A stable map label. Prefer env_source basename; fall back to folder."""
    src = data.get("env_source", None)
    if src is not None:
        src = src.decode() if isinstance(src, bytes) else str(src)
        name = os.path.basename(src.rstrip("/")).replace(".json", "")
        if name and name != "NA":
            return name
    return Path(run_dir).name


def _start_tag(data):
    """First start cell, from traj_rc at t=0, as 'r,c' — distinguishes replicates."""
    traj_rc = np.asarray(data["traj_rc_agents_all"])
    r, c = int(traj_rc[0, 0, 0]), int(traj_rc[0, 0, 1])
    return f"{r},{c}"


def roi_name_to_cells(data):
    """Build {roi_name: set((r,c))} from saved roi_rc / roi_name."""
    roi_rc = np.asarray(data["roi_rc"])
    roi_nm = np.asarray(data["roi_name"])
    out = defaultdict(set)
    for (r, c), nm in zip(roi_rc, roi_nm):
        nm = nm.decode() if isinstance(nm, bytes) else str(nm)
        out[nm].add((int(r), int(c)))
    return dict(out)


def segment_bounds(data, T):
    """
    Return list of (seg_idx, t_start, t_end) segments.
    Static -> one segment [0, T). Dynamic -> split at env_change_steps.
    """
    env_dyn = data.get("env_dyn", b"static")
    env_dyn = env_dyn.decode() if isinstance(env_dyn, bytes) else str(env_dyn)
    if env_dyn != "dynamic":
        return [(0, 0, T)]
    changes = np.asarray(data.get("env_change_steps", [])).astype(int).tolist()
    bounds = [0] + [c for c in changes if 0 < c < T] + [T]
    return [(i, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# ── metric 1: per-ROI first visit ────────────────────────────────────────────
def roi_first_visits(data):
    """
    For each (segment, ROI): first step (relative to segment start) any live
    agent enters that ROI. NaN if never. Per-ROI (not union).
    """
    traj_rc = np.asarray(data["traj_rc_agents_all"])      # (T, N, 2)
    T = traj_rc.shape[0]
    roi_cells = roi_name_to_cells(data)
    rows = []
    for seg_idx, t0, t1 in segment_bounds(data, T):
        for roi, cells in roi_cells.items():
            found = np.nan
            for t in range(t0, t1):
                hit = False
                for a in range(traj_rc.shape[1]):
                    r, c = int(traj_rc[t, a, 0]), int(traj_rc[t, a, 1])
                    if r < 0:                      # dead agent sentinel
                        continue
                    if (r, c) in cells:
                        hit = True
                        break
                if hit:
                    found = t - t0
                    break
            rows.append((seg_idx, roi, found))
    return rows


# ── metric 2: full-map coverage ──────────────────────────────────────────────
def full_map_cover(data):
    """
    Per segment: first step (relative to segment start) when every free cell has
    been visited, with the visited-set reset at each segment start. NaN if never.
    """
    traj_states = np.asarray(data["traj_states"])         # (T, N)
    T = traj_states.shape[0]
    nstates = int(_item(data["nstates"]))
    rows = []
    for seg_idx, t0, t1 in segment_bounds(data, T):
        visited = np.zeros(nstates, dtype=bool)
        cover_t = np.nan
        for t in range(t0, t1):
            for a in range(traj_states.shape[1]):
                s = int(traj_states[t, a])
                if s >= 0:
                    visited[s] = True
            if visited.all():
                cover_t = t - t0
                break
        rows.append((seg_idx, cover_t, int(visited.sum()), nstates))
    return rows


# ── build tables ─────────────────────────────────────────────────────────────
def build_tables(batch_dir: Path):
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name != "analysis"]
    print(f"{len(run_dirs)} run folders in {batch_dir}")

    roi_records, cov_records = [], []

    for run_dir in run_dirs:
        data = load_run(run_dir)
        if data is None:
            print(f"  [skip] no run.h5 in {run_dir.name}")
            continue

        planner = _planner(data)
        map_id  = _map_id(data, run_dir)
        start   = _start_tag(data)
        n_ag    = int(_item(data["n_agents"]))
        meta = dict(planner=planner, map=map_id, start=start, n_agents=n_ag,
                    run=run_dir.name)

        for seg_idx, roi, t in roi_first_visits(data):
            roi_records.append({**meta, "segment": seg_idx, "roi": roi,
                                "first_visit": t})

        for seg_idx, cov_t, n_vis, nstates in full_map_cover(data):
            cov_records.append({**meta, "segment": seg_idx,
                                "cover_time": cov_t,
                                "cells_visited": n_vis, "nstates": nstates})

    roi_df = pd.DataFrame(roi_records)
    cov_df = pd.DataFrame(cov_records)
    return roi_df, cov_df


def summarize(roi_df, cov_df):
    """Average over runs, grouped by (planner, map). Keeps segment & ROI detail."""
    # ROI: average first-visit across replicate runs, per (planner, map, segment, roi)
    roi_avg = (
        roi_df.groupby(["planner", "map", "segment", "roi"], as_index=False)
        .agg(first_visit_mean=("first_visit", "mean"),
             first_visit_std=("first_visit", "std"),
             n_runs=("first_visit", "size"),
             n_reached=("first_visit", lambda s: int(s.notna().sum())))
    )
    # Coverage: average cover-time across replicate runs, per (planner, map, segment)
    cov_avg = (
        cov_df.groupby(["planner", "map", "segment"], as_index=False)
        .agg(cover_time_mean=("cover_time", "mean"),
             cover_time_std=("cover_time", "std"),
             n_runs=("cover_time", "size"),
             n_fully_covered=("cover_time", lambda s: int(s.notna().sum())))
    )
    return roi_avg, cov_avg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roi_df, cov_df = build_tables(BATCH_DIR)

    if roi_df.empty:
        print("No runs found — check BATCH_DIR.")
        return

    roi_avg, cov_avg = summarize(roi_df, cov_df)

    roi_df.to_csv(OUT_DIR / "roi_first_visit_per_run.csv", index=False)
    roi_avg.to_csv(OUT_DIR / "roi_first_visit_avg.csv", index=False)
    cov_df.to_csv(OUT_DIR / "coverage_per_run.csv", index=False)
    cov_avg.to_csv(OUT_DIR / "coverage_avg.csv", index=False)

    print(f"\nwrote 4 CSVs to {OUT_DIR}")
    print("\n=== ROI first-visit (avg by planner, map) ===")
    print(roi_avg.to_string(index=False))
    print("\n=== Full-map coverage (avg by planner, map) ===")
    print(cov_avg.to_string(index=False))


if __name__ == "__main__":
    main()
