#!/usr/bin/env python3
"""
tables_planner_comparison.py

Summarize a planner-comparison batch (ergodic-mh vs mac_dt) from run .h5 files.
Three CSVs are written under <BATCH_DIR>/analysis/:

  1. roi_time_in_roi.csv
        One row per (run, ROI, env-change segment).  `time_in_roi` = raw
        agent-steps spent inside the ROI DURING that segment only.  The ROI's
        cells for a segment come from that segment's env_00k.json (see below),
        so a relocating ROI is split into roi1_cfg0, roi1_cfg1, ... and an ROI
        that switches on later simply has no row for the cfgs before it exists.
        `active_steps` is the segment length (t1 - t0), NOT a run-spanning count.

  2. roi_first_visit.csv
        One row per (run, ROI, env-change segment).  `first_visit_time` =
        timesteps, relative to that segment's start, until any agent first
        enters the ROI's THIS-cfg cells, or NA if not entered before the next
        change.  Static runs collapse to a single segment (cfg 0).

  3. coverage_time.csv
        One row per (run, env-change segment).  `coverage_time` = timesteps,
        relative to that segment's start, until every state has been visited at
        least once, or NA if the map is not fully (re)covered before the next
        change.  The visited-set resets at each env change.  Static runs collapse
        to a single segment (cfg 0, change_step 0) = whole-run coverage.

Why env JSONs for #1/#2:  the h5 `roi_rc`/`roi_name` datasets store only the
FINAL config, so a relocating ROI (roi1: (0,0)->(1,0)->(1,1)->(2,0)) cannot be
named per-segment from the h5 alone.  Each segment's env_00k.json supplies the
per-config name->cell map; we cross-check its cells against that segment's
true_phi "hot" set (true_phi > 0) and warn on any mismatch.

Segment boundaries come from env_change_steps (falling back to
true_phi_change_times); static runs collapse to one segment.  Segment i is bound
to the i-th env config JSON (sorted ENV_GLOB).

Reads run .h5 files with:
  traj_states            (T, N)            state index per agent per step (-1 = dead)
  state_index_map_keys   (S, 2)            (r,c) of every valid cell
  state_index_map_vals   (S,)              its state index
  true_phi_map_time      (E, S) or (T, S)  reward map; event-keyed or dense
  true_phi_change_times  (E,)  optional    activation step per event row (if event-keyed)
  env_change_steps       (K,)  optional    segment-start steps
  attrs: planner, n_agents, env_source, nstates (optional)
"""

import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

# ------------------------------- CONFIG ------------------------------------
BATCH_DIR = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/"
    "results/PLANNER"
)
RUN_GLOB    = "**/run.h5"     # recursive under BATCH_DIR
MAP_FROM    = "attr"          # "attr" -> h5.attrs['env_source']; else parent-dir name
MAP_FILTER  = None            # e.g. "dynamic_v0" to keep only ONE map; None = all

# Folder holding the dynamic env config JSONs (env_000.json, env_001.json, ...).
# Segment i is bound to the i-th file in sorted(ENV_GLOB).  The run's own
# `env_source` attr directory is tried first; ENV_DIR is the fallback.
ENV_DIR     = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/envs"
)
ENV_GLOB    = "env_*.json"
# ---------------------------------------------------------------------------


# -- import the existing coverage helper (single source of truth) -----------
def _load_metrics():
    """Locate viz/metrics.py by walking up from this file, then import."""
    try:                                   # already importable (run from repo root)
        from viz.metrics import full_map_cover_time
        return full_map_cover_time
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        m = cand / "viz" / "metrics.py"
        if m.exists():
            sys.path.insert(0, str(m.parent))
            break
    from metrics import full_map_cover_time
    return full_map_cover_time


full_map_cover_time = _load_metrics()


# -- env-config (JSON) readers ----------------------------------------------
def _expand_rect(rect):
    """[r0,c0,r1,c1] half-open -> set of (r,c).  Matches the env-builder and the
    h5 roi_rc exactly (verified: rect [2,0,3,1] -> {(2,0)})."""
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _load_env_roi_sets(json_path):
    """env_00k.json -> {roi_name: set((r,c))} for that config's rect ROIs."""
    with open(json_path) as fh:
        d = json.load(fh)
    out = {}
    for roi in d.get("rois", []):
        shp = roi.get("shape", {})
        if shp.get("type") == "rect":
            out.setdefault(roi["name"], set()).update(_expand_rect(shp["rect"]))
    return out


def _env_files(f):
    """Sorted list of env config JSONs for this run: prefer the directory of the
    run's `env_source` attr, else ENV_DIR."""
    dirs = []
    if "env_source" in f.attrs:
        p = Path(str(f.attrs["env_source"]))
        dirs.append(p if p.is_dir() else p.parent)
    if ENV_DIR:
        dirs.append(ENV_DIR)
    for d in dirs:
        if d and d.exists():
            fs = sorted(d.glob(ENV_GLOB))
            if fs:
                return fs
    return []


# -- h5 readers -------------------------------------------------------------
def _rc_to_idx_map(f):
    keys = np.asarray(f["state_index_map_keys"])
    vals = np.asarray(f["state_index_map_vals"])
    return {(int(r), int(c)): int(v) for (r, c), v in zip(keys, vals)}


def _idx_to_rc_map(rc2idx):
    return {v: rc for rc, v in rc2idx.items()}


def _row_of_t(f, T):
    """For each timestep t in [0,T), which row of true_phi_map_time is active."""
    tp_rows = f["true_phi_map_time"].shape[0]
    if "true_phi_change_times" in f:             # event-keyed
        ct    = np.asarray(f["true_phi_change_times"]).ravel()
        valid = ct >= 0
        rows  = np.nonzero(valid)[0]
        ct    = ct[valid]
        order = np.argsort(ct)
        ct, rows = ct[order], rows[order]
        k = np.searchsorted(ct, np.arange(T), side="right") - 1
        k = np.clip(k, 0, len(ct) - 1)
        return rows[k]
    return np.clip(np.arange(T), 0, tp_rows - 1)  # dense (T,S): row == timestep


def _nstates(f, rc2idx):
    if "nstates" in f.attrs:
        return int(f.attrs["nstates"])
    return (max(rc2idx.values()) + 1) if rc2idx else 0


def _cfg_starts(f, T):
    """Absolute segment-start steps: [0, change1, change2, ...].
    Static runs (no change dataset) collapse to a single segment [0]."""
    ch = None
    if "env_change_steps" in f:
        ch = np.asarray(f["env_change_steps"]).ravel()
    elif "true_phi_change_times" in f:
        ch = np.asarray(f["true_phi_change_times"]).ravel()
    if ch is None:
        return np.array([0], dtype=int)
    ch = ch[(ch > 0) & (ch < T)]
    ch = np.unique(ch.astype(int))
    return np.concatenate([[0], ch]).astype(int)


# -- per-run processing -----------------------------------------------------
def process_run(path):
    dwell_rows, visit_rows, cover_rows = [], [], []
    with h5py.File(path, "r") as f:
        # ergodic runs store this under `method` (ergodic/mh); only mac_dt
        # writes `planner`. Try planner first, then fall back to method.
        planner  = str(f.attrs.get("planner", f.attrs.get("method", "unknown")))
        n_agents = int(f.attrs.get("n_agents", 1))
        if MAP_FROM == "attr" and "env_source" in f.attrs:
            mp = os.path.basename(str(f.attrs["env_source"])).split(".")[0]
        else:
            mp = os.path.basename(os.path.dirname(path))
        run_id = (os.path.splitext(os.path.basename(path))[0]
                  + "::" + os.path.basename(os.path.dirname(path)))

        traj = np.asarray(f["traj_states"])      # (T, N)
        T, N = traj.shape

        rc2idx   = _rc_to_idx_map(f)
        nstates  = _nstates(f, rc2idx)

        tp       = np.asarray(f["true_phi_map_time"], dtype=float)   # (rows, S)
        row_of_t = _row_of_t(f, T)               # (T,)

        alive   = traj >= 0                      # (T, N)
        safe    = np.where(alive, traj, 0)       # dead -> dummy idx 0 (masked below)

        cfg_starts = _cfg_starts(f, T)

        # ---- (3) full-map coverage time, PER env-change segment ----
        # visited-set resets at each change; coverage_time is relative to the
        # segment start t0, NA if the map isn't fully re-covered before the next
        # change.  Static runs collapse to one segment (cfg 0, t0=0) = whole-run.
        for i, t0 in enumerate(cfg_starts):
            t1  = cfg_starts[i + 1] if i + 1 < len(cfg_starts) else T
            cov = full_map_cover_time(traj[t0:t1], nstates)   # relative to t0
            cover_rows.append(dict(
                planner=planner, map=mp, run=run_id, n_agents=n_agents,
                nstates=nstates, cfg=i, change_step=int(t0),
                coverage_time=(None if (cov is None or np.isnan(cov)) else int(cov)),
            ))

        # ---- (1)+(2) per-(ROI, cfg) dwell & discovery, from env JSONs ----
        # roi_rc in the h5 stores only the FINAL config, so a relocating ROI
        # can't be named per-segment from the h5 alone.  Read each segment's
        # env config for its name->cell map; cross-check against true_phi.
        env_files = _env_files(f)
        if not env_files:
            print(f"[warn] {run_id}: no env JSONs found "
                  f"(env_source / ENV_DIR); ROI dwell+discovery skipped")

        for i, t0 in enumerate(cfg_starts):
            t1      = cfg_starts[i + 1] if i + 1 < len(cfg_starts) else T
            seg_row = tp[row_of_t[t0]]                 # (S,) reward during this segment
            hot_idx = set(np.nonzero(seg_row > 0)[0].tolist())

            env_rois = _load_env_roi_sets(env_files[i]) if i < len(env_files) else {}
            if env_files and i >= len(env_files):
                print(f"[warn] {run_id}: no env JSON for cfg{i} "
                      f"({len(env_files)} files, {len(cfg_starts)} segments)")

            # name -> state-index array for THIS config (drop obstacle/invalid cells)
            roi_idx = {}
            for nm, rcset in env_rois.items():
                idxs = sorted({rc2idx[(r, c)] for (r, c) in rcset if (r, c) in rc2idx})
                if idxs:
                    roi_idx[nm] = np.array(idxs, dtype=int)

            # sanity: env-config cells for this cfg should match the reward map
            union = set().union(*[set(v.tolist()) for v in roi_idx.values()]) \
                    if roi_idx else set()
            if roi_idx and union != hot_idx:
                print(f"[warn] {run_id} cfg{i}: env {env_files[i].name} ROI cells "
                      f"{sorted(union)} != true_phi hot {sorted(hot_idx)} "
                      f"(check env/segment binding or rect convention)")

            seg_alive = alive[t0:t1]                    # (seg, N)
            seg_safe  = safe[t0:t1]                     # (seg, N)
            seg_avail = int(seg_alive.sum())            # alive agent-steps in segment

            for nm, idxs in roi_idx.items():
                in_cell = np.zeros(nstates, dtype=bool)
                in_cell[idxs] = True
                present = in_cell[seg_safe] & seg_alive # (seg, N) agent inside ROI?

                # (1) raw dwell: agent-steps inside the ROI during this segment
                dwell_rows.append(dict(
                    planner=planner, map=mp, run=run_id, n_agents=n_agents,
                    roi=nm, roi_cfg=f"{nm}_cfg{i}", cfg=i, change_step=int(t0),
                    weight=float(seg_row[idxs].sum()),
                    active_steps=int(t1 - t0),
                    avail_agent_steps=seg_avail,
                    time_in_roi=int(present.sum()),
                ))

                # (2) discovery: first step (rel. to t0) any agent enters, else NA
                hit = present.any(axis=1)               # (seg,)
                fv  = int(np.argmax(hit)) if hit.any() else None
                visit_rows.append(dict(
                    planner=planner, map=mp, run=run_id, n_agents=n_agents,
                    roi=nm, roi_cfg=f"{nm}_cfg{i}", cfg=i, change_step=int(t0),
                    first_visit_time=fv,
                ))
    return dwell_rows, visit_rows, cover_rows


def main():
    base    = Path(sys.argv[1]) if len(sys.argv) > 1 else BATCH_DIR
    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(str(p) for p in base.glob(RUN_GLOB))
    if not paths:
        print(f"no run files matched: {base / RUN_GLOB}"); return

    dwell_rows, visit_rows, cover_rows = [], [], []
    for p in paths:
        try:
            d, v, c = process_run(p)
            dwell_rows += d
            visit_rows += v
            cover_rows += c
        except Exception as ex:
            print(f"[skip] {p}: {ex}")

    df_dwell = pd.DataFrame(dwell_rows)
    df_visit = pd.DataFrame(visit_rows)
    df_cover = pd.DataFrame(cover_rows)

    if MAP_FILTER:
        for d in (df_dwell, df_visit, df_cover):
            if not d.empty:
                d.drop(d[d["map"] != MAP_FILTER].index, inplace=True)

    # nullable Int so "not found" stays an integer column with a clean NA
    if not df_visit.empty:
        df_visit["first_visit_time"] = df_visit["first_visit_time"].astype("Int64")
    if not df_cover.empty:
        df_cover["coverage_time"] = df_cover["coverage_time"].astype("Int64")

    df_dwell.to_csv(out_dir / "roi_time_in_roi.csv", index=False, na_rep="NA")
    df_visit.to_csv(out_dir / "roi_first_visit.csv", index=False, na_rep="NA")
    df_cover.to_csv(out_dir / "coverage_time.csv",  index=False, na_rep="NA")

    n_runs = df_cover["run"].nunique() if not df_cover.empty else 0
    print(f"runs processed: {n_runs}")
    if not df_dwell.empty:
        print("\nmean time_in_roi (agent-steps) per (planner, roi_cfg):")
        print(df_dwell.groupby(["planner", "roi_cfg"])["time_in_roi"].mean().round(1))
    if not df_visit.empty:
        nfound = (df_visit["first_visit_time"].isna()
                  .groupby([df_visit["planner"], df_visit["cfg"]]).sum())
        print("\nROIs not discovered (NA), per (planner, cfg):")
        print(nfound)
    if not df_cover.empty:
        print("\nsegments never fully covered (NA), per (planner, cfg):")
        print(df_cover["coverage_time"].isna()
              .groupby([df_cover["planner"], df_cover["cfg"]]).sum())

    print(f"\nwrote 3 CSVs to {out_dir}/")


if __name__ == "__main__":
    main()