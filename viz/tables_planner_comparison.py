#!/usr/bin/env python3
"""
tables_planner_comparison.py

Summarize a planner-comparison batch (ergodic-mh vs mac_dt) from run .h5 files.
Three CSVs are written under <BATCH_DIR>/analysis/:

  1. roi_time_in_roi.csv
        One row per (run, ROI).  `time_in_roi` = raw agent-steps spent inside
        the ROI, counted ONLY over the ROI's active lifetime (see ACTIVE_FRAC).
        This replaces the old (incorrect) fraction-of-time metric.

  2. roi_first_visit.csv
        One row per (run, ROI, env-change segment).  `first_visit_time` =
        timesteps, relative to that segment's start, until any agent first
        enters the ROI in that segment, or NA if it is not entered before the
        next change.  Static runs collapse to a single segment (cfg 0).

  3. coverage_time.csv
        One row per (run, env-change segment).  `coverage_time` = timesteps,
        relative to that segment's start, until every state has been visited at
        least once, or NA if the map is not fully (re)covered before the next
        change.  The visited-set resets at each env change.  Static runs collapse
        to a single segment (cfg 0, change_step 0) = whole-run coverage.

"active" (for #1) = the ROI's reward mass (true_phi summed over its cells) is at
least ACTIVE_FRAC of that ROI's own peak reward mass, so an ROI that switches on
later is judged only over its own lifetime.

Discovery (#2) and coverage (#3) reuse the existing helpers in viz/metrics.py,
applied per env-change segment (segment boundaries from env_change_steps, falling
back to true_phi_change_times; static runs collapse to one segment):
    roi_first_visit_per_cfg(traj_rc, roi_cellset, cfg_starts) -> [time|nan, ...]
    full_map_cover_time(traj_states[t0:t1], nstates)          -> time | nan
Times are relative to each segment's start, NA if not reached before the next
change.

Reads run .h5 files with:
  traj_states            (T, N)            state index per agent per step (-1 = dead)
  traj_rc                (T, N, 2)         optional; derived from traj_states if absent
  roi_rc                 (M, 2)            (r,c) of every ROI cell
  roi_name               (M,)  bytes       ROI name per cell (parallel to roi_rc)
  state_index_map_keys   (S, 2)            (r,c) of every valid cell
  state_index_map_vals   (S,)              its state index
  true_phi_map_time      (E, S) or (T, S)  reward map; event-keyed or dense
  true_phi_change_times  (E,)  optional    activation step per event row (if event-keyed)
  attrs: planner, n_agents, env_source, nstates (optional)
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

# ------------------------------- CONFIG ------------------------------------
BATCH_DIR = Path(
    "/home/gaby/Documents/UCB/AAM/GIT/sastryrepo/MarkovErgodic/MAES/"
    "results/PlannerComp_variableUCB"
)
RUN_GLOB    = "**/run.h5"     # recursive under BATCH_DIR
MAP_FROM    = "attr"          # "attr" -> h5.attrs['env_source']; else parent-dir name
MAP_FILTER  = None            # e.g. "dynamic_v0" to keep only ONE map; None = all
ACTIVE_FRAC = 0.5             # ROI "exists" when reward mass >= ACTIVE_FRAC * its own peak
# ---------------------------------------------------------------------------


# ── import the existing metrics helpers (single source of truth) ───────────
def _load_metrics():
    """Locate viz/metrics.py by walking up from this file, then import."""
    try:                                   # already importable (run from repo root)
        from viz.metrics import (roi_first_visit_static, roi_first_visit_per_cfg,
                                 full_map_cover_time)
        return roi_first_visit_static, roi_first_visit_per_cfg, full_map_cover_time
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        m = cand / "viz" / "metrics.py"
        if m.exists():
            sys.path.insert(0, str(m.parent))
            break
    from metrics import (roi_first_visit_static, roi_first_visit_per_cfg,
                         full_map_cover_time)
    return roi_first_visit_static, roi_first_visit_per_cfg, full_map_cover_time


roi_first_visit_static, roi_first_visit_per_cfg, full_map_cover_time = _load_metrics()


# ── h5 readers ─────────────────────────────────────────────────────────────
def _rc_to_idx_map(f):
    keys = np.asarray(f["state_index_map_keys"])
    vals = np.asarray(f["state_index_map_vals"])
    return {(int(r), int(c)): int(v) for (r, c), v in zip(keys, vals)}


def _idx_to_rc_map(rc2idx):
    return {v: rc for rc, v in rc2idx.items()}


def _roi_index_sets(f, rc2idx):
    """name -> sorted np.array of state indices (obstacle ROI cells dropped)."""
    rc   = np.asarray(f["roi_rc"])
    name = [n.decode() if isinstance(n, bytes) else str(n)
            for n in np.asarray(f["roi_name"])]
    sets = {}
    for (r, c), nm in zip(rc, name):
        idx = rc2idx.get((int(r), int(c)))
        if idx is not None:                      # skip ROI cells that are obstacles
            sets.setdefault(nm, []).append(idx)
    return {nm: np.array(sorted(set(v)), dtype=int) for nm, v in sets.items()}


def _roi_rc_sets(f):
    """name -> set of (r,c) tuples (for roi_first_visit_static)."""
    rc   = np.asarray(f["roi_rc"])
    name = [n.decode() if isinstance(n, bytes) else str(n)
            for n in np.asarray(f["roi_name"])]
    sets = {}
    for (r, c), nm in zip(rc, name):
        sets.setdefault(nm, set()).add((int(r), int(c)))
    return sets


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


def _build_traj_rc(f, traj, idx2rc):
    """Use stored traj_rc if present, else map traj_states -> (r,c) per agent.
    Dead agents (state < 0) map to (-1,-1), which is never an ROI cell."""
    if "traj_rc" in f:
        return np.asarray(f["traj_rc"])
    T, N = traj.shape
    out = np.full((T, N, 2), -1, dtype=int)
    for t in range(T):
        for a in range(N):
            s = int(traj[t, a])
            if s >= 0:
                r, c = idx2rc.get(s, (-1, -1))
                out[t, a, 0] = r
                out[t, a, 1] = c
    return out


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


# ── per-run processing ─────────────────────────────────────────────────────
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
        idx2rc   = _idx_to_rc_map(rc2idx)
        roi_sets = _roi_index_sets(f, rc2idx)    # name -> state-index array
        roi_rcs  = _roi_rc_sets(f)               # name -> set of (r,c)
        nstates  = _nstates(f, rc2idx)

        tp       = np.asarray(f["true_phi_map_time"], dtype=float)   # (rows, S)
        row_of_t = _row_of_t(f, T)               # (T,)

        alive   = traj >= 0                      # (T, N)
        safe    = np.where(alive, traj, 0)       # dead -> dummy idx 0 (masked below)
        avail_t = alive.sum(axis=1)              # (T,) alive agents per step

        roi_max = max((int(c.max()) for c in roi_sets.values() if len(c)), default=0)
        maxidx  = int(max(int(safe.max()) if safe.size else 0, roi_max))

        traj_rc = _build_traj_rc(f, traj, idx2rc)

        # ---- (3) full-map coverage time, PER env-change segment ----
        # visited-set resets at each change; coverage_time is relative to the
        # segment start t0, NA if the map isn't fully re-covered before the next
        # change.  Static runs collapse to one segment (cfg 0, t0=0) = whole-run.
        cfg_starts = _cfg_starts(f, T)
        for i, t0 in enumerate(cfg_starts):
            t1  = cfg_starts[i + 1] if i + 1 < len(cfg_starts) else T
            cov = full_map_cover_time(traj[t0:t1], nstates)   # relative to t0
            cover_rows.append(dict(
                planner=planner, map=mp, run=run_id, n_agents=n_agents,
                nstates=nstates, cfg=i, change_step=int(t0),
                coverage_time=(None if (cov is None or np.isnan(cov)) else int(cov)),
            ))

        for nm, cells in roi_sets.items():
            # ---- (2) ROI discovery, PER env-change segment ----
            # roi_first_visit_per_cfg returns one time per segment, relative to
            # that segment's start (NaN if the ROI isn't entered in the segment).
            fvs = roi_first_visit_per_cfg(traj_rc, roi_rcs.get(nm, set()), cfg_starts)
            for i, fv in enumerate(fvs):
                visit_rows.append(dict(
                    planner=planner, map=mp, run=run_id, n_agents=n_agents, roi=nm,
                    cfg=i, change_step=int(cfg_starts[i]),
                    first_visit_time=(None if (fv is None or np.isnan(fv)) else int(fv)),
                ))

            # ---- (1) raw dwell time inside ROI over its active lifetime ----
            if len(cells) == 0:
                continue
            w_snap = tp[:, cells].sum(axis=1)    # reward mass per snapshot row
            w_t    = w_snap[row_of_t]            # (T,) reward mass at each timestep
            peak   = float(w_t.max())
            if peak <= 0:
                continue                         # ROI never carries reward -> skip dwell
            active = w_t >= ACTIVE_FRAC * peak   # (T,) the ROI's lifetime mask

            in_cell = np.zeros(maxidx + 1, dtype=bool)
            in_cell[cells] = True
            inroi_t = (in_cell[safe] & alive).sum(axis=1)   # (T,) agents in ROI per step

            dwell = int(inroi_t[active].sum())              # raw agent-steps in ROI
            avail = int(avail_t[active].sum())              # alive agent-steps available
            dwell_rows.append(dict(
                planner=planner, map=mp, run=run_id, n_agents=n_agents, roi=nm,
                weight=peak, active_steps=int(active.sum()),
                avail_agent_steps=avail, time_in_roi=dwell,
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
        print("\nmean time_in_roi (agent-steps, over ROI active lifetime):")
        print(df_dwell.groupby(["planner", "roi"])["time_in_roi"].mean().round(1))
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