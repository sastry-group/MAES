"""
export_dynamic_metrics.py
-------------------------
Per-event metrics from run .h5 files (static or dynamic). Two CSVs, one row per event:

  1. cover_cycles.csv
     One row per back-to-back full-map cover by the TEAM (union of all agents).
     The visited set resets each time every state has been seen.
       cycle      : cycle index within the run
       t_start    : step the cycle started
       length     : steps to visit every state (team union)
       complete   : False for the trailing cycle cut off by the end of the run
                    (length is then a lower bound -> exclude or treat as censored)

  2. roi_episodes.csv
     One row per ROI episode. An episode opens when an ROI appears or relocates,
     and closes when it moves or disappears (or the run ends).
       t_start    : step the ROI appeared / moved there
       dur        : steps the ROI stayed in that location
       t_hit      : steps from t_start until ANY agent first enters it (NaN = never)
       end        : 'moved' (moved/gone before t_end) or 'run_end'
                    -> a NaN t_hit with end='run_end' is censored, not a miss

ROI cells come from the env JSONs (h5 roi_rc only stores the final config).
Dynamic runs bind change i to the i-th sorted JSON in env_source's directory;
the sim keeps the last JSON when change steps outrun the files, so those no-op
changes are skipped. Static runs use only their own env_source JSON.

Usage (from MAES repo root):
    python viz/export_dynamic_metrics.py [BATCH_DIR]
"""

import sys, os, json, glob, re
import numpy as np
import h5py
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_DIR    = Path("results/PLANNER_TEST3")   
COVER_OUTPUT = "cover_cycles_planner.csv"
ROI_OUTPUT   = "roi_episodes_planner.csv"

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


def get_change_starts(data, T):
    """[0, change1, change2, ...] from saved env_change_steps; [0] if static."""
    if _s(data.get("env_dyn", "static")) != "dynamic" or "env_change_steps" not in data:
        return [0]
    ch = sorted({int(t) for t in np.atleast_1d(data["env_change_steps"]) if 0 < int(t) < T})
    return [0] + ch


def _rois_from_json(path):
    with open(path) as f:
        cfg = json.load(f)
    d = {}
    for r in cfg.get("rois", []):
        shp = r.get("shape", {})
        if shp.get("type") != "rect":
            continue
        r0, c0, r1, c1 = shp["rect"]
        d.setdefault(r.get("name", "roi"), set()).update(
            (i, j) for i in range(r0, r1) for j in range(c0, c1))
    return d


_SEG_ROIS = {}

def segment_rois(src, dynamic):
    """List of {roi_name: set((r,c))}, one per env config, in sim load order."""
    key = (src, dynamic)
    if key in _SEG_ROIS:
        return _SEG_ROIS[key]
    if not dynamic and src.endswith(".json"):
        out = [_rois_from_json(src)]
    else:
        cfg_dir = os.path.dirname(src) if src.endswith(".json") else src
        out = [_rois_from_json(p) for p in sorted(glob.glob(os.path.join(cfg_dir, "*.json")))]
    _SEG_ROIS[key] = out
    return out


def cover_cycles(traj_states, nstates):
    """Back-to-back team covers. Returns [(t_start, length, complete), ...];
    the last entry is the incomplete trailing cycle (if any steps remain)."""
    seg = np.asarray(traj_states)
    if seg.ndim == 1:
        seg = seg[:, None]
    visited = np.zeros(nstates, dtype=bool)
    t_last, out = 0, []
    for t, row in enumerate(seg):
        visited[row[row >= 0]] = True
        if visited.all():
            out.append((t_last, t + 1 - t_last, True))
            t_last = t + 1
            visited[:] = False
    if t_last < len(seg):
        out.append((t_last, len(seg) - t_last, False))
    return out


def roi_episodes(seg_rois, change_starts, T):
    """[(name, cells, t_start, t_end, end_reason), ...]"""
    n = len(seg_rois)
    if n == 0:
        return []
    # skip no-op changes (sim reuses the last json once files run out)
    real   = [i for i in range(len(change_starts))
              if i == 0 or min(i, n - 1) != min(i - 1, n - 1)]
    bounds = [int(change_starts[i]) for i in real] + [T]

    open_ep, eps = {}, []
    for k, i in enumerate(real):
        t0, rois = bounds[k], seg_rois[min(i, n - 1)]
        for name in list(open_ep):
            if rois.get(name) != open_ep[name][0]:          # moved or gone -> close
                cells, ts = open_ep.pop(name)
                eps.append((name, cells, ts, t0, "moved"))
        for name, cells in rois.items():
            if name not in open_ep:                         # new or just moved -> open
                open_ep[name] = (cells, t0)
    eps += [(name, cells, ts, T, "run_end") for name, (cells, ts) in open_ep.items()]
    return eps


# ── MAIN ──────────────────────────────────────────────────────────────────────

def collect_all(batch_dir: Path):
    cover_rows, hit_rows = [], []

    for run_dir in sorted(d for d in batch_dir.iterdir() if d.is_dir()):
        data = load_h5(run_dir)
        if data is None:
            continue

        meta = dict(
            filename    = run_dir.name,
            planner     = _s(data.get("planner", data.get("method", "unknown"))),
            env_dyn     = _s(data.get("env_dyn", "static")),
            n_agents    = int(_s(data.get("n_agents", 1))),
            comm_radius = float(_s(data.get("comm_radius", np.nan))),
            comm_delay  = int(_s(data.get("comm_delay", data.get("comms_delay", 0)))),
            attrition   = int(re.search(r"_attrition(\d+)", run_dir.name).group(1))
                          if "_attrition" in run_dir.name else 0,
        )
        nstates     = int(_s(data.get("nstates", 0)))
        traj_rc     = np.asarray(data["traj_rc_agents_all"]).astype(int)   # (T, N, 2)
        traj_states = np.asarray(data["traj_states"]).astype(int)          # (T, N)
        active_mask = data.get("active_mask", None)
        if active_mask is not None:                                       # dead agents -> -1
            dead = ~np.asarray(active_mask, dtype=bool)
            traj_rc = traj_rc.copy();         traj_rc[dead] = -1
            traj_states = traj_states.copy(); traj_states[dead] = -1
        T = traj_rc.shape[0]

        # ── Team cover cycles ─────────────────────────────────────────────
        for c, (ts, length, complete) in enumerate(cover_cycles(traj_states, nstates)):
            cover_rows.append(dict(meta, cycle=c, t_start=ts, length=length, complete=complete))

        # ── ROI time-to-hit ───────────────────────────────────────────────
        src      = _s(data.get("env_source", ""))
        dynamic  = meta["env_dyn"] == "dynamic"
        seg_rois = segment_rois(src, dynamic)
        if not seg_rois:
            print(f"[WARN] {run_dir.name}: no env JSONs at {src}; ROI episodes skipped")
            continue
        eps = roi_episodes(seg_rois, get_change_starts(data, T), T)

        # grid big enough for every ROI cell and every visited cell
        H = max([traj_rc[..., 0].max()] + [r for _, cells, *_ in eps for r, _ in cells]) + 1
        W = max([traj_rc[..., 1].max()] + [c for _, cells, *_ in eps for _, c in cells]) + 1
        valid = (traj_rc[..., 0] >= 0) & (traj_rc[..., 1] >= 0)            # (T, N)
        r_idx = np.where(valid, traj_rc[..., 0], 0)
        c_idx = np.where(valid, traj_rc[..., 1], 0)

        for name, cells, ts, te, end in eps:
            mask = np.zeros((H, W), dtype=bool)
            rr, cc = zip(*cells)
            mask[list(rr), list(cc)] = True
            hit = (mask[r_idx[ts:te], c_idx[ts:te]] & valid[ts:te]).any(axis=1)   # (te-ts,)
            hit_rows.append(dict(meta, roi=name, t_start=ts, dur=te - ts,
                                 t_hit=float(np.argmax(hit)) if hit.any() else np.nan,
                                 end=end))

    return cover_rows, hit_rows


if __name__ == "__main__":
    batch = Path(sys.argv[1]) if len(sys.argv) > 1 else BATCH_DIR
    cover_rows, hit_rows = collect_all(batch)

    c = pd.DataFrame(cover_rows)
    h = pd.DataFrame(hit_rows)
    if not h.empty:
        h["kind"] = h.roi.str.extract(r"^([a-z]+)")
    c.to_csv(batch / COVER_OUTPUT, index=False, na_rep="NA")
    h.to_csv(batch / ROI_OUTPUT,   index=False, na_rep="NA")
    print(f"{COVER_OUTPUT}: {len(c)} cycles from {c.filename.nunique() if len(c) else 0} runs")
    print(f"{ROI_OUTPUT}:   {len(h)} ROI episodes")
    print(f"written to {batch}/")