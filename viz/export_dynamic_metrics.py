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

import sys, os, re, csv, json, glob
import numpy as np
import h5py
from pathlib import Path
from collections import OrderedDict
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# from viz.metrics import roi_first_visit_per_cfg, roi_first_visit_static

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_DIR = Path("results/COMMS_SWEEP")   # <-- edit as needed

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

_SEG_ROIS = {}

def segment_rois(cfg_dir):
    """One {roi_name: set((r,c))} per json, in the order the sim loads them."""
    if cfg_dir in _SEG_ROIS:
        return _SEG_ROIS[cfg_dir]
    out = []
    for p in sorted(glob.glob(os.path.join(cfg_dir, "*.json"))):
        with open(p) as f:
            cfg = json.load(f)
        d = {}
        for r in cfg.get("rois", []):
            shp = r.get("shape", {})
            if shp.get("type") != "rect":
                continue
            r0, c0, r1, c1 = shp["rect"]
            d.setdefault(r.get("name", "roi"), set()).update(
                (i, j) for i in range(r0, r1) for j in range(c0, c1))
        out.append(d)
    _SEG_ROIS[cfg_dir] = out
    return out


def cover_cycles(traj_states, nstates):
    """Back-to-back full-map covers over the whole run: reset the visited set
    each time every state has been seen. Returns list of cycle lengths (steps)."""
    seg = np.asarray(traj_states)
    if seg.ndim == 1:
        seg = seg[:, None]
    visited = np.zeros(nstates, dtype=bool)
    t_last, cycles = 0, []
    for t, row in enumerate(seg):
        visited[row[row >= 0]] = True
        if visited.all():
            cycles.append(t + 1 - t_last)
            t_last = t + 1
            visited[:] = False
    return cycles

# ── MAIN ──────────────────────────────────────────────────────────────────────

def collect_all(batch_dir: Path):
    """
    Returns:
        cover_rows   : one dict per run (full-map cover per cfg + cover cycles)
        hit_rows     : one dict per ROI episode (ROI keeps an episode until it
                       moves or disappears; t_hit = steps to first visit, NaN = never)
        all_cfg_keys : sorted cfg column keys
    """
    cover_rows, hit_rows = [], []
    all_cfg_keys = set()

    run_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
    # print(f"Scanning {len(run_dirs)} runs in {batch_dir.name} ...")

    for run_dir in run_dirs:
        data = load_h5(run_dir)
        if data is None:
            continue
        if _s(data.get("env_dyn", "static")) != "dynamic":
            continue

        n_agents    = int(_s(data.get("n_agents", 1)))
        comm_radius = (float(_s(data["comm_radius"])) if "comm_radius" in data
                       else float(re.search(r"_comm_radius([\d.]+?)_concenIter", run_dir.name).group(1)))
        comm_delay  = int(_s(data.get("comm_delay", data.get("comms_delay", 0))))
        nstates     = int(_s(data.get("nstates", 0)))

        traj_rc     = np.asarray(data.get("traj_rc_agents_all"))   # (T, N, 2)
        traj_states = np.asarray(data.get("traj_states"))          # (T, N)
        active_mask = data.get("active_mask", None)
        if active_mask is not None:                                # dead agents -> -1
            traj_rc = traj_rc.copy()
            traj_rc[~np.asarray(active_mask)] = -1

        cfg_starts = get_cfg_starts(data)
        T_total    = traj_rc.shape[0]
        cfg_keys   = [f"cfg{i}_t{int(cfg_starts[i])}" for i in range(len(cfg_starts))]
        all_cfg_keys.update(cfg_keys)

        # ── Full-map coverage ──────────────────────────────────────────────
        cover_row = OrderedDict([("filename", run_dir.name), ("n_agents", n_agents),
                                 ("comm_radius", comm_radius), ("comm_delay", comm_delay)])
        for i, key in enumerate(cfg_keys):
            t0 = int(cfg_starts[i])
            t1 = int(cfg_starts[i + 1]) if i + 1 < len(cfg_starts) else T_total
            cover_row[key] = full_map_cover_time_segment(traj_states, nstates, t0, t1)
        cyc = cover_cycles(traj_states, nstates)
        cover_row["cycle_mean"] = float(np.mean(cyc)) if cyc else np.nan
        cover_row["n_cycles"]   = len(cyc)
        cover_rows.append(cover_row)

        # ── ROI episodes (ROIs read from the jsons, same order as the sim) ─
        src = _s(data.get("env_source", ""))
        cfg_dir = os.path.dirname(src) if src.endswith(".json") else src
        seg_rois = segment_rois(cfg_dir)
        n = len(seg_rois)
        # sim keeps the last json when change steps outrun the files -> skip no-op changes
        real   = [i for i in range(len(cfg_starts)) if i == 0 or min(i, n - 1) != min(i - 1, n - 1)]
        bounds = [int(cfg_starts[i]) for i in real] + [T_total]

        open_ep, episodes = {}, []
        for k, i in enumerate(real):
            t0, rois = bounds[k], seg_rois[min(i, n - 1)]
            for name in list(open_ep):
                if rois.get(name) != open_ep[name][0]:              # moved or gone -> close
                    cells, ts = open_ep.pop(name)
                    episodes.append((name, cells, ts, t0))
            for name, cells in rois.items():
                if name not in open_ep:                             # new or just moved -> open
                    open_ep[name] = (cells, t0)
        episodes += [(name, cells, ts, T_total) for name, (cells, ts) in open_ep.items()]

        for name, cells, ts, te in episodes:
            hit = [any((int(r), int(c)) in cells for r, c in step) for step in traj_rc[ts:te]]
            hit_rows.append(dict(filename=run_dir.name, comm_radius=comm_radius, roi=name,
                                 t_start=ts, dur=te - ts,
                                 t_hit=float(np.argmax(hit)) if any(hit) else np.nan))

    all_cfg_keys = sorted(all_cfg_keys, key=lambda k: int(re.search(r"cfg(\d+)", k).group(1)))
    # print(f"  {len(cover_rows)} runs, {len(hit_rows)} ROI episodes")
    return cover_rows, hit_rows, all_cfg_keys





def write_cover_csv(cover_rows, all_cfg_keys, output_path):
    """
    Wide format: one row per run.
    Columns: filename, n_agents, comm_radius, comm_delay,
             cfg0_t0, cfg1_t500, ...
    """
    if not cover_rows:
        print("[SKIP] No cover rows to write.")
        return

    # fieldnames = ["filename", "n_agents", "comm_radius", "comm_delay"] + all_cfg_keys
    fieldnames = ["filename", "n_agents", "comm_radius", "comm_delay",
                  "cycle_mean", "n_cycles"] + all_cfg_keys
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in cover_rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    print(f"Cover CSV saved → {output_path}  ({len(cover_rows)} rows)")


if __name__ == "__main__":
    cover_rows, hit_rows, all_cfg_keys = collect_all(BATCH_DIR)

    # full-map cover cycles
    write_cover_csv(cover_rows, all_cfg_keys, COVER_OUTPUT)
    c = pd.read_csv(COVER_OUTPUT)
    print(c.groupby("comm_radius")[["cycle_mean", "n_cycles"]].agg(["mean", "std"]).round(1))

    # ROI episodes: time from appearance/move to first visit; miss = moved/gone before visited
    h = pd.DataFrame(hit_rows)
    h["kind"] = h.roi.str.extract(r"^([a-z]+)")
    h.to_csv("roi_episodes.csv", index=False)
    print(h.groupby(["comm_radius", "kind"]).agg(
        median=("t_hit", "median"), p90=("t_hit", lambda s: s.quantile(.9)),
        miss=("t_hit", lambda s: s.isna().mean()), n=("t_hit", "size"),
        dur_med=("dur", "median")).round(2))
    print(h[h.t_hit.isna()][["filename", "roi", "t_start", "dur"]].to_string())