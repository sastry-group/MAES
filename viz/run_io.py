# viz/run_io.py
"""
Processing layer: load a run folder and return dense, aligned arrays.
No plotting here. plot_run.py, animate_run.py and check_noise.py all use this,
so densify / true-phi / belief alignment lives in exactly one place.
"""
import os
import glob
import numpy as np
import h5py

from env.grid import make_env_from_json


# ── loading ──────────────────────────────────────────────────────────────────
def load_h5(path):
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
        for key, val in f.attrs.items():
            data[key] = val
    return data


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    z.close()
    return data


def load_run(folder):
    """run.h5 (current) or legacy run_*.npz. Returns (data, path)."""
    h5 = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if h5:
        return load_h5(h5[0]), h5[0]
    npz = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if npz:
        return load_npz(npz[0]), npz[0]
    raise FileNotFoundError(f"No .h5 or .npz file found in {folder}")


def _str(v):
    v = v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def planner_name(data):
    return _str(data.get("planner", data.get("method", "ergodic")))


def is_event_driven(data):
    """mac_dt / doubling trick: belief changes at irregular refit steps."""
    return planner_name(data) == "mac_dt" or _str(data.get("update_rule", "fixed")) == "doubling"


# ── alignment ────────────────────────────────────────────────────────────────
def densify(snaps, times, T):
    """Event-keyed snapshots -> dense (T, ...). Row k is active from times[k];
    unused rows have times == -1."""
    snaps, times = np.asarray(snaps), np.asarray(times)
    valid = times >= 0
    snaps, times = snaps[valid], times[valid]
    order = np.argsort(times, kind="stable")
    snaps, times = snaps[order], times[order]
    rows = np.clip(np.searchsorted(times, np.arange(T), side="right") - 1, 0, len(snaps) - 1)
    return snaps[rows]


def true_phi_dense(data, T):
    """(T, nstates). Uses true_phi_change_times when present (static AND dynamic)."""
    tp = np.asarray(data["true_phi_map_time"], dtype=float)
    if "true_phi_change_times" in data:
        return densify(tp, data["true_phi_change_times"], T)
    return tp if tp.shape[0] == T else np.repeat(tp[:1], T, axis=0)


def belief_dense(data, key, T):
    """(T, N, nstates) for belief_phi_map / belief_mu_map / belief_sigma_map.
    ergodic: sparse snapshots + belief_update_times -> densify.
    mac_dt:  already dense per step -> returned as is.
    Returns None if the key is absent (older runs have no mu/sigma)."""
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=float)
    if "belief_update_times" in data:
        return densify(arr, data["belief_update_times"], T)
    if arr.ndim == 2:                       # legacy (T, nstates)
        arr = arr[:, None, :]
    return arr


def team_mean(arr_TNS):
    """(T, N, nstates) -> (T, nstates), ignoring dead-agent NaNs."""
    return np.nanmean(arr_TNS, axis=1)


def normalize_rows(M, eps=1e-12):
    return M / np.maximum(M.sum(axis=-1, keepdims=True), eps)


def mask_dead_agents(traj_states, active_mask):
    masked = np.array(traj_states, copy=True)
    if active_mask is not None:
        masked[~np.asarray(active_mask, dtype=bool)] = -1
    return masked


# ── environment ──────────────────────────────────────────────────────────────
def env_from_run(data, fallback="configs/simple5x5_static.json"):
    src = _str(data.get("env_source", ""))
    if _str(data.get("env_dyn", "static")) == "dynamic":
        env = make_env_from_json(sorted(glob.glob(os.path.join(src, "*.json")))[0])
    elif os.path.isfile(src):
        env = make_env_from_json(src)
    else:
        print(f"[run_io] env_source '{src}' not found, using {fallback}")
        env = make_env_from_json(fallback)
    env._indexing()
    env._precompute_graph()
    return env


def cfg_starts(data, T):
    """Segment start times. Uses the saved env_change_steps (not steps_per_cfg,
    which is never written to h5)."""
    if _str(data.get("env_dyn", "static")) != "dynamic":
        return [0]
    ch = [int(t) for t in np.atleast_1d(data.get("env_change_steps", [])) if 0 < int(t) < T]
    return [0] + sorted(ch)


# ── empirical visitation ─────────────────────────────────────────────────────
# One definition, one knob. `window` is the number of past steps that count
# toward rho_hat: window=None (or 0) means the full history, i.e. the same
# thing as metrics.empirical_full. Everything downstream -- frames, conv, kl,
# animation -- goes through this, so a run's rho_hat is the same everywhere.

class StreamingEmpirical:
    """
    Sliding-window empirical distribution, evaluated at increasing k in O(N)
    per step instead of O(window*N). Dead agents (state -1) are skipped.

    Callable with the metrics.empirical_fn signature, so it can be handed
    straight to convergence_errors_from_run / kl_series_emp_vs_true:

        emp = StreamingEmpirical(traj_states, nstates, window=4000)
        rho_hat_t = emp(traj_states=traj, env=env, k=t)
    """

    def __init__(self, traj_states, nstates, window=None):
        self.traj = np.asarray(traj_states)
        self.nstates = int(nstates)
        self.window = None if not window else int(window)
        self._reset()

    def _reset(self):
        self.counts = np.zeros(self.nstates, dtype=float)
        self.last_k = -1

    def at(self, k):
        k = int(k)
        if k < self.last_k:                      # went backwards: rebuild
            self._reset()
        for t in range(self.last_k + 1, k + 1):
            row = self.traj[t]
            np.add.at(self.counts, row[row >= 0], 1.0)
            if self.window is not None and t - self.window >= 0:
                old = self.traj[t - self.window]
                np.add.at(self.counts, old[old >= 0], -1.0)
        self.last_k = k
        s = self.counts.sum()
        return self.counts / s if s > 0 else np.full(self.nstates, 1.0 / self.nstates)

    def __call__(self, traj_states=None, env=None, k=0, **kwargs):
        return self.at(k)


def empirical_series(traj_states, nstates, frames, window=None):
    """(len(frames), nstates) at the requested frames only. frames ascending."""
    emp = StreamingEmpirical(traj_states, nstates, window=window)
    return np.stack([emp.at(int(k)) for k in np.asarray(frames)])


def update_steps(data, T, kind="p"):
    """
    Steps at which P (or the GP) was rebuilt.
    Uses the saved dataset if the run has one, else the fixed cadence attr,
    else the belief snapshot times.
    """
    key = f"{kind}_update_steps"
    if key in data:
        st = np.asarray(data[key], dtype=int)
        return sorted(set(int(t) for t in st if 0 <= t < T))
    cad = data.get("p_update" if kind == "p" else "gp_update", None)
    try:
        cad = int(cad)
    except (TypeError, ValueError):
        cad = -1
    if cad > 0:
        return list(range(cad, T, cad))
    if "belief_update_times" in data:
        st = np.asarray(data["belief_update_times"], dtype=int)
        return sorted(set(int(t) for t in st if 0 <= t < T))
    return []


def frame_indices(data, T, stride=None):
    """Snapshot frames: refit steps for mac_dt/doubling, else every stride
    (default min(gp_update, p_update))."""
    if is_event_driven(data):
        refit = np.asarray(data.get("gp_update_steps", []), dtype=int)
        return sorted(set(refit.tolist()) | {0, T - 1})
    if stride is None:
        stride = min(int(data.get("gp_update", 100)), int(data.get("p_update", 100)))
    return list(range(0, T, max(int(stride), 1)))