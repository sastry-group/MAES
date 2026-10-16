# sim/plot_kinematics.py
"""Animation (and optional static path plot) for kinematics.npz. Matplotlib only.
RUN=results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05

# 1) fly the planner's waypoints -> kinematics.npz
python3 -m sim.kinematics --h5 $RUN/run.h5 --out $RUN/kinematics.npz

# 2) animate it: drones over empirical visitation | true importance
python3 -m viz.plot_kinematics --npz $RUN/kinematics.npz --h5 $RUN/run.h5 --speedup 300 --emp_window 2000

# add the team belief as a third panel
python3 -m viz.plot_kinematics --npz $RUN/kinematics.npz --h5 $RUN/run.h5 --panels traj true belief

Default: kin_anim.mp4 (.gif if no ffmpeg) -- top-down view, every drone from
the run, paths drawn as trails (dark greys = altitude). With --h5, the grid
under the drones is the empirical visitation rho_hat (run_io.StreamingEmpirical,
same window as plot_run), and side panels show the true importance rho* and,
optionally, the team-mean belief -- all on one lightened viridis and one norm
(plot_run colors, softer), so
"left looks like right" reads as ergodic. Both follow the env changes
(true_phi_map_time + true_phi_change_times), not the initial true_phi_map.
  --static   also kin_paths.png (full paths colored by altitude)
  --stats    also kin_stats.png (speed, altitude, separation, timing check)
  --agents / --window   optional subset of drones / time window [s]
  --fire     analytic fire overlay from the env JSON (off by default)
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, ListedColormap


def _light(name, white=0.45, lo=0.0, hi=1.0, n=256):
    """Colormap blended toward white: same hues as plot_run, softer."""
    c = plt.get_cmap(name)(np.linspace(lo, hi, n))
    c[:, :3] = white + (1.0 - white) * c[:, :3]
    return ListedColormap(c)


DIST_CMAP = _light("viridis", 0.45)                     # empirical / true / belief: plot_run convention, lighter
CMAP = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.5, 1.0, 256)))   # trails / dots: altitude, dark greys
FP_COLOR = "tab:red"                                   # sensing footprint
TITLES = {"true": r"true importance $\rho^*$", "belief": "team belief (normalized)"}


class RunMaps:
    """Planner-side maps from run.h5, looked up by planner step k (h5 row).
    True phi and belief stay event-keyed (per env change / per GP update),
    nothing is densified to (T, S), so 50x50 / 40k-step runs fit in memory."""

    def __init__(self, h5_path, emp_window=0, belief=False):
        import h5py
        from viz import run_io
        with h5py.File(h5_path, "r") as f:
            env = run_io.env_from_run({k: f.attrs[k] for k in ("env_source", "env_dyn") if k in f.attrs})
            self.H, self.W = env.H, env.W
            self.idx = np.full((self.H, self.W), -1, dtype=int)         # state index per cell, -1 = obstacle
            for i, (r, c) in env._index_state_map.items():
                self.idx[r, c] = i
            traj = f["traj_states"][()]
            self.T = traj.shape[0]
            self.emp = run_io.StreamingEmpirical(traj, env.nstates, window=emp_window)

            tp = f["true_phi_map_time"][()].astype(float)
            if "true_phi_change_times" in f:
                ct = f["true_phi_change_times"][()]
            else:                                                       # legacy: first row only
                tp, ct = tp[:1], np.zeros(1, dtype=int)
            self.true_t, self.true_rho = self._sorted(ct, tp, run_io)

            self.bel_t = self.bel_rho = None
            if belief and "belief_phi_map" in f:
                bt = f["belief_update_times"][()]
                ds = f["belief_phi_map"]
                rows = np.flatnonzero(bt >= 0)
                bel = np.stack([np.nanmean(ds[i], axis=0) for i in rows])   # team mean, one snapshot at a time
                self.bel_t, self.bel_rho = self._sorted(bt[rows], bel, run_io)

    @staticmethod
    def _sorted(times, vecs, run_io):
        ok = times >= 0
        o = np.argsort(times[ok], kind="stable")
        return times[ok][o], run_io.normalize_rows(vecs[ok][o])

    @staticmethod
    def _at(times, rows, k):
        return rows[max(np.searchsorted(times, k, side="right") - 1, 0)]

    def grid(self, vec):
        return np.where(self.idx >= 0, vec[np.maximum(self.idx, 0)], np.nan)

    def empirical(self, k):
        return self.grid(self.emp.at(k))

    def true(self, k):
        return self.grid(self._at(self.true_t, self.true_rho, k))

    def belief(self, k):
        return self.grid(self._at(self.bel_t, self.bel_rho, k))

    def cfg(self, k):
        return max(int(np.searchsorted(self.true_t, k, side="right")) - 1, 0)

    def vmax(self):
        return max(float(self.true_rho.max()), 1e-12)


def _load(npz_path, h5_path=None, emp_window=0, belief=False):
    d = dict(np.load(npz_path))
    maps = RunMaps(h5_path, emp_window, belief) if h5_path else None
    return d, float(d["cfg_cell_m"]), maps


def _rows(t, d, maps):
    """Kinematic time [s] -> planner step (h5 row). During step k the drones
    fly to centers[k+1], which is h5 row kin_step0 + k + 1."""
    K = len(d["centers"]) - 1
    k = int(d.get("kin_step0", 0)) + np.minimum((t // float(d["cfg_dt_s"])).astype(int) + 1, K)
    return np.minimum(k, maps.T - 1) if maps is not None else k


def _select(d, agents, window):
    t = d["t"]
    i0, i1 = 0, len(t)
    if window is not None:
        i0 = np.searchsorted(t, window[0]); i1 = np.searchsorted(t, window[1], side="right")
    ag = np.arange(d["p"].shape[1]) if agents is None else np.asarray(agents)
    return t[i0:i1], d["p"][i0:i1][:, ag], d["v"][i0:i1][:, ag], ag


def _shape(d, cell, maps):
    if maps is not None:
        return maps.H, maps.W
    return (int(round(d["centers"][..., 1].max() / cell + 0.5)),
            int(round(d["centers"][..., 0].max() / cell + 0.5)))


def _axes(ax, cell, H, W, km):
    s = 1e-3 if km else 1.0
    if max(H, W) <= 30:
        for i in range(H + 1):
            ax.axhline(i * cell * s, lw=0.3, color="0.75")
        for j in range(W + 1):
            ax.axvline(j * cell * s, lw=0.3, color="0.75")
    ax.set_xlim(0, W * cell * s); ax.set_ylim(H * cell * s, 0); ax.set_aspect("equal")   # row 0 at top
    u = "km" if km else "m"
    ax.set_xlabel(f"x [{u}]"); ax.set_ylabel(f"y [{u}]")
    return s


def _image(ax, grid, cell, s, cmap, norm=None, alpha=1.0):
    H, W = grid.shape
    return ax.imshow(grid, origin="upper", cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest",
                     extent=[0, W * cell * s, H * cell * s, 0], zorder=0)


def _znorm(p):
    z = p[..., 2]
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1.0:
        lo, hi = lo - 0.5, hi + 0.5
    return Normalize(lo, hi)


def _segments(xy):
    return np.stack([xy[:-1], xy[1:]], axis=1)

def _footprints(xy, fp):
    """(N, 2) centers -> (N, 4, 2) axis-aligned camera-footprint rectangles, fp = (w, h)."""
    hw, hh = 0.5 * fp[0], 0.5 * fp[1]
    return xy[:, None, :] + np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])


def animate(d, cell, maps, agents, window, out, km, panels=("traj", "true"), fps=20, speedup=30.0,
            trail_s=180.0, fp=None, fire=None, bg_alpha=1.0):
    t, p, _, ag = _select(d, agents, window)
    H, W = _shape(d, cell, maps)
    rows = _rows(t, d, maps)
    side = [q for q in panels if q != "traj"] if maps is not None else []
    if "belief" in side and maps.bel_rho is None:
        print("no belief_phi_map in h5 -> belief panel skipped"); side.remove("belief")
    fig, axs = plt.subplots(1, 1 + len(side), figsize=(6.6 + 5.0 * len(side), 6), squeeze=False,
                            layout="constrained")
    axs = axs[0]; ax = axs[0]
    s = _axes(ax, cell, H, W, km)
    norm = _znorm(p)
    dt_log = t[1] - t[0]
    stride = max(1, int(round(speedup / (fps * dt_log))))
    trail = max(2, int(trail_s / dt_log))
    many = len(ag) > 30

    emp_im, side_im = None, {}
    if maps is not None:                                      # empirical rho_hat under the drones
        dnorm = Normalize(0.0, maps.vmax())
        emp_im = _image(ax, maps.empirical(rows[0]), cell, s, DIST_CMAP, dnorm, bg_alpha)
        ax.set_title(r"UAVs over empirical visitation $\hat\rho$", fontsize=10)
        for a_, q in zip(axs[1:], side):
            _axes(a_, cell, H, W, km); a_.set_ylabel("")
            side_im[q] = _image(a_, getattr(maps, q)(rows[0]), cell, s, DIST_CMAP, dnorm)
            a_.set_title(TITLES[q], fontsize=10)
        fig.colorbar(emp_im if not side_im else side_im[side[-1]], ax=axs[-1], fraction=0.046, pad=0.04,
                     label="share of visits / importance")

    lc = LineCollection([], cmap=CMAP, norm=norm, linewidths=0.4 if many else 1.0, alpha=0.8)
    ax.add_collection(lc)
    fire_im = None
    if fire is not None:                                      # fire = (t_arr, sub_m, residence_s)
        from viz.fire_spread import _state, CMAP as FIRE_CMAP
        t_arr, sub_m, res = fire
        fire_im = ax.imshow(_state(t_arr, t[0], res), cmap=FIRE_CMAP, vmin=0, vmax=2, origin="upper",
                            extent=[0, t_arr.shape[1] * sub_m * s, t_arr.shape[0] * sub_m * s, 0],
                            interpolation="nearest", alpha=0.8, zorder=0.5)
    fps_pc = None
    if fp is not None:                                        # sensing footprint, drawn under trails
        fps_pc = PolyCollection(_footprints(p[0, :, :2], fp) * s, facecolor=FP_COLOR,
                                edgecolor="none", alpha=0.06 if many else 0.15, zorder=1)
        ax.add_collection(fps_pc)
    dots = ax.scatter(p[0, :, 0] * s, p[0, :, 1] * s, c=p[0, :, 2], cmap=CMAP, norm=norm,
                      s=6 if many else 18, edgecolors="k", linewidths=0.3, zorder=3)
    fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.04, label="altitude [m]")
    title = fig.suptitle("")

    def upd(i):
        j0 = max(0, i - trail)
        if i > j0:
            seg = np.concatenate([_segments(p[j0:i + 1, m, :2] * s) for m in range(len(ag))])
            zc = np.concatenate([0.5 * (p[j0:i, m, 2] + p[j0 + 1:i + 1, m, 2]) for m in range(len(ag))])
            lc.set_segments(seg); lc.set_array(zc)
        dots.set_offsets(p[i, :, :2] * s); dots.set_array(p[i, :, 2])
        if fps_pc is not None:
            fps_pc.set_verts(_footprints(p[i, :, :2], fp) * s)
        if fire_im is not None:
            fire_im.set_data(_state(fire[0], t[i], fire[2]))
        txt = f"{len(ag)} UAVs   t = {t[i]/60:5.1f} min   ({speedup:.0f}x real time)"
        if maps is not None:
            k = int(rows[i])
            emp_im.set_data(maps.empirical(k))
            for q, im in side_im.items():
                im.set_data(getattr(maps, q)(k))
            txt += f"   step {k}   env {maps.cfg(k)}"
        title.set_text(txt)
        return [lc, dots, title]

    anim = animation.FuncAnimation(fig, upd, frames=range(0, len(t), stride), blit=False)
    if animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps), dpi=150)
    else:
        out = os.path.splitext(out)[0] + ".gif"
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out


def plot_paths(d, cell, maps, agents, window, out, km):
    t, p, _, ag = _select(d, agents, window)
    H, W = _shape(d, cell, maps)
    fig, ax = plt.subplots(figsize=(6.6, 6))
    s = _axes(ax, cell, H, W, km)
    if maps is not None:                                      # true importance at the window start
        _image(ax, maps.true(int(_rows(t[:1], d, maps)[0])), cell, s, "Greys", alpha=0.45)
    norm = _znorm(p)
    seg = np.concatenate([_segments(p[:, m, :2] * s) for m in range(len(ag))])
    zc = np.concatenate([0.5 * (p[:-1, m, 2] + p[1:, m, 2]) for m in range(len(ag))])
    lc = LineCollection(seg, cmap=CMAP, norm=norm, linewidths=0.8)
    lc.set_array(zc); ax.add_collection(lc)
    ax.scatter(p[-1, :, 0] * s, p[-1, :, 1] * s, c=p[-1, :, 2], cmap=CMAP, norm=norm,
               s=18, edgecolors="k", linewidths=0.3, zorder=3)
    fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="altitude [m]")
    ax.set_title(f"{len(ag)} UAV paths, t = {t[0]/60:.0f}-{t[-1]/60:.0f} min")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_stats(d, agents, window, out):
    t, p, v, ag = _select(d, agents, window)
    dt_s = float(d["cfg_dt_s"])
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.5))
    tm = t / 60.0
    axs[0, 0].plot(tm, np.linalg.norm(v[..., :2], axis=-1), lw=0.5)
    axs[0, 0].axhline(float(d["cfg_v_max"]), ls="--", c="k", lw=0.8); axs[0, 0].set_ylabel("horiz. speed [m/s]")
    axs[0, 1].plot(tm, p[..., 2], lw=0.5); axs[0, 1].set_ylabel("altitude [m]")
    k = np.arange(len(d["min_sep"])) * dt_s
    kw = (k >= t[0]) & (k <= t[-1])
    axs[1, 0].plot(k[kw] / 60.0, d["min_sep"][kw], lw=0.8)
    axs[1, 0].axhline(float(d["cfg_d_safe"]), ls="--", c="r", lw=0.8)
    axs[1, 0].set_ylabel("min separation [m]")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_xlabel("time [min]")
    sl = d["slack"].ravel()
    axs[1, 1].hist(sl, bins=40); axs[1, 1].axvline(0, c="r", lw=0.8)
    axs[1, 1].set_xlabel("spare time per planner step [s]")
    axs[1, 1].set_title(f"late: {(sl < 0).mean()*100:.2f}% of steps", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--h5", default=None, help="run.h5: empirical background + true/belief panels")
    ap.add_argument("--panels", nargs="+", default=["traj", "true"], choices=["traj", "true", "belief"],
                    help="side panels next to the drones (needs --h5)")
    ap.add_argument("--emp_window", type=int, default=0,
                    help="steps counted in rho_hat; 0 = full history (same as plot_run)")
    ap.add_argument("--bg_alpha", type=float, default=1.0, help="opacity of rho_hat under the drones (1 = same colors as the side panels)")
    ap.add_argument("--agents", type=int, nargs="+", default=None, help="default: all drones in the run")
    ap.add_argument("--window", type=float, nargs=2, default=None, help="seconds [t0 t1]; default: whole run")
    ap.add_argument("--speedup", type=float, default=30.0)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--km", action="store_true")
    ap.add_argument("--footprint", type=float, nargs=2, default=[200.0, 200.0],
                    help="camera footprint w h [m] at cruise altitude; 0 0 to hide")
    ap.add_argument("--fire", action="store_true", help="analytic fire overlay (env JSON from the h5 unless --json)")
    ap.add_argument("--json", default=None, help="env JSON; enables the fire overlay")
    ap.add_argument("--fire_rois", nargs="+", default=None, help="ROI names that spread, e.g. roi2 roi3")
    ap.add_argument("--fire_rate", default="slow", help="slow | medium | fast")
    ap.add_argument("--sub_m", type=float, default=10.0)
    ap.add_argument("--residence_s", type=float, default=1e9)
    a = ap.parse_args()
    d, cell, maps = _load(a.npz, a.h5, a.emp_window, belief="belief" in a.panels)
    od = os.path.dirname(os.path.abspath(a.npz))
    fire = None
    json_path = a.json
    if json_path is None and a.h5 and a.fire:                 # planner saved the env path in the h5
        import h5py, glob
        with h5py.File(a.h5, "r") as f:
            src = f.attrs.get("env_source", "")
        src = src.decode() if isinstance(src, bytes) else str(src)
        json_path = sorted(glob.glob(os.path.join(src, "*.json")))[0] if os.path.isdir(src) else src
    if json_path and os.path.isfile(json_path):
        import json
        from viz.fire_spread import load_fires, arrival_times
        with open(json_path) as f:
            names = a.fire_rois or [r["name"] for r in json.load(f).get("rois", [])]   # default: all ROIs
        H, W, fires, _, _ = load_fires(json_path, fire_names=names, rate=a.fire_rate)
        if fires:
            fire = (arrival_times(H, W, fires, cell, a.sub_m)[0], a.sub_m, a.residence_s)
        print("fire json:", json_path, "| fire rois:", [f["name"] for f in fires] if fire else None)
    print("wrote", animate(d, cell, maps, a.agents, a.window, os.path.join(od, "kin_anim.mp4"), a.km,
                           panels=a.panels, speedup=a.speedup, bg_alpha=a.bg_alpha,
                           fp=a.footprint if min(a.footprint) > 0 else None, fire=fire))
    if a.static:
        plot_paths(d, cell, maps, a.agents, a.window, os.path.join(od, "kin_paths.png"), a.km)
    if a.stats:
        plot_stats(d, a.agents, a.window, os.path.join(od, "kin_stats.png"))


if __name__ == "__main__":
    main()