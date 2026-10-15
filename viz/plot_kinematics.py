# sim/plot_kinematics.py
"""Animation (and optional static path plot) for kinematics.npz. Matplotlib only.
RUN=results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05

# 1) fly the planner's waypoints -> kinematics.npz
python3 -m sim.kinematics --h5 $RUN/run.h5 --out $RUN/kinematics.npz --altitudes 120 135 150

# 2) animate it
python3 -m viz.plot_kinematics --npz $RUN/kinematics.npz --h5 $RUN/run.h5 --speedup 300
results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05

python3 -m viz.plot_kinematics --npz results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05/kinematics.npz --h5 results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05/run.h5 --speedup 300

Default: kin_anim.mp4 (.gif if no ffmpeg) -- top-down view, every drone from
the run, paths drawn as trails and colored by altitude.
  --static   also kin_paths.png (full paths colored by altitude)
  --stats    also kin_stats.png (speed, altitude, separation, timing check)
  --agents / --window   optional subset of drones / time window [s]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from viz.fire_spread import load_fires, arrival_times, _state, CMAP as FIRE_CMAP
from matplotlib.colors import Normalize

CMAP = "viridis"


def _load(npz_path, h5_path=None):
    d = dict(np.load(npz_path))
    phi = None
    if h5_path:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if "true_phi_map" in f:
                phi = f["true_phi_map"][()]
    return d, float(d["cfg_cell_m"]), phi


def _select(d, agents, window):
    t = d["t"]
    i0, i1 = 0, len(t)
    if window is not None:
        i0 = np.searchsorted(t, window[0]); i1 = np.searchsorted(t, window[1], side="right")
    ag = np.arange(d["p"].shape[1]) if agents is None else np.asarray(agents)
    return t[i0:i1], d["p"][i0:i1][:, ag], d["v"][i0:i1][:, ag], ag


def _axes(d, cell, phi, km, figsize=(6.6, 6)):
    if phi is not None:
        H, W = phi.shape
    else:
        H = int(round(d["centers"][..., 1].max() / cell + 0.5))
        W = int(round(d["centers"][..., 0].max() / cell + 0.5))
    s = 1e-3 if km else 1.0
    fig, ax = plt.subplots(figsize=figsize)
    if phi is not None:
        ax.imshow(phi, origin="upper", cmap="Greys", alpha=0.45, extent=[0, W * cell * s, H * cell * s, 0])
    if max(H, W) <= 30:
        for i in range(H + 1):
            ax.axhline(i * cell * s, lw=0.3, color="0.75")
        for j in range(W + 1):
            ax.axvline(j * cell * s, lw=0.3, color="0.75")
    ax.set_xlim(0, W * cell * s); ax.set_ylim(H * cell * s, 0); ax.set_aspect("equal")   # row 0 at top
    u = "km" if km else "m"
    ax.set_xlabel(f"x [{u}]"); ax.set_ylabel(f"y [{u}]")
    return fig, ax, s


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


def animate(d, cell, phi, agents, window, out, km, fps=20, speedup=30.0, trail_s=180.0, 
            fp=None, fire=None):
    t, p, _, ag = _select(d, agents, window)
    fig, ax, s = _axes(d, cell, phi, km)
    norm = _znorm(p)
    dt_log = t[1] - t[0]
    stride = max(1, int(round(speedup / (fps * dt_log))))
    trail = max(2, int(trail_s / dt_log))
    many = len(ag) > 30
    lc = LineCollection([], cmap=CMAP, norm=norm, linewidths=0.4 if many else 1.0, alpha=0.8)
    ax.add_collection(lc)
    fire_im = None
    if fire is not None:                                      # fire = (t_arr, sub_m, residence_s)
        t_arr, sub_m, res = fire
        fire_im = ax.imshow(_state(t_arr, t[0], res), cmap=FIRE_CMAP, vmin=0, vmax=2, origin="upper",
                            extent=[0, t_arr.shape[1] * sub_m * s, t_arr.shape[0] * sub_m * s, 0],
                            interpolation="nearest", alpha=0.8, zorder=0.5)
    fps_pc = None
    if fp is not None:                                        # sensing footprint, drawn under trails
        fps_pc = PolyCollection(_footprints(p[0, :, :2], fp) * s, facecolor="tab:orange",
                                edgecolor="none", alpha=0.06 if many else 0.15, zorder=1)
        ax.add_collection(fps_pc)
    dots = ax.scatter(p[0, :, 0] * s, p[0, :, 1] * s, c=p[0, :, 2], cmap=CMAP, norm=norm,
                      s=6 if many else 18, edgecolors="k", linewidths=0.3, zorder=3)
    fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.04, label="altitude [m]")
    title = ax.set_title("")

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
        title.set_text(f"{len(ag)} UAVs   t = {t[i]/60:5.1f} min   ({speedup:.0f}x real time)")
        return lc, dots, title

    anim = animation.FuncAnimation(fig, upd, frames=range(0, len(t), stride), blit=False)
    if animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps), dpi=150)
    else:
        out = os.path.splitext(out)[0] + ".gif"
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out


def plot_paths(d, cell, phi, agents, window, out, km):
    t, p, _, ag = _select(d, agents, window)
    fig, ax, s = _axes(d, cell, phi, km)
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
    ap.add_argument("--h5", default=None, help="run.h5 for the true-phi background")
    ap.add_argument("--agents", type=int, nargs="+", default=None, help="default: all drones in the run")
    ap.add_argument("--window", type=float, nargs=2, default=None, help="seconds [t0 t1]; default: whole run")
    ap.add_argument("--speedup", type=float, default=30.0)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--km", action="store_true")
    ap.add_argument("--footprint", type=float, nargs=2, default=[200.0, 200.0],
                    help="camera footprint w h [m] at cruise altitude; 0 0 to hide")
    ap.add_argument("--json", default=None, help="env JSON; enables the fire overlay")
    ap.add_argument("--fire_rois", nargs="+", default=None, help="ROI names that spread, e.g. roi2 roi3")
    ap.add_argument("--fire_rate", default="slow", help="slow | medium | fast")
    ap.add_argument("--sub_m", type=float, default=10.0)
    ap.add_argument("--residence_s", type=float, default=1e9) 
    a = ap.parse_args()
    d, cell, phi = _load(a.npz, a.h5)
    od = os.path.dirname(os.path.abspath(a.npz))
    fire = None
    json_path = a.json
    if json_path is None and a.h5:                            # planner saved the env path in the h5
        import h5py, glob
        with h5py.File(a.h5, "r") as f:
            src = f.attrs.get("env_source", "")
        src = src.decode() if isinstance(src, bytes) else str(src)
        json_path = sorted(glob.glob(os.path.join(src, "*.json")))[0] if os.path.isdir(src) else src
    if json_path and os.path.isfile(json_path):
        import json
        with open(json_path) as f:
            names = a.fire_rois or [r["name"] for r in json.load(f).get("rois", [])]   # default: all ROIs
        H, W, fires, _, _ = load_fires(json_path, fire_names=names, rate=a.fire_rate)
        if fires:
            fire = (arrival_times(H, W, fires, cell, a.sub_m)[0], a.sub_m, a.residence_s)
    print("fire json:", json_path, "| fire rois:", [f["name"] for f in fires] if fire else None)
    print("wrote", animate(d, cell, phi, a.agents, a.window, os.path.join(od, "kin_anim.mp4"),
                           a.km, speedup=a.speedup, fp=a.footprint if min(a.footprint) > 0 else None, fire=fire))
    if a.static:
        plot_paths(d, cell, phi, a.agents, a.window, os.path.join(od, "kin_paths.png"), a.km)
    if a.stats:
        plot_stats(d, a.agents, a.window, os.path.join(od, "kin_stats.png"))


if __name__ == "__main__":
    main()