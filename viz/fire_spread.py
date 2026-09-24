# sim/fire_spread.py
"""Sub-grid fire spread inside the planner's regions. Visual only; planner unchanged.

Each ROI with a "fire" block in the env JSON ignites at its rect center at t=0 and
the front advances radially at rate_mps (SWUIFT, Masoudvaziri 2021: slow 0.06,
medium 0.21, fast 0.49 m/s). ROIs without "fire" stay as they are (static).

    "rois": [ { "name": "roi2", "weight": 6.0,
                "shape": { "type": "rect", "rect": [0, 3, 1, 4] },
                "fire":  { "rate": "medium" } } ]          # or "rate_mps": 0.21

Optional per-fire keys: "ignite": [row, col] (region; default rect center),
"confine": true (default) keeps the fire inside its rect; false lets it spread
into neighboring regions.

Each sub-cell gets an arrival time t_arr = t_ign + dist / rate. At time t it is
    burning  if t_arr <= t < t_arr + residence_s
    burned   if t >= t_arr + residence_s.
Per region we save the burning and burned fractions (for the region value later).

python3 -m sim.fire_spread --json envs/simple5x5_static_v0.json --T 120 --out results/fire
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap


## chekc this rate slow was 0.06
RATES = {"slow": 0.015, "medium": 0.21, "fast": 0.49}   # m/s, SWUIFT Table 1


def load_fires(json_path, fire_names=None, rate=None, confine=True):
    """fire_names: ROI names that spread (overrides any "fire" block in the JSON);
    rate: 'slow'|'medium'|'fast' or m/s."""
    with open(json_path) as f:
        cfg = json.load(f)
    H, W = cfg["grid"]["height"], cfg["grid"]["width"]
    fires, static = [], []
    for r in cfg.get("rois", []):
        rect = r["shape"]["rect"]                        # [r0, c0, r1, c1], half-open
        fb = r.get("fire")
        if fire_names is not None:
            fb = ({"rate_mps": rate if not isinstance(rate, str) else RATES[rate], "confine": confine}
                  if r.get("name") in fire_names else None)
        if fb:
            rate = fb.get("rate_mps", RATES.get(fb.get("rate", "medium")))
            ign = fb.get("ignite", [(rect[0] + rect[2]) / 2.0 - 0.5, (rect[1] + rect[3]) / 2.0 - 0.5])
            fires.append(dict(name=r.get("name"), rect=rect, rate=float(rate), ign=ign,
                              t_ign=float(fb.get("t_ign_s", 0.0)), confine=fb.get("confine", True),
                              weight=r.get("weight", 1.0)))
        elif fb is None and r.get("mode") == "fire":
            fb = {"rate": r.get("rate", rate or "medium"), "confine": confine}
        else:
            static.append(dict(rect=rect, weight=r.get("weight", 1.0)))
    obst = [tuple(o) for o in cfg.get("obstacles", {}).get("cells", [])]
    return H, W, fires, static, obst


def arrival_times(H, W, fires, cell_m, sub_m):
    """(H*n, W*n) sub-cell arrival time [s]; inf = never burns."""
    n = int(round(cell_m / sub_m))
    yy, xx = np.meshgrid((np.arange(H * n) + 0.5) * sub_m, (np.arange(W * n) + 0.5) * sub_m, indexing="ij")
    t_arr = np.full(yy.shape, np.inf)
    for f in fires:
        cy, cx = (f["ign"][0] + 0.5) * cell_m, (f["ign"][1] + 0.5) * cell_m     # region -> meters, row 0 at top
        t = f["t_ign"] + np.hypot(yy - cy, xx - cx) / f["rate"]
        if f["confine"]:
            r0, c0, r1, c1 = f["rect"]
            mask = np.zeros_like(t, bool); mask[r0 * n:r1 * n, c0 * n:c1 * n] = True
            t = np.where(mask, t, np.inf)
        t_arr = np.minimum(t_arr, t)
    return t_arr, n


def region_fractions(t_arr, n, times, residence_s):
    """(T, H, W) burning and burned fractions per region."""
    Hn, Wn = t_arr.shape
    blk = t_arr.reshape(Hn // n, n, Wn // n, n)
    burning = np.stack([((blk <= t) & (t < blk + residence_s)).mean(axis=(1, 3)) for t in times])
    burned = np.stack([(t >= blk + residence_s).mean(axis=(1, 3)) for t in times])
    return burning, burned


def _state(t_arr, t, residence_s):
    """1 burning, 2 burned; nan (transparent) where not burning yet or never."""
    s = np.where(t_arr <= t, 1.0, np.nan)
    s[t >= t_arr + residence_s] = 2.0
    return s


def _base(ax, H, W, static, obst, fires, cell_m, n):
    ax.set_xlim(0, W * cell_m); ax.set_ylim(H * cell_m, 0); ax.set_aspect("equal")
    for i in range(H + 1):
        ax.axhline(i * cell_m, lw=0.4, color="0.75")
    for j in range(W + 1):
        ax.axvline(j * cell_m, lw=0.4, color="0.75")
    wmax = max([s["weight"] for s in static] + [f["weight"] for f in fires] + [1.0])
    for s in static:                                    # static ROIs: flat fill, alpha ~ weight
        r0, c0, r1, c1 = s["rect"]
        ax.add_patch(plt.Rectangle((c0 * cell_m, r0 * cell_m), (c1 - c0) * cell_m, (r1 - r0) * cell_m,
                                   color="#fdae61", alpha=0.3 + 0.5 * s["weight"] / wmax, lw=0, zorder=0))
    for r, c in obst:
        ax.plot((c + 0.5) * cell_m, (r + 0.5) * cell_m, "kx", ms=12, mew=2, zorder=4)
    for f in fires:                                     # fire ROIs: outline only, fire drawn on top
        r0, c0, r1, c1 = f["rect"]
        ax.add_patch(plt.Rectangle((c0 * cell_m, r0 * cell_m), (c1 - c0) * cell_m, (r1 - r0) * cell_m,
                                   fill=False, ec="#b30000", lw=1.2, zorder=3))
        ax.plot((f["ign"][1] + 0.5) * cell_m, (f["ign"][0] + 0.5) * cell_m, "*", c="gold",
                mec="k", ms=10, zorder=5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


CMAP = ListedColormap(["#ffffff", "#e31a1c", "#404040"])   # (unused), burning, burned


def snapshots(t_arr, n, H, W, static, obst, fires, cell_m, times, residence_s, out):
    fig, axs = plt.subplots(1, len(times), figsize=(3.2 * len(times), 3.4))
    for ax, t in zip(np.atleast_1d(axs), times):
        _base(ax, H, W, static, obst, fires, cell_m, n)
        ax.imshow(_state(t_arr, t, residence_s), cmap=CMAP, vmin=0, vmax=2, origin="upper",
                  extent=[0, W * cell_m, H * cell_m, 0], interpolation="nearest", zorder=1)
        ax.set_title(f"t = {t/60:.0f} min", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def animate(t_arr, n, H, W, static, obst, fires, cell_m, times, residence_s, out, fps=10):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _base(ax, H, W, static, obst, fires, cell_m, n)
    im = ax.imshow(_state(t_arr, times[0], residence_s), cmap=CMAP, vmin=0, vmax=2, origin="upper",
                   extent=[0, W * cell_m, H * cell_m, 0], interpolation="nearest", zorder=1)
    title = ax.set_title("")

    def upd(i):
        im.set_data(_state(t_arr, times[i], residence_s))
        title.set_text(f"t = {times[i]/60:5.1f} min")
        return im, title

    anim = animation.FuncAnimation(fig, upd, frames=len(times), blit=False)
    if animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps), dpi=120)
    else:
        out = os.path.splitext(out)[0] + ".gif"
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--T", type=int, default=120, help="planner steps to show")
    ap.add_argument("--dt_s", type=float, default=30.0)
    ap.add_argument("--cell_m", type=float, default=200.0)
    ap.add_argument("--sub_m", type=float, default=10.0, help="sub-cell side [m]")
    ap.add_argument("--residence_s", type=float, default=1800.0, help="time a sub-cell stays burning [s]")
    ap.add_argument("--out", default="fire_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    H, W, fires, static, obst = load_fires(a.json)
    t_arr, n = arrival_times(H, W, fires, a.cell_m, a.sub_m)
    times = np.arange(a.T + 1) * a.dt_s
    burning, burned = region_fractions(t_arr, n, times, a.residence_s)
    np.savez_compressed(os.path.join(a.out, "fire.npz"), t=times, burning=burning, burned=burned,
                        t_arr=t_arr, sub_m=a.sub_m, cell_m=a.cell_m, residence_s=a.residence_s)
    snap_t = np.linspace(0, times[-1], 5)
    snapshots(t_arr, n, H, W, static, obst, fires, a.cell_m, snap_t, a.residence_s,
              os.path.join(a.out, "fire_snapshots.png"))
    print("wrote", animate(t_arr, n, H, W, static, obst, fires, a.cell_m, times, a.residence_s,
                           os.path.join(a.out, "fire_anim.mp4")))
    for f in fires:
        print(f"{f['name']}: {f['rate']} m/s -> edge in {a.cell_m/2/f['rate']/a.dt_s:.0f} steps, "
              f"corner in {a.cell_m/np.sqrt(2)/f['rate']/a.dt_s:.0f} steps")


if __name__ == "__main__":
    main()