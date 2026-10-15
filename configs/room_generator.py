"""Rooms scenario: sources spread out, each enclosed in a walled room with
DOORS openings (Chinmay's test: is HEDAC's greedy climb hurt by enclosures?).

Writes env_000.json ... env_{N-1}.json in the usual cell/JSON format, plus plots/env_NNN.png
(plot_grid, same colors and scale as dynamic_map_generator.py). Walls are identical in every snapshot (HEDAC assumes fixed
obstacles); only which rooms hold a source changes. Timing stays in the
simulator (--env_change_steps), as for the other dynamic maps.

  python rooms_map_generator.py --n_snap 4     # dynamic, 3 changes, 1 door per room
  python rooms_map_generator.py --doors 2
  python rooms_map_generator.py --static       # every room active, 1 snapshot
"""
import argparse, json, random, sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from env.grid import make_env_from_json
from viz.plotting import plot_grid

H = W = 10
# interior rects [r0, c0, r1, c1) (exclusive end, same as "rect" in the JSON)
# and door candidates (wall cells to open), first DOORS of them are used
ROOMS = {
    "room1": dict(rect=(0, 0, 2, 2), doors=[(1, 2), (2, 0)]),   # top-left,     opens east / south
    "room2": dict(rect=(0, 8, 2, 10), doors=[(2, 8), (1, 7)]),  # top-right,    opens south / west
    "room3": dict(rect=(8, 8, 10, 10), doors=[(8, 7), (7, 9)]), # bottom-right, opens west / north
    "room4": dict(rect=(4, 2, 6, 4), doors=[(6, 3), (3, 2)]),   # left-middle,  opens south / north
}
FIRE_W, HUMAN_W = 6.0, 3.0


def ring(r0, c0, r1, c1):
    """Wall cells around an interior rect, clipped to the grid (corners included)."""
    cells = set()
    for r in range(r0 - 1, r1 + 1):
        for c in range(c0 - 1, c1 + 1):
            inside = r0 <= r < r1 and c0 <= c < c1
            if not inside and 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
    return cells


def walls(n_doors):
    obs = set()
    for spec in ROOMS.values():
        obs |= ring(*spec["rect"]) - set(spec["doors"][:n_doors])
    return obs


def connected(obs):
    free = [(r, c) for r in range(H) for c in range(W) if (r, c) not in obs]
    seen, q = {free[0]}, deque([free[0]])
    while q:
        r, c = q.popleft()
        for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= n[0] < H and 0 <= n[1] < W and n not in obs and n not in seen:
                seen.add(n); q.append(n)
    return len(seen) == len(free), free


def roi(name, kind, w):
    return {"name": f"{kind}_{name}", "weight": w,
            "shape": {"type": "rect", "rect": list(ROOMS[name]["rect"])}, "mode": "static"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doors", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n_snap", type=int, default=12)
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    obs = walls(a.doors)
    ok, free = connected(obs)
    assert ok, "walls disconnect the grid"
    interior = {(r, c) for s in ROOMS.values() for r in range(s["rect"][0], s["rect"][2])
                for c in range(s["rect"][1], s["rect"][3])}
    doors = {d for s in ROOMS.values() for d in s["doors"][:a.doors]}
    starts_ok = sorted(set(free) - interior - doors)  # valid starts: free, outside rooms, not a doorway

    tag = "static" if a.static else "dynamic"
    out = Path(a.out or f"generated_maps/{tag}_rooms{W}x{H}_d{a.doors}")
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    names = list(ROOMS)

    snaps = []
    perm = rng.sample(range(4), 4)
    for k in range(1 if a.static else a.n_snap):
        if a.static:
            rois = [roi(n, "fire", FIRE_W if i < 2 else HUMAN_W) for i, n in enumerate(names)]
        else:                                          # rotation: every change moves BOTH sources
            f = names[perm[k % 4]]                     # into rooms that were empty before it,
            h = names[perm[(k + 2) % 4]]               # and the fire visits all 4 rooms every 4 snapshots
            rois = [roi(f, "fire", FIRE_W), roi(h, "human", HUMAN_W)]
        cfg = {"grid": {"width": W, "height": H, "neighbors": "4", "rng_seed": 1},
               "start": {"type": "rect", "rect": [H - 1, 0, H, 1]},
               "obstacles": {"cells": [list(c) for c in sorted(obs)]},
               "rois": rois}
        json_path = out / f"env_{k:03d}.json"
        json_path.write_text(json.dumps(cfg, indent=2))
        snaps.append(rois)

        env = make_env_from_json(json_path.resolve())            # same plot as dynamic_map_generator
        plot_grid(env, plot_dir / f"env_{k:03d}.png",
                  title=f"{W}x{H} rooms d{a.doors} | snapshot {k}",
                  roi_vmin=0.0, roi_vmax=FIRE_W)

    print(f"{out}: {len(snaps)} snapshot(s), {len(obs)} wall cells, doors/room = {a.doors}")
    print("valid starts outside rooms (r, c):", starts_ok)
    for k, rois in enumerate(snaps):
        print(f"  env_{k:03d}:", ", ".join(f"{r['name']}={r['weight']:g}" for r in rois))


if __name__ == "__main__":
    main()