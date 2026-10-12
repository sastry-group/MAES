import json
import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from env.grid import make_env_from_json
from viz.plotting import plot_grid




OUT_DIR = Path("generated_maps")
OUT_DIR.mkdir(exist_ok=True)

RNG = random.Random(1)



GRID_SIZES = [
    # (5, 5),
    (10, 10),
    # (20, 20),
]

MAPS_PER_GRID = 1
NEIGHBORS = "4"

# one JSON per snapshot; the simulator decides how many steps each lasts.
# Generate more than you need: a shorter horizon just uses the first ones.
N_SNAPSHOTS = {
    # (5, 5): 15,
    (10, 10): 20,
    # (20, 20): 21,
}

OBSTACLE_COUNT = {      # ~2-4% of cells
    (5, 5): 1,
    (10, 10): 5,
    (20, 20): 10,
}

HUMAN_WEIGHT = 3.0

# Value of a burning cell as the fire fills it, one entry per snapshot.
# A new cell starts at FIRE_RAMP[0] and reaches FIRE_RAMP[-1]; only then does the
# fire advance to the next cell -> one new cell every len(FIRE_RAMP) snapshots.
# All entries > HUMAN_WEIGHT, so fire is always the highest ROI.
# [6.0] = old all-or-nothing behaviour.
FIRE_RAMP = [5.0, 6.0]

# Fire front: moves along a main heading, one cell at a time, with side randomness.
P_TURN = 0.3     # chance the head steps sideways instead of straight ahead
P_FLANK = 0.3    # chance a side cell next to the new head also ignites (burns in place, does not spread)

# Humans: presence, not motion. They change on their own, slower clock:
# every HUMAN_EVERY snapshots (default = once per new fire cell), new humans
# ~ Poisson(HUMAN_RATE) appear and each stays HUMAN_DWELL updates.
# Mean number present ~ HUMAN_RATE * HUMAN_DWELL.
HUMAN_EVERY = len(FIRE_RAMP)
HUMAN_RATE = {
    # (5, 5): 0.5,      # ~1 present
    (10, 10): 0.5,    # ~1 present
    # (20, 20): 1.0,    # ~2 present
}
HUMAN_DWELL = 10       # updates (= 2 fire cells of time)
HUMAN_PLACEMENT = "uniform"   # "near_fire" or "uniform"
NEAR_FIRE_DIST = 10              # cells (Manhattan) from any burning cell

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]



def cell_rect(r, c):
    # half-open [r0, c0, r1, c1], same as env_000.json
    return [r, c, r + 1, c + 1]


def generate_obstacles(width, height, n_cells, blocked):
    obstacles = set()
    while len(obstacles) < n_cells:
        cell = (RNG.randint(0, height - 1), RNG.randint(0, width - 1))
        if cell not in blocked:
            obstacles.add(cell)
    return obstacles


def random_free_cell(width, height, occupied):
    while True:
        cell = (RNG.randint(0, height - 1), RNG.randint(0, width - 1))
        if cell not in occupied:
            return cell


def _free(cell, width, height, obstacles, fire):
    r, c = cell
    return 0 <= r < height and 0 <= c < width and cell not in obstacles and cell not in fire


def spread_fire(fire, head, direction, width, height, obstacles):
    """fire: dict (r, c) -> index into FIRE_RAMP. Every burning cell steps up the
    ramp. Only the head spreads, once fully burning, to the next cell along the
    main heading `direction`; with prob P_TURN it steps sideways instead, and with
    prob P_FLANK a side cell of the new head also ignites. If the way ahead is
    blocked, the heading turns to a free side (back only as a last resort)."""
    top = len(FIRE_RAMP) - 1
    new_fire = {cell: min(top, i + 1) for cell, i in fire.items()}

    if fire[head] < top:
        return new_fire, head, direction

    dr, dc = direction
    sides = RNG.sample([(dc, dr), (-dc, -dr)], 2)          # the two perpendicular dirs
    order = sides + [direction] if RNG.random() < P_TURN else [direction] + sides
    order.append((-dr, -dc))                                # backwards: last resort

    for d in order:
        nb = (head[0] + d[0], head[1] + d[1])
        if not _free(nb, width, height, obstacles, new_fire):
            continue
        new_fire[nb] = 0
        ahead = (head[0] + dr, head[1] + dc)
        if d != direction and not _free(ahead, width, height, obstacles, new_fire):
            direction = d                                   # forced turn: new heading
        if RNG.random() < P_FLANK:
            fr, fc = direction
            flank = RNG.choice([(nb[0] + fc, nb[1] + fr), (nb[0] - fc, nb[1] - fr)])
            if _free(flank, width, height, obstacles, new_fire):
                new_fire[flank] = 0
        return new_fire, nb, direction
    return new_fire, head, direction   # boxed in: stops spreading


def poisson(lam):
    # Knuth; fine for small rates
    L, k, p = pow(2.718281828459045, -lam), 0, 1.0
    while True:
        p *= RNG.random()
        if p <= L:
            return k
        k += 1


def human_cells(width, height, obstacles, fire):
    blocked = obstacles | set(fire)
    free = [(r, c) for r in range(height) for c in range(width) if (r, c) not in blocked]
    if HUMAN_PLACEMENT == "near_fire":
        near = [(r, c) for (r, c) in free
                if min(abs(r - fr) + abs(c - fc) for (fr, fc) in fire) <= NEAR_FIRE_DIST]
        if near:
            return near
    return free


def update_humans(humans, s, rate, width, height, obstacles, fire, at_least_one=False):
    """humans: list of (r, c, t_off). Drop expired, add Poisson arrivals."""
    humans = [h for h in humans if h[2] > s]
    n_new = poisson(rate)
    if at_least_one:
        n_new = max(1, n_new)
    cand = [c for c in human_cells(width, height, obstacles, fire)
            if c not in {(h[0], h[1]) for h in humans}]
    for _ in range(min(n_new, len(cand))):
        r, c = cand.pop(RNG.randrange(len(cand)))
        humans.append((r, c, s + HUMAN_DWELL))
    return humans


def build_rois(fire, humans):
    rois = []
    for i, (r, c, _) in enumerate(humans):
        rois.append({
            "name": f"human{i+1}",
            "weight": HUMAN_WEIGHT,
            "shape": {"type": "rect", "rect": cell_rect(r, c)},
            "mode": "static"
        })
    for (r, c), i in sorted(fire.items()):
        rois.append({
            "name": f"fire_{r}_{c}",
            "weight": FIRE_RAMP[i],
            "shape": {"type": "rect", "rect": cell_rect(r, c)},
            "mode": "static"
        })
    return rois



def generate_all_maps():
    for (w, h) in GRID_SIZES:
        grid_key = (w, h)

        for k in range(1, MAPS_PER_GRID + 1):
            start = (h - 1, 0)
            obstacles = generate_obstacles(w, h, OBSTACLE_COUNT[grid_key], {start})

            fire_cell = random_free_cell(w, h, obstacles | {start})
            fire = {fire_cell: len(FIRE_RAMP) - 1}   # snapshot 0: one fully burning cell
            head, direction = fire_cell, RNG.choice(DIRS)
            humans = []

            grid_cfg = {
                "width": w,
                "height": h,
                "neighbors": NEIGHBORS,
                "rng_seed": RNG.randint(0, 10_000)
            }

            folder = OUT_DIR / f"dynamic_simple{w}x{h}_v{k}"
            folder.mkdir(exist_ok=True)
            plot_dir = folder / "plots"
            plot_dir.mkdir(exist_ok=True)

            for s in range(N_SNAPSHOTS[grid_key]):
                # 0: fire in one cell | 1: first humans appear
                # 2+: fire steps every snapshot, humans every HUMAN_EVERY snapshots
                if s >= 2:
                    fire, head, direction = spread_fire(fire, head, direction, w, h, obstacles)
                    humans = [hm for hm in humans if (hm[0], hm[1]) not in fire]  # overrun by fire
                if s == 1 or (s > 1 and (s - 1) % HUMAN_EVERY == 0):
                    humans = update_humans(humans, (s - 1) // HUMAN_EVERY,
                                           HUMAN_RATE[grid_key], w, h, obstacles, fire,
                                           at_least_one=(s == 1))
                map_data = {
                    "grid": grid_cfg,
                    "start": {"type": "rect", "rect": cell_rect(*start)},
                    "obstacles": {"cells": [list(o) for o in sorted(obstacles)]},
                    "rois": build_rois(fire, humans)
                }

                json_path = folder / f"env_{s:03d}.json"
                with open(json_path, "w") as f:
                    json.dump(map_data, f, indent=2)

                env = make_env_from_json(json_path)
                plot_grid(env, plot_dir / f"env_{s:03d}.png",
                          title=f"{w}x{h} v{k} | snapshot {s}",
                          roi_vmin=0.0, roi_vmax=max(FIRE_RAMP))

            print(f"Saved {folder} ({N_SNAPSHOTS[grid_key]} snapshots)")


if __name__ == "__main__":
    generate_all_maps()