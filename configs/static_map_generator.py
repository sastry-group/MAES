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

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RNG = random.Random(42)



GRID_SIZES = [
    # (20, 20),
    # (50, 50),
    (100, 100),
]

MAPS_PER_GRID = 3
NEIGHBORS = "4"
ROI_WEIGHT_RANGE = (1.0, 8.0)


ROI_COUNT_RANGE = {
    # (20, 20): (10, 12),
    # (50, 50): (15, 25),
    (100, 100): (30, 45),
}

OBSTACLE_DENSITY = {
    # (20, 20): 0.01,
    # (50, 50): 0.01,
    (100, 100): 0.001,
}

ROI_MAX_SIZE = {
    # (20, 20): 4,
    # (50, 50): 6,
    (100, 100): 8,
}



def random_rect(width, height, max_size):
    w = RNG.randint(1, max_size)
    h = RNG.randint(1, max_size)

    x0 = RNG.randint(0, width - w)
    y0 = RNG.randint(0, height - h)

    return [x0, y0, x0 + w - 1, y0 + h - 1]


def generate_obstacles(width, height, density):
    n_cells = int(width * height * density)
    obstacles = set()

    while len(obstacles) < n_cells:
        obstacles.add((
            RNG.randint(0, width - 1),
            RNG.randint(0, height - 1)
        ))

    return [[x, y] for x, y in obstacles]

def rect_cells(rect):
    x0, y0, x1, y1 = rect
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def generate_rois(width, height, grid_key, obstacle_cells):
    n_rois = RNG.randint(*ROI_COUNT_RANGE[grid_key])
    max_size = ROI_MAX_SIZE[grid_key]

    rois = []
    occupied = set(obstacle_cells)  # block ROIs from obstacles

    for i in range(n_rois):
        for _ in range(50):  # retry limit
            rect = random_rect(width, height, max_size)
            cells = rect_cells(rect)

            if cells.isdisjoint(occupied):
                occupied |= cells
                rois.append({
                    "name": f"roi{i+1}",
                    "weight": round(RNG.uniform(*ROI_WEIGHT_RANGE), 2),
                    "shape": {
                        "type": "rect",
                        "rect": rect
                    },
                    "mode": "static"
                })
                break
        else:
            raise RuntimeError("Could not place ROI without obstacle overlap")

    return rois, n_rois



def generate_all_maps():
    for (w, h) in GRID_SIZES:
        grid_key = (w, h)

        for k in range(1, MAPS_PER_GRID + 1):
            obstacles = generate_obstacles(w, h, OBSTACLE_DENSITY[grid_key])
            obstacle_cells = {tuple(c) for c in obstacles}

            rois, n_rois = generate_rois(w, h, grid_key, obstacle_cells)

            map_data = {
                "grid": {
                    "width": w,
                    "height": h,
                    "neighbors": NEIGHBORS,
                    "rng_seed": RNG.randint(0, 10_000)
                },
                "start": {
                    "type": "rect",
                    "rect": random_rect(w, h, max_size=3)
                },
                "obstacles": {"cells": obstacles},
                "rois": rois
            }

            filename = f"simple{w}x{h}_static_v{k}.json"
            json_path = OUT_DIR / filename

            with open(json_path, "w") as f:
                json.dump(map_data, f, indent=2)

            print(f"Saved {filename} (ROIs={n_rois})")

            env = make_env_from_json(json_path)
            plot_path = PLOT_DIR / filename.replace(".json", ".png")

            plot_grid(
                env,
                plot_path,
                title=f"{w}x{h} | ROIs={n_rois}"
            )

            print(f"    ↳ plotted {plot_path}")


if __name__ == "__main__":
    generate_all_maps()
