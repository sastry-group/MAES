
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from core.types import GridSpec
import os
import json
import glob

# ----------------------------
# Cell encodings
# ----------------------------
EMPTY = 0
OBSTACLE = 1
ROI = 2
START = 3



# def load_env_sequence(dir_path):
#     """
#     Given a directory, load all .json files and return a list of cfg dicts
#     sorted by filename.
#     Example: env_000.json, env_001.json, ...
#     """
#     if not os.path.isabs(dir_path):
#         base_dir = os.path.dirname(os.path.dirname(__file__))  # MAES/
#         dir_path = os.path.join(base_dir, dir_path)
#     abs_dir = os.path.abspath(dir_path)

#     if not os.path.isdir(abs_dir):
#         raise NotADirectoryError(f"Dynamic env config path is not a directory: {abs_dir}")

#     pattern = os.path.join(abs_dir, "*.json")
#     files = sorted(glob.glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No .json files found in dynamic env folder: {abs_dir}")

#     cfgs = []
#     for fpath in files:
#         with open(fpath, "r") as f:
#             cfg = json.load(f)
#         cfgs.append(cfg)
#     return cfgs


# might remove this later, as I will pass json files instead
def load_dynamic_config(path):
    """Returns: list of {'name','mode','rate'} for each ROI that has 'dynamic'."""
    if path is None:
        return []

    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # MAES/
        path = os.path.join(base_dir, path)
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"config file not found: {abs_path}")

    with open(abs_path, "r") as f:
        cfg = json.load(f)

    rois = cfg.get("rois", [])
    roi_dict = {}
    for r in rois:
        d = r.get("dynamic")
        if not d:
            continue
        roi_dict[r.get("name", "roi")] = {
            "mode": d.get("mode", "static"),
            "rate": int(d.get("rate", 0)),
        }
        roi_dict[r.get("name", "roi")]["shape"] = r.get("shape", {})
        roi_dict[r.get("name", "roi")]["weight"] = r.get("weight", 1.0)
    return roi_dict



# def load_env_sequence(dir_path):
#     """
#     Given a directory, load all .json files and return a list of cfg dicts
#     sorted by filename.
#     Example: env_000.json, env_001.json, ...
#     """
#     if not os.path.isabs(dir_path):
#         base_dir = os.path.dirname(os.path.dirname(__file__))  # MAES/
#         dir_path = os.path.join(base_dir, dir_path)
#     abs_dir = os.path.abspath(dir_path)

#     if not os.path.isdir(abs_dir):
#         raise NotADirectoryError(f"Dynamic env config path is not a directory: {abs_dir}")

#     pattern = os.path.join(abs_dir, "*.json")
#     files = sorted(glob.glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No .json files found in dynamic env folder: {abs_dir}")

#     cfgs = []
#     for fpath in files:
#         with open(fpath, "r") as f:
#             cfg = json.load(f)
#         cfgs.append(cfg)
#     return cfgs


def make_env_from_json(path):
    if not os.path.isabs(path):
        # otherwise, interpret relative to this script’s parent (MAES/)
        base_dir = os.path.dirname(os.path.dirname(__file__))  # moves up from experiments/ to MAES/
        path = os.path.join(base_dir, path)

    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    
    with open(abs_path, "r") as f:
        cfg = json.load(f)

    g = cfg["grid"]
    spec = GridSpec(width=g["width"], height=g["height"],
                    neighbors=g.get("neighbors", "4"),
                    rng_seed=g.get("rng_seed", 1))
    env = GridMC(spec)

    # ROI setup
    rois = cfg.get("rois", [])
    for r in rois:
        shp = r.get("shape", {})
        if shp.get("type") == "rect":
            x0, y0, x1, y1 = shp["rect"]
            env.set_roi_rect(x0, y0, x1, y1, weight=r.get("weight", 1.0))
            if not hasattr(env, "roi_cells"):
                    env.roi_cells = []   # list of dicts

            env.roi_cells.append({
                "name": r.get("name", "roi"),
                "rect": (x0, y0, x1, y1),
                "weight": r.get("weight", 1.0)
            })

    # Start
    s = cfg["start"]["rect"]
    env.set_start_rect(*s)

    # Obstacles
    obs = [tuple(o) for o in cfg["obstacles"]["cells"]]
    env.set_obstacles(obs)



    # noise higher around ROIs
    H, W = env.H, env.W
    if not hasattr(env, "noise_sigma"):
        env.noise_sigma = np.zeros((H, W), dtype=float)

    high_mask = np.zeros((H, W), dtype=bool)
    mid_mask  = np.zeros((H, W), dtype=bool)

    if rois:

        rois_sorted = sorted(rois, key=lambda r: r.get("weight", 0.0))

        mid = rois_sorted[0]
        shp_mid = mid.get("shape", {})
        if shp_mid.get("type") == "rect":
            mr0, mc0, mr1, mc1 = shp_mid["rect"]
            mid_mask[mr0:mr1, mc0:mc1] = True


        high = rois_sorted[-1]
        shp_high = high.get("shape", {})
        if shp_high.get("type") == "rect":
            hr0, hc0, hr1, hc1 = shp_high["rect"]
            high_mask[hr0:hr1, hc0:hc1] = True


    ring_mask = np.zeros((H, W), dtype=bool)
    if env.spec.neighbors == "8":
        deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]

    for r in range(H):
        for c in range(W):
            if high_mask[r, c]:
                for dr, dc in deltas:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        if env.grid[rr, cc] != OBSTACLE and not high_mask[rr, cc]:
                            ring_mask[rr, cc] = True

    sigma_low  = 0.005
    sigma_mid  = 0.015
    sigma_ring = 0.020
    sigma_high = 0.030

    env.noise_sigma[:] = sigma_low
    env.noise_sigma[mid_mask]  = sigma_mid
    env.noise_sigma[ring_mask] = np.maximum(env.noise_sigma[ring_mask], sigma_ring)
    env.noise_sigma[high_mask] = sigma_high

    return env


# def make_env(neighbors="4", seed=1):
#     spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
#     env = GridMC(spec)

#     env.set_roi_rect(0, 0, 1, 1, weight=3.0)   # MID ROI (1x1)
#     env.set_roi_rect(0, 3, 2, 4, weight=6.0)   # HIGH ROI (2x1: rows 0..1, col 3)
#     env.set_start_rect(3, 1, 4, 1)
#     env.set_obstacles([(2, 2)])


#     H, W = env.H, env.W
#     if not hasattr(env, "noise_sigma"):
#         env.noise_sigma = np.zeros((H, W), dtype=float)

#     # masks for ROIs
#     high_mask = np.zeros((H, W), dtype=bool)
#     high_mask[0:2, 3:4] = True   # matches the high-weight rect above

#     mid_mask = np.zeros((H, W), dtype=bool)
#     mid_mask[0:1, 0:1] = True    # matches the mid-weight rect above

#     # smoke around the high ROI
#     ring_mask = np.zeros((H, W), dtype=bool)
#     if neighbors == "8":
#         deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
#     else:
#         deltas = [(-1,0),(1,0),(0,-1),(0,1)]

#     for r in range(H):
#         for c in range(W):
#             if high_mask[r, c]:
#                 for dr, dc in deltas:
#                     rr, cc = r + dr, c + dc
#                     if 0 <= rr < H and 0 <= cc < W:
#                         if env.grid[rr, cc] != OBSTACLE and not high_mask[rr, cc]:
#                             ring_mask[rr, cc] = True

#     sigma_low  = 0.005
#     sigma_mid  = 0.015
#     sigma_ring = 0.020
#     sigma_high = 0.030

#     env.noise_sigma[:] = sigma_low
#     env.noise_sigma[mid_mask]  = sigma_mid
#     env.noise_sigma[ring_mask] = np.maximum(env.noise_sigma[ring_mask], sigma_ring)
#     env.noise_sigma[high_mask] = sigma_high

#     return env



# # this is a bandaid to update map quickly
# def update_env(neighbors="4", seed=1, time=0):
#     spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
#     env = GridMC(spec)

#     # --- base ROIs (same as make_env) ---
#     mid_rect_base  = (0, 0, 1, 1)   # r0, c0, r1, c1  (half-open)
#     high_rect_base = (0, 3, 2, 4)   # rows 0..1, col 3

#     # --- changed ROIs at/after t=100: move mid, expand high ---
#     mid_rect_t100  = (1, 1, 2, 2)   # move mid 1 step down-right
#     high_rect_t100 = (0, 3, 3, 5)   # expand high to rows 0..2, cols 3..4 (3x2)

#     if time >= 200:
#         mid_rect  = mid_rect_t100
#         high_rect = high_rect_t100
#     else:
#         mid_rect  = mid_rect_base
#         high_rect = high_rect_base

#     # Set ROIs
#     env.set_roi_rect(*mid_rect,  weight=3.0)  # MID ROI
#     env.set_roi_rect(*high_rect, weight=6.0)  # HIGH ROI

#     # start & obstacles (same as make_env)
#     env.set_start_rect(3, 1, 4, 1)
#     env.set_obstacles([(2, 2)])

#     # ---- noise model derived from ROIs ----
#     H, W = env.H, env.W
#     if not hasattr(env, "noise_sigma"):
#         env.noise_sigma = np.zeros((H, W), dtype=float)

#     # masks that match current ROI rectangles
#     high_mask = np.zeros((H, W), dtype=bool)
#     hr0, hc0, hr1, hc1 = high_rect
#     high_mask[hr0:hr1, hc0:hc1] = True

#     mid_mask = np.zeros((H, W), dtype=bool)
#     mr0, mc0, mr1, mc1 = mid_rect
#     mid_mask[mr0:mr1, mc0:mc1] = True

#     # ring around the high ROI (4- or 8-neighborhood)
#     ring_mask = np.zeros((H, W), dtype=bool)
#     if neighbors == "8":
#         deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
#     else:
#         deltas = [(-1,0),(1,0),(0,-1),(0,1)]

#     for r in range(H):
#         for c in range(W):
#             if high_mask[r, c]:
#                 for dr, dc in deltas:
#                     rr, cc = r + dr, c + dc
#                     if 0 <= rr < H and 0 <= cc < W:
#                         if env.grid[rr, cc] != OBSTACLE and not high_mask[rr, cc]:
#                             ring_mask[rr, cc] = True

#     # assign sigmas
#     sigma_low  = 0.005
#     sigma_mid  = 0.015
#     sigma_ring = 0.020
#     sigma_high = 0.030

#     env.noise_sigma[:] = sigma_low
#     env.noise_sigma[mid_mask]  = sigma_mid
#     env.noise_sigma[ring_mask] = np.maximum(env.noise_sigma[ring_mask], sigma_ring)
#     env.noise_sigma[high_mask] = sigma_high

#     return env


### To update dynamic maps ###
def _clip_rect(r0, c0, r1, c1, H, W):
    # Rects are half-open: [r0:r1, c0:c1]
    r0 = max(0, r0); c0 = max(0, c0)
    r1 = min(H, r1); c1 = min(W, c1)
    return r0, c0, r1, c1

def _rect_has_obstacle(env, r0, c0, r1, c1):
    if r0 >= r1 or c0 >= c1:
        return True  # empty/invalid rect -> treat as blocked
    return np.any(env.grid[r0:r1, c0:c1] == OBSTACLE)

def _shift_rect(r0, c0, r1, c1, dr, dc, H, W):
    nr0, nc0 = r0 + dr, c0 + dc
    nr1, nc1 = r1 + dr, c1 + dc
    nr0, nc0, nr1, nc1 = _clip_rect(nr0, nc0, nr1, nc1, H, W)
    return nr0, nc0, nr1, nc1

def _dilate_rect(r0, c0, r1, c1, pad, H, W):
    return _clip_rect(r0 - pad, c0 - pad, r1 + pad, c1 + pad, H, W)

def _paint_true_phi_from_rois(env, roi_specs):
    # Rebuild from scratch to avoid bookkeeping headaches
    env.true_phi[:] = 0.0
    for spec in roi_specs.values():
        weight = float(spec.get("weight", 0.0))
        shp = spec.get("shape", {})
        if shp.get("type") != "rect": 
            continue
        r0, c0, r1, c1 = map(int, shp.get("rect", [0, 0, 0, 0]))
        if r0 < r1 and c0 < c1:
            # add weight; obstacles will be zeroed below
            env.true_phi[r0:r1, c0:c1] += weight
    # enforce obstacles to 0
    env.true_phi[env.grid == OBSTACLE] = 0.0

def update_map(env, roi_names, roi_specs):
    """
    Mutates env.true_phi (and implicitly keeps ROI positions updated in roi_specs).
    - 'move': shift rect by 1 cell in a random valid direction (UDLR).
    - 'spread': dilate rect by 1 cell (clip to bounds). Obstacles are never painted.
    """
    H, W = env.H, env.W
    rng = getattr(env, "rng", np.random.default_rng())

    for name in roi_names:
        spec = roi_specs[name]
        mode = spec.get("mode", "")
        shp = spec.get("shape", {})
        if shp.get("type") != "rect":
            continue

        r0, c0, r1, c1 = map(int, shp.get("rect", [0, 0, 0, 0]))

        if mode == "move":
            # Try 4-connected moves
            candidates = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr0, nc0, nr1, nc1 = _shift_rect(r0, c0, r1, c1, dr, dc, H, W)
                # valid if area preserved and no obstacles in new rect
                if (nr1 - nr0 == r1 - r0) and (nc1 - nc0 == c1 - c0) and not _rect_has_obstacle(env, nr0, nc0, nr1, nc1):
                    candidates.append((nr0, nc0, nr1, nc1))
            if candidates:
                r0, c0, r1, c1 = candidates[rng.integers(len(candidates))]

            # update rect back into specs (stateful)
            spec["shape"]["rect"] = [int(r0), int(c0), int(r1), int(c1)]

        elif mode == "spread":
            print("Spreading ROI:", name)
            nr0, nc0, nr1, nc1 = _dilate_rect(r0, c0, r1, c1, pad=1, H=H, W=W)
            # Keep the rect; obstacles will be masked out at paint time
            spec["shape"]["rect"] = [int(nr0), int(nc0), int(nr1), int(nc1)]

        else:
            # Unknown mode -> no-op; keep rect as-is
            pass

    # Rebuild true_phi from all current ROI rects
    _paint_true_phi_from_rois(env, roi_specs)


class GridMC:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.H, self.W = spec.height, spec.width
        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.rng = np.random.default_rng(spec.rng_seed)
        
        self.phi_belief = np.ones((self.H, self.W), dtype=float)
        self.true_phi = self.phi_belief.copy()
        self.noise_sigma = np.full((self.H, self.W), 0.0, dtype=float)

        
        self._state_index_map = None
        self._index_state_map = None
        # graph caches
        self._edge_src = None
        self._edge_dst = None
        self._deg = None
        self.nstates = 0
        assert spec.neighbors in ("4", "8") # check this



    # ---- Map editing ----
    def set_roi_rect(self, r0: int, c0: int, r1: int, c1: int, weight: float = 1.0):
        self.grid[r0:r1, c0:c1] = ROI
        self.true_phi[r0:r1, c0:c1] += float(weight)

    def set_start_rect(self, r0: int, c0: int, r1: int, c1: int):
        self.grid[r0:r1, c0:c1] = START

    def set_obstacles(self, cells: List[Tuple[int,int]]):
        for r, c in cells:
            if 0 <= r < self.H and 0 <= c < self.W:
                self.grid[r, c] = OBSTACLE
                self.true_phi[r, c] = 0.0

    # ---- State graph ----
    def _neighbors(self, r: int, c: int) -> List[Tuple[int,int]]:
        if self.grid[r, c] == OBSTACLE:
            return []
        deltas4 = [(-1,0),(1,0),(0,-1),(0,1)]
        deltas8 = deltas4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
        deltas = deltas8 if self.spec.neighbors == "8" else deltas4
        nbrs = []
        for dr, dc in deltas:
            rr, cc = r+dr, c+dc
            if 0 <= rr < self.H and 0 <= cc < self.W and self.grid[rr, cc] != OBSTACLE:
                nbrs.append((rr, cc))
        return [(r, c)] + nbrs

    def _indexing(self):
        idx = 0
        s2i, i2s = {}, {}
        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r, c] != OBSTACLE:
                    s2i[(r, c)] = idx
                    i2s[idx] = (r, c)
                    idx += 1
        self._state_index_map = s2i
        self._index_state_map = i2s
        return idx
    

    def _precompute_graph(self):
        """Build an directed adjancy matrix of the graph"""
        if self._edge_src is not None and self._edge_dst is not None and self._deg is not None:
            return

        assert self._index_state_map is not None, "Call _indexing() first."
        nstates = len(self._index_state_map) # gets the valid states
        deg = np.zeros(nstates, dtype=np.int32) # init the degree of neighbors
        src_list, dst_list = [], []

        for i in range(nstates):
            r, c = self._index_state_map[i]
            # neighbors() currently includes self; exclude it for proposals
            nbrs = self._neighbors(r, c)
            Ni = [self._state_index_map[(rr, cc)]
                for (rr, cc) in nbrs if not (rr == r and cc == c)]
            deg[i] = len(Ni)
            for j in Ni:
                src_list.append(i)
                dst_list.append(j)

        self._edge_src = np.asarray(src_list, dtype=np.int32)
        self._edge_dst = np.asarray(dst_list, dtype=np.int32)
        self._deg = deg
        self.nstates = nstates



    def build_map_vec(self, field2d: np.ndarray) -> np.ndarray:
        self._indexing()
        self._precompute_graph()
        vec = np.empty(self.nstates, dtype=float)
        for i in range(self.nstates):
            r, c = self._index_state_map[i]
            vec[i] = max(field2d[r, c], 0.0)
        return vec

    def write_vec_to_phi(self, phi_vec: np.ndarray):
        if self._index_state_map is None:
            self.finalize_indexing()
        for i in range(self.nstates):
            r, c = self._index_state_map[i]
            self.phi_belief[r, c] = float(max(phi_vec[i], 0.0))
        self.phi_belief[self.grid == OBSTACLE] = 0.0

    def target_distribution(self) -> np.ndarray:
        pi = np.array([max(self.true_phi[self._index_state_map[i]], 0.0) for i in range(len(self._index_state_map))])
        s = pi.sum()
        return pi/s if s > 0 else np.ones_like(pi)/len(pi)
    
