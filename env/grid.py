
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from core.types import GridSpec

# ----------------------------
# Cell encodings
# ----------------------------
EMPTY = 0
OBSTACLE = 1
ROI = 2
START = 3



def make_env(neighbors="4", seed=1):
    spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
    env = GridMC(spec)
    # Define the “true world” (oracle)
    env.set_roi_rect(0, 0, 1, 1, weight=3.0)
    env.set_roi_rect(0, 3, 2, 4, weight=6.0)
    env.set_start_rect(3, 1, 4, 1)
    env.set_obstacles([(2, 2)])
    return env


class GridMC:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.H, self.W = spec.height, spec.width
        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.rng = np.random.default_rng(spec.rng_seed)
        
        self.phi_belief = np.ones((self.H, self.W), dtype=float)
        self.true_phi = self.phi_belief.copy()

        
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
    
