
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


@dataclass
class MovingROI:
    r0:int; c0:int; r1:int; c1:int
    dr:int; dc:int; every_k:int=1; weight:float=1.0; bounce:bool=True

@dataclass
class SpreadingROI:
    r0:int; c0:int; r1:int; c1:int
    p:float=0.25; every_k:int=1; weight:float=1.0
    # internal state for spread mask (initialized on first step)
    mask: Optional[np.ndarray]=None


class GridMC:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.H, self.W = spec.height, spec.width
        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.rng = np.random.default_rng(spec.rng_seed)
        
        self.phi_belief = np.ones((self.H, self.W), dtype=float)
        self.true_phi = self.phi_belief.copy()
        self.base_phi   = np.zeros((self.H, self.W), dtype=float) 

        self.moving_rois: List[MovingROI] = []
        self.spreading_rois: List[SpreadingROI] = []
        
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
    

    def add_moving_roi(self, r0,c0,r1,c1, *, dr=0, dc=0, every_k=1, weight=1.0, bounce=True):
        self.moving_rois.append(MovingROI(r0,c0,r1,c1,dr,dc,every_k,weight,bounce))
        self.grid[r0:r1, c0:c1] = ROI

    def add_spreading_roi(self, r0,c0,r1,c1, *, p=0.25, every_k=1, weight=1.0):
        self.spreading_rois.append(SpreadingROI(r0,c0,r1,c1,p,every_k,weight,mask=None))
        self.grid[r0:r1, c0:c1] = ROI

    def add_static_roi_rect(self, r0,c0,r1,c1, weight=1.0):
        self.base_phi[r0:r1, c0:c1] += float(weight)
        self.grid[r0:r1, c0:c1] = ROI

    def _rebuild_true_phi_from_rois(self):
        phi = self.base_phi.copy()
        # add movers
        for m in self.moving_rois:
            phi[m.r0:m.r1, m.c0:m.c1] += m.weight
        # add spreaders via their current mask; if mask None, seed with rect
        for s in self.spreading_rois:
            if s.mask is None:
                mask = np.zeros_like(phi, dtype=bool)
                mask[s.r0:s.r1, s.c0:s.c1] = True
                s.mask = mask
            phi[s.mask] += s.weight
        phi[self.grid == OBSTACLE] = 0.0
        self.true_phi = phi

    def step_dynamic_rois(self, update_idx:int):
        # Move each MovingROI when its cadence hits
        for m in self.moving_rois:
            if m.every_k>0 and (update_idx % m.every_k)==0:
                nr0, nc0 = m.r0 + m.dr, m.c0 + m.dc
                h, w = (m.r1 - m.r0), (m.c1 - m.c0)
                nr1, nc1 = nr0 + h, nc0 + w

                def clamp_or_bounce(v0,v1,dv,lim,bounce):
                    if 0 <= v0 and v1 <= lim:
                        return v0, v1, dv
                    if bounce:
                        ndv = -dv
                        v0b, v1b = v0 + 2*ndv, v1 + 2*ndv
                        span = v1 - v0
                        v0b = max(0, min(v0b, lim - span))
                        v1b = v0b + span
                        return v0b, v1b, ndv
                    span = v1 - v0
                    v0c = max(0, min(v0, lim - span))
                    v1c = v0c + span
                    return v0c, v1c, 0

                nr0, nr1, m.dr = clamp_or_bounce(nr0,nr1,m.dr,self.H,m.bounce)
                nc0, nc1, m.dc = clamp_or_bounce(nc0,nc1,m.dc,self.W,m.bounce)
                m.r0, m.r1, m.c0, m.c1 = int(nr0), int(nr1), int(nc0), int(nc1)

        # Spread each SpreadingROI when its cadence hits
        for s in self.spreading_rois:
            if s.mask is None:
                mask = np.zeros((self.H,self.W), dtype=bool)
                mask[s.r0:s.r1, s.c0:s.c1] = True
                s.mask = mask
            if s.every_k>0 and (update_idx % s.every_k)==0:
                newmask = s.mask.copy()
                # 4-neighborhood spread, obstacles block
                deltas = [(-1,0),(1,0),(0,-1),(0,1)]
                rs, cs = np.where(s.mask)
                for r, c in zip(rs, cs):
                    for dr, dc in deltas:
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < self.H and 0 <= cc < self.W and self.grid[rr,cc] != OBSTACLE:
                            if not s.mask[rr,cc] and (self.rng.random() < s.p):
                                newmask[rr,cc] = True
                s.mask = newmask

        # Rebuild phi
        self._rebuild_true_phi_from_rois()