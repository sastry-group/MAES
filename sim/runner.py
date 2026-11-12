import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple
from env.grid import GridMC, EMPTY, OBSTACLE, ROI, START



# class ObservationLogger:
#     def __init__(self):
#         self.t, self.r, self.c, self.y = [], [], [], []

#     def add(self, t, r, c, y):
#         self.t.append(int(t)); self.r.append(int(r)); self.c.append(int(c)); self.y.append(y)

#     def arrays(self):
#         return (np.asarray(self.t), np.asarray(self.r), np.asarray(self.c), np.asarray(self.y, dtype=float))
    
#     def observe_cell(self, r: int, c: int, t: int):
#         """
#         Observe the current cell type and record it.
#         - EMPTY: returns 0.0
#         - ROI: returns 1.0
#         - OBSTACLE: returns None (can't observe)
#         - START: returns 0.5 (just a neutral value for testing)
#         """
#         if not hasattr(self, "obs_log"):
#             self.obs_log = {"t": [], "r": [], "c": [], "val": []}

#         cell_type = self.grid[r, c]
#         if cell_type == OBSTACLE:
#             val = None
#         elif cell_type == EMPTY:
#             val = 0.0
#         elif cell_type == ROI:
#             val = float(self.phi[r, c])
#         elif cell_type == START:
#             val = 0.5
#         else:
#             val = 0.0

#         self.obs_log["t"].append(int(t))
#         self.obs_log["r"].append(int(r))
#         self.obs_log["c"].append(int(c))
#         self.obs_log["val"].append(val)

#         return val


#     def get_observations(self):
#         if not hasattr(self, "obs_log") or len(self.obs_log["t"]) == 0:
#             return (np.array([]), np.array([]), np.array([]), np.array([]))
#         r = np.asarray(self.obs_log["r"], dtype=int)
#         c = np.asarray(self.obs_log["c"], dtype=int)
#         t = np.asarray(self.obs_log["t"], dtype=int)
#         val = np.asarray(self.obs_log["val"], dtype=float)
#         return r, c, t, val


# class Sensor:
#     def __init__(self, sigma=0.0, sigma_map=None):
#         self.sigma = float(sigma)
#         self.sigma_map = sigma_map

#     def measure(self, true_value, rng, r=None, c=None):
#         if np.isnan(true_value): true_value = 0.0
#         if self.sigma_map is not None and r is not None and c is not None:
#             sigma = float(self.sigma_map[r, c])
#         else:
#             sigma = self.sigma
#         noise = rng.normal(0.0, sigma) if sigma > 0 else 0.0
#         return float(true_value + noise)
    


class Simulator:
    def __init__(self, env):
        self.env = env
        self.rng = np.random.default_rng(env.spec.rng_seed)
        print(f"Using seed: {env.spec.rng_seed}")
        print(self.rng)

    def sample_start(self) -> Tuple[int,int]:
        starts = np.argwhere(self.env.grid == 3)  # START
        if len(starts) == 0:
            free = np.argwhere(self.env.grid != 1)  # not OBSTACLE
            r, c = free[self.rng.integers(len(free))]
            return int(r), int(c)
        r, c = starts[self.rng.integers(len(starts))]
        return int(r), int(c)

    def simulate(self, P_csr, T: int, start: Tuple[int,int] | None = None):
        assert P_csr is not None, "Provide a transition matrix P"
        if start is None: start = self.sample_start()
        s = self.env._state_index_map[start]
        traj_idx = np.empty(T, dtype=int)
        traj_rc  = np.empty((T, 2), dtype=int)
        counts   = np.zeros(self.env.nstates, dtype=int)

        for t in range(T):
            traj_idx[t] = s
            r, c = self.env._index_state_map[s]
            traj_rc[t] = (r, c)
            counts[s] += 1

            
            if t == T - 1:
                break    
            # # observe from TRUE world
            # true_val = 0.0 if self.env.true_phi is None else self.env.true_phi[r, c]
            # meas = self.sensor.measure(true_val, self.rng, r=r, c=c)
            # self.logger.add(t, r, c, meas)

            # step by P
            beg, end = P_csr.indptr[s], P_csr.indptr[s+1]
            cols = P_csr.indices[beg:end]; probs = P_csr.data[beg:end]
            s = int(self.rng.choice(cols, p=probs))

        return traj_idx, traj_rc, counts

    def empirical_distribution(self, counts: np.ndarray) -> np.ndarray:
        return counts/counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)          