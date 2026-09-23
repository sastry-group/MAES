# experiments/planners/mac_dt_planner.py
from __future__ import annotations
import numpy as np
import os

from .base import BasePlanner
from env.grid import make_env_from_json
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from sim.sensing import GaussianSensor
from viz.plotting import plot_grid


class MACDTPlanner(BasePlanner):
    """

    """

    def __init__(self, args):
        self.args = args
        self.n_agents = int(getattr(args, "n_agents", 1))
        self.total_steps = int(args.T)
        self.env_dyn = getattr(args, "env_dyn", "static")
        # self.gp_update_every = int(getattr(args, "update_GP_every", 1))

        # Decision Tree parameters
        # self.horizon = int(getattr(args, "dt_horizon", 5))
        # self.gamma = float(getattr(args, "dt_gamma", 0.98))
        # self.overlap_penalty = float(getattr(args, "dt_overlap_penalty", 0.0))

        self.l_time = args.l_time

    def _neighbors(self, rc):
        r, c = rc
        mode = getattr(self.env.spec, "neighbors", "4")
        if str(mode) == "8":
            deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        else:
            deltas = [(-1,0),(1,0),(0,-1),(0,1)]

        out = [rc]  # allow stay
        for dr, dc in deltas:
            nxt = (r + dr, c + dc)
            if nxt in self.env._state_index_map:
                out.append(nxt)
        return out

    def _idx_to_rc_map(self):
        # reverse map idx -> (r,c)
        return {v: k for k, v in self.env._state_index_map.items()}

    def _rc_to_idx(self, rc):
        return int(self.env._state_index_map[rc])

    def _coverage_set_from_dest_rc(self, dest_rc, radius=0):
        """
        Paper's s_i(v): cells 'covered' if agent chooses destination v.

        """
        r0, c0 = dest_rc
        covered = []
        for (r, c), idx in self.env._state_index_map.items():
            if abs(r - r0) + abs(c - c0) <= radius:  # Manhattan ball; change if you want Euclidean
                covered.append(idx)
        return np.array(covered, dtype=int)


    def _greedy_oracle_destinations(self, w_ucb_vec, radius=0):
        """
        Greedy max-coverage oracle (1-1/e)-approx:
        sequentially pick destinations maximizing marginal covered UCB reward.
        Returns list of dest_rc for each agent (length n_agents).
        """
        w = np.asarray(w_ucb_vec, dtype=float)
        idx_to_rc = self._idx_to_rc

        # candidate destinations: all free cells (all state indices)
        candidates = list(range(self.env.nstates))

        chosen = []
        covered_mask = np.zeros(self.env.nstates, dtype=bool)

        for i in range(self.n_agents):
            best_gain = -1e18
            best_dest_idx = candidates[0]

            for v_idx in candidates:
                v_rc = idx_to_rc[v_idx]
                cover_idxs = self._coverage_set_from_dest_rc(v_rc, radius=radius)
                already = covered_mask[cover_idxs]  
                # marginal gain = sum w on newly covered cells
                new_cells = cover_idxs[~covered_mask[cover_idxs]]
                gain = float(w[new_cells].sum())

                if gain > best_gain:
                    best_gain = gain
                    best_dest_idx = v_idx
                    best_raw = float(w[cover_idxs].sum())    # this is to check the overlap
                    best_overlap_n = int(already.sum())  # check the overlap


            chosen.append(best_dest_idx)
            # print(f"[ORACLE] agent {i}: dest={idx_to_rc[best_dest_idx]} "
            #     f"raw={best_raw:.4f} marginal={best_gain:.4f} "
            #     f"discount={best_raw - best_gain:.4f} overlap_cells={best_overlap_n}")
            # update covered set
            best_rc = idx_to_rc[best_dest_idx]
            cover_idxs = self._coverage_set_from_dest_rc(best_rc, radius=radius)
            covered_mask[cover_idxs] = True

            # # optional: prevent choosing same destination again
            # if best_dest_idx in candidates:
            #     candidates.remove(best_dest_idx)

        return [idx_to_rc[i] for i in chosen]

    # def _best_first_move_to_dest(self, start_rc, util_vec, dest_rc, reserved=None):
    #     """
    #     Your DT planner, but destination-conditioned:
    #     same as _best_first_move, except each step also gets a penalty for being far from dest.
    #     """
    #     if reserved is None:
    #         reserved = set()

    #     best_score = -1e18
    #     best_first = start_rc

    #     start_idx = self._rc_to_idx(start_rc)
    #     stack = [(0, start_rc, 0.0, start_rc, {start_idx})]

    #     # tune this (add as args if you want)
    #     lam = float(getattr(self.args, "dt_to_dest_lambda", 0.5))

    #     while stack:
    #         depth, cur_rc, score, first_rc, visited = stack.pop()
    #         if depth == self.horizon:
    #             if score > best_score:
    #                 best_score = score
    #                 best_first = first_rc
    #             continue

    #         for nxt_rc in self._neighbors(cur_rc):
    #             nxt_idx = self._rc_to_idx(nxt_rc)
    #             if nxt_idx in visited:
    #                 continue

    #             util = float(util_vec[nxt_idx])
    #             overlap = 1.0 if nxt_idx in reserved else 0.0

    #             # distance-to-destination shaping (Manhattan)
    #             dist = abs(nxt_rc[0] - dest_rc[0]) + abs(nxt_rc[1] - dest_rc[1])

    #             step_score = (self.gamma ** depth) * (
    #                 util - self.overlap_penalty * overlap - lam * dist
    #             )

    #             nxt_first = nxt_rc if depth == 0 else first_rc
    #             new_visited = set(visited)
    #             new_visited.add(nxt_idx)

    #             stack.append((depth + 1, nxt_rc, score + step_score, nxt_first, new_visited))

    #     return best_first

    # def _best_first_move(self, start_rc, util_vec, reserved=None):
    #     if reserved is None:
    #         reserved = set()

    #     best_score = -1e18
    #     best_first = start_rc

    #     start_idx = self._rc_to_idx(start_rc)
    #     stack = [(0, start_rc, 0.0, start_rc, {start_idx})]

    #     while stack:
    #         depth, cur_rc, score, first_rc, visited = stack.pop()

    #         if depth == self.horizon:
    #             if score > best_score:
    #                 best_score = score
    #                 best_first = first_rc
    #             continue

    #         for nxt_rc in self._neighbors(cur_rc):
    #             nxt_idx = self._rc_to_idx(nxt_rc)
    #             if nxt_idx in visited:
    #                 continue

    #             util = float(util_vec[nxt_idx])
    #             overlap = 1.0 if nxt_idx in reserved else 0.0
    #             step_score = (self.gamma ** depth) * (util - self.overlap_penalty * overlap)

    #             nxt_first = nxt_rc if depth == 0 else first_rc
    #             new_visited = set(visited)
    #             new_visited.add(nxt_idx)

    #             stack.append(
    #                 (depth + 1, nxt_rc, score + step_score, nxt_first, new_visited)
    #             )

    #     return best_first


    def _shortest_first_move(self, start_rc, dest_rc):
        """
        Paper Alg 1, line 4: shortest path on the graph from current cell to the
        oracle destination; return the first step along it. No lookahead horizon.
        """
        if start_rc == dest_rc or dest_rc not in self.env._state_index_map:
            return start_rc  # at dest (stay & sample) or dest no longer valid

        from collections import deque
        prev = {start_rc: None}
        q = deque([start_rc])
        while q:
            cur = q.popleft()
            if cur == dest_rc:
                break
            for nxt in self._neighbors(cur):
                if nxt == cur or nxt in prev:   # skip the 'stay' entry and visited
                    continue
                prev[nxt] = cur
                q.append(nxt)

        if dest_rc not in prev:
            return start_rc  # unreachable; stay put

        node = dest_rc                          # walk back to the first step out of start
        while prev[node] != start_rc:
            node = prev[node]
        return node
    



    def _episode_index(self):
        return max(1, len(self.episode_starts))   # paper's e starts at 1

    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, stem=None,config_path=None, h5_file=None, **kwargs):
        self.env = env
        self.run_dir = stem
        self.steps_per_cfg = int(steps_per_cfg)
        self.h5_file = h5_file
        self.cfg_paths = cfg_paths
        self.config_path = config_path
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        
        self.current_change_idx = 0
        self.current_cfg_idx = 0
        self.next_map_change = self.steps_per_cfg
        self.env_change_steps = sorted(getattr(args, "env_change_steps",  [i * self.steps_per_cfg for i in range(1, self.n_cfg)]))


        # coverage radius for oracle (paper s_i(v)); set 0 for "destination cell only"
        self.oracle_radius = int(getattr(self.args, "oracle_radius", 0))

        # how often to re-run oracle; in the paper it's "each episode"
        # simplest: use gp_update_every as episode length
        # self.episode_len = int(getattr(self.args, "episode_len", self.gp_update_every))
        self.t0 = 0
        self.global_t = 0  
        # self.since_gp = 0
        self.new_episode = True 
        


        # Env datasets

        T = self.total_steps
        N = self.n_agents
        


        # Initial belief
        phi_vec_uni = UniformPhi().phi_vec(self.env) * 6.0 # we need to change this from hard-coding

        # Agent positions
        if self.n_agents == 1:
            self.start_rc_agents = [agents_initial_positions[0]]
            self.last_phi_vec = phi_vec_uni.copy()
        else:
            self.start_rc_agents = list(agents_initial_positions)
            # self.last_phi_vecs = [phi_vec_uni.copy() for _ in range(self.n_agents)]
        
        self.last_phi_vec = phi_vec_uni.copy() # single global belief
        self.dest_rc_agents = [rc for rc in self.start_rc_agents]


        # # GP per agent
        # self.gps = [
        #     GPInfoMap(l_space=2.0, l_time=self.l_time, nu=1.5, noise=1e-12)
        #     for _ in range(self.n_agents)
        # ]
        # self.gp_phi_loggers = [
        #     GPPhi(
        #         self.gps[a],
        #         mode=args.gp_mode,
        #         beta_m=args.ucb_beta_m,
        #         beta_u=args.ucb_beta_u,
        #         base=1e-3,
        #         beta_schedule="mac_dt",
        #         nstates=len(self.env._state_index_map),
        #         delta=getattr(args, "ucb_delta", 0.05),
        #     )
        #     for a in range(self.n_agents)
        # ]
        # Single shared GP (centralized per Remark 1)
        self.gp = GPInfoMap(l_space=self.args.l_space, l_time=self.l_time, nu=1.5, noise=1e-12)
        self.sensor = GaussianSensor.from_args(args, self.env)   # Eq. (3)
        self.h5_file.attrs["sigma_eps"] = self.sensor.sigma_eps
        self.h5_file.attrs["gp_noise_scale"] = self.sensor.gp_noise_scale
        self.gp_phi_logger = GPPhi(
            self.gp,
            mode=args.gp_mode,
            beta_m=args.ucb_beta_m,
            beta_u=args.ucb_beta_u,
            base=1e-3,
            beta_schedule="mac_dt",
            nstates=len(self.env._state_index_map),
            delta=getattr(args, "ucb_delta", 0.05),
        )    
        self.last_mu_vec  = np.zeros(self.env.nstates)
        self.last_std_vec = np.ones(self.env.nstates)
        self.gp_fit_steps = []
        self.grid_shape = np.array(self.env.true_phi.shape, dtype=int)
        self.true_phi_map = self.env.true_phi.astype(float)

        init_true_vec = self.env.build_map_vec(self.env.true_phi)
        self.current_true_phi = init_true_vec

        self._idx_to_rc = self._idx_to_rc_map()
        self.w_ucb = phi_vec_uni.copy()


        # ---- HDF5 datasets (streaming writes) ----
        nstates = self.env.nstates
        T = self.total_steps
        N = self.n_agents
        # for the double-trick
        self._init_sample_counts()  



        self.d_traj_states = self.h5_file.create_dataset(
            "traj_states", shape=(T, N), dtype="i",
            compression="gzip", compression_opts=4
        )
        self.d_obs_y = self.h5_file.create_dataset(
            "obs_y", shape=(T, N), dtype="f", fillvalue=np.nan,
            compression="gzip", compression_opts=4
        )
        self.d_traj_rc = self.h5_file.create_dataset(
            "traj_rc_agents_all", shape=(T, N, 2), dtype="i",
            compression="gzip", compression_opts=4
        )
        self.d_belief_phi = self.h5_file.create_dataset(
            "belief_phi_map", shape=(T, N, nstates), dtype="f",
            compression="gzip", compression_opts=4
        )
        # self.d_belief_phi_avg = self.h5_file.create_dataset(
        #     "belief_phi_map_avg", shape=(T, nstates), dtype="f",
        #     compression="gzip", compression_opts=4
        # )
        # self.d_true_phi = self.h5_file.create_dataset(
        #     "true_phi_map_time", shape=(T, nstates), dtype="f",
        #     compression="gzip", compression_opts=4
        # )

        self.d_belief_mu = self.h5_file.create_dataset(
            "belief_mu_map", shape=(T, N, nstates), dtype="f",
            compression="gzip", compression_opts=4
        )
        self.d_belief_sigma = self.h5_file.create_dataset(
            "belief_sigma_map", shape=(T, N, nstates), dtype="f",
            compression="gzip", compression_opts=4
        )

        # Write initial positions
        for a in range(self.n_agents):
            rc = self.start_rc_agents[a]
            idx = self._rc_to_idx(rc)
            self.d_traj_states[0, a] = idx
            self.d_traj_rc[0, a] = rc
            # if self.n_agents == 1:
            self.d_belief_phi[0, a] = phi_vec_uni # single initial
            self.d_belief_mu[0, a]    = self.last_mu_vec
            self.d_belief_sigma[0, a] = self.last_std_vec
            # else:
            #     self.d_belief_phi[0, a] = self.last_phi_vecs[a]

        # self.d_true_phi[0] = self.current_true_phi
        n_true = self.n_cfg + 1          # static: n_cfg=1 -> 2 rows
        self.d_true_phi = self.h5_file.create_dataset(
            "true_phi_map_time", shape=(n_true, nstates), dtype="f",
            chunks=(1, nstates), compression="gzip", compression_opts=4,
        )
        self.d_true_phi_change_t = self.h5_file.create_dataset(
            "true_phi_change_times", shape=(n_true,), dtype="i"
        )
        self.d_true_phi_change_t[:] = -1
        self.d_true_phi[0] = self.current_true_phi
        self.d_true_phi_change_t[0] = 0
        self.true_phi_row = 1


    ## These functions support the doubling-trick approach
    def _init_sample_counts(self):
        """Initializtion to track vertex sampling counters n_v(t) for the doubling trick (Alg 1, line 6)."""
        nstates = len(self.env._state_index_map)  
        self.n_v = np.zeros(nstates, dtype=np.int64)          # n_v(t)
        self.n_v_episode_start = np.zeros(nstates, dtype=np.int64)  # n_v(t_e - 1)
        self.episode_starts = []   # for verification: timestep each episode begins

    def _start_episode(self):
        """Snapshot counts at the episode boundary: n_v(t_e - 1)."""
        self.n_v_episode_start = self.n_v.copy()
        self.episode_starts.append(int(self.t0))

    def _record_samples(self, idxs):
        """
        n_v(t) = sum_tau 1{v in s_eval_tau}: tracking sample per steps
        So a cell sampled by >=1 agent this step increments by exactly 1.
        """
        for idx in set(idxs):
            self.n_v[idx] += 1

    def _doubling_triggered(self):
        """
        Paper's termination (Alg 1, line 6): episode ends at the minimum t with
        some vertex v s.t.  n_v(t) >= max{ 2 * n_v(t_e - 1), 1 }.
        """
        threshold = np.maximum(2 * self.n_v_episode_start, 1)
        return bool(np.any(self.n_v >= threshold))



    def step(self):
        # print(f"Current time {self.t0}")
        if self.t0 >= self.total_steps - 1:
            return

        # if self.n_agents == 1:
        phi_for_plot = self.last_phi_vec # single for all agents
        # else:
        #     phi_for_plot = sum(self.last_phi_vecs) / float(self.n_agents)
        current_phi_vec_norm = phi_for_plot / phi_for_plot.sum()


        # ---- EPISODE START: compute destinations via greedy oracle
        if self.new_episode:
            self._start_episode()     # snapshot n_v(t_e - 1)
            # Build the "UCB reward map" for the oracle.
            # Paper assumes a common w_UCB(e). Simplest: use average across agents.
            e = self._episode_index()
            print(f"e={e}  beta={self.gp_phi_logger._beta(e):.3f}")
            # w_list = [self.gp_phi_loggers[a].phi_vec(self.env t=self.t0, e=self._episode_index())  for a in range(self.n_agents)]
            # self.w_ucb = np.mean(np.stack(w_list, axis=0), axis=0)
            self.w_ucb = self.gp_phi_logger.phi_vec(self.env, t=self.t0, e=e)

            # idx_max = int(np.argmax(w_ucb))
            # idx_min = int(np.argmin(w_ucb))
            # rc_max = self._idx_to_rc[idx_max]
            # rc_min = self._idx_to_rc[idx_min]
            # print(f"[DEBUG] t={self.t0} w_ucb min={w_ucb[idx_min]:.6f} at {rc_min}, max={w_ucb[idx_max]:.6f} at {rc_max}")
            # print(f"[DEBUG] top10 idx: {np.argsort(w_ucb)[-10:][::-1]}")

            # Greedy oracle destinations
            self.dest_rc_agents = self._greedy_oracle_destinations(self.w_ucb, radius=self.oracle_radius)

            # print(f"[DEBUG] t={self.t0} destinations: {self.dest_rc_agents}")
            # print(f"[DEBUG] t={self.t0} starts: {self.start_rc_agents}")
            self.new_episode = False


        # ---- Choose actions
        # reserved = set()
        chosen_next = []

        for a in range(self.n_agents):
            rc = self.start_rc_agents[a]
            dest_rc = self.dest_rc_agents[a]

            # util = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0,  e=self._episode_index())
            # print(f"time {self.t0}: util.min() = {util.min()}, util.max() = {util.max()}")

            # DT transient controller *toward dest*
            # nxt = self._best_first_move_to_dest(rc, util, dest_rc, reserved)
            nxt = self._shortest_first_move(rc, dest_rc)

            chosen_next.append(nxt)
            # reserved.add(self._rc_to_idx(nxt))

        # ---- Execute actions
        t_next = self.t0 + 1
        sampled_idxs = []
        for a in range(self.n_agents):
            r, c = chosen_next[a]
            agent_rc = (r, c)
            idx = self._rc_to_idx(agent_rc)
            self.start_rc_agents[a] = agent_rc

            self.d_traj_states[t_next, a] = idx
            self.d_traj_rc[t_next, a] = agent_rc
            sampled_idxs.append(idx) 

            # Observation
            # cell_noise = float(self.env.noise_sigma[r, c])
            # y_true = float(self.env.true_phi[r, c])
            # y_noisy = y_true + np.random.normal(0.0, cell_noise)

            # k = 0.7
            # proxy_var = float(np.clip((k * cell_noise) ** 2, 1e-6, 1e-1))

            y_noisy, proxy_var = self.sensor.observe(self.env, [r], [c])
            self.d_obs_y[t_next, a] = y_noisy[0]
            self.gp.add_batch(
                r=np.array([r], dtype=float),
                c=np.array([c], dtype=float),
                t=np.array([self.t0], dtype=float),
                y=y_noisy,
                noise_var=proxy_var,
            )


            # self.gp.add_batch(
            #     r=np.array([r], dtype=float),
            #     c=np.array([c], dtype=float),
            #     t=np.array([self.t0], dtype=float),
            #     y=np.array([y_noisy], dtype=float),
            #     noise_var=np.array([proxy_var], dtype=float),
            # )
    
        self._record_samples(sampled_idxs)
        if self._doubling_triggered():
            self.new_episode = True

        # ---- GP refit at episode boundary (Alg 1, line 7)
        # self.new_episode was just set True by the doubling trick above iff the
        # episode terminated this step; refit once, here, on all data so far.
        # self.since_gp += 1
        if self.new_episode:
            # if self.n_agents == 1:
            #     self.gps[0].fit_if_ready(min_pts=1)
            #     self.last_phi_vec = self.gp_phi_loggers[0].phi_vec(self.env, t=self.t0,  e=self._episode_index()).copy()
                
            # else:
            #     for a in range(self.n_agents):
            #         self.gps[a].fit_if_ready(min_pts=1)
            #         self.last_phi_vecs[a] = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0,  e=self._episode_index()).copy()   
            #             self.gp.fit_if_ready(min_pts=1)
            # one global GP
            self.gp.fit_if_ready(min_pts=1)
            self.last_phi_vec = self.gp_phi_logger.phi_vec(self.env, t=self.t0, e=self._episode_index()).copy()
            self.last_mu_vec  = self.gp_phi_logger.last_mu.copy()
            self.last_std_vec = self.gp_phi_logger.last_std.copy()
            self.gp_fit_steps.append(t_next)   
            
                    # phi = sum(self.last_phi_vecs) / self.n_agents
            # self.since_gp = 0




        # DYNAMIC ENVIRONMENT CHANGES      
        if self.env_dyn == "dynamic":
            while (
                self.current_change_idx < len(self.env_change_steps)
                and self.t0 >= self.env_change_steps[self.current_change_idx]
            ):

                self.current_cfg_idx += 1
                if self.current_cfg_idx >= len(self.cfg_paths):
                    break

                self.env = make_env_from_json(
                    self.cfg_paths[self.current_cfg_idx]
                )
                self.current_true_phi = self.env.build_map_vec(
                    self.env.true_phi
                )

                plot_grid(
                    self.env,
                    os.path.join(
                        self.run_dir,
                        f"gp_grid_true_cfg{self.current_cfg_idx}.png"
                    ),
                    title=f"Grid (true info map) cfg {self.current_cfg_idx}"
                )

                print(
                    f"[t={self.env_change_steps[self.current_change_idx]}] Environment updated, "
                    f"loaded config {self.cfg_paths[self.current_cfg_idx]}"
                )


                if self.true_phi_row < self.d_true_phi.shape[0]:
                    self.d_true_phi[self.true_phi_row] = self.current_true_phi
                    self.d_true_phi_change_t[self.true_phi_row] = self.t0
                    self.true_phi_row += 1

                self.current_change_idx += 1
 
                # self.true_phi_map_time[self.t0:self.next_map_change, :] = self.current_true_phi

        # self.belief_phi_map[self.t0] = current_phi_vec_norm
        
        for a in range(self.n_agents):
            
            # if self.n_agents == 1:
            self.d_belief_phi[t_next, a] = self.last_phi_vec
            self.d_belief_mu[t_next, a]    = self.last_mu_vec
            self.d_belief_sigma[t_next, a] = self.last_std_vec
            
            # else:
                
            #     self.d_belief_phi[t_next, a] = self.last_phi_vecs[a]

        # self.d_belief_phi_avg[t_next] = current_phi_vec_norm
        # self.d_true_phi[t_next] = self.current_true_phi
        self.t0 = t_next



    def finalize(self):
        # assert np.all(self.traj_states >= 0), "traj_states has unwritten entries (-1)."
        # assert self.traj_rc_time.shape == (self.total_steps, self.n_agents, 2)

        # if self.n_agents == 1:
        pi_vec = self.last_phi_vec / np.maximum(self.last_phi_vec.sum(), 1e-12)
        # else:
        #     phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
        #     pi_vec = phi_avg_vec / np.maximum(phi_avg_vec.sum(), 1e-12)
        # phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
        # pi_vec = phi_avg_vec / phi_avg_vec.sum()
        # ep = np.array(self.episode_starts)
        # print("episode lengths:", np.diff(ep))
        # print("num episodes:", len(ep), " bound |V|logT+1:", self.env.nstates*np.log(self.total_steps)+1)

        # to keep track of the first time reaching the ROIs
        roi_rc = []
        roi_name = []

        for roi in getattr(self.env, "roi_cells", []):
            name = roi["name"]
            r0, c0, r1, c1 = roi["rect"]

            for r in range(r0, r1):
                for c in range(c0, c1):
                    if (r, c) in self.env._state_index_map:
                        roi_rc.append((r, c))
                        roi_name.append(name)


        # ---- Write final datasets ----
        self.h5_file.create_dataset("pi_vec_final", data=pi_vec)
        self.h5_file.create_dataset("roi_rc", data=np.array(roi_rc, dtype=int))
        self.h5_file.create_dataset("roi_name", data=np.array(roi_name, dtype="S"))
        self.h5_file.create_dataset("grid_shape", data=self.grid_shape)
        self.h5_file.create_dataset("true_phi_map", data=self.true_phi_map)
        self.h5_file.create_dataset(
            "gp_update_steps", data=np.array(self.gp_fit_steps, dtype=int)
        )
        self.h5_file.create_dataset(
            "state_index_map_keys",
            data=np.array(list(self.env._state_index_map.keys()), dtype=int)
        )
        self.h5_file.create_dataset(
            "state_index_map_vals",
            data=np.array(list(self.env._state_index_map.values()), dtype=int)
        )

        self.h5_file.create_dataset(
            "env_change_steps",
            data=np.array([i * self.steps_per_cfg for i in range(1, self.n_cfg)], dtype=int)
        )
        # ---- Attributes (metadata) ----
        self.h5_file.attrs["stem"] = self.run_dir
        self.h5_file.attrs["planner"] = "mac_dt"
        self.h5_file.attrs["mode"] = str(self.args.mode)
        self.h5_file.attrs["env_dyn"] = str(self.env_dyn)
        self.h5_file.attrs["env_source"] = str(getattr(self.args, "config", "NA"))
        self.h5_file.attrs["T"] = self.total_steps
        self.h5_file.attrs["n_agents"] = self.n_agents
        self.h5_file.attrs["gp_mode"] = str(self.args.gp_mode)
        self.h5_file.attrs["nstates"] = int(self.env.nstates)
        self.h5_file.attrs["oracle_radius"] = self.oracle_radius
        self.h5_file.attrs["ucb_beta_m"] = float(self.args.ucb_beta_m)
        self.h5_file.attrs["ucb_beta_u"] = float(self.args.ucb_beta_u)

        self.h5_file.close()
        print(f"[MAC-DT] HDF5 file saved and closed.")

