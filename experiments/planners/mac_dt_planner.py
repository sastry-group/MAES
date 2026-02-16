# experiments/planners/mac_dt_planner.py
from __future__ import annotations
import numpy as np
import os

from .base import BasePlanner
from env.grid import make_env_from_json
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
# from experiments.concensus import run_phi_consensus
from viz.plotting import plot_grid


class MACDTPlanner(BasePlanner):
    """

    """

    def __init__(self, args):
        self.args = args
        self.n_agents = int(getattr(args, "n_agents", 1))
        self.total_steps = int(args.T)

        self.env_dyn = getattr(args, "env_dyn", "static")
        self.gp_update_every = int(getattr(args, "update_GP_every", 1))

        # Decision Tree parameters
        self.horizon = int(getattr(args, "dt_horizon", 5))
        self.gamma = float(getattr(args, "dt_gamma", 0.98))
        self.overlap_penalty = float(getattr(args, "dt_overlap_penalty", 0.0))

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

                # marginal gain = sum w on newly covered cells
                new_cells = cover_idxs[~covered_mask[cover_idxs]]
                gain = float(w[new_cells].sum())

                if gain > best_gain:
                    best_gain = gain
                    best_dest_idx = v_idx

            chosen.append(best_dest_idx)

            # update covered set
            best_rc = idx_to_rc[best_dest_idx]
            cover_idxs = self._coverage_set_from_dest_rc(best_rc, radius=radius)
            covered_mask[cover_idxs] = True

            # optional: prevent choosing same destination again
            if best_dest_idx in candidates:
                candidates.remove(best_dest_idx)

        return [idx_to_rc[i] for i in chosen]

    def _best_first_move_to_dest(self, start_rc, util_vec, dest_rc, reserved=None):
        """
        Your DT planner, but destination-conditioned:
        same as _best_first_move, except each step also gets a penalty for being far from dest.
        """
        if reserved is None:
            reserved = set()

        best_score = -1e18
        best_first = start_rc

        start_idx = self._rc_to_idx(start_rc)
        stack = [(0, start_rc, 0.0, start_rc, {start_idx})]

        # tune this (add as args if you want)
        lam = float(getattr(self.args, "dt_to_dest_lambda", 0.5))

        while stack:
            depth, cur_rc, score, first_rc, visited = stack.pop()
            if depth == self.horizon:
                if score > best_score:
                    best_score = score
                    best_first = first_rc
                continue

            for nxt_rc in self._neighbors(cur_rc):
                nxt_idx = self._rc_to_idx(nxt_rc)
                if nxt_idx in visited:
                    continue

                util = float(util_vec[nxt_idx])
                overlap = 1.0 if nxt_idx in reserved else 0.0

                # distance-to-destination shaping (Manhattan)
                dist = abs(nxt_rc[0] - dest_rc[0]) + abs(nxt_rc[1] - dest_rc[1])

                step_score = (self.gamma ** depth) * (
                    util - self.overlap_penalty * overlap - lam * dist
                )

                nxt_first = nxt_rc if depth == 0 else first_rc
                new_visited = set(visited)
                new_visited.add(nxt_idx)

                stack.append((depth + 1, nxt_rc, score + step_score, nxt_first, new_visited))

        return best_first

    def _best_first_move(self, start_rc, util_vec, reserved=None):
        if reserved is None:
            reserved = set()

        best_score = -1e18
        best_first = start_rc

        start_idx = self._rc_to_idx(start_rc)
        stack = [(0, start_rc, 0.0, start_rc, {start_idx})]

        while stack:
            depth, cur_rc, score, first_rc, visited = stack.pop()

            if depth == self.horizon:
                if score > best_score:
                    best_score = score
                    best_first = first_rc
                continue

            for nxt_rc in self._neighbors(cur_rc):
                nxt_idx = self._rc_to_idx(nxt_rc)
                if nxt_idx in visited:
                    continue

                util = float(util_vec[nxt_idx])
                overlap = 1.0 if nxt_idx in reserved else 0.0
                step_score = (self.gamma ** depth) * (util - self.overlap_penalty * overlap)

                nxt_first = nxt_rc if depth == 0 else first_rc
                new_visited = set(visited)
                new_visited.add(nxt_idx)

                stack.append(
                    (depth + 1, nxt_rc, score + step_score, nxt_first, new_visited)
                )

        return best_first

    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, stem=None, **kwargs):
        self.env = env
        self.run_dir = stem
        self.steps_per_cfg = int(steps_per_cfg)
        self.cfg_paths = cfg_paths
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        self.current_cfg_idx = 0
        self.next_map_change = self.steps_per_cfg



        # coverage radius for oracle (paper s_i(v)); set 0 for "destination cell only"
        self.oracle_radius = int(getattr(self.args, "oracle_radius", 0))

        # how often to re-run oracle; in the paper it's "each episode"
        # simplest: use gp_update_every as episode length
        self.episode_len = int(getattr(self.args, "episode_len", self.gp_update_every))
        self.t0 = 0
        self.global_t = 0  
        self.since_gp = 0


        
        # Initial belief
        phi_vec_uni = UniformPhi().phi_vec(self.env) * 6.0

        # Agent positions
        if self.n_agents == 1:
            self.start_rc_agents = [agents_initial_positions[0]]
            self.last_phi_vec = phi_vec_uni.copy()
        else:
            self.start_rc_agents = list(agents_initial_positions)
            self.last_phi_vecs = [phi_vec_uni.copy() for _ in range(self.n_agents)]

        self.dest_rc_agents = [rc for rc in self.start_rc_agents]


        # GP per agent
        self.gps = [
            GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
            for _ in range(self.n_agents)
        ]
        self.gp_phi_loggers = [
            GPPhi(
                self.gps[a],
                mode=args.gp_mode,
                beta_m=args.ucb_beta_m,
                beta_u=args.ucb_beta_u,
                base=1e-3,
            )
            for a in range(self.n_agents)
        ]

        # Logging
        # self.cr = np.zeros(self.env.nstates, dtype=int)
        # self.counts_total = np.zeros(self.env.nstates, dtype=int)
        # self.traj_rc_agents = [[] for _ in range(self.n_agents)]
        # self.traj_idx_agents = [[] for _ in range(self.n_agents)]
        self.traj_states = np.full((self.total_steps, self.n_agents), -1, dtype=int)
        self.traj_rc_time = np.zeros((self.total_steps, self.n_agents, 2), dtype=int)

        for a in range(self.n_agents):
            rc = self.start_rc_agents[a]
            idx = self._rc_to_idx(rc)

            self.traj_states[0, a] = idx
            self.traj_rc_time[0, a, :] = rc

        # self.emp_series = []
        # self.phi_norm_series = []
        # self.error_series = []

        # self.first_visit_step = np.full(self.env.nstates, -1, dtype=int)
        # self.unvisited_count = self.env.nstates
        # self.cover_time = -1


        # True map logging
        init_true = self.env.build_map_vec(self.env.true_phi)
        self.true_phi_map_time = np.zeros((self.total_steps, init_true.size))
        self.true_phi_map_time[:self.steps_per_cfg] = init_true
        self.current_true_phi = init_true

        self.belief_phi_map = np.zeros((self.total_steps, self.env.nstates))
        self.obs_r_all, self.obs_c_all, self.obs_t_all, self.obs_y_all = [], [], [], []
        self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
        self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)
        self.gp_fit_steps = []
        self.grid_shape = np.array(self.env.true_phi.shape, dtype=int)
        self.true_phi_map = self.env.true_phi.astype(float)



        # r_hat0 = self.cr / self.cr.sum()

        phi0 = phi_vec_uni / phi_vec_uni.sum()
        self.belief_phi_map[0] = phi0
        # self.phi_norm_series = [phi0.copy()]
        # self.error_series = [float(np.linalg.norm(r_hat0 - phi0))]
        self._idx_to_rc = self._idx_to_rc_map()
        # self.emp_series = [r_hat0.copy()]



    def step(self):
        if self.t0 >= self.total_steps - 1:
            return

        if self.n_agents == 1:
            phi_for_plot = self.last_phi_vec
        else:
            phi_for_plot = sum(self.last_phi_vecs) / float(self.n_agents)
        current_phi_vec_norm = phi_for_plot / phi_for_plot.sum()

        # for a in range(self.n_agents):
        #     print(f"[DEBUG] t={self.t0} agent {a} GP X points= {self.gps[a].X.shape if self.gps[a].X is not None else None } y points= {self.gps[a].y.shape if self.gps[a].y is not None else None }")

        # ---- EPISODE START: compute destinations via greedy oracle
        if (self.t0 % self.episode_len) == 0:
            # Build the "UCB reward map" for the oracle.
            # Paper assumes a common w_UCB(e). Simplest: use average across agents.
            w_list = [self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0) for a in range(self.n_agents)]
            w_ucb = np.mean(np.stack(w_list, axis=0), axis=0)
            # idx_max = int(np.argmax(w_ucb))
            # idx_min = int(np.argmin(w_ucb))
            # rc_max = self._idx_to_rc[idx_max]
            # rc_min = self._idx_to_rc[idx_min]
            # print(f"[DEBUG] t={self.t0} w_ucb min={w_ucb[idx_min]:.6f} at {rc_min}, max={w_ucb[idx_max]:.6f} at {rc_max}")
            # print(f"[DEBUG] top10 idx: {np.argsort(w_ucb)[-10:][::-1]}")

            # Greedy oracle destinations
            self.dest_rc_agents = self._greedy_oracle_destinations(w_ucb, radius=self.oracle_radius)
            # print(f"[DEBUG] t={self.t0} destinations: {self.dest_rc_agents}")
            # print(f"[DEBUG] t={self.t0} starts: {self.start_rc_agents}")


        # ---- Choose actions
        reserved = set()
        chosen_next = []

        for a in range(self.n_agents):
            rc = self.start_rc_agents[a]
            dest_rc = self.dest_rc_agents[a]

            util = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0)
            # print(f"time {self.t0}: util.min() = {util.min()}, util.max() = {util.max()}")

            # DT transient controller *toward dest*
            nxt = self._best_first_move_to_dest(rc, util, dest_rc, reserved)

            chosen_next.append(nxt)
            reserved.add(self._rc_to_idx(nxt))

        # ---- Execute actions
        for a in range(self.n_agents):
            r, c = chosen_next[a]
            idx = self._rc_to_idx((r, c))

            # self.traj_idx_agents[a].append(idx)
            # self.traj_rc_agents[a].append((r, c))
            self.start_rc_agents[a] = (r, c)
            self.traj_states[self.t0 + 1, a] = idx
            self.traj_rc_time[self.t0 +1, a, :] = (r, c)

            # self.counts_total[idx] += 1
            # self.cr[idx] += 1
            # self.global_t += 1
            # env_t = self.t0  # current environment step

            # if self.first_visit_step[idx] == -1:
            #     self.first_visit_step[idx] = env_t + 1
            #     self.unvisited_count -= 1
            #     if self.unvisited_count == 0 and self.cover_time == -1:
            #         self.cover_time = env_t + 1

            # Observation
            cell_noise = float(self.env.noise_sigma[r, c])
            y_true = float(self.env.true_phi[r, c])

            noise = np.random.normal(0.0, cell_noise)
            y_noisy = y_true + noise
            k = 0.7
            proxy_var = float(np.clip((k * cell_noise) ** 2, 1e-6, 1e-1))

            self.gps[a].add_batch(
                r=np.array([r], dtype=float),
                c=np.array([c], dtype=float),
                t=np.array([self.t0], dtype=float),
                y=np.array([y_noisy], dtype=float),
                noise_var=np.array([proxy_var], dtype=float),
            )

        # r_hat = self.cr / self.cr.sum()

        # if self.env_dyn == "dynamic":
        #     self.emp_series.append(r_hat.copy())
        #     self.phi_norm_series.append(current_phi_vec_norm.copy())
        #     self.error_series.append(np.linalg.norm(r_hat - current_phi_vec_norm))
        # else:
        #     self.emp_series.append(r_hat.copy())
        #     self.phi_norm_series.append(current_phi_vec_norm.copy())
        #     self.error_series.append(np.linalg.norm(r_hat - current_phi_vec_norm))


        # phi_norm = phi


        # ---- GP update
        self.since_gp += 1
        if self.since_gp >= self.gp_update_every:
            print(f"[MAC-DT] Step {self.t0}: updating GPs")
            if self.n_agents == 1:
                    fitted = self.gp.fit_if_ready(min_pts=1)
                    if fitted:
                        self.last_phi_vec = self.gp_phi_logger.phi_vec(self.env, t=self.t0 - 1).copy()
                        self.gp_fit_steps.append(self.t0)
            else:
                any_fitted = False
                for a in range(self.n_agents):
                    fitted_a = self.gps[a].fit_if_ready(min_pts=1)
                    if fitted_a:
                        self.last_phi_vecs[a] = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0 - 1).copy()
                        self.phi_agents_time[self.t0 - 1, a, :] = self.last_phi_vecs[a]
                        any_fitted = True
                if any_fitted:
                    self.gp_fit_steps.append(self.t0 - 1)
                    # phi = sum(self.last_phi_vecs) / self.n_agents
            

            self.since_gp = 0


        # ---- Dynamic env 
        if self.env_dyn == "dynamic":
            while self.t0 >= self.next_map_change and self.current_cfg_idx < self.n_cfg - 1:
                desired_cfg_idx = min(self.next_map_change // self.steps_per_cfg, self.n_cfg - 1)
                if desired_cfg_idx != self.current_cfg_idx:
                    self.current_cfg_idx = desired_cfg_idx
                    self.env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
                    self.current_true_phi = self.env.build_map_vec(self.env.true_phi) 

                plot_grid(
                    self.env,
                    os.path.join(self.run_dir, f"true_cfg{self.current_cfg_idx}.png"),
                    title=f"True map cfg {self.current_cfg_idx}",
                )

                self.next_map_change += self.steps_per_cfg
                self.true_phi_map_time[self.t0:self.next_map_change, :] = self.current_true_phi

        self.belief_phi_map[self.t0] = current_phi_vec_norm
        self.t0 += 1


    def finalize(self):
        assert np.all(self.traj_states >= 0), "traj_states has unwritten entries (-1)."
        assert self.traj_rc_time.shape == (self.total_steps, self.n_agents, 2)
        phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
        pi_vec = phi_avg_vec / phi_avg_vec.sum()
        # emp = self.counts_total / max(1, self.counts_total.sum())

        # if self.n_agents == 1:
        #     pi = self.last_phi_vec / self.last_phi_vec.sum()
        #     traj_rc_all = np.asarray(self.traj_rc_agents[0], dtype=int)
        #     traj_idx_all = np.asarray(self.traj_idx_agents[0], dtype=int)
        #     traj_rc_agents_all = np.array(self.traj_rc_agents[0])[:, None, :]
        #     traj_idx_agents_all = np.array(self.traj_idx_agents[0])[:, None]
        # else:
        #     phi_avg = sum(self.last_phi_vecs) / self.n_agents
        #     pi = phi_avg / phi_avg.sum()
        #     traj_rc_agents_all = np.stack(
        #         [np.asarray(t, int) for t in self.traj_rc_agents], axis=1
        #     )
        #     traj_idx_agents_all = np.stack(
        #         [np.asarray(t, int) for t in self.traj_idx_agents], axis=1
        #     )
        #     traj_rc_all = traj_rc_agents_all.reshape(-1, 2)
        #     traj_idx_all = traj_idx_agents_all.reshape(-1)
        #     assert traj_rc_all.shape[0] == self.total_steps * self.n_agents, \
        #     f"Trajectory length mismatch: got {traj_rc_all.shape[0]}, expected {self.total_steps * self.n_agents}"

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




        save_dir = self.run_dir
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, f"run_{os.path.basename(save_dir)}.npz"),

            # --- metadata ---
            stem=save_dir,
            planner="mac_dt",
            mode=str(self.args.mode),
            env_dyn=str(self.env_dyn),
            env_source=str(getattr(self.args, "config", "NA")),
            state_index_map=self.env._state_index_map,
            T=self.total_steps,
            n_agents=self.n_agents,
            gp_update = int(self.gp_update_every),
            gp_mode=str(self.args.gp_mode),


            # ROI info
            roi_rc=np.array(roi_rc, dtype=int),
            roi_name=np.array(roi_name, dtype=object),
            
            # --- env ---
            nstates=np.array(self.env.nstates, dtype=int),
            grid_shape=self.grid_shape,
            true_phi_map=self.true_phi_map,
            true_phi_map_time=self.true_phi_map_time,


            # --- distributions ---
            # emp_final=emp,
            pi_vec_final=pi_vec,
            # emp_series=np.vstack(self.emp_series),
            # phi_norm_series=np.vstack(self.phi_norm_series),
            belief_phi_map=self.belief_phi_map,
            phi_agents_time=self.phi_agents_time,
            # error_series=np.array(self.error_series),


            # trajectory
            traj_states=self.traj_states,
            traj_rc_agents_all=self.traj_rc_time,

            # --- GP ---
            # gp_mode=str(self.args.gp_mode),
            gp_update_steps=np.array(self.gp_fit_steps, dtype=int),



        )

        print(f"[MAC-DT] Saved results to {save_dir}")
