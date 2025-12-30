# experiments/planners/mac_dt_planner.py
from __future__ import annotations

import numpy as np

from .base import BasePlanner
from env.grid import make_env_from_json
from sim.runner import Simulator, Sensor, ObservationLogger
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from experiments.concensus import run_phi_consensus
from viz.plotting import plot_grid


class MACDTPlanner(BasePlanner):
    """
    MAC-DT = receding-horizon planning with a decision tree.
    Here: one action per step() call (per agent).
    Belief comes from GP (GPPhi produces a utility per cell, optionally UCB).
    """

    def __init__(self, args):
        self.args = args
        self.n_agents = getattr(args, "n_agents", 1)
        self.total_steps = int(args.T)

        self.env_dyn = getattr(args, "env_dyn", "static")
        self.gp_update_every = int(getattr(args, "update_GP_every", 1))
        self.p_update_every = int(getattr(args, "update_P_every", 1))  # not used in MAC-DT, but keep counters consistent

        # DT parameters (pick defaults if not in args yet)
        self.horizon = int(getattr(args, "dt_horizon", 6))
        self.gamma = float(getattr(args, "dt_gamma", 0.98))
        self.overlap_penalty = float(getattr(args, "dt_overlap_penalty", 0.0))  # set >0 to discourage overlap

    # -----------------------
    # Helpers (grid neighbors)
    # -----------------------
    def _neighbors(self, rc: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Use env._state_index_map to validate free cells (works regardless of obstacle encoding).
        Includes 'stay' as an action.
        """
        r, c = rc
        neighbors_mode = getattr(getattr(self.env, "spec", None), "neighbors", "4")
        if str(neighbors_mode) == "8":
            deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        else:
            deltas = [(-1,0),(1,0),(0,-1),(0,1)]

        out = [rc]  # allow stay
        for dr, dc in deltas:
            nxt = (r + dr, c + dc)
            if nxt in self.env._state_index_map:
                out.append(nxt)
        return out

    def _rc_to_idx(self, rc: tuple[int, int]) -> int:
        return int(self.env._state_index_map[rc])

    # -----------------------
    # DT planning
    # -----------------------
    def _best_first_move(self, start_rc: tuple[int, int], util_vec: np.ndarray, reserved: set[int] | None = None) -> tuple[int, int]:
        """
        Enumerate all paths of length self.horizon (small horizons only).
        Score = discounted sum util(state) - overlap_penalty * overlap_count.
        Return the first move of best-scoring path.
        """
        if reserved is None:
            reserved = set()

        best_score = -1e18
        best_first = start_rc

        # DFS stack: (depth, current_rc, score_so_far, first_move_rc, visited_idxs)
        start_idx = self._rc_to_idx(start_rc)
        stack = [(0, start_rc, 0.0, start_rc, {start_idx})]

        while stack:
            depth, cur_rc, score, first_move, visited = stack.pop()

            if depth == self.horizon:
                if score > best_score:
                    best_score = score
                    best_first = first_move
                continue

            # expand
            for nxt_rc in self._neighbors(cur_rc):
                nxt_idx = self._rc_to_idx(nxt_rc)
                step_util = float(util_vec[nxt_idx])

                # overlap penalty (very simple): penalize if agent goes somewhere already "reserved"
                ov = 1.0 if (nxt_idx in reserved) else 0.0
                step_score = (self.gamma ** depth) * (step_util - self.overlap_penalty * ov)

                nxt_first = nxt_rc if depth == 0 else first_move

                # (Optional) avoid cycles; you can relax this if you want revisits
                if nxt_idx in visited:
                    continue

                new_visited = set(visited)
                new_visited.add(nxt_idx)
                stack.append((depth + 1, nxt_rc, score + step_score, nxt_first, new_visited))

        return best_first

    # -----------------------
    # Core lifecycle
    # -----------------------
    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, stem=None):
        self.env = env
        self.stem = stem
        self.steps_per_cfg = int(steps_per_cfg)
        self.cfg_paths = cfg_paths
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        self.current_cfg_idx = 0
        self.next_map_change = self.steps_per_cfg

        self.t0 = 0
        self.since_gp = 0
        self.since_p = 0  # keep for symmetry even if MAC-DT doesn't rebuild P

        # sim + belief init
        phi_vec_uni = UniformPhi().phi_vec(self.env)
        self.sim = Simulator(self.env, Sensor(sigma_map=self.env.noise_sigma), ObservationLogger())

        # starts
        if self.n_agents == 1:
            self.start_rc = agents_initial_positions[0] if isinstance(agents_initial_positions, list) else agents_initial_positions
        else:
            self.start_rc_agents = agents_initial_positions

        # Per-agent GP (simplest)
        self.gps = [GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12) for _ in range(self.n_agents)]
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

        # Track last phi vecs for plotting/logging
        if self.n_agents == 1:
            self.last_phi_vec = phi_vec_uni.copy()
        else:
            self.last_phi_vecs = [phi_vec_uni.copy() for _ in range(self.n_agents)]

        # logging arrays
        self.counts_total = np.zeros(self.env.nstates, dtype=int)
        self.counts_running = np.zeros(self.env.nstates, dtype=int)
        self.traj_rc_agents = [[] for _ in range(self.n_agents)]
        self.traj_rc_all = []

        self.emp_series, self.phi_norm_series, self.error_series = [], [], []
        self.cr = np.zeros(self.env.nstates, dtype=int)
        self.dcr = np.zeros(self.env.nstates, dtype=float)
        self.beta_discount = 0.999
        self.global_t = 0

        self.first_hit_step = np.full(self.env.nstates, -1, dtype=int)
        self.unseen = self.env.nstates
        self.cover_time = -1

        # obs logs
        self.obs_r_all, self.obs_c_all, self.obs_t_all, self.obs_y_all = [], [], [], []
        self.gp_fit_steps = []

        # true map time-series (for plots/animations)
        init_true_vec = self.env.build_map_vec(self.env.true_phi)
        self.true_phi_map_time = np.zeros((self.total_steps, init_true_vec.shape[0]), dtype=float)
        self.true_phi_map_time[0:self.steps_per_cfg, :] = init_true_vec
        self.current_true_phi = init_true_vec

    def step(self, t: int):
        if self.t0 >= self.total_steps:
            return

        # -----------------------
        # (A) Dynamic env update - unsure maybe later
        # -----------------------
        if self.env_dyn == "dynamic":
            while self.t0 >= self.next_map_change and self.current_cfg_idx < self.n_cfg - 1:
                desired_cfg_idx = min(self.next_map_change // self.steps_per_cfg, self.n_cfg - 1)
                if desired_cfg_idx != self.current_cfg_idx:
                    self.current_cfg_idx = desired_cfg_idx

                    # Update ONLY true_phi + noise_sigma (keep geometry fixed)
                    new_env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
                    self.env.true_phi = new_env.true_phi
                    self.env.noise_sigma = new_env.noise_sigma
                    self.sim.sensor.sigma_map = self.env.noise_sigma

                    self.current_true_phi = self.env.build_map_vec(self.env.true_phi)
                    end = min(self.total_steps, self.next_map_change + self.steps_per_cfg)
                    self.true_phi_map_time[self.t0:end, :] = self.current_true_phi

                    plot_grid(
                        self.env,
                        f"results/{self.stem}_gp_grid_true_cfg{self.current_cfg_idx}.png",
                        title=f"Grid (true info map) cfg {self.current_cfg_idx}",
                    )

                    self.next_map_change += self.steps_per_cfg

        # -----------------------
        # (B) Choose actions (DT)
        # -----------------------
        reserved: set[int] = set()
        chosen_next = []

        for a in range(self.n_agents):
            cur_rc = self.start_rc if self.n_agents == 1 else self.start_rc_agents[a]

            # use latest belief utility vector
            util_vec = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0).copy()

            nxt_rc = self._best_first_move(cur_rc, util_vec, reserved=reserved)
            chosen_next.append(nxt_rc)

            reserved.add(self._rc_to_idx(nxt_rc))

        # -----------------------
        # (C) Execute 1 step + observe + log
        # -----------------------
        for a in range(self.n_agents):
            rc = chosen_next[a]
            idx = self._rc_to_idx(rc)

            # update trajectories
            self.traj_rc_agents[a].append(rc)
            self.traj_rc_all.append(rc)

            # counts
            self.counts_total[idx] += 1
            self.counts_running[idx] += 1

            # empirical distro logs (match your style)
            self.global_t += 1
            self.cr[idx] += 1
            r_hat = self.cr / self.global_t

            self.dcr *= self.beta_discount
            self.dcr[idx] += 1.0
            r_hat_disc = self.dcr / self.dcr.sum()

            # phi_for_plot = avg belief if multiagent
            if self.n_agents == 1:
                phi_for_plot = self.last_phi_vec
            else:
                phi_for_plot = sum(self.last_phi_vecs) / float(self.n_agents)

            phi_norm = phi_for_plot / phi_for_plot.sum()

            if self.env_dyn == "dynamic":
                self.emp_series.append(r_hat_disc.copy())
                self.phi_norm_series.append(phi_norm.copy())
                self.error_series.append(float(np.linalg.norm(r_hat_disc - phi_norm)))
            else:
                self.emp_series.append(r_hat.copy())
                self.phi_norm_series.append(phi_norm.copy())
                self.error_series.append(float(np.linalg.norm(r_hat - phi_norm)))

            # coverage tracking
            if self.first_hit_step[idx] == -1:
                self.first_hit_step[idx] = self.global_t
                self.unseen -= 1
                if self.unseen == 0 and self.cover_time == -1:
                    self.cover_time = self.global_t

            # observation at rc
            r, c = rc
            ts = float(self.t0)

            sigma = float(self.env.noise_sigma[r, c])
            y_true = float(self.env.true_phi[r, c])
            y_noisy = y_true + float(np.random.normal(0.0, sigma))

            # log obs
            self.obs_r_all.append(float(r))
            self.obs_c_all.append(float(c))
            self.obs_t_all.append(ts)
            self.obs_y_all.append(y_noisy)

            # GP update (per agent)
            k = 0.7
            proxy_var = float(np.clip((k * sigma) ** 2, 1e-6, 1e-1))
            self.gps[a].add_batch(
                r=np.array([float(r)]),
                c=np.array([float(c)]),
                t=np.array([ts]),
                y=np.array([y_noisy]),
                noise_var=np.array([proxy_var]),
            )

            # move agent
            if self.n_agents == 1:
                self.start_rc = rc
            else:
                self.start_rc_agents[a] = rc

        # -----------------------
        # (D) Periodic GP refit + belief sync
        # -----------------------
        self.since_gp += 1
        self.since_p += 1  # kept for symmetry

        if self.since_gp >= self.gp_update_every:
            any_fitted = False
            for a in range(self.n_agents):
                if self.gps[a].fit_if_ready(min_pts=1):
                    any_fitted = True
                    phi_vec = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0).copy()
                    if self.n_agents == 1:
                        self.last_phi_vec = phi_vec
                    else:
                        self.last_phi_vecs[a] = phi_vec

            if any_fitted:
                self.gp_fit_steps.append(self.t0)

                # optional consensus hook (only meaningful multi-agent)
                if self.n_agents > 1 and getattr(self.args, "consensus_iters", 0):
                    run_phi_consensus(self.args, self.last_phi_vecs, self.start_rc_agents)

            self.since_gp = 0

        # advance time by ONE step
        self.t0 += 1

    def finalize(self):
        emp = self.counts_total / max(1, self.counts_total.sum())

        if self.n_agents == 1:
            pi_vec = self.last_phi_vec / self.last_phi_vec.sum()
        else:
            phi_avg = sum(self.last_phi_vecs) / float(self.n_agents)
            pi_vec = phi_avg / phi_avg.sum()

        return {
            "emp_final": emp,
            "pi_vec_final": pi_vec,
            "error_series": np.array(self.error_series, dtype=float),
            "traj_rc_agents": self.traj_rc_agents,
            "gp_update_steps": np.array(self.gp_fit_steps, dtype=int),
        }
