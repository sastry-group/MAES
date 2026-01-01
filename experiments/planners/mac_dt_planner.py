# experiments/planners/mac_dt_planner.py
from __future__ import annotations
import numpy as np

from .base import BasePlanner
from env.grid import make_env_from_json
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from experiments.concensus import run_phi_consensus
from viz.plotting import plot_grid


class MACDTPlanner(BasePlanner):
    """

    """

    # =========================
    # Initialization
    # =========================
    def __init__(self, args):
        self.args = args
        self.n_agents = int(getattr(args, "n_agents", 1))
        self.total_steps = int(args.T)

        self.env_dyn = getattr(args, "env_dyn", "static")
        self.gp_update_every = int(getattr(args, "update_GP_every", 1))

        # Decision Tree parameters
        self.horizon = int(getattr(args, "dt_horizon", 6))
        self.gamma = float(getattr(args, "dt_gamma", 0.98))
        self.overlap_penalty = float(getattr(args, "dt_overlap_penalty", 0.0))

    # =========================
    # Grid helpers
    # =========================
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

    def _rc_to_idx(self, rc):
        return int(self.env._state_index_map[rc])

    # =========================
    # Decision Tree
    # =========================
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

    # =========================
    # Reset
    # =========================
    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, stem=None, **kwargs):
        self.env = env
        self.stem = stem
        self.steps_per_cfg = int(steps_per_cfg)
        self.cfg_paths = cfg_paths
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        self.current_cfg_idx = 0
        self.next_map_change = self.steps_per_cfg

        self.t0 = 0
        self.since_gp = 0

        # Initial belief
        phi0 = UniformPhi().phi_vec(self.env)

        # Agent positions
        if self.n_agents == 1:
            self.start_rc_agents = [agents_initial_positions[0]]
        else:
            self.start_rc_agents = list(agents_initial_positions)

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

        if self.n_agents == 1:
            self.last_phi_vec = phi0.copy()
        else:
            self.last_phi_vecs = [phi0.copy() for _ in range(self.n_agents)]

        # Logging
        self.counts_total = np.zeros(self.env.nstates, dtype=int)
        self.traj_rc_agents = [[] for _ in range(self.n_agents)]

        self.emp_series = []
        self.phi_norm_series = []
        self.error_series = []

        self.cr = np.zeros(self.env.nstates, dtype=int)
        self.global_t = 0

        # True map logging
        init_true = self.env.build_map_vec(self.env.true_phi)
        self.true_phi_map_time = np.zeros((self.total_steps, init_true.size))
        self.true_phi_map_time[:self.steps_per_cfg] = init_true

    # =========================
    # Step (ONE env step)
    # =========================
    def step(self):
        if self.t0 >= self.total_steps:
            return

        # ---- Dynamic env (not using for now)
        if self.env_dyn == "dynamic":
            while self.t0 >= self.next_map_change and self.current_cfg_idx < self.n_cfg - 1:
                self.current_cfg_idx += 1
                new_env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
                self.env.true_phi = new_env.true_phi
                self.env.noise_sigma = new_env.noise_sigma

                true_vec = self.env.build_map_vec(self.env.true_phi)
                end = min(self.total_steps, self.next_map_change + self.steps_per_cfg)
                self.true_phi_map_time[self.t0:end] = true_vec

                plot_grid(
                    self.env,
                    f"results/{self.stem}_true_cfg{self.current_cfg_idx}.png",
                    title=f"True map cfg {self.current_cfg_idx}",
                )

                self.next_map_change += self.steps_per_cfg

        # ---- Choose actions
        reserved = set()
        chosen_next = []

        for a in range(self.n_agents):
            rc = self.start_rc_agents[a]
            util = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0)
            nxt = self._best_first_move(rc, util, reserved)
            chosen_next.append(nxt)
            reserved.add(self._rc_to_idx(nxt))

        # ---- Execute actions
        for a in range(self.n_agents):
            r, c = chosen_next[a]
            idx = self._rc_to_idx((r, c))

            self.traj_rc_agents[a].append((r, c))
            self.start_rc_agents[a] = (r, c)

            self.counts_total[idx] += 1
            self.cr[idx] += 1
            self.global_t += 1

            r_hat = self.cr / self.global_t

            if self.n_agents == 1:
                phi = self.last_phi_vec
            else:
                phi = sum(self.last_phi_vecs) / self.n_agents

            phi_norm = phi / phi.sum()
            self.emp_series.append(r_hat.copy())
            self.phi_norm_series.append(phi_norm.copy())
            self.error_series.append(float(np.linalg.norm(r_hat - phi_norm)))

            # Observation
            sigma = self.env.noise_sigma[r, c]
            y_true = self.env.true_phi[r, c]
            y_noisy = y_true + np.random.normal(0.0, sigma)

            self.gps[a].add_batch(
                r=np.array([r], float),
                c=np.array([c], float),
                t=np.array([self.t0], float),
                y=np.array([y_noisy], float),
                noise_var=np.array([(0.7 * sigma) ** 2], float),
            )

        # ---- GP update
        self.since_gp += 1
        if self.since_gp >= self.gp_update_every:
            for a in range(self.n_agents):
                if self.gps[a].fit_if_ready(min_pts=1):
                    phi = self.gp_phi_loggers[a].phi_vec(self.env, t=self.t0)
                    if self.n_agents == 1:
                        self.last_phi_vec = phi
                    else:
                        self.last_phi_vecs[a] = phi

            if self.n_agents > 1 and getattr(self.args, "consensus_iters", 0):
                run_phi_consensus(self.args, self.last_phi_vecs, self.start_rc_agents)

            self.since_gp = 0

        self.t0 += 1

    # =========================
    # Finalize
    # =========================
    def finalize(self):
        emp = self.counts_total / max(1, self.counts_total.sum())
        if self.n_agents == 1:
            pi = self.last_phi_vec / self.last_phi_vec.sum()
        else:
            phi_avg = sum(self.last_phi_vecs) / self.n_agents
            pi = phi_avg / phi_avg.sum()

        return {
            "emp_final": emp,
            "pi_vec_final": pi,
            "error_series": np.array(self.error_series),
            "traj_rc_agents": self.traj_rc_agents,
        }