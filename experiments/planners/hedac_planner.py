"""Mantovani et al., RA-L 2026 (distributed HEDAC ergodic coverage for
time-varying spatial processes), implemented on the region graph as a baseline.

Same belief (GP), sensing, observation sharing, h5 logging and env-change
timing as ErgodicPlanner. Only the motion rule changes: instead of sampling
from P_a, each agent climbs the HEDAC potential u_a, re-solved every step.

  every tau_P : GP refit -> mu, sigma -> g_a                   Eqs. (11)-(13)
  every step  : c~_a = c_a + sum_j w * c_j, normalized         Eqs. (14)-(16)
                s = max(g_a - c~_a, 0)^2                       Eqs. (17)-(18)
                (ell^2 L + I) u = s,  L = graph Laplacian      Eq. (19), Delta -> -L
                next = argmax_{N(z) U {z}} u                   discrete Eq. (3)

argmax u is invariant to scaling rho and beta together, so the heat equation
has a single knob: ell = sqrt(rho / beta), in cells (--hedac_ell).
Neumann BC is implicit: no edges to obstacles or outside the grid.
Collision avoidance (their potential field [26]) -> grid analogue: agents decide
in a random order each step, and an agent may not move into a region already
claimed this step, or currently occupied, by a neighbor in comm range (so agents
never swap through each other). Without it, two agents that meet
see near-identical c~ and g, pick the same argmax, and move together forever.
Dropped (grid-level planner): dynamics Eqs. (4)-(5), FoV,
sigma_in/sigma_out data filtering, online hyperparameter learning.
"""
import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .ergodic_planner import ErgodicPlanner
from env.grid import make_env_from_json
from viz.plotting import plot_grid
from info_maps.hedac_gp import HedacGP

class HEDACPlanner(ErgodicPlanner):

    def __init__(self, args):
        super().__init__(args)
        assert self.update_rule == "fixed", "HEDAC baseline: use a fixed --update_GP_every"
        assert self.n_agents > 1, "HEDAC baseline is implemented for n_agents > 1"
        self.ell = float(getattr(args, "hedac_ell", 1.0))
        self.gamma = float(getattr(args, "hedac_gamma", 0.0))
        self.diag_w = float(getattr(args, "hedac_diag_w", 0.5))
        self.w_kind = str(getattr(args, "hedac_w", "hard"))          # Eq. (16) w_r: "hard" | "gauss"
        self.rng_move = np.random.default_rng(int(getattr(args, "noise_seed", 0)) + 1)  # tie-breaking
        self.dbg_every = int(getattr(args, "hedac_debug_every", 0))     # 0 = off
        self.u_every = int(getattr(args, "hedac_save_u_every", 0))      # 0 = off
        self.gp_kind = getattr(args, "hedac_gp", "shared")
        self.block = bool(int(getattr(args, "hedac_block", 1)))

    # ------------------------------------------------------------------ setup
    def _build_graph(self):
        env = self.env
        if env._edge_src is None:
            env._indexing()
            env._precompute_graph()
        n = env.nstates
        src, dst = env._edge_src, env._edge_dst
        rc = np.array([env._index_state_map[i] for i in range(n)], dtype=float)

        w = np.where(np.abs(rc[src] - rc[dst]).sum(1) == 2, self.diag_w, 1.0)   # diagonal edges
        A = sp.csr_matrix((w, (src, dst)), shape=(n, n))
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A                            # L = -Delta
        self._solve = spla.factorized((self.ell ** 2 * L + sp.eye(n)).tocsc())    # Eq. (19), factor once

        self._rc = rc
        self._nbrs = np.split(dst, np.cumsum(env._deg)[:-1])                      # edges grouped by src
        self._Kg = None
        if self.gamma > 0:                                                        # Eq. (13)
            D2 = ((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)
            self._Kg = np.exp(-D2 / (2.0 * self.gamma ** 2))

    def reset(self, env, args, *a, **kw):
        super().reset(env, args, *a, **kw)
        if self.gp_kind == "hedac_gp":
            
            self.gps = [HedacGP.from_args(args) for _ in range(self.n_agents)]
            for a in range(self.n_agents):
                self.gp_phi_loggers[a].gp = self.gps[a]
            self.h5_file.attrs.update(self.gps[0].attrs())
        self.h5_file.attrs["hedac_gp"] = self.gp_kind
        self._build_graph()
        n = self.env.nstates
        self.cov = np.zeros((self.n_agents, n))            # c_a, own visits only, Eq. (14)
        self.n_cov = np.zeros(self.n_agents, dtype=int)
        self.g = np.full((self.n_agents, n), 1.0 / n)      # goal density before first GP fit
        self.solve_s = np.zeros(self.n_agents)             # HEDAC time since last GP update
        self.last_phi_vecs = [g.copy() for g in self.g]    # belief_phi_map logs g for this planner

        self.h5_file.attrs["planner"] = "hedac"
        self.h5_file.attrs["hedac_ell"] = self.ell
        self.h5_file.attrs["hedac_gamma"] = self.gamma
        self.h5_file.attrs["hedac_diag_w"] = self.diag_w
        self.h5_file.attrs["hedac_w"] = self.w_kind
        self.h5_file.attrs["compute_log_cols"] = "agent,t,gp_s,hedac_solve_s_since_last_gp"
        self.h5_file.attrs["hedac_block"] = int(self.block)
        # debug: u, s, c~ snapshots for inspection (off by default)
        self._dbg = {}
        if self.u_every > 0:
            n_snap = self.total_steps // self.u_every + 1
            kw = dict(shape=(n_snap, self.n_agents, n), dtype="f", fillvalue=np.nan,
                      chunks=(1, self.n_agents, n), compression="gzip", compression_opts=4)
            self.d_hedac_u = self.h5_file.create_dataset("hedac_u", **kw)
            self.d_hedac_s = self.h5_file.create_dataset("hedac_s", **kw)
            self.d_hedac_ct = self.h5_file.create_dataset("hedac_ctilde", **kw)
            self.d_hedac_u_t = self.h5_file.create_dataset("hedac_u_times", shape=(n_snap,), dtype="i", fillvalue=-1)

    # --------------------------------------------------------------- pieces
    def _goal(self, mu, sd):                                # Eqs. (11)-(13)
        mu = np.clip(mu, 0.0, None); mu = mu / (mu.max() + 1e-12)
        sd = np.clip(sd, 0.0, None); sd = sd / (sd.max() + 1e-12)
        F = np.expm1(mu) + np.expm1(sd)
        if self._Kg is not None:
            F = self._Kg @ F
        tot = F.sum()
        return F / tot if tot > 0 else np.full_like(F, 1.0 / F.size)

    def _comm_neighbors(self, pos, active_list):           # same rule as ErgodicPlanner.step
        R = self.args.comm_radius
        return {
            a: [b for b in active_list
                if np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) <= R]
            for a in active_list
        }

    def _hedac_next(self, a, s_idx, nbr_agents, blocked=()):
        d2 = ((self._rc - self._rc[s_idx]) ** 2).sum(1)
        R = self.args.comm_radius
        if self.w_kind == "gauss":                          # w_r, Eq. (16): smooth decay, sigma = r/2
            w = np.exp(-d2 / (2.0 * (R / 2.0) ** 2))
        else:                                               # w_r, Eq. (16): hard indicator
            w = (d2 <= R ** 2).astype(float)
        ct = self.cov[a].copy()
        for b in nbr_agents:
            ct += w * self.cov[b]
        tot = ct.sum()
        if tot > 0:
            ct /= tot                                       # unit mass, comparable to g
        e = self.g[a] - ct                                  # Eq. (17)
        src = np.where(e > 0.0, e * e, 0.0)                 # Eq. (18)
        u = self._solve(src)                                # Eq. (19)
        cand = np.concatenate(([s_idx], self._nbrs[s_idx]))
        if blocked:                                         # collision avoidance (grid analogue)
            keep = ~np.isin(cand, list(blocked))
            if keep.any():
                cand = cand[keep]
        uc = u[cand]
        best = cand[uc >= uc.max() - 1e-15]
        self._dbg[a] = (u, src, ct, cand, len(best), s_idx)
        return int(self.rng_move.choice(best))

    def _debug_log(self, t, active_list, pos, nxt):
        if self.u_every and t % self.u_every == 0:
            row = t // self.u_every
            for a in active_list:
                u, src, ct, _, _, _ = self._dbg[a]
                self.d_hedac_u[row, a] = u
                self.d_hedac_s[row, a] = src
                self.d_hedac_ct[row, a] = ct
            self.d_hedac_u_t[row] = t
        if self.dbg_every and t % self.dbg_every == 0:
            i2s = self.env._index_state_map
            for a in active_list:
                u, src, ct, cand, n_tie, here = self._dbg[a]
                du = u[cand] - u[here]                       # discrete gradient: candidate minus here
                move = "stay" if nxt[a] == here else f"-> {i2s[nxt[a]]}"
                print(f"[hedac t={t:5d} a={a}] at {pos[a]} {move}  "
                      f"max du={du.max():+.2e}  ties={n_tie}  blocked={len(cand) < 1 + len(self._nbrs[here])}  "
                      f"sum s={src.sum():.2e}  g@here={self.g[a, here]:.3f}  c~@here={ct[here]:.3f}")

    def _apply_attrition(self, t):
        if self.attrition and t in self.attrition_schedule:
            a = self.attrition_schedule[t]
            if a in self.active_agents:
                self.active_agents.discard(a)
                self.death_log.append((t, a))
                print(f"[t={t}] Agent {a} removed. Active: {sorted(self.active_agents)}")

    # ----------------------------------------------------------------- step
    def step(self):
        if self.t0 >= self.total_steps:
            return
        next_chg = (self.env_change_steps[self.current_change_idx]
                    if self.env_dyn == "dynamic"
                    and self.current_change_idx < len(self.env_change_steps)
                    else self.total_steps)
        T_dt = min(self.total_steps - self.t0,
                   self.gp_update_every - self.since_gp,
                   max(1, next_chg - self.t0))      # end chunk exactly at the env change
        s2i, i2s = self.env._state_index_map, self.env._index_state_map

        for k in range(T_dt):
            t = self.t0 + k
            self._apply_attrition(t)
            active_list = sorted(self._iter_agents())

            # 1) decide all moves from current positions and last-step coverage (synchronous)
            pos = {a: tuple(self.start_rc_agents[a]) for a in active_list}
            nbrs = self._comm_neighbors(pos, active_list)
            nxt = {}
            order = list(active_list)
            self.rng_move.shuffle(order)                    # random priority each step
            for a in order:
                t_s = time.perf_counter()
                others = [b for b in nbrs[a] if b != a]

                # blocked = {nxt[b] for b in others if b in nxt}   # regions claimed by neighbors this step
                # blocked |= {s2i[pos[b]] for b in others}          # regions occupied by neighbors (no swaps)
                if self.block:
                    blocked = {nxt[b] for b in others if b in nxt}   # neighbors pick distinct next regions
                else:
                    # print("no separation running")
                    blocked = set()                                  # no separation: plain argmax
                nxt[a] = self._hedac_next(a, s2i[pos[a]], others, blocked)
                self.solve_s[a] += time.perf_counter() - t_s
            self._debug_log(t, active_list, pos, nxt)

            # 2) move, observe, log (same bookkeeping as ErgodicPlanner)
            rows = np.array([i2s[nxt[a]][0] for a in active_list], dtype=int)
            cols = np.array([i2s[nxt[a]][1] for a in active_list], dtype=int)
            y, nv = self.sensor.observe(self.env, rows, cols)
            for i, a in enumerate(active_list):
                s, ra, ca = nxt[a], int(rows[i]), int(cols[i])
                self.start_rc_agents[a] = (ra, ca)
                self.d_traj_states[t, a] = s
                self.d_traj_rc[t, a] = (ra, ca)
                if self.attrition:
                    self.d_active_mask[t, a] = True
                self.d_obs_y[t, a] = y[i]
                obs_id = (a, t)
                if obs_id not in self.seen_obs[a]:
                    pkt = (obs_id, ra, ca, t, y[i], nv[i])
                    self.seen_obs[a].add(obs_id)
                    self.obs_buffer[a].append(pkt)
                    self.local_history[a].append(pkt)
                m = self.n_cov[a]                           # running mean, Eq. (14)
                self.cov[a] *= m / (m + 1.0)
                self.cov[a, s] += 1.0 / (m + 1.0)
                self.n_cov[a] += 1

            if self.attrition:
                for a in range(self.n_agents):
                    if a not in self.active_agents:
                        self.d_traj_states[t, a] = -1
                        self.d_traj_rc[t, a] = (-1, -1)
                        self.d_active_mask[t, a] = False

            # 3) share observations at the new positions
            pos = {a: tuple(self.start_rc_agents[a]) for a in active_list}
            self._share_and_feed(t, active_list, self._comm_neighbors(pos, active_list))

        t1 = self.t0 + T_dt
        self.since_gp += T_dt

        # GP refit -> new goal density, every tau_P (same cadence as ErgodicPlanner)
        if self.since_gp >= self.gp_update_every:
            t_update = t1 - 1
            self.gp_episode += 1
            for a in self._iter_agents():
                self.gps[a].z = self.start_rc_agents[a]
                t_s = time.perf_counter()
                self.gps[a].fit_if_ready(min_pts=1)
                self.gps[a].trim(max_pts=self.gp_max_obs)
                self.gp_phi_loggers[a].phi_vec(self.env, t=t_update, e=self.gp_episode)
                mu = self.gp_phi_loggers[a].last_mu.copy()
                sd = self.gp_phi_loggers[a].last_std.copy()
                self.g[a] = self._goal(mu, sd)
                gp_dt = time.perf_counter() - t_s

                self.last_mu_vecs[a] = mu
                self.last_std_vecs[a] = sd
                self.last_phi_vecs[a] = self.g[a].copy()
                g = self.gps[a]
                self.gp_timing.append((a, t_update, g.last_n, g.last_fit_s, g.last_predict_s))
                self.compute_log.append((a, t_update, gp_dt, self.solve_s[a]))
                self.solve_s[a] = 0.0
            self._log_belief_snapshot(t_update)
            self.since_gp = 0

        self._apply_env_changes(t1)
        self.t0 = t1

    # ------------------------------------------------- env change (as parent)
    def _apply_env_changes(self, t1):
        if self.env_dyn != "dynamic":
            return
        while (self.current_change_idx < len(self.env_change_steps)
               and t1 >= self.env_change_steps[self.current_change_idx]):
            self.current_cfg_idx += 1
            if self.current_cfg_idx >= len(self.cfg_paths):
                break
            n_old = self.env.nstates
            self.env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
            self.sim.env = self.env
            self.current_true_phi = self.env.build_map_vec(self.env.true_phi)
            assert self.env.nstates == n_old, "HEDAC baseline assumes obstacles do not change"

            plot_grid(self.env,
                      os.path.join(self.run_dir, f"gp_grid_true_cfg{self.current_cfg_idx}.png"),
                      title=f"Grid (true info map) cfg {self.current_cfg_idx}")
            print(f"[t={self.env_change_steps[self.current_change_idx]}] Environment updated, "
                  f"loaded config {self.cfg_paths[self.current_cfg_idx]}")

            if self.true_phi_row < self.d_true_phi.shape[0]:
                self.d_true_phi[self.true_phi_row] = self.current_true_phi
                self.d_true_phi_change_t[self.true_phi_row] = t1
                self.true_phi_row += 1
            self.current_change_idx += 1