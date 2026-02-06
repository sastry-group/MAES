
import os
import numpy as np
from .base import BasePlanner
from sim.runner import Simulator, Sensor, ObservationLogger
from info_maps.info_map_sources import UniformPhi, GPPhi
from info_maps.gp_model import GPInfoMap
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from viz.animation import animate_trajectory, animate_distributions, animate_distributions_multi
# from experiments.concensus import run_phi_consensus
from core.types import GridSpec
from env.grid import GridMC,  make_env_from_json
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import counts_to_field, plot_grid, plot_trajectory, plot_distributions, plot_convergence, vector_to_field, counts_to_field

class ErgodicPlanner(BasePlanner):
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.total_steps = args.T
        self.gp_update_every = args.update_GP_every
        self.p_update_every = args.update_P_every
        self.method = args.method
        self.env_dyn = args.env_dyn


    def _normalize(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        return v / (s + eps)


    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, config_path=None,stem=None, **kwargs):
        self.env = env
        self.run_dir = stem
        self.steps_per_cfg = steps_per_cfg
        self.cfg_paths = cfg_paths
        self.config_path = config_path
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        self.current_cfg_idx = 0
        self.next_map_change = self.steps_per_cfg

        self.t0 = 0
        self.since_gp = 0
        self.since_p = 0

        # 1) INITIAL phi
        phi_vec_uni = UniformPhi().phi_vec(self.env) * 6.0
        self.sim = Simulator(self.env, Sensor(sigma_map=self.env.noise_sigma), ObservationLogger())



        if self.n_agents > 1:
            # per-agent beliefs
            self.last_phi_vecs = [phi_vec_uni.copy() for _ in range(self.n_agents)]
            # per-agent transition matrices
            self.P_agents = []

            if self.method == "mh":
                for a in range(self.n_agents):
                    self.P_agents.append(build_transition_matrix_MH(self.env, self.last_phi_vecs[a]))
            else:
                for a in range(self.n_agents):
                    self.P_agents.append(build_transition_matrix_REMC(self.env, self.last_phi_vecs[a], max_iters=5000, verbose=False))

            self.gps = [GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
                for _ in range(self.n_agents)
                ]
            self.gp_phi_loggers = [
                GPPhi(
                    self.gps[a],
                    mode=self.args.gp_mode,
                    beta_m=self.args.ucb_beta_m,
                    beta_u=self.args.ucb_beta_u,
                    base=1e-3,
                )
                for a in range(self.n_agents)
            ]
            self.belief_phi_map = np.zeros((self.total_steps, phi_vec_uni.shape[0]), dtype=float)
            # initial average belief (across agents) for logging
            phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
            self.belief_phi_map[0, :] = phi_avg_vec.copy()
            self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
            self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)
            self.obs_buffer = [ [] for _ in range(self.n_agents) ]
            self.seen_obs = [set() for _ in range(self.n_agents)]

        else:   
            
            self.last_phi_vec = phi_vec_uni.copy()   
        
            if self.method == "mh":
                # might need to normalize and oass it
                # self._normalize(phi_vec_uni)
                self.P = build_transition_matrix_MH(self.env, phi_vec_uni)
            else:
                self.P = build_transition_matrix_REMC(self.env, self.last_phi_vec, max_iters=5000, verbose=False)

            self.gp = GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
            self.gp_phi_logger = GPPhi(
                self.gp,
                mode=self.args.gp_mode,
                beta_m=self.args.ucb_beta_m,
                beta_u=self.args.ucb_beta_u,
                base=1e-3
            )
            self.belief_phi_map = np.zeros((self.total_steps, self.last_phi_vec.shape[0]), dtype=float)
            self.belief_phi_map[0, :] = self.last_phi_vec.copy() # not normalized
            self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
            self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)



        # counters and loggers
        self.counts_total = np.zeros(self.env.nstates, dtype=int)
        self.counts_running = np.zeros(self.env.nstates, dtype=int) 
        self.traj_idx_all, self.traj_rc_all = [], []


        # for animation
        # self.emp_series, 
        self.phi_norm_series = []
        # self.cr = np.zeros(self.env.nstates, dtype=int)
        # self.dcr = np.zeros(self.env.nstates, dtype=float)
        # self.beta_discount = 1.0
        # self.global_t = 0

        # self.error_series = []
        # self.first_visit_step = np.full(self.env.nstates, -1, dtype=int)
        # self.unvisited_count = self.env.nstates
        # self.cover_time = -1

        if self.n_agents == 1:
            if len(agents_initial_positions) != 1:
                raise ValueError(f"n_agents=1 but --starts provides {len(agents_initial_positions)} starts")
            start_rc = agents_initial_positions[0]
            self.start_rc_agents = self.validate_start(start_rc)
        else:
            if len(agents_initial_positions) != self.n_agents:
                raise ValueError(f"--starts provides {len(agents_initial_positions)} starts but n_agents={self.n_agents}")
            start_rc_agents = agents_initial_positions
            self.start_rc_agents = [self.validate_start(rc) for rc in start_rc_agents]

        self.traj_rc_agents = [[] for _ in range(self.n_agents)]  # should work even if agents=1

        
        # saving 
        self.obs_r_all, self.obs_c_all, self.obs_t_all, self.obs_y_all = [], [], [], []
        self.gp_fit_steps = []         # step indices at which GP was (re)fit
        self.p_rebuild_steps = []      # step indices at which P was rebuilt
        self.P_snapshots = []          # store P at each rebuild

        # true map logging
        self.grid_shape = np.array(self.env.true_phi.shape, dtype=int)
        self.true_phi_map = getattr(self.env, "true_phi", None)
        if self.true_phi_map is not None:
            self.true_phi_map = self.true_phi_map.astype(float)


        init_true_vec = self.env.build_map_vec(self.env.true_phi)  # (nstates,)
        self.true_phi_map_time = np.zeros((self.total_steps, init_true_vec.shape[0]), dtype=float)
        self.true_phi_map_time[0:steps_per_cfg, :] = init_true_vec
        self.current_true_phi = init_true_vec




    def step(self):
        if self.t0 >= self.total_steps:
            return

        ### 1. epoch length 
        print(f"ErgodicPlanner step at t={self.t0} / {self.total_steps}")
        T_dt = min(
        self.total_steps - self.t0,
        self.gp_update_every - self.since_gp,
        self.p_update_every  - self.since_p)

        # Target distribution for logging
        if self.n_agents == 1:
            phi_for_plot = self.last_phi_vec
        else:
            phi_for_plot = sum(self.last_phi_vecs) / float(self.n_agents)
        current_phi_vec_norm = phi_for_plot / phi_for_plot.sum()

        

        # 2. Rollout policy for T_dt env steps
        if self.n_agents == 1:
            # current_phi_vec_norm = last_phi_vec / last_phi_vec.sum() # normalized
            traj_idx, traj_rc, counts = self.sim.simulate(self.P, T_dt, start=self.start_rc_agents)
            self.counts_total += counts
            self.counts_running += counts
            # print(f"[t={t0}], {counts_running}")
            self.traj_idx_all.append(traj_idx)
            self.traj_rc_all.append(traj_rc)

            # env steps for animation
            # for s in traj_idx:
            #     self._log_env_step(s, current_phi_vec_norm)

            for s in traj_idx:
                # self.global_t += 1

                # self.cr[s] += 1
                # Cesàro average (empirical)
                # r_hat = self.cr / self.global_t
                # self.dcr *= self.beta_discount          # decay old visits
                # self.dcr[s] += 1.0        # add current visit
                # r_hat_disc = self.dcr / self.dcr.sum()

                if self.env_dyn == "dynamic":
                    # self.emp_series.append(r_hat_disc.copy())
                    self.phi_norm_series.append(current_phi_vec_norm.copy())
                    # self.error_series.append(np.linalg.norm(r_hat_disc - current_phi_vec_norm))     
                else:               
                    # self.emp_series.append(r_hat.copy())
                    self.phi_norm_series.append(current_phi_vec_norm.copy())
                    # err = np.linalg.norm(r_hat - current_phi_vec_norm)
                    # self.error_series.append(err)

                # if self.first_visit_step[s] == -1:
                #     self.first_visit_step[s] = self.global_t
                #     self.unvisited_count -= 1
                #     if self.unvisited_count == 0 and self.cover_time == -1:
                #         self.cover_time = self.global_t
        
            # add observations to GP 
            ts = self.t0 + np.arange(len(traj_rc), dtype=float) # time
            rs = traj_rc[:, 0].astype(float) # row
            cs = traj_rc[:, 1].astype(float) # column

            cell_noise = self.env.noise_sigma[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            y_true = self.env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            noise   = np.random.normal(0.0, cell_noise)
            y_noisy = y_true + noise


            self.obs_r_all.append(rs)
            self.obs_c_all.append(cs)
            self.obs_t_all.append(ts)
            self.obs_y_all.append(y_noisy)

            k = 0.7
            proxy_var = np.clip((k * cell_noise)**2, 1e-6, 1e-1)
            self.gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=proxy_var)

            self.start_rc_agents = tuple(traj_rc[-1])
            
        

        else:
            # ---------- MULTI-AGENT BRANCH ----------
            # we keep one global GP and aggregate all agents' trajectories
            # phi_avg_vec = sum(last_phi_vecs) / float(n_agents)
            # current_phi_vec_norm = phi_avg_vec / phi_avg_vec.sum()

            # initialize per-step accumulators
            counts_step_total = np.zeros(self.env.nstates, dtype=int)
            traj_idx_batch_list = []
            traj_rc_batch_list = []
            # neighbor_obs = for every agent create a dictionary of neighords at every timestep

            local_obs = {}
            for a in range(self.n_agents):
                start_a = self.start_rc_agents[a]
                traj_idx_a, traj_rc_a, counts_a = self.sim.simulate(
                    self.P_agents[a], T_dt, start=start_a
                )
                self.start_rc_agents[a] = tuple(traj_rc_a[-1])
                self.traj_rc_agents[a].extend(traj_rc_a.tolist())

                traj_idx_batch_list.append(traj_idx_a)
                traj_rc_batch_list.append(traj_rc_a)
                counts_step_total += counts_a

                # local GP observations
                ts_a = self.t0 + np.arange(len(traj_rc_a), dtype=float)
                rs_a = traj_rc_a[:, 0].astype(float)
                cs_a = traj_rc_a[:, 1].astype(float)

                cell_noise_a = self.env.noise_sigma[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                y_true_a = self.env.true_phi[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                noise_a = np.random.normal(0.0, cell_noise_a)
                y_noisy_a = y_true_a + noise_a


                self.obs_r_all.append(rs_a)
                self.obs_c_all.append(cs_a)
                self.obs_t_all.append(ts_a)
                self.obs_y_all.append(y_noisy_a)

                k = 0.7
                proxy_var_a = np.clip((k * cell_noise_a) ** 2, 1e-6, 1e-1)

                local_obs[a] = {
                    "r": rs_a,
                    "c": cs_a,
                    "t": ts_a,
                    "y": y_noisy_a,
                    "noise": proxy_var_a,
                }

                # moved this below
                # self.gps[a].add_batch(
                #     r=rs_a, c=cs_a, t=ts_a, y=y_noisy_a, noise_var=proxy_var_a
                # )

            traj_idx = np.concatenate(traj_idx_batch_list, axis=0)
            traj_rc = np.vstack(traj_rc_batch_list)

            self.counts_total += counts_step_total
            self.counts_running += counts_step_total

            self.traj_idx_all.append(traj_idx)
            self.traj_rc_all.append(traj_rc)

            idx_stack = np.stack(traj_idx_batch_list, axis=1)   # each traj_idx_a is (T_dt,)

            for k in range(T_dt):

                positions_k = [traj_rc_batch_list[a][k] for a in range(self.n_agents)]
                neighbors = {
                    a: [
                        b for b in range(self.n_agents)
                        if np.hypot(
                            positions_k[a][0] - positions_k[b][0],
                            positions_k[a][1] - positions_k[b][1]
                        ) <= self.args.comm_radius
                    ]
                    for a in range(self.n_agents)
                }
                
                # self.global_t += 1
                for a in range(self.n_agents):
                    # print(f"Agent {a} at time {self.t0 + k} in cell {positions_k[a]} with neighbors {neighbors[a]}")
                    ra, ca = positions_k[a]
                    t = self.t0 + k
                    y = local_obs[a]["y"][k]
                    noise_var = local_obs[a]["noise"][k]

                    obs_id = (a, t)
                    if obs_id not in self.seen_obs[a]:
                        self.seen_obs[a].add(obs_id)
                        self.obs_buffer[a].append((obs_id, ra, ca, t, y, noise_var))
                    # for b in neighbors[a]:
                    #     if b == a:
                    #         continue
                    #     self.gps[b].add_batch(
                    #         r=np.array([ra]),
                    #         c=np.array([ca]),
                    #         t=np.array([t]),
                    #         y=np.array([y]),
                    #         noise_var=np.array([noise_var]),
                    #     )

                    
                    
                    
                    s = int(idx_stack[k, a]) # time adn agent
                    self.obs_count_agents[self.t0 + k, a, s] += 1

                    
                    # self.cr[s] += 1

                    # self.dcr *= self.beta_discount
                    # self.dcr[s] += 1.0

                    # if self.first_visit_step[s] == -1:
                    #     self.first_visit_step[s] = self.t0 + k
                    #     self.unvisited_count -= 1
                    #     if self.unvisited_count == 0 and self.cover_time == -1:
                    #         self.cover_time = self.t0 + k

                new_buffers = [ [] for _ in range(self.n_agents) ]

                for a in range(self.n_agents):
                    for b in neighbors[a] + [a]:
                        for pkt in self.obs_buffer[b]:
                            obs_id = pkt[0]
                            if obs_id not in self.seen_obs[a]:
                                self.seen_obs[a].add(obs_id)
                                new_buffers[a].append(pkt)
                self.obs_buffer = new_buffers  

                for a in range(self.n_agents):
                    for (_, r_o, c_o, t_o, y_o, nv_o) in self.obs_buffer[a]:
                        self.gps[a].add_batch(
                            r=np.array([r_o]),
                            c=np.array([c_o]),
                            t=np.array([t_o]),
                            y=np.array([y_o]),
                            noise_var=np.array([nv_o]),
                        )
                    self.obs_buffer[a].clear()
       

                # r_hat = self.cr / self.global_t
                # r_hat_disc = self.dcr / self.dcr.sum()


                if self.env_dyn == "dynamic":
                    # self.emp_series.append(r_hat_disc.copy())
                    self.phi_norm_series.append(current_phi_vec_norm.copy())
                    # self.error_series.append(np.linalg.norm(r_hat_disc - current_phi_vec_norm))
                else:
                    # self.emp_series.append(r_hat.copy())
                    self.phi_norm_series.append(current_phi_vec_norm.copy())
                    # self.error_series.append(np.linalg.norm(r_hat - current_phi_vec_norm))



    
        # ---------
        # TIME STEP
        # ---------
        t1 = self.t0 + T_dt
        self.since_gp += T_dt
        self.since_p += T_dt


        # keep average belief for logging (even in single-agent, phi_for_plot = last_phi_vec)        
        self.belief_phi_map[self.t0:t1, :] = phi_for_plot
        
        # ------------------
        # GP UPDATE + phi SYNC
        # ------------------
        if self.since_gp >= self.gp_update_every:
            if self.n_agents == 1:
                fitted = self.gp.fit_if_ready(min_pts=1)
                if fitted:
                    self.last_phi_vec = self.gp_phi_logger.phi_vec(self.env, t=t1 - 1).copy()
                    self.gp_fit_steps.append(t1 - 1)
                    # print(f"[t={t1}] GP refit done (single agent).")
            else:
                any_fitted = False
                for a in range(self.n_agents):
                    fitted_a = self.gps[a].fit_if_ready(min_pts=1)
                    if fitted_a:
                        self.last_phi_vecs[a] = self.gp_phi_loggers[a].phi_vec(self.env, t=t1 - 1).copy()
                        self.phi_agents_time[t1 - 1, a, :] = self.last_phi_vecs[a]
                        any_fitted = True
                if any_fitted:
                    self.gp_fit_steps.append(t1 - 1)
                    
                    # run_phi_consensus(self.args, self.last_phi_vecs, self.start_rc_agents)
                    # print(f"[t={t1}] GPs refit + phi consensus done for {n_agents} agents.")

            self.since_gp = 0

        # -------------
        # P UPDATE (REMC)
        # -------------
        if self.since_p >= self.p_update_every:
            if self.n_agents == 1:
                if self.method == "mh":
                    #  might need to normalize and pass it
                    pi_norm = self._normalize(self.last_phi_vec)
                    self.P = build_transition_matrix_MH(self.env, pi_norm)
                else:
                    self.P = build_transition_matrix_REMC(self.env, self.last_phi_vec, max_iters=10000, verbose=False)
                self.P_snapshots.append(self.P.copy())
            else:
                P_snap = []
                if self.method == "mh":
                    for a in range(self.n_agents):
                        pi_norm_a = self._normalize(self.last_phi_vecs[a])
                        P_a = build_transition_matrix_MH(self.env, pi_norm_a)
                        self.P_agents[a] = P_a
                        P_snap.append(P_a.copy())
                else:
                    for a in range(self.n_agents):
                        P_a = build_transition_matrix_REMC(self.env, self.last_phi_vecs[a], max_iters=10000, verbose=False)
                        self.P_agents[a] = P_a
                        P_snap.append(P_a.copy())
                self.P_snapshots.append(np.array(P_snap, dtype=object))

            self.p_rebuild_steps.append(t1 - 1)
            self.since_p = 0
    
        # ---------------------------
        # DYNAMIC ENVIRONMENT CHANGES
        # ---------------------------        
        if self.env_dyn == "dynamic":
            while t1 >= self.next_map_change and self.current_cfg_idx < self.n_cfg - 1:
                desired_cfg_idx = min(self.next_map_change // self.steps_per_cfg, self.n_cfg - 1)
                if desired_cfg_idx != self.current_cfg_idx:
                    self.current_cfg_idx = desired_cfg_idx
                    self.env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
                    self.current_true_phi = self.env.build_map_vec(self.env.true_phi) 

                    # new_env = make_env_from_json(self.cfg_paths[self.current_cfg_idx])
                    # self.env.true_phi = new_env.true_phi
                    # self.env.noise_sigma = new_env.noise_sigma  

                    # current_true_phi = self.env.build_map_vec(self.env.true_phi)
                    # self.sim.sensor.sigma_map = self.env.noise_sigma
                    # print(" step", os.path.join(self.run_dir, f"gp_grid_true_cfg{self.current_cfg_idx}.png"))

                    plot_grid(
                        self.env,
                        os.path.join(self.run_dir, f"gp_grid_true_cfg{self.current_cfg_idx}.png"),
                        title=f"Grid (true info map) cfg {self.current_cfg_idx}"
                    )
                    print(f"[t={t1}] Environment updated, loaded config {self.cfg_paths[self.current_cfg_idx]}")
                    # cr[:] = 0        # reset empirical counts
                    # global_t = 0     # reset time for metric
                
                    self.next_map_change += self.steps_per_cfg
                    self.true_phi_map_time[t1:self.next_map_change, :] = self.current_true_phi # not normalized

        self.t0 = t1



    def validate_start(self, rc):
        if rc not in self.env._state_index_map:
            raise ValueError(f"Invalid start {rc}: not a valid free cell (out of bounds or obstacle).")
        return rc


    def finalize(self):
        if self.n_agents > 1:
            for a in range(self.n_agents):
                assert len(self.traj_rc_agents[a]) == self.total_steps, f"Mismatch in recorded trajectory length for agent {a}"
            traj_rc_agents_all = np.stack([np.asarray(self.traj_rc_agents[a], dtype=int) for a in range(self.n_agents)], axis=1)

        # 3) Final evaluation/plots vs lastphi_hat(if no GP fit ever happened)
        # emp = self.counts_total / self.counts_total.sum()
        if self.n_agents == 1:
            pi_vec = self.last_phi_vec / self.last_phi_vec.sum()
        else:
            phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
            pi_vec = phi_avg_vec / phi_avg_vec.sum()


        traj_idx_all = np.concatenate(self.traj_idx_all, axis=0)
        traj_rc_all = np.vstack(self.traj_rc_all)

        # to keep track of ROI
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

        print("Periodic-GP run complete:", self.run_dir)
        # print(f"Final L2 error ||r_hat_T - pi|| = {self.error_series[-1]:.4f}")
        print(f"[GP] Saved plots + data under results/ with name: {self.run_dir}")



        if self.args.animation:
            if self.n_agents == 1:
                animate_distributions(
                    self.env,
                    # emp_series=self.emp_series,
                    pi_series=self.phi_norm_series,
                    filename=os.path.join(self.run_dir, f"anim_{os.path.basename(self.run_dir)}_emp_vs_phi_norm.mp4"),
                    fps=self.args.fps,
                    stride=self.args.stride,
                    dpi=150,
                    true_phi_map= self.true_phi_map_time,
                    traj_rc=traj_rc_all
            )
            else:
                traj_rc_agents_all = np.stack(
                    [np.asarray(self.traj_rc_agents[a], dtype=int) for a in range(self.n_agents)],
                    axis=1
                )  # (T, n_agents, 2)

                animate_distributions_multi(
                    self.env,
                    # emp_series=self.emp_series,
                    pi_series=self.phi_norm_series,
                    filename=os.path.join(self.run_dir, f"anim_{os.path.basename(self.run_dir)}_emp_vs_phi_norm.mp4"),
                    fps=self.args.fps,
                    stride=self.args.stride,
                    dpi=150,
                    true_phi_map=self.true_phi_map_time,
                    traj_rc_agents=traj_rc_agents_all
                )

        # flatten obs streams
        if len(self.obs_r_all) > 0:
            obs_r_all = np.concatenate(self.obs_r_all)
            obs_c_all = np.concatenate(self.obs_c_all)
            obs_t_all = np.concatenate(self.obs_t_all)
            obs_y_all = np.concatenate(self.obs_y_all)
        else:
            obs_r_all = np.zeros((0,), dtype=float)
            obs_c_all = np.zeros((0,), dtype=float)
            obs_t_all = np.zeros((0,), dtype=float)
            obs_y_all = np.zeros((0,), dtype=float)

        # P snapshots can vary in size/type; store as object array for safety
        P_snapshots_obj = np.array(self.P_snapshots, dtype=object)


        save_dir = self.run_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_dir, f"run_{os.path.basename(save_dir)}.npz"),
            # run metadata
            stem=self.run_dir,
            method=str(self.args.method),
            neighbors=int(self.args.neighbors),
            env_dyn=str(self.env_dyn),
            env_source=str(self.config_path),
            state_index_map=self.env._state_index_map,

            mode=str(self.args.mode),             
            T=int(self.total_steps),
            
            n_agents=int(self.n_agents),
            comm_mode=str(getattr(self.args, 'comm_mode', 'NA')),
            comm_radius=float(getattr(self.args, 'comm_radius', np.nan)),
            consensus_iters=int(getattr(self.args, 'consensus_iters', np.nan)),

            gp_mode=str(self.args.gp_mode),
            gp_update = int(self.gp_update_every),
            p_update = int(self.p_update_every),
            ucb_beta_m=float(self.args.ucb_beta_m),
            ucb_beta_u=float(self.args.ucb_beta_u),

            # # coverage tracking
            # first_visit_step=self.first_visit_step,  # (nstates,) first time each cell was visited - per cell discovery
            # cover_time=np.array(self.cover_time, dtype=int), # scalar, time when ALL cells were visited

            roi_rc=np.array(roi_rc, dtype=int), # (n_roi_cells, 2) (row, col) of ROI cells
            roi_name=np.array(roi_name, dtype=object),

            # env/meta
            nstates=np.array(self.env.nstates, dtype=int),
            grid_shape=self.grid_shape,
            true_phi_map=self.true_phi_map,
            # true_phi_vec=initial_true_phi,
            true_phi_map_time=self.true_phi_map_time,
            # final distributions
            # emp_final=emp,
            pi_vec_final=pi_vec,
            # time series
            # emp_series=np.vstack(self.emp_series),
            phi_norm_series=np.vstack(self.phi_norm_series),
            phi_agents_time=self.phi_agents_time,
            belief_phi_map=self.belief_phi_map,

            # trajectory
            traj_idx_all=traj_idx_all,
            traj_rc_all=traj_rc_all,
            traj_rc_agents_all=traj_rc_agents_all if self.n_agents > 1 else np.zeros((0,0,0), dtype=int),
            
            # observations stream
            obs_r=obs_r_all,
            obs_c=obs_c_all,
            obs_t=obs_t_all,
            obs_y=obs_y_all,
            obs_count_agents=self.obs_count_agents,
            # updates & policies
            gp_update_steps=np.array(self.gp_fit_steps, dtype=int),
            p_update_steps=np.array(self.p_rebuild_steps, dtype=int),
            P_snapshots=P_snapshots_obj
            
        )
        