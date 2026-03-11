
import os
import numpy as np
import time
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
        self.compute_log = []  


    def _normalize(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        return v / (s + eps)


    def reset(self, env, args, agents_initial_positions, steps_per_cfg, cfg_paths=None, config_path=None,stem=None, h5_file=None, **kwargs):
        self.env = env
        self.run_dir = stem
        self.h5_file = h5_file

        self.steps_per_cfg = steps_per_cfg
        self.cfg_paths = cfg_paths
        self.config_path = config_path
        self.n_cfg = len(cfg_paths) if cfg_paths else 1
        
        self.current_change_idx = 0
        self.current_cfg_idx = 0
        
        self.next_map_change = self.steps_per_cfg
        self.env_change_steps = sorted(getattr(args, "env_change_steps",  [i * self.steps_per_cfg for i in range(1, self.n_cfg)]))
        

        self.t0 = 0
        self.since_gp = 0
        self.since_p = 0
        self.second_largest_eigs = []  # for logging spectral gap of P over time


        #initial believe
        phi_vec_uni = UniformPhi().phi_vec(self.env) * 6 # Initial phi

        # initial sim
        self.sim = Simulator(self.env, Sensor(sigma_map=self.env.noise_sigma), ObservationLogger())



        if self.n_agents > 1:
            # per-agent beliefs
            self.last_phi_vecs = [phi_vec_uni.copy() for _ in range(self.n_agents)]
            # per-agent transition matrices
            self.P_agents = []

            if self.method == "mh":
                for a in range(self.n_agents):
                    P, second_largest_eig = build_transition_matrix_MH(self.env, self.last_phi_vecs[a])
                    self.P_agents.append(P)
                    self.second_largest_eigs.append(second_largest_eig)
            else:
                for a in range(self.n_agents):
                    P, second_largest_eig = build_transition_matrix_REMC(self.env, self.last_phi_vecs[a], max_iters=5000, verbose=False)
                    self.P_agents.append(P)
                    self.second_largest_eigs.append(second_largest_eig)

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
            # self.belief_phi_map = np.zeros((self.total_steps, phi_vec_uni.shape[0]), dtype=float)
            # initial average belief (across agents) for logging
            # phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
            # self.belief_phi_map[0, :] = phi_avg_vec.copy()
            # self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
            # self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)
            # self.obs_buffer = [ [] for _ in range(self.n_agents) ]
            # self.seen_obs = [set() for _ in range(self.n_agents)]

        else:   
            
            self.last_phi_vec = phi_vec_uni.copy()   
        
            if self.method == "mh":
                # might need to normalize and oass it
                # self._normalize(phi_vec_uni)
                self.P, self.second_largest_eig = build_transition_matrix_MH(self.env, phi_vec_uni)
            else:
                self.P, self.second_largest_eig = build_transition_matrix_REMC(self.env, self.last_phi_vec, max_iters=5000, verbose=False)

            self.gp = GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
            self.gp_phi_logger = GPPhi(
                self.gp,
                mode=self.args.gp_mode,
                beta_m=self.args.ucb_beta_m,
                beta_u=self.args.ucb_beta_u,
                base=1e-3
            )
            # self.belief_phi_map = np.zeros((self.total_steps, self.last_phi_vec.shape[0]), dtype=float)
            # self.belief_phi_map[0, :] = self.last_phi_vec.copy() # not normalized
            # self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
            # self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)
  
        # self.traj_states = np.full((self.total_steps, self.n_agents),-1, dtype=int)
        # self.traj_rc_time = np.zeros((self.total_steps, self.n_agents, 2), dtype=int)


        # self.phi_norm_series = []

        if len(agents_initial_positions) == 1:
            # One start provided → all agents start there
            start_rc = self.validate_start(agents_initial_positions[0])
            self.start_rc_agents = [start_rc for _ in range(self.n_agents)]

        elif len(agents_initial_positions) == self.n_agents:
            # One start per agent
            self.start_rc_agents = [self.validate_start(rc) for rc in agents_initial_positions]

        else:
            raise ValueError(
                f"--starts provides {len(agents_initial_positions)} starts but must be 1 or n_agents={self.n_agents}"
            )



        # # saving 
        # self.obs_r_all, self.obs_c_all, self.obs_t_all, self.obs_y_all = [], [], [], []
        # self.gp_fit_steps = []         # step indices at which GP was (re)fit
        # self.p_rebuild_steps = []      # step indices at which P was rebuilt
        # self.P_snapshots = []          # store P at each rebuild
        # for observation 
        self.obs_buffer = [[] for _ in range(self.n_agents)]
        self.seen_obs = [set() for _ in range(self.n_agents)]

        # true map logging
        self.grid_shape = np.array(self.env.true_phi.shape, dtype=int)
        self.true_phi_map = getattr(self.env, "true_phi", None)
        if self.true_phi_map is not None:
            self.true_phi_map = self.true_phi_map.astype(float)


        # true map init
        init_true_vec = self.env.build_map_vec(self.env.true_phi)  # (nstates,)
        # self.true_phi_map_time = np.zeros((self.total_steps, init_true_vec.shape[0]), dtype=float)
        # self.true_phi_map_time[0:steps_per_cfg, :] = init_true_vec
        self.current_true_phi = init_true_vec


        # Env datasets
        nstates = self.env.nstates
        T = self.total_steps
        N = self.n_agents



        self.d_traj_states = self.h5_file.create_dataset(
            "traj_states",
            shape=(T, N),
            dtype="i",
            compression="gzip", compression_opts=4
        )

        self.d_traj_rc = self.h5_file.create_dataset(
            "traj_rc_agents_all",
            shape=(T, N, 2),
            dtype="i",
            compression="gzip", compression_opts=4
        )

        self.d_belief_phi = self.h5_file.create_dataset(
            "belief_phi_map",
            shape=(T, N, nstates),   # per-agent belief
            dtype="f",
            compression="gzip", compression_opts=4
        )

        self.d_true_phi = self.h5_file.create_dataset(
            "true_phi_map_time",
            shape=(T, nstates),
            dtype="f",
            compression="gzip", compression_opts=4
        )


        self.d_belief_phi_avg = self.h5_file.create_dataset(
            "belief_phi_map_avg",
            shape=(T, nstates),
            dtype="f",
            compression="gzip", compression_opts=4
        )

        self.d_second_largest_eig = self.h5_file.create_dataset(
            "second_largest_eig",
            shape=(T, N),   # per-agent second largest eigenvalue of P
            dtype="f",
            compression="gzip", compression_opts=4
        )



    def step(self):
        if self.t0 >= self.total_steps:
            return

        ### 1. epoch length 
        print(f"ErgodicPlanner step at t={self.t0} / {self.total_steps}")
        T_dt = min(
        self.total_steps - self.t0,
        self.gp_update_every - self.since_gp,
        self.p_update_every  - self.since_p)
        # print(f"t0={self.t0}, T_dt={T_dt}, t1={self.t0+T_dt}, since_gp={self.since_gp}, since_p={self.since_p}")



        # Target distribution for logging
        # if self.n_agents == 1:
        #     phi_for_plot = self.last_phi_vec
        # else:
        #     phi_for_plot = sum(self.last_phi_vecs) / float(self.n_agents)
        # current_phi_vec_norm = phi_for_plot / phi_for_plot.sum()

        

        
        if self.n_agents == 1:
            # current_phi_vec_norm = last_phi_vec / last_phi_vec.sum() # normalized
            traj_idx, traj_rc, _ = self.sim.simulate(self.P, T_dt, start=self.start_rc_agents)

            for k in range(T_dt):
                t = self.t0 + k
                s = int(traj_idx[k])
                ra, ca = map(int, traj_rc[k])

                # self.traj_states[t, 0] = s
                # self.traj_rc_time[t, 0, :] = traj_rc[k]
                self.d_traj_states[t, 0] = s
                self.d_traj_rc[t, 0] = (ra, ca)
                # self.phi_norm_series.append(current_phi_vec_norm.copy())
                self.d_belief_phi[t, 0] = self.last_phi_vec
                self.d_belief_phi_avg[t] = self.last_phi_vec
                self.d_true_phi[t] = self.current_true_phi
        
            # add observations to GP 
            ts = self.t0 + np.arange(len(traj_rc), dtype=float) # time
            rs = traj_rc[:, 0].astype(float) # row
            cs = traj_rc[:, 1].astype(float) # column

            cell_noise = self.env.noise_sigma[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            y_true = self.env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            noise   = np.random.normal(0.0, cell_noise)
            y_noisy = y_true + noise
            # print(f"adding obs to GP: t={ts}, r={rs}, c={cs}, y_true={y_true}, noise={noise}, y_noisy={y_noisy}")


            # self.obs_r_all.append(rs)
            # self.obs_c_all.append(cs)
            # self.obs_t_all.append(ts)
            # self.obs_y_all.append(y_noisy)

            k = 0.7
            proxy_var = np.clip((k * cell_noise)**2, 1e-6, 1e-1)
            self.gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=proxy_var)

            self.start_rc_agents = tuple(traj_rc[-1])
            
        

        else:
            # ---------- MULTI-AGENT BRANCH ----------

            traj_idx_batch = []
            traj_rc_batch = []

            local_obs = {}
            for a in range(self.n_agents):
                start_a = self.start_rc_agents[a]
                traj_idx_a, traj_rc_a, counts_a = self.sim.simulate(
                    self.P_agents[a], T_dt, start=start_a
                )
                self.start_rc_agents[a] = tuple(traj_rc_a[-1])


                traj_idx_batch.append(traj_idx_a)
                traj_rc_batch.append(traj_rc_a)
                # counts_step_total += counts_a

                # local GP observations
                ts_a = self.t0 + np.arange(len(traj_rc_a), dtype=float)
                rs_a = traj_rc_a[:, 0].astype(float)
                cs_a = traj_rc_a[:, 1].astype(float)

                cell_noise_a = self.env.noise_sigma[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                y_true_a = self.env.true_phi[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                noise_a = np.random.normal(0.0, cell_noise_a)
                y_noisy_a = y_true_a + noise_a


                # self.obs_r_all.append(rs_a)
                # self.obs_c_all.append(cs_a)
                # self.obs_t_all.append(ts_a)
                # self.obs_y_all.append(y_noisy_a)

                k = 0.7
                proxy_var_a = np.clip((k * cell_noise_a) ** 2, 1e-6, 1e-1)

                local_obs[a] = {
                    "r": rs_a,
                    "c": cs_a,
                    "t": ts_a,
                    "y": y_noisy_a,
                    "noise": proxy_var_a,
                }


            idx_stack = np.stack(traj_idx_batch, axis=1)   # each traj_idx_a is (T_dt,)

            for k in range(T_dt):
                t = self.t0 + k
                # self.phi_norm_series.append(current_phi_vec_norm.copy())

                positions_k = [traj_rc_batch[a][k] for a in range(self.n_agents)]
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
                

                for a in range(self.n_agents):
                    s = int(idx_stack[k, a]) # time and state index for agent a at step k
                    ra, ca = map(int, positions_k[a])
                    self.d_traj_states[t, a] = s
                    self.d_traj_rc[t, a] = (ra, ca)

                    y = local_obs[a]["y"][k]
                    noise_var = local_obs[a]["noise"][k]
                    obs_id = (a, t)

                    if obs_id not in self.seen_obs[a]:
                        self.seen_obs[a].add(obs_id)
                        self.obs_buffer[a].append((obs_id, ra, ca, t, y, noise_var))
     
                    
                    self.d_belief_phi[t, a] = self.last_phi_vecs[a]
                self.d_true_phi[t] = self.current_true_phi     
                phi_avg = np.mean(self.last_phi_vecs, axis=0)
                self.d_belief_phi_avg[t] = phi_avg

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



    

        # TIME STEP
        t1 = self.t0 + T_dt
        self.since_gp += T_dt
        self.since_p += T_dt     

        # keep average belief for logging (even in single-agent, phi_for_plot = last_phi_vec)   
        # self.belief_phi_map[self.t0:t1, :] = phi_for_plot
        # self.d_belief_phi[t] = phi_for_plot

        # GP UPDATE + phi SYNC
        if self.since_gp >= self.gp_update_every:
            t_update = t1 - 1
            if self.n_agents == 1:
                t0_track = time.perf_counter()
                self.gp.fit_if_ready(min_pts=1)
                t1_track = time.perf_counter()
                # print(
                #     f"phi updated at t={t1-1}, "
                #     f"will be visible from t={t1}"
                # )
                gp_dt = t1_track - t0_track
                self.compute_log.append(
                    (0, t_update, gp_dt,0.0)
                )
                self.last_phi_vec = self.gp_phi_logger.phi_vec(self.env, t=t1 - 1).copy()
                # phi_norm = self.last_phi_vec / self.last_phi_vec.sum()
                # self.phi_norm_series[-1] = phi_norm

                # self.gp_fit_steps.append(t1 - 1)

                            # print(f"[t={t1}] GP refit done (single agent).")
            else:

                for a in range(self.n_agents):
                    t0_track = time.perf_counter()
                    self.gps[a].fit_if_ready(min_pts=1)
                    t1_track = time.perf_counter()
                    gp_dt = t1_track - t0_track
                    self.compute_log.append(
                        (a, t_update, gp_dt, 0.0)
                    )
                    self.last_phi_vecs[a] = self.gp_phi_loggers[a].phi_vec(self.env, t=t1 - 1).copy()
                    # self.phi_agents_time[t1 - 1, a, :] = self.last_phi_vecs[a]
                # self.gp_fit_steps.append(t1 - 1)

            self.since_gp = 0



        # P UPDATE (REMC)
        if self.since_p >= self.p_update_every:
            t_update = t1 - 1
            if self.n_agents == 1:
                if self.method == "mh":
                    #  might need to normalize and pass it
                    pi_norm = self._normalize(self.last_phi_vec)
                    self.P, self.second_largest_eig = build_transition_matrix_MH(self.env, pi_norm)
                else:
                    t0_track = time.perf_counter()
                    self.P, self.second_largest_eig = build_transition_matrix_REMC(self.env, self.last_phi_vec, max_iters=5000, verbose=False)
                    t1_track = time.perf_counter()
                    p_dt = t1_track - t0_track
                    self.compute_log.append(
                        (0, t_update, 0.0, p_dt)
                    )
                # self.P_snapshots.append(self.P.copy())
            else:
                P_snap = []
                if self.method == "mh":
                    for a in range(self.n_agents):
                        pi_norm_a = self._normalize(self.last_phi_vecs[a])
                        P_a, second_largest_eig_a = build_transition_matrix_MH(self.env, pi_norm_a)
                        self.P_agents[a] = P_a
                        self.second_largest_eigs[a] = second_largest_eig_a
                        P_snap.append(P_a.copy())
                else:
                    for a in range(self.n_agents):
                        t0_track = time.perf_counter()
                        P_a, second_largest_eig_a = build_transition_matrix_REMC(self.env, self.last_phi_vecs[a], max_iters=5000, verbose=False)
                        t1_track = time.perf_counter()
                        p_dt = t1_track - t0_track
                        self.compute_log.append(
                            (a, t_update, 0.0, p_dt)
                        )
                        self.P_agents[a] = P_a
                        self.second_largest_eigs[a] = second_largest_eig_a
                        P_snap.append(P_a.copy())
                # self.P_snapshots.append(np.array(P_snap, dtype=object))

            # self.p_rebuild_steps.append(t1 - 1)
            self.since_p = 0
    

        # DYNAMIC ENVIRONMENT CHANGES      
        if self.env_dyn == "dynamic":
            while (
                self.current_change_idx < len(self.env_change_steps)
                and t1 >= self.env_change_steps[self.current_change_idx]
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

                # Determine next boundary
                # if self.current_change_idx + 1 < len(self.env_change_steps):
                #     next_boundary = self.env_change_steps[self.current_change_idx + 1]
                # else:
                #     next_boundary = self.total_steps

                # Log true map for this interval
                

                self.current_change_idx += 1

            # assert len(self.phi_norm_series) == t1, (len(self.phi_norm_series), t1)
        # self.true_phi_map_time[self.t0:t1, :] = self.current_true_phi
        # self.d_true_phi[t] = self.current_true_phi
        # assert np.all(self.traj_states[self.t0:t1] >= 0)
        self.t0 = t1



    def validate_start(self, rc):
        if rc not in self.env._state_index_map:
            raise ValueError(f"Invalid start {rc}: not a valid free cell (out of bounds or obstacle).")
        return rc



    def finalize(self):

        # ---- Compute final pi_vec ----
        if self.n_agents == 1:
            pi_vec = self.last_phi_vec / np.maximum(self.last_phi_vec.sum(), 1e-12)
        else:
            phi_avg_vec = np.mean(self.last_phi_vecs, axis=0)
            pi_vec = phi_avg_vec / np.maximum(phi_avg_vec.sum(), 1e-12)

        # ---- ROI extraction ----
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

        # ---- Store datasets ----
        self.h5_file.create_dataset("pi_vec_final", data=pi_vec)
        self.h5_file.create_dataset("roi_rc", data=np.array(roi_rc, dtype=int))
        self.h5_file.create_dataset("roi_name", data=np.array(roi_name, dtype="S"))

        self.h5_file.create_dataset(
            "env_change_steps",
            data=np.array(self.env_change_steps, dtype=int)
        )


        self.h5_file.create_dataset(
            "state_index_map_keys",
            data=np.array(list(self.env._state_index_map.keys()), dtype=int)
        )

        self.h5_file.create_dataset(
            "state_index_map_vals",
            data=np.array(list(self.env._state_index_map.values()), dtype=int)
        )

        self.h5_file.create_dataset("grid_shape", data=self.grid_shape)
        self.h5_file.create_dataset("true_phi_map", data=self.true_phi_map)

        # ---- Save update steps ----
        self.h5_file.create_dataset(
            "gp_update",
            data=np.array(getattr(self, "gp_update_every", []), dtype=int)
        )

        self.h5_file.create_dataset(
            "p_update",
            data=np.array(getattr(self, "p_update_every", []), dtype=int)
        )

        # ---- Attributes (metadata) ----
        self.h5_file.attrs["stem"] = self.run_dir
        self.h5_file.attrs["method"] = str(self.args.method)
        self.h5_file.attrs["neighbors"] = int(self.args.neighbors)
        self.h5_file.attrs["mode"] = str(self.args.mode)
        self.h5_file.attrs["gp_mode"] = str(self.args.gp_mode)

        self.h5_file.attrs["T"] = self.total_steps
        self.h5_file.attrs["n_agents"] = self.n_agents
        self.h5_file.attrs["gp_update"] = self.gp_update_every
        self.h5_file.attrs["p_update"] = self.p_update_every

        self.h5_file.attrs["env_dyn"] = self.env_dyn
        self.h5_file.attrs["env_source"] = self.config_path

        self.h5_file.attrs["comm_mode"] = str(getattr(self.args, "comm_mode", "NA"))
        self.h5_file.attrs["comm_radius"] = float(getattr(self.args, "comm_radius", np.nan))
        self.h5_file.attrs["consensus_iters"] = int(getattr(self.args, "consensus_iters", 0))

        self.h5_file.attrs["ucb_beta_m"] = float(self.args.ucb_beta_m)
        self.h5_file.attrs["ucb_beta_u"] = float(self.args.ucb_beta_u)

        self.h5_file.attrs["nstates"] = int(self.env.nstates)

        self.h5_file.close()

        print("HDF5 file saved and closed.")
    # def finalize(self):
    #     assert np.all(self.traj_states >= 0), "traj_states has unwritten entries (-1)."
    #     assert self.traj_rc_time.shape == (self.total_steps, self.n_agents, 2)
    #     # assert len(self.phi_norm_series) == self.total_steps

    #     if self.n_agents == 1:
    #         pi_vec = self.last_phi_vec / self.last_phi_vec.sum()
    #     else:
    #         phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
    #         pi_vec = phi_avg_vec / phi_avg_vec.sum()


    #     out_path = os.path.join(self.run_dir, "compute_per_agent.csv")

    #     with open(out_path, "w") as f:
    #         f.write("agent_id,t,gp_fit_sec,p_build_sec\n")
    #         for a, t, gp_dt, p_dt in self.compute_log:
    #             f.write(f"{a},{t},{gp_dt:.8f},{p_dt:.8f}\n")

    #     print(f"[compute] wrote {out_path}")

    #     # to keep track of ROI
    #     roi_rc = []
    #     roi_name = []

    #     for roi in getattr(self.env, "roi_cells", []):
    #         name = roi["name"]
    #         r0, c0, r1, c1 = roi["rect"]

    #         for r in range(r0, r1):
    #             for c in range(c0, c1):
    #                 if (r, c) in self.env._state_index_map:
    #                     roi_rc.append((r, c))
    #                     roi_name.append(name)

    #     print("Periodic-GP run complete:", self.run_dir)
    #     # print(f"Final L2 error ||r_hat_T - pi|| = {self.error_series[-1]:.4f}")
    #     print(f"[GP] Saved plots + data under results/ with name: {self.run_dir}")


    #     # flatten obs streams
    #     if len(self.obs_r_all) > 0:
    #         obs_r_all = np.concatenate(self.obs_r_all)
    #         obs_c_all = np.concatenate(self.obs_c_all)
    #         obs_t_all = np.concatenate(self.obs_t_all)
    #         obs_y_all = np.concatenate(self.obs_y_all)
    #     else:
    #         obs_r_all = np.zeros((0,), dtype=float)
    #         obs_c_all = np.zeros((0,), dtype=float)
    #         obs_t_all = np.zeros((0,), dtype=float)
    #         obs_y_all = np.zeros((0,), dtype=float)

    #     # P snapshots can vary in size/type; store as object array for safety
    #     P_snapshots_obj = np.array(self.P_snapshots, dtype=object)


    #     save_dir = self.run_dir
    #     os.makedirs(save_dir, exist_ok=True)
    #     np.savez_compressed(
    #         os.path.join(save_dir, f"run_{os.path.basename(save_dir)}.npz"),
    #         # run metadata
    #         stem=self.run_dir,
    #         method=str(self.args.method),
    #         neighbors=int(self.args.neighbors),
    #         env_dyn=str(self.env_dyn),
    #         env_source=str(self.config_path),
    #         state_index_map=self.env._state_index_map,
    #         env_change_steps=np.array(self.env_change_steps, dtype=int),

    #         mode=str(self.args.mode),             
    #         T=int(self.total_steps),
            
    #         n_agents=int(self.n_agents),
    #         comm_mode=str(getattr(self.args, 'comm_mode', 'NA')),
    #         comm_radius=float(getattr(self.args, 'comm_radius', np.nan)),
    #         consensus_iters=int(getattr(self.args, 'consensus_iters', np.nan)),

    #         gp_mode=str(self.args.gp_mode),
    #         gp_update = int(self.gp_update_every),
    #         p_update = int(self.p_update_every),
    #         ucb_beta_m=float(self.args.ucb_beta_m),
    #         ucb_beta_u=float(self.args.ucb_beta_u),

    #         # ROI info
    #         roi_rc=np.array(roi_rc, dtype=int), # (n_roi_cells, 2) (row, col) of ROI cells
    #         roi_name=np.array(roi_name, dtype=object),

    #         # env/meta
    #         nstates=np.array(self.env.nstates, dtype=int),
    #         grid_shape=self.grid_shape,
    #         true_phi_map=self.true_phi_map,

    #         true_phi_map_time=self.true_phi_map_time,

    #         pi_vec_final=pi_vec,

    #         # phi_norm_series=np.vstack(self.phi_norm_series),
    #         phi_agents_time=self.phi_agents_time,
    #         belief_phi_map=self.belief_phi_map,

    #         # trajectory
    #         traj_states=self.traj_states,              # (T, n_agents)
    #         traj_rc_agents_all=self.traj_rc_time,      # (T, n_agents, 2)
            
    #         # observations stream
    #         obs_r=obs_r_all,
    #         obs_c=obs_c_all,
    #         obs_t=obs_t_all,
    #         obs_y=obs_y_all,
    #         obs_count_agents=self.obs_count_agents,
    #         # updates & policies
    #         gp_update_steps=np.array(self.gp_fit_steps, dtype=int),
    #         p_update_steps=np.array(self.p_rebuild_steps, dtype=int),
    #         P_snapshots=P_snapshots_obj
            
    #     )
        