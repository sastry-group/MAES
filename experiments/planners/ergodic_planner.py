
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
        self.large_scale = getattr(args, "large_scale", False)
        self.attrition = getattr(args, "attrition", False)
        self.attrition_schedule = {}      
        self.active_agents = None          
        self.death_log = []      
        self.comm_delay = getattr(args, "comm_delay", 0)   
        # self.comm_delay_prob = getattr(args, "comm_delay_prob", 0.5) 
        self.delay_rng = np.random.default_rng(42)     
        # self.comm_history_window = getattr(args, "comm_history_window", 5)
        self.history_window = args.share_obs_hist ###self.gp_update_every * 2
        self.gp_max_obs = args.gp_obs_hist ##self.gp_update_every * 10
        self.l_time = args.l_time

        self._configure_runtime_mode()


    def _configure_runtime_mode(self):
        # self.shared_P = False
        self.skip_lambda2 = False
        self.log_stride = 1
        self.save_full_agent_beliefs = True
        self.save_true_phi_every_step = True

        if self.large_scale:
            self.method = "mh"              # force MH
            # self.shared_P = True            # one P for all agents
            self.skip_lambda2 = True        # no spectral computation
            self.log_stride = 20            # for saving
            self.save_full_agent_beliefs = False
            self.save_true_phi_every_step = False

    def _normalize(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.clip(v, 0.0, None)
        s = float(v.sum())
        return v / (s + eps)

    def _iter_agents(self):
        """Return the set of agents to simulate this step."""
        if self.attrition:

            return self.active_agents
        return range(self.n_agents)


    def _share_and_feed(self, t, active_list, neighbors):
        # DEBUG_AGENT = 0

        to_deliver_now = [[] for _ in range(self.n_agents)]
        to_delay = [[] for _ in range(self.n_agents)]

        for a in active_list:
            for b in neighbors[a]:
                if b == a:
                    continue

                if self.comm_delay > 0:
                    link_delay = int(self.delay_rng.integers(1, self.comm_delay + 1))
                else:
                    link_delay = 0

                for pkt in self.local_history[b]:
                    obs_id = pkt[0]
                    obs_time = pkt[3]
                    if t - obs_time > self.history_window:
                        continue
                    if obs_id not in self.seen_obs[a]:
                        self.seen_obs[a].add(obs_id)

                        if link_delay > 0:
                            to_delay[a].append((t + link_delay, pkt))
                            # if a == DEBUG_AGENT:
                            #     print(f"  [t={t}] agent {a}: QUEUED obs from agent {obs_id[0]} "
                            #         f"(obs_t={int(obs_time)}, cell=({pkt[1]},{pkt[2]})), "
                            #         f"link_delay={link_delay}, delivers at t={t+link_delay}")
                        else:
                            to_deliver_now[a].append(pkt)
                            # if a == DEBUG_AGENT:
                            #     print(f"  [t={t}] agent {a}: INSTANT obs from agent {obs_id[0]} "
                            #         f"(obs_t={int(obs_time)}, cell=({pkt[1]},{pkt[2]}))")

        # Stage 2: apply
        for a in active_list:
            for pkt in to_deliver_now[a]:
                self.local_history[a].append(pkt)
                (_, r_o, c_o, t_o, y_o, nv_o) = pkt
                self.gps[a].add_batch(
                    r=np.array([r_o]), c=np.array([c_o]),
                    t=np.array([t_o]), y=np.array([y_o]),
                    noise_var=np.array([nv_o]),
                )
                # if a == DEBUG_AGENT:
                #     print(f"  [t={t}] agent {a}: FED TO GP (instant) obs from agent {pkt[0][0]} "
                #         f"(obs_t={int(t_o)}, cell=({r_o},{c_o}))")

            for item in to_delay[a]:
                self.delay_queue[a].append(item)

        # Stage 3: flush delay queue
        for a in active_list:
            still_waiting = []
            for (deliver_at, pkt) in self.delay_queue[a]:
                if t >= deliver_at:
                    (_, r_o, c_o, t_o, y_o, nv_o) = pkt
                    self.gps[a].add_batch(
                        r=np.array([r_o]), c=np.array([c_o]),
                        t=np.array([t_o]), y=np.array([y_o]),
                        noise_var=np.array([nv_o]),
                    )
                    self.local_history[a].append(pkt)
                    # if a == DEBUG_AGENT:
                    #     print(f"  [t={t}] agent {a}: DELIVERED from delay queue, "
                    #         f"obs from agent {pkt[0][0]} "
                    #         f"(obs_t={int(t_o)}, cell=({r_o},{c_o})), "
                    #         f"was scheduled for t={deliver_at}")
                else:
                    still_waiting.append((deliver_at, pkt))
            self.delay_queue[a] = still_waiting

        # Stage 4: feed own observations
        for a in active_list:
            for pkt in self.obs_buffer[a]:
                (_, r_o, c_o, t_o, y_o, nv_o) = pkt
                self.gps[a].add_batch(
                    r=np.array([r_o]), c=np.array([c_o]),
                    t=np.array([t_o]), y=np.array([y_o]),
                    noise_var=np.array([nv_o]),
                )
                # if a == DEBUG_AGENT:
                #     print(f"  [t={t}] agent {a}: FED OWN obs to GP "
                #         f"(cell=({r_o},{c_o}))")
            self.obs_buffer[a].clear()


        # Stage 5: trim old data
        for a in active_list:
            self.local_history[a] = [
                pkt for pkt in self.local_history[a]
                if t - pkt[3] <= self.history_window
            ]
            self.seen_obs[a] = {
                obs_id for obs_id in self.seen_obs[a]
                if obs_id[1] >= t - self.history_window
            }

        # # debug agent
        # if self.comm_delay > 0:
        #     print(f"  [t={t}] agent {DEBUG_AGENT} summary: "
        #         f"local_history={len(self.local_history[DEBUG_AGENT])}, "
        #         f"delay_queue={len(self.delay_queue[DEBUG_AGENT])}, "
        #         f"seen={len(self.seen_obs[DEBUG_AGENT])}, ")

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

        # Env datasets
        nstates = self.env.nstates
        T = self.total_steps
        N = self.n_agents

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

            self.gps = [GPInfoMap(l_space=2.0, l_time=self.l_time, nu=1.5, noise=1e-12)
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

            ##Attrition
            self.death_log = []
            self.attrition_schedule = kwargs.get("attrition_schedule", {})
            if self.attrition:
                self.active_agents = set(range(self.n_agents))
                self.d_active_mask = self.h5_file.create_dataset(
                    "active_mask",
                    shape=(T, N),
                    dtype="bool",
                    compression="gzip", compression_opts=4,
                )
            else:
                self.active_agents = None  
            # self.belief_phi_map = np.zeros((self.total_steps, phi_vec_uni.shape[0]), dtype=float)
            # initial average belief (across agents) for logging
            # phi_avg_vec = sum(self.last_phi_vecs) / float(self.n_agents)
            # self.belief_phi_map[0, :] = phi_avg_vec.copy()
            # self.phi_agents_time = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=float)
            # self.obs_count_agents = np.zeros((self.total_steps, self.n_agents, self.env.nstates),dtype=int)
            # self.obs_buffer = [ [] for _ in range(self.n_agents) ]
            # self.seen_obs = [set() for _ in range(self.n_agents)]
            # self.delay_queue = [[] for _ in range(self.n_agents)]
            # self.local_history = [[] for _ in range(self.n_agents)]

        else:   
            
            self.last_phi_vec = phi_vec_uni.copy()   
        
            if self.method == "mh":
                # might need to normalize and oass it
                # self._normalize(phi_vec_uni)
                self.P, self.second_largest_eig = build_transition_matrix_MH(self.env, phi_vec_uni)
            else:
                self.P, self.second_largest_eig = build_transition_matrix_REMC(self.env, self.last_phi_vec, max_iters=5000, verbose=False)

            self.gp = GPInfoMap(l_space=2.0, l_time=self.l_time, nu=1.5, noise=1e-12)
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
        elif len(agents_initial_positions) < self.n_agents:
            # round-robin distribution - clean already in n_gp
            self.start_rc_agents = [
                self.validate_start(agents_initial_positions[i % len(agents_initial_positions)])
                for i in range(self.n_agents)
            ]
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
        self.delay_queue = [[] for _ in range(self.n_agents)]  # list of delay messages (deliver at step, obs_msg)
        self.local_history = [[] for _ in range(self.n_agents)]

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

        if self.attrition:
            for t_kill in sorted(self.attrition_schedule.keys()):
                if self.t0 <= t_kill < self.t0 + T_dt:
                    agent_to_kill = self.attrition_schedule[t_kill]
                    if agent_to_kill in self.active_agents:
                        self.active_agents.discard(agent_to_kill)
                        self.death_log.append((t_kill, agent_to_kill))
                        print(f"[t={t_kill}] Agent {agent_to_kill} removed. "
                            f"Active: {sorted(self.active_agents)}")

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

            traj_idx_batch = {}
            traj_rc_batch = {}

            local_obs = {}
            # for a in range(self.n_agents):
            for a in self._iter_agents():
                start_a = self.start_rc_agents[a]
                traj_idx_a, traj_rc_a, counts_a = self.sim.simulate(
                    self.P_agents[a], T_dt, start=start_a
                )
                self.start_rc_agents[a] = tuple(traj_rc_a[-1])



                traj_idx_batch[a] = traj_idx_a   # keyed by agent id
                traj_rc_batch[a] = traj_rc_a 
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


            # idx_stack = np.stack(traj_idx_batch, axis=1)   # each traj_idx_a is (T_dt,)

            for k in range(T_dt):
                t = self.t0 + k
                active_now = self._iter_agents()
                # self.phi_norm_series.append(current_phi_vec_norm.copy())

                # positions_k = [traj_rc_batch[a][k] for a in range(self.n_agents)]
                positions_k = {a: traj_rc_batch[a][k] for a in active_now}
                active_list = sorted(active_now)
                neighbors = {
                    a: [
                        # b for b in range(self.n_agents)
                        b for b in active_list
                        if np.hypot(
                            positions_k[a][0] - positions_k[b][0],
                            positions_k[a][1] - positions_k[b][1]
                        ) <= self.args.comm_radius
                    ]
                    # for a in range(self.n_agents)
                    for a in active_list

                }
                

                for a in range(self.n_agents):
                    if a in active_now:
                        s = int(traj_idx_batch[a][k])
                        # s = int(idx_stack[k, a]) # time and state index for agent a at step k
                        ra, ca = map(int, positions_k[a])
                        self.d_traj_states[t, a] = s
                        self.d_traj_rc[t, a] = (ra, ca)
                        if self.attrition: 
                            self.d_active_mask[t, a] = True 

                        y = local_obs[a]["y"][k]
                        noise_var = local_obs[a]["noise"][k]
                        obs_id = (a, t)

                        if obs_id not in self.seen_obs[a]:
                            self.seen_obs[a].add(obs_id)
                            self.obs_buffer[a].append((obs_id, ra, ca, t, y, noise_var))
                            self.local_history[a].append((obs_id, ra, ca, t, y, noise_var)) 
        
                        
                        self.d_belief_phi[t, a] = self.last_phi_vecs[a]
                    elif self.attrition:
                        # dead agent sentinels
                        self.d_traj_states[t, a] = -1
                        self.d_traj_rc[t, a] = (-1, -1)
                        self.d_active_mask[t, a] = False
                        self.d_belief_phi[t, a] = np.nan  


                self.d_true_phi[t] = self.current_true_phi     
                # average belief: only active
                active_beliefs = [self.last_phi_vecs[a] for a in active_now]
                if active_beliefs:
                    phi_avg = np.mean(active_beliefs, axis=0)
                else:
                    phi_avg = np.zeros(self.env.nstates)
                self.d_belief_phi_avg[t] = phi_avg

                # handles delays in comms and attrition
                self._share_and_feed(t, active_list, neighbors)

                # new_buffers = [ [] for _ in range(self.n_agents) ]
                # for a in active_list:
                # # for a in range(self.n_agents):
                #     for b in neighbors[a] + [a]:
                #         for pkt in self.obs_buffer[b]:
                #             obs_id = pkt[0]
                #             if obs_id not in self.seen_obs[a]:
                #                 self.seen_obs[a].add(obs_id)
                #                 new_buffers[a].append(pkt)
                # self.obs_buffer = new_buffers  

                # this is ebcause for large scale sims, the agetns starrint in one location all accumulating obs
                # if k > 0 and k % 25 == 0:
                #     for a in active_list:
                #         self.gps[a].trim(max_pts=self.gp_max_obs)


                # for a in active_list:
                # # for a in range(self.n_agents):
                #     for (_, r_o, c_o, t_o, y_o, nv_o) in self.obs_buffer[a]:
                #         self.gps[a].add_batch(
                #             r=np.array([r_o]),
                #             c=np.array([c_o]),
                #             t=np.array([t_o]),
                #             y=np.array([y_o]),
                #             noise_var=np.array([nv_o]),
                #         )
                #     self.obs_buffer[a].clear()


        if self.n_agents > 1 and self.t0 % 50 == 0:
            gp_sizes  = [len(self.gps[a].y) if self.gps[a].y is not None else 0 for a in range(min(3, self.n_agents))]
            hist_sizes = [len(self.local_history[a]) for a in range(min(3, self.n_agents))]
            seen_sizes = [len(self.seen_obs[a]) for a in range(min(3, self.n_agents))]
            dq_sizes   = [len(self.delay_queue[a]) for a in range(min(3, self.n_agents))]
            print(f"  [MEM Past sim t={self.t0}] GP:{gp_sizes} hist:{hist_sizes} seen:{seen_sizes} dq:{dq_sizes}")
    

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
                for a in self._iter_agents():

                # for a in range(self.n_agents):
                    t0_track = time.perf_counter()
                    self.gps[a].fit_if_ready(min_pts=1)

                    t1_track = time.perf_counter()
                    gp_dt = t1_track - t0_track
                    self.compute_log.append(
                        (a, t_update, gp_dt, 0.0)
                    )
                    self.gps[a].trim(max_pts=self.gp_max_obs) 
                    self.last_phi_vecs[a] = self.gp_phi_loggers[a].phi_vec(self.env, t=t1 - 1).copy()
                    # self.phi_agents_time[t1 - 1, a, :] = self.last_phi_vecs[a]
                # self.gp_fit_steps.append(t1 - 1)

            self.since_gp = 0


        if self.n_agents > 1 and self.t0 % 50 == 0:
            gp_sizes  = [len(self.gps[a].y) if self.gps[a].y is not None else 0 for a in range(min(3, self.n_agents))]
            hist_sizes = [len(self.local_history[a]) for a in range(min(3, self.n_agents))]
            seen_sizes = [len(self.seen_obs[a]) for a in range(min(3, self.n_agents))]
            dq_sizes   = [len(self.delay_queue[a]) for a in range(min(3, self.n_agents))]
            print(f"  [MEM past GP t={self.t0}] GP:{gp_sizes} hist:{hist_sizes} seen:{seen_sizes} dq:{dq_sizes}")

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
                    # for a in range(self.n_agents):
                    for a in self._iter_agents():
                        pi_norm_a = self._normalize(self.last_phi_vecs[a])
                        P_a, second_largest_eig_a = build_transition_matrix_MH(self.env, pi_norm_a)
                        self.P_agents[a] = P_a
                        self.second_largest_eigs[a] = second_largest_eig_a
                        P_snap.append(P_a.copy())
                else:
                    # for a in range(self.n_agents):
                    for a in self._iter_agents():
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
    
        if self.n_agents > 1 and self.t0 % 50 == 0:
            gp_sizes  = [len(self.gps[a].y) if self.gps[a].y is not None else 0 for a in range(min(3, self.n_agents))]
            hist_sizes = [len(self.local_history[a]) for a in range(min(3, self.n_agents))]
            seen_sizes = [len(self.seen_obs[a]) for a in range(min(3, self.n_agents))]
            dq_sizes   = [len(self.delay_queue[a]) for a in range(min(3, self.n_agents))]
            print(f"  [MEM, past P t={self.t0}] GP:{gp_sizes} hist:{hist_sizes} seen:{seen_sizes} dq:{dq_sizes}")

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


        if self.attrition:
            if self.death_log:
                self.h5_file.create_dataset(
                    "attrition_log",
                    data=np.array(self.death_log, dtype=int),
                )
            self.h5_file.attrs["attrition"] = True
            self.h5_file.attrs["attrition_steps"] = str(
                getattr(self.args, "attrition_steps", "")
            )
        else:
            self.h5_file.attrs["attrition"] = False


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
        self.h5_file.attrs["gp_obs_history"] = float(self.args.gp_obs_hist)
        self.h5_file.attrs["share_obs_history"] = float(self.args.share_obs_hist)
        self.h5_file.attrs["l_time"] = float(self.args.l_time)

        self.h5_file.attrs["nstates"] = int(self.env.nstates)
        self.h5_file.attrs["comm_delay"] = int(getattr(self.args, "comm_delay", 0))

        self.h5_file.close()

        print("HDF5 file saved and closed.")
    