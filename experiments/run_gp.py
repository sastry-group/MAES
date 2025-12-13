import os
import numpy as np
import glob

from core.types import GridSpec
from env.grid import GridMC, make_env, update_env, make_env_from_json, load_env_sequence
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import counts_to_field, plot_grid, plot_trajectory, plot_distributions, plot_convergence, vector_to_field, counts_to_field
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from viz.animation import animate_trajectory, animate_distributions, animate_distributions_multi
from experiments.concensus import run_phi_consensus


def parse_starts(starts_str: str) -> list[tuple[int,int]]:
    out = []
    for chunk in starts_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        r_s, c_s = chunk.split(",")
        out.append((int(r_s), int(c_s)))
    return out

def validate_start(env, rc):
    if rc not in env._state_index_map:
        raise ValueError(f"Invalid start {rc}: not a valid free cell (out of bounds or obstacle).")
    return rc


def normalize_nonneg(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    return v / (s + eps)


def run(args):
    os.makedirs("results", exist_ok=True)

    n_agents = getattr(args, "n_agents", 1)
    stem = f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}_GPupdate{args.update_GP_every}_Pupdate{args.update_P_every}_gp{args.gp_mode}_{args.env_dyn}"
    if n_agents > 1:
        stem = (
            f"{stem}_nagents{n_agents}_commMode{getattr(args, 'comm_mode', 'global')}"
            f"_comm_radius{getattr(args, 'comm_radius', 'NA')}"
            f"_concenIter{getattr(args, 'consensus_iters', 'NA')}"
        )

    print("Starting periodic-GP run:", stem)
    print(f"Running with n_agents = {n_agents}")

    # Checking if dynamic or static environment
    config_path = args.config
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        config_path = os.path.join(base_dir, config_path)
    config_path = os.path.abspath(config_path)


    GP_UPDATE_EVERY = args.update_GP_every  
    P_UPDATE_EVERY  = args.update_P_every  
    total_steps = args.T

    if args.env_dyn == "dynamic":
        pattern = os.path.join(config_path, "*.json")
        json_files = sorted(glob.glob(pattern))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in dynamic env folder: {config_path}")

        cfg_paths = json_files
        n_cfg = len(cfg_paths)
        env = make_env_from_json(cfg_paths[0])
        current_cfg_idx = 0
        steps_per_cfg = int(np.ceil(total_steps / n_cfg))

    else:  # args.env_dyn == "static"
        env = make_env_from_json(config_path)
        cfg_paths = None
        n_cfg = 1
        current_cfg_idx = 0
        steps_per_cfg = total_steps
    plot_grid(env, f"results/{stem}_gp_grid_true.png", title="Grid (true info map)")


    # 1) INITIAL phi
    phi_vec_uni = UniformPhi().phi_vec(env)  
    sim = Simulator(env, Sensor(sigma_map=env.noise_sigma), ObservationLogger())

    if n_agents > 1:
        # per-agent beliefs
        last_phi_vecs = [phi_vec_uni.copy() for _ in range(n_agents)]
        # per-agent transition matrices
        P_agents = []

        if args.method == "mh":
            for a in range(n_agents):
                P_agents.append(build_transition_matrix_MH(env, last_phi_vecs[a]))
        else:
            for a in range(n_agents):
                P_agents.append(build_transition_matrix_REMC(env, last_phi_vecs[a], max_iters=5000, verbose=False))

        gps = [GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
            for _ in range(n_agents)
            ]
        gp_phi_loggers = [
            GPPhi(
                gps[a],
                mode=args.gp_mode,
                beta_m=args.ucb_beta_m,
                beta_u=args.ucb_beta_u,
                base=1e-3,
            )
            for a in range(n_agents)
        ]
        belief_phi_map = np.zeros((total_steps, phi_vec_uni.shape[0]), dtype=float)
        # initial average belief (across agents) for logging
        phi_avg_vec = sum(last_phi_vecs) / float(n_agents)
        belief_phi_map[0, :] = phi_avg_vec.copy()

    else:   
        last_phi_vec = phi_vec_uni.copy()   
    
        if args.method == "mh":
            P = build_transition_matrix_MH(env, phi_vec_uni)
        else:
            P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=5000, verbose=False)

        gp = GPInfoMap(l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-12)
        gp_phi_logger = GPPhi(
            gp,
            mode=args.gp_mode,
            beta_m=args.ucb_beta_m,
            beta_u=args.ucb_beta_u,
            base=1e-3
        )
        belief_phi_map = np.zeros((total_steps, last_phi_vec.shape[0]), dtype=float)
        belief_phi_map[0, :] = last_phi_vec.copy() # not normalized


    # counters and loggers
    counts_total = np.zeros(env.nstates, dtype=int)
    counts_running = np.zeros(env.nstates, dtype=int) 
    traj_idx_all, traj_rc_all = [], []


    # for animation
    emp_series = []                 # list of (nstates,) empirical distros over time
    phi_norm_series  = []                 # list of (nstates,) target distros used at that time
    cr = np.zeros(env.nstates, dtype=int)  # running counts for empirical distro
    error_series = []              # ergodic error || r_hat - pi || over time
    traj_rc_agents = [ [] for _ in range(n_agents) ]  # per-agent trajectory record, multiagent
    dcr = np.zeros(env.nstates, dtype=float)  # discounted counts
    beta = 0.999                               # discount factor 
    global_t = 0                    # total steps elapsed
    first_hit_step = np.full(env.nstates, -1, dtype=int)  # -1 = never visited
    unseen = env.nstates                                 # how many still unvisited
    cover_time = -1  


    starts_list = parse_starts(args.starts)

    if n_agents == 1:
        if len(starts_list) != 1:
            raise ValueError(f"n_agents=1 but --starts provides {len(starts_list)} starts")
        start_rc = starts_list[0]
        start_rc = validate_start(env, start_rc)
    else:
        if len(starts_list) != n_agents:
            raise ValueError(f"--starts provides {len(starts_list)} starts but n_agents={n_agents}")
        start_rc_agents = starts_list
        start_rc_agents = [validate_start(env, rc) for rc in start_rc_agents]

    t0 = 0
    
    # saving 
    obs_r_all, obs_c_all, obs_t_all, obs_y_all = [], [], [], []
    gp_fit_steps = []         # step indices at which GP was (re)fit
    p_rebuild_steps = []      # step indices at which P was rebuilt
    P_snapshots = []          # store P at each rebuild

    grid_shape = np.array(env.true_phi.shape, dtype=int)
    try:
        true_phi_map = env.true_phi.astype(float)
    except Exception:
        true_phi_map = None

    # Saving the map for convergence plots
    if n_agents == 1:
        true_phi_map_time = np.zeros((total_steps, last_phi_vec.shape[0]), dtype=float) #not normalized, vector form per time
        initial_true_phi = env.build_map_vec(env.true_phi) # not normalized, vector form
        true_phi_map_time[0:steps_per_cfg, :] = initial_true_phi
        current_true_phi = initial_true_phi
    else:
        true_phi_map_time = np.zeros((total_steps, last_phi_vecs[0].shape[0]), dtype=float) #not normalized, vector form per time
        initial_true_phi = env.build_map_vec(env.true_phi) # not normalized, vector form
        true_phi_map_time[0:steps_per_cfg, :] = initial_true_phi
        current_true_phi = initial_true_phi

    since_gp = 0
    since_p  = 0
    next_map_change = steps_per_cfg



    while t0 < total_steps:

        T_dt = min(
            total_steps - t0,
            GP_UPDATE_EVERY - since_gp,
            P_UPDATE_EVERY  - since_p
        )

        # current target for error plots: average phi if multi-agent
        if n_agents == 1:
            phi_for_plot = last_phi_vec
        else:
            phi_for_plot = sum(last_phi_vecs) / float(n_agents)
        current_phi_vec_norm = phi_for_plot / phi_for_plot.sum()

        

        # 2a) simulate for K steps under current P
        if n_agents == 1:
            # current_phi_vec_norm = last_phi_vec / last_phi_vec.sum() # normalized
            traj_idx, traj_rc, counts = sim.simulate(P, T_dt, start=start_rc)
            counts_total += counts
            counts_running += counts
            # print(f"[t={t0}], {counts_running}")
            traj_idx_all.append(traj_idx)
            traj_rc_all.append(traj_rc)

            # appending for animation
            for s in traj_idx:
                global_t += 1
                cr[s] += 1
                # Cesàro average (empirical)
                r_hat = cr / global_t
                dcr *= beta          # decay old visits
                dcr[s] += 1.0        # add current visit
                r_hat_disc = dcr / dcr.sum()

                if args.env_dyn == "dynamic":
                    emp_series.append(r_hat_disc.copy())
                    phi_norm_series.append(current_phi_vec_norm.copy())
                    error_series.append(np.linalg.norm(r_hat_disc - current_phi_vec_norm))     
                else:               
                    emp_series.append(r_hat.copy())
                    phi_norm_series.append(current_phi_vec_norm.copy())
                    err = np.linalg.norm(r_hat - current_phi_vec_norm)
                    error_series.append(err)

                if first_hit_step[s] == -1:
                    first_hit_step[s] = global_t
                    unseen -= 1
                    if unseen == 0 and cover_time == -1:
                        cover_time = global_t

            # add observations to GP 
            ts = t0 + np.arange(len(traj_rc), dtype=float) # time
            rs = traj_rc[:, 0].astype(float) # row
            cs = traj_rc[:, 1].astype(float) # column

            cell_noise = env.noise_sigma[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            y_true = env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
            noise   = np.random.normal(0.0, cell_noise)
            y_noisy = y_true + noise


            obs_r_all.append(rs)
            obs_c_all.append(cs)
            obs_t_all.append(ts)
            obs_y_all.append(y_noisy)

            k = 0.7
            proxy_var = np.clip((k * cell_noise)**2, 1e-6, 1e-1)
            gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=proxy_var)

            start_rc = tuple(traj_rc[-1])
            
        

        else:
            # ---------- MULTI-AGENT BRANCH ----------
            # we keep one global GP and aggregate all agents' trajectories
            # phi_avg_vec = sum(last_phi_vecs) / float(n_agents)
            # current_phi_vec_norm = phi_avg_vec / phi_avg_vec.sum()

            # initialize per-step accumulators
            counts_step_total = np.zeros(env.nstates, dtype=int)
            traj_idx_batch_list = []
            traj_rc_batch_list = []


            for a in range(n_agents):
                start_a = start_rc_agents[a]
                traj_idx_a, traj_rc_a, counts_a = sim.simulate(
                    P_agents[a], T_dt, start=start_a
                )
                start_rc_agents[a] = tuple(traj_rc_a[-1])
                traj_rc_agents[a].extend(traj_rc_a.tolist())

                traj_idx_batch_list.append(traj_idx_a)
                traj_rc_batch_list.append(traj_rc_a)
                counts_step_total += counts_a

                # local GP observations
                ts_a = t0 + np.arange(len(traj_rc_a), dtype=float)
                rs_a = traj_rc_a[:, 0].astype(float)
                cs_a = traj_rc_a[:, 1].astype(float)

                cell_noise_a = env.noise_sigma[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                y_true_a = env.true_phi[traj_rc_a[:, 0], traj_rc_a[:, 1]].astype(float)
                noise_a = np.random.normal(0.0, cell_noise_a)
                y_noisy_a = y_true_a + noise_a

                obs_r_all.append(rs_a)
                obs_c_all.append(cs_a)
                obs_t_all.append(ts_a)
                obs_y_all.append(y_noisy_a)

                k = 0.7
                proxy_var_a = np.clip((k * cell_noise_a) ** 2, 1e-6, 1e-1)
                gps[a].add_batch(
                    r=rs_a, c=cs_a, t=ts_a, y=y_noisy_a, noise_var=proxy_var_a
                )

            traj_idx = np.concatenate(traj_idx_batch_list, axis=0)
            traj_rc = np.vstack(traj_rc_batch_list)

            counts_total += counts_step_total
            counts_running += counts_step_total

            traj_idx_all.append(traj_idx)
            traj_rc_all.append(traj_rc)

            # empirical distro over all agents combined
            # for s in traj_idx:
            #     global_t += 1
            #     cr[s] += 1
            #     r_hat = cr / global_t
            #     dcr *= beta
            #     dcr[s] += 1.0
            #     r_hat_disc = dcr / dcr.sum()
            idx_stack = np.stack(traj_idx_batch_list, axis=1)   # each traj_idx_a is (T_dt,)

            for k in range(T_dt):
                # apply all agents' visits at this SAME environment time step
                for a in range(n_agents):
                    s = int(idx_stack[k, a])
                    global_t += 1
                    cr[s] += 1

                    dcr *= beta
                    dcr[s] += 1.0

                    if first_hit_step[s] == -1:
                        first_hit_step[s] = global_t
                        unseen -= 1
                        if unseen == 0 and cover_time == -1:
                            cover_time = global_t

                # now log ONE empirical snapshot per environment step k
                r_hat = cr / global_t
                r_hat_disc = dcr / dcr.sum()


                if args.env_dyn == "dynamic":
                    emp_series.append(r_hat_disc.copy())
                    phi_norm_series.append(current_phi_vec_norm.copy())
                    error_series.append(np.linalg.norm(r_hat_disc - current_phi_vec_norm))
                else:
                    emp_series.append(r_hat.copy())
                    phi_norm_series.append(current_phi_vec_norm.copy())
                    error_series.append(np.linalg.norm(r_hat - current_phi_vec_norm))

                if first_hit_step[s] == -1:
                    first_hit_step[s] = global_t
                    unseen -= 1
                    if unseen == 0 and cover_time == -1:
                        cover_time = global_t

     
        # ---------
        # TIME STEP
        # ---------
        t1 = t0 + T_dt
        since_gp += T_dt
        since_p += T_dt


       # keep average belief for logging (even in single-agent, phi_for_plot = last_phi_vec)        
        belief_phi_map[t0:t1, :] = phi_for_plot
        
        # ------------------
        # GP UPDATE + phi SYNC
        # ------------------
        if since_gp >= GP_UPDATE_EVERY:
            if n_agents == 1:
                fitted = gp.fit_if_ready(min_pts=1)
                if fitted:
                    last_phi_vec = gp_phi_logger.phi_vec(env, t=t1 - 1).copy()
                    gp_fit_steps.append(t1 - 1)
                    # print(f"[t={t1}] GP refit done (single agent).")
            else:
                any_fitted = False
                for a in range(n_agents):
                    fitted_a = gps[a].fit_if_ready(min_pts=1)
                    if fitted_a:
                        last_phi_vecs[a] = gp_phi_loggers[a].phi_vec(env, t=t1 - 1).copy()
                        any_fitted = True
                if any_fitted:
                    gp_fit_steps.append(t1 - 1)
                    run_phi_consensus(args, last_phi_vecs, start_rc_agents)
                    # print(f"[t={t1}] GPs refit + phi consensus done for {n_agents} agents.")

            since_gp = 0

        # -------------
        # P UPDATE (REMC)
        # -------------
        if since_p >= P_UPDATE_EVERY:
            if n_agents == 1:
                if args.method == "mh":
                    P = build_transition_matrix_MH(env, last_phi_vec)
                else:
                    P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=10000, verbose=False)
                P_snapshots.append(P.copy())
            else:
                P_snap = []
                if args.method == "mh":
                    for a in range(n_agents):
                        P_a = build_transition_matrix_MH(env, last_phi_vecs[a])
                        P_agents[a] = P_a
                        P_snap.append(P_a.copy())
                else:
                    for a in range(n_agents):
                        P_a = build_transition_matrix_REMC(env, last_phi_vecs[a], max_iters=10000, verbose=False)
                        P_agents[a] = P_a
                        P_snap.append(P_a.copy())
                P_snapshots.append(np.array(P_snap, dtype=object))

            p_rebuild_steps.append(t1 - 1)
            since_p = 0
    
        # ---------------------------
        # DYNAMIC ENVIRONMENT CHANGES
        # ---------------------------        
        if args.env_dyn == "dynamic":
            while t1 >= next_map_change and current_cfg_idx < n_cfg - 1:
                desired_cfg_idx = min(next_map_change // steps_per_cfg, n_cfg - 1)
                if desired_cfg_idx != current_cfg_idx:
                    current_cfg_idx = desired_cfg_idx
                    env = make_env_from_json(cfg_paths[current_cfg_idx])
                    current_true_phi = env.build_map_vec(env.true_phi) 
                    plot_grid(
                        env,
                        f"results/{stem}_gp_grid_true_cfg{current_cfg_idx}.png",
                        title=f"Grid (true info map) cfg {current_cfg_idx}"
                    )
                    print(f"[t={t1}] Environment updated, loaded config {cfg_paths[current_cfg_idx]}")
                    # cr[:] = 0        # reset empirical counts
                    # global_t = 0     # reset time for metric
                
                    next_map_change += steps_per_cfg
                    true_phi_map_time[t1:next_map_change, :] = current_true_phi # not normalized

        t0 = t1

        # if t0 == 500:
        #     ## this is hardcoded test
        #     env = update_env(neighbors=args.neighbors, seed=1, time=500)
        #     # env.build_map_vec(env.true_phi)
        #     # true_phi_map_time[500:,:] = env.build_map_vec(env.true_phi)

        #     plot_grid(env, f"results/{stem}_{t0}_gp_grid_true.png", title="Grid (true info map)")
            # current_true_phi = env.build_map_vec(env.true_phi) 
     
    if n_agents > 1:
        for a in range(n_agents):
            assert len(traj_rc_agents[a]) == total_steps, f"Mismatch in recorded trajectory length for agent {a}"
        traj_rc_agents_all = np.stack([np.asarray(traj_rc_agents[a], dtype=int) for a in range(n_agents)], axis=1)

    # 3) Final evaluation/plots vs lastphi_hat(if no GP fit ever happened)
    emp = counts_total / counts_total.sum()
    if n_agents == 1:
        pi_vec = last_phi_vec / last_phi_vec.sum()
    else:
        phi_avg_vec = sum(last_phi_vecs) / float(n_agents)
        pi_vec = phi_avg_vec / phi_avg_vec.sum()


    traj_idx_all = np.concatenate(traj_idx_all, axis=0)
    traj_rc_all = np.vstack(traj_rc_all)

    # plot_trajectory(env, traj_rc_all, f"results/{stem}_trajectory.png",
    #                 title="Trajectory with periodic GP updates")
    # plot_distributions(env, emp, pi_vec, f"results/{stem}_emp_vs_pi_final.png")

    # if len(emp_series) > 0:
    #     emp_arr = np.vstack(emp_series)
    #     phi_norm_arr = np.vstack(phi_norm_series)
    #     error_series = [np.linalg.norm(emp_arr[t] - phi_norm_arr[t]) for t in range(len(emp_arr))]
    # else:
    #     emp_arr = np.zeros((0, env.nstates), dtype=float)
    #     phi_norm_arr = np.zeros((0, env.nstates), dtype=float)
    #     error_series = []


    # plot_convergence(emp_arr, phi_norm_arr, f"results/{stem}_convergence.png", true_phi, belief_phi_map)

    print("Periodic-GP run complete:", stem)
    print(f"Final L2 error ||r_hat_T - pi|| = {error_series[-1]:.4f}")
    print(f"[GP] Saved plots + data under results/ with name: {stem}")


    # np.savez_compressed(
    #     f"results/run_{stem}.npz",
    #      pi_vec, emp, true_phi,
    #     counts, traj_idx_all, traj_rc_all
    # )

#     animate_trajectory(
#         env, traj_rc_all,
#         filename=f"results/anim_{stem}.mp4",
#         fps=args.fps, stride=args.stride
#     )



    if args.animation:
        if n_agents == 1:
            animate_distributions(
                env,
                emp_series=emp_series,
                pi_series=phi_norm_series,
                filename=f"results/anim_{stem}__emp_vs_phi_norm.mp4",
                fps=args.fps,
                stride=args.stride,
                dpi=150,
                true_phi_map= true_phi_map_time,
                traj_rc=traj_rc_all
        )
        else:
            traj_rc_agents_all = np.stack(
                [np.asarray(traj_rc_agents[a], dtype=int) for a in range(n_agents)],
                axis=1
            )  # (T, n_agents, 2)

            animate_distributions_multi(
                env,
                emp_series=emp_series,
                pi_series=phi_norm_series,
                filename=f"results/anim_{stem}__emp_vs_phi_norm.mp4",
                fps=args.fps,
                stride=args.stride,
                dpi=150,
                true_phi_map=true_phi_map_time,
                traj_rc_agents=traj_rc_agents_all
            )

    # -------------------------
    # SAVE EVERYTHING TO NPZ
    # -------------------------
    # flatten obs streams
    if len(obs_r_all) > 0:
        obs_r_all = np.concatenate(obs_r_all)
        obs_c_all = np.concatenate(obs_c_all)
        obs_t_all = np.concatenate(obs_t_all)
        obs_y_all = np.concatenate(obs_y_all)
    else:
        obs_r_all = np.zeros((0,), dtype=float)
        obs_c_all = np.zeros((0,), dtype=float)
        obs_t_all = np.zeros((0,), dtype=float)
        obs_y_all = np.zeros((0,), dtype=float)

    # P snapshots can vary in size/type; store as object array for safety
    P_snapshots_obj = np.array(P_snapshots, dtype=object)


    save_dir = f"results/{stem}"
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        f"{save_dir}/run_{stem}.npz",
        # run metadata
        stem=stem,
        method=str(args.method),
        neighbors=int(args.neighbors),
        mode=str(args.mode),             
        T=int(total_steps),
        
        n_agents=int(n_agents),
        comm_mode=str(getattr(args, 'comm_mode', 'NA')),
        comm_radius=float(getattr(args, 'comm_radius', np.nan)),
        consensus_iters=int(getattr(args, 'consensus_iters', np.nan)),

        gp_mode=str(args.gp_mode),
        gp_update = int(GP_UPDATE_EVERY),
        p_update = int(P_UPDATE_EVERY),
        ucb_beta_m=float(args.ucb_beta_m),
        ucb_beta_u=float(args.ucb_beta_u),

        # coverage tracking
        first_hit_step=first_hit_step,
        cover_time=np.array(cover_time, dtype=int),


        # env/meta
        nstates=np.array(env.nstates, dtype=int),
        grid_shape=grid_shape,
        true_phi_map=true_phi_map,
        # true_phi_vec=initial_true_phi,
        true_phi_map_time=true_phi_map_time,
        # final distributions
        emp_final=emp,
        pi_vec_final=pi_vec,
        # time series
        emp_series=np.vstack(emp_series),
        phi_norm_series=np.vstack(phi_norm_series),
        belief_phi_map=belief_phi_map,
        error_series=np.array(error_series),
        # trajectory
        traj_idx_all=traj_idx_all,
        traj_rc_all=traj_rc_all,
        traj_rc_agents_all=traj_rc_agents_all if n_agents > 1 else np.zeros((0,0,0), dtype=int),
        # observations stream
        obs_r=obs_r_all,
        obs_c=obs_c_all,
        obs_t=obs_t_all,
        obs_y=obs_y_all,
        # updates & policies
        gp_update_steps=np.array(gp_fit_steps, dtype=int),
        p_update_steps=np.array(p_rebuild_steps, dtype=int),
        P_snapshots=P_snapshots_obj
    )