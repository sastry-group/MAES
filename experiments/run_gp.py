import os
import numpy as np
import glob

from core.types import GridSpec
from env.grid import GridMC, make_env, update_env, make_env_from_json, load_env_sequence, apply_env_config_to_env
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import counts_to_field, plot_grid, plot_trajectory, plot_distributions, plot_convergence, vector_to_field, counts_to_field
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from viz.animation import animate_trajectory, animate_distributions




def normalize_nonneg(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    return v / (s + eps)


def run(args):
    os.makedirs("results", exist_ok=True)
    stem = f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}_GPupdate{args.update_GP_every}_Pupdate{args.update_P_every}_gp{args.gp_mode}_{args.env_dyn}"
    print("Starting periodic-GP run:", stem)



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

        # build env from the first JSON file
        env = make_env_from_json(cfg_paths[0])
        current_cfg_idx = 0
        steps_per_cfg = int(np.ceil(total_steps / n_cfg))

    else:  # args.env_dyn == "static"
        env = make_env_from_json(config_path)
        cfg_seq = None
        n_cfg = 1
        current_cfg_idx = 0
        # 0) Environment + plotting of true map for reference - previous
        # env = make_env(neighbors=args.neighbors, seed=1)
        steps_per_cfg = total_steps
    plot_grid(env, f"results/{stem}_gp_grid_true.png", title="Grid (true info map)")

    # 1) INITIAL P: using a uniform phi over free cells to explore
    phi_vec_uni = UniformPhi().phi_vec(env)  
    last_phi_vec = phi_vec_uni.copy() 
    
    if args.method == "mh":
        P = build_transition_matrix_MH(env, phi_vec_uni)
    else:
        P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=5000, verbose=False)

    
    sim = Simulator(env, Sensor(sigma_map=env.noise_sigma), ObservationLogger())


    # gp = GPInfoMap(l_space=2.0, l_time=10.0, nu=1.5, noise=max(args.sensor_sigma**2, 1e-3))
    gp = GPInfoMap(l_space=2.0, l_time=100.0, nu=1.5, noise=1e-12)
    gp_phi_logger = GPPhi(
        gp,
        mode=args.gp_mode,
        beta_m=args.ucb_beta_m,
        beta_u=args.ucb_beta_u,
        base=1e-3
    )


    counts_total = np.zeros(env.nstates, dtype=int)
    counts_running = np.zeros(env.nstates, dtype=int) 
    traj_idx_all, traj_rc_all = [], []



    # for animation
    emp_series = []                 # list of (nstates,) empirical distros over time
    pi_series  = []                 # list of (nstates,) target distros used at that time
    cr = np.zeros(env.nstates, dtype=int)  # running counts for empirical distro
    global_t = 0                    # total steps elapsed
    first_hit_step = np.full(env.nstates, -1, dtype=int)  # -1 = never visited
    unseen = env.nstates                                 # how many still unvisited
    cover_time = -1  


    start_rc = None
    t0 = 0
    # n_updates = (total_steps + dt - 1) // dt
    belief_phi_map = np.zeros((total_steps, last_phi_vec.shape[0]), dtype=float)
    

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
    true_phi_map_time = np.zeros((total_steps, last_phi_vec.shape[0]), dtype=float)
    initial_true_phi = env.build_map_vec(env.true_phi)
    true_phi_map_time[0:steps_per_cfg, :] = initial_true_phi
    current_true_phi = initial_true_phi


    since_gp = 0
    since_p  = 0
    belief_phi_map[0, :] = last_phi_vec.copy()
    next_map_change = steps_per_cfg

    while t0 < total_steps:

        # initial distro
        # phi_loop_init = last_phi_vec.copy()


        # T_dt = min(dt, total_steps - t0)
        T_dt = min(
            total_steps - t0,
            GP_UPDATE_EVERY - since_gp,
            P_UPDATE_EVERY  - since_p
        )



        current_pi_vec = last_phi_vec / last_phi_vec.sum()
        pi_t = last_phi_vec / last_phi_vec.sum()
        denom = counts_total.sum()
        emp_t = (counts_total / denom) if denom > 0 else np.zeros_like(counts_total, dtype=float)
        # plot_distributions(env, emp_t, pi_t, f"results/{stem}__T{t0}emp_vs_pi.png")

        # 2a) simulate for K steps under current P
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
            r_hat = cr / global_t
            emp_series.append(r_hat.copy())
            pi_series.append(current_pi_vec.copy())
            if first_hit_step[s] == -1:
                first_hit_step[s] = global_t
                unseen -= 1
                if unseen == 0 and cover_time == -1:
                    cover_time = global_t
                    # print("=== COVERAGE ACHIEVED at step", cover_time, "===")

        # 2b) add observations to GP 
        ts = t0 + np.arange(len(traj_rc), dtype=float)
        rs = traj_rc[:, 0].astype(float)
        cs = traj_rc[:, 1].astype(float)

        cell_noise = env.noise_sigma[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
        y_true = env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
        noise   = np.random.normal(0.0, cell_noise)
        y_noisy = y_true + noise
        # yvals = env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float) #


        obs_r_all.append(rs)
        obs_c_all.append(cs)
        obs_t_all.append(ts)
        obs_y_all.append(y_noisy)

        k = 0.7
        proxy_var = np.clip((k * cell_noise)**2, 1e-6, 1e-1)
        gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=proxy_var)
        # gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=cell_noise**2)
        fitted = gp.fit_if_ready(min_pts=1)
        # fitted = True
        
        t1 = t0 + T_dt
        since_gp += T_dt
        since_p  += T_dt
        
        phi_belief_now = gp_phi_logger.phi_vec(env, t=t1 - 1)
        belief_phi_map[t0:t1, :] = phi_belief_now

        if since_gp >= GP_UPDATE_EVERY:
            fitted = gp.fit_if_ready(min_pts=1)
            if fitted:
                # refresh last_phi_vec from the *current* GP posterior
                last_phi_vec = gp_phi_logger.phi_vec(env, t=t1 - 1).copy()
                gp_fit_steps.append(t1 - 1)
                # belief_phi_map[t1 - 1, :] = last_phi_vec
                # belief_phi_map[t1-GP_UPDATE_EVERY : t1, :] = last_phi_vec
                print(f"[t={t1}] GP refit done. Updated target phi_vec., GP update {GP_UPDATE_EVERY}")
            since_gp = 0

            # last_phi_vec = phi_vec.copy()
            # env.write_vec_to_phi(last_phi_vec)
            # plot_distributions(env, counts_to_field(env, counts_total / counts_total.sum()), gaussian_phi, f"results/{stem}__T{t0+T_dt}_GPBel_emp_vs_pi.png")

        if since_p >= P_UPDATE_EVERY:
            if args.method == "mh":
                P = build_transition_matrix_MH(env, last_phi_vec)
            else:
                P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=10000, verbose=False)
            p_rebuild_steps.append(t1 - 1)
            # store a snapshot of P (object array later)
            P_snapshots.append(P.copy())
            since_p = 0
            print(f"[t={t1}] P updated.  P update {P_UPDATE_EVERY}")
     
        
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
                
                next_map_change += steps_per_cfg
        
        true_phi_map_time[t0:t1, :] = current_true_phi

        # if t0 == 500:
        #     ## this is hardcoded test
        #     env = update_env(neighbors=args.neighbors, seed=1, time=500)
        #     # env.build_map_vec(env.true_phi)
        #     # true_phi_map_time[500:,:] = env.build_map_vec(env.true_phi)

        #     plot_grid(env, f"results/{stem}_{t0}_gp_grid_true.png", title="Grid (true info map)")
            # current_true_phi = env.build_map_vec(env.true_phi) 
            
        
 

        # advance
        start_rc = tuple(traj_rc[-1])
        t0 = t1

        
        

    # 3) Final evaluation/plots vs lastphi_hat(if no GP fit ever happened)
    emp = counts_total / counts_total.sum()
    pi_vec = last_phi_vec / last_phi_vec.sum()

    traj_idx_all = np.concatenate(traj_idx_all, axis=0)
    traj_rc_all = np.vstack(traj_rc_all)

    # plot_trajectory(env, traj_rc_all, f"results/{stem}_trajectory.png",
    #                 title="Trajectory with periodic GP updates")
    # plot_distributions(env, emp, pi_vec, f"results/{stem}_emp_vs_pi_final.png")

    if len(emp_series) > 0:
        emp_arr = np.vstack(emp_series)
        pi_arr = np.vstack(pi_series)
        error_series = [np.linalg.norm(emp_arr[t] - pi_arr[t]) for t in range(len(emp_arr))]
    else:
        emp_arr = np.zeros((0, env.nstates), dtype=float)
        pi_arr = np.zeros((0, env.nstates), dtype=float)
        error_series = []

    # error_series = [np.linalg.norm(emp_series[t] - pi_series[t]) for t in range(len(emp_series))]
    # true_phi = env.build_map_vec(env.true_phi) 

    # plot_convergence(emp_arr, pi_vec, f"results/{stem}_convergence.png", true_phi, belief_phi_map)

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
        animate_distributions(
            env,
            emp_series=emp_series,
            pi_series=pi_series,
            filename=f"results/anim_{stem}__emp_vs_pi.mp4",
            fps=args.fps,
            stride=args.stride,
            dpi=150
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
        emp_series=emp_arr,
        pi_series=pi_arr,
        belief_phi_map=belief_phi_map,
        error_series=error_series,
        # trajectory
        traj_idx_all=traj_idx_all,
        traj_rc_all=traj_rc_all,
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