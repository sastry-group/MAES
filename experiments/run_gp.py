import os
import numpy as np

from core.types import GridSpec
from env.grid import GridMC, OBSTACLE
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import counts_to_field, plot_grid, plot_trajectory, plot_distributions, plot_convergence, vector_to_field, counts_to_field
from info_maps.gp_model import GPInfoMap
from info_maps.info_map_sources import UniformPhi, GPPhi
from viz.animation import animate_trajectory, animate_distributions


def make_env(neighbors="4", seed=1):
    spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
    env = GridMC(spec)
    # Define the “true world” (oracle)
    env.set_roi_rect(0, 0, 1, 1, weight=3.0)
    env.set_roi_rect(0, 3, 2, 4, weight=6.0)
    env.set_start_rect(3, 1, 4, 1)
    env.set_obstacles([(2, 2)])
    return env

def normalize_nonneg(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    return v / (s + eps)


def run(args):
    os.makedirs("results", exist_ok=True)
    stem = f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}"

    # 0) Environment + plotting of true map for reference
    env = make_env(neighbors=args.neighbors, seed=1)
    plot_grid(env, f"results/gp_grid_true.png", title="Grid (true info map)")

    # 1) INITIAL P: using a uniform phi over free cells to explore
    phi_vec_uni = UniformPhi().phi_vec(env)  
    last_phi_vec = phi_vec_uni.copy() 
    
    if args.method == "mh":
        P = build_transition_matrix_MH(env, phi_vec_uni)
    else:
        P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=5000, verbose=False)

    
    sim = Simulator(env, Sensor(sigma=args.sensor_sigma), ObservationLogger())

    sensor_sigma = float(args.sensor_sigma)

    # gp = GPInfoMap(l_space=2.0, l_time=10.0, nu=1.5, noise=max(args.sensor_sigma**2, 1e-3))
    gp = GPInfoMap(l_space=2.0, l_time=100.0, nu=1.5, noise=1e-12)
    gp_phi_logger = GPPhi(
        gp,
        mode=args.gp_mode,
        beta_m=args.ucb_beta_m,
        beta_u=args.ucb_beta_u,
        base=1e-3
    )


    # Accumulate across the whole run
    total_steps = args.T
    dt= max(1, args.update_every) # steps between updates
    counts_total = np.zeros(env.nstates, dtype=int)
    counts_running = np.zeros(env.nstates, dtype=int) 
    traj_idx_all, traj_rc_all = [], []
    GP_UPDATE_EVERY = 25
    P_UPDATE_EVERY  = args.update_every  # e.g., rebuild P every 500 steps

    # for animation
    emp_series = []                 # list of (nstates,) empirical distros over time
    pi_series  = []                 # list of (nstates,) target distros used at that time
    cr = np.zeros(env.nstates, dtype=int)  # running counts for empirical distro
    global_t = 0                    # total steps elapsed



    start_rc = None
    t0 = 0
    # n_updates = (total_steps + dt - 1) // dt
    belief_phi_map = np.zeros((total_steps, last_phi_vec.shape[0]), dtype=float)
    since_gp = 0
    since_p  = 0
    belief_phi_map[0, :] = last_phi_vec.copy()

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
        emp_t = counts_total / counts_total.sum() 
        plot_distributions(env, emp_t, pi_t, f"results/{stem}__T{t0}emp_vs_pi.png")

        # 2a) simulate for K steps under current P
        traj_idx, traj_rc, counts = sim.simulate(P, T_dt, start=start_rc)
        counts_total += counts
        
        counts_running += counts
        traj_idx_all.append(traj_idx)
        traj_rc_all.append(traj_rc)

        # appending for animation
        for s in traj_idx:
            global_t += 1
            cr[s] += 1
            r_hat = cr / global_t
            emp_series.append(r_hat.copy())
            pi_series.append(current_pi_vec.copy())

        # 2b) add observations to GP 
        ts = t0 + np.arange(len(traj_rc), dtype=float)
        rs = traj_rc[:, 0].astype(float)
        cs = traj_rc[:, 1].astype(float)
        y_true = env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float)
        y_noisy = y_true + np.random.normal(0.0, sensor_sigma, size=len(y_true))
        # yvals = env.true_phi[traj_rc[:, 0], traj_rc[:, 1]].astype(float) #
        gp.add_batch(r=rs, c=cs, t=ts, y=y_noisy, noise_var=np.full(len(y_noisy), sensor_sigma**2))
        fitted = gp.fit_if_ready(min_pts=20)
        # fitted = True
        
        t1 = t0 + T_dt
        since_gp += T_dt
        since_p  += T_dt

        phi_belief_now = gp_phi_logger.phi_vec(env, t=t1 - 1)
        belief_phi_map[t0:t1, :] = phi_belief_now

        if since_gp >= GP_UPDATE_EVERY:
            fitted = gp.fit_if_ready(min_pts=5)
            if fitted:
                # refresh last_phi_vec from the *current* GP posterior
                last_phi_vec = gp_phi_logger.phi_vec(env, t=t1 - 1).copy()
            since_gp = 0

            # last_phi_vec = phi_vec.copy()
            # env.write_vec_to_phi(last_phi_vec)
            # plot_distributions(env, counts_to_field(env, counts_total / counts_total.sum()), gaussian_phi, f"results/{stem}__T{t0+T_dt}_GPBel_emp_vs_pi.png")

        if since_p >= P_UPDATE_EVERY:
            if args.method == "mh":
                P = build_transition_matrix_MH(env, last_phi_vec)
            else:
                P = build_transition_matrix_REMC(env, last_phi_vec, max_iters=10000, verbose=False)
            since_p = 0
 

        # advance
        start_rc = tuple(traj_rc[-1])
        t0 = t1

        
        

    # 3) Final evaluation/plots vs lastphi_hat(if no GP fit ever happened)
    emp = counts_total / counts_total.sum()
    pi_vec = last_phi_vec / last_phi_vec.sum()

    traj_idx_all = np.concatenate(traj_idx_all, axis=0)
    traj_rc_all = np.vstack(traj_rc_all)

    plot_trajectory(env, traj_rc_all, f"results/{stem}_trajectory.png",
                    title="Trajectory with periodic GP updates")
    plot_distributions(env, emp, pi_vec, f"results/{stem}_emp_vs_pi_final.png")


    error_series = [np.linalg.norm(emp_series[t] - pi_series[t]) for t in range(len(emp_series))]
    true_phi = env.build_map_vec(env.true_phi) 
    plot_convergence(error_series, pi_vec, f"results/{stem}_convergence.png", true_phi, belief_phi_map)

    print("Periodic-GP run complete:", stem)
    print(f"Final L2 error ||r_hat_T - pi|| = {error_series[-1]:.4f}")
    print(f"[GP] Saved plots + data under results/ with name: {stem}")

#     animate_trajectory(
#         env, traj_rc_all,
#         filename=f"results/anim_{stem}.mp4",
#         fps=args.fps, stride=args.stride
#     )
#     animate_distributions(
#         env,
#         emp_series=emp_series,
#         pi_series=pi_series,
#         filename=f"results/anim_{stem}__emp_vs_pi.mp4",
#         fps=args.fps,
#         stride=args.stride,
#         dpi=150
# )