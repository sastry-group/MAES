import os
import numpy as np


from core.types import GridSpec
from env.grid import GridMC, OBSTACLE
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import plot_grid, plot_trajectory, plot_distributions, plot_convergence
from viz.animation import animate_trajectory

def make_env(neighbors="4", seed=1):
    spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
    env = GridMC(spec)
    # Define the “true world” (oracle)
    env.set_roi_rect(0, 0, 1, 1, weight=3.0)
    env.set_roi_rect(0, 3, 2, 4, weight=6.0)
    env.set_start_rect(3, 1, 4, 1)
    env.set_obstacles([(2, 2)])
    return env

def run(args):
    os.makedirs("results", exist_ok=True)
    file_name= f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}"

    # 1) Build env
    env = make_env(neighbors=args.neighbors, seed=1)

    # 2) Freeze true map & build phi_vec
    #    (true_phi shows the real world you want to target)
    env.true_phi[env.grid == OBSTACLE] = 0.0
    phi_vec = env.build_map_vec(env.true_phi)   

    # 3) Build P from oracle map 
    if args.method == "mh":
        P = build_transition_matrix_MH(env, phi_vec)
    else: 
        P = build_transition_matrix_REMC(env, phi_vec, max_iters=5000, verbose=True)

    # 4) Simulate
    sim = Simulator(env, Sensor(sigma=0.0), ObservationLogger())
    traj_idx, traj_rc, counts = sim.simulate(P, args.T, start=(4,0))

    # 5) Distributions (pi from the same phi_vec you used to build P)
    pi = env.target_distribution()
    emp = counts / counts.sum()

    # 6) Convergence series
    emp_series = []
    error_series = []          # store L2 error at each step
    cr = np.zeros_like(counts)
    for t, s in enumerate(traj_idx, start=1):
        cr[s] += 1
        # emp_series.append(cr / t)
        r_hat = cr / t
        emp_series.append(r_hat)
        error_series.append(np.linalg.norm(r_hat - pi)) 

    # 7) Plots
    plot_grid(env, f"results/map_{file_name}.png", title="Grid")
    plot_trajectory(env, traj_rc, filename = f"results/trajectory_{file_name}.png")
    plot_distributions(env, emp, pi, f"results/distributions_{file_name}.png")
    plot_convergence(error_series, pi, f"results/convergence_{file_name}.png")

    np.savez_compressed(
        f"results/run_{file_name}.npz",
        phi_vec=phi_vec, pi=pi, emp=emp,
        counts=counts, traj_idx=traj_idx, traj_rc=traj_rc
    )

    # print(f"L1(emp, pi) = {np.abs(emp - pi).sum():.4f}")
    print(f"Final L2 error ||r_hat_T - pi|| = {error_series[-1]:.4f}")
    print(f"[oracle] Saved plots + data under results/ with name: {file_name}")

    # animate_trajectory(
    #     env, traj_rc,
    #     filename=f"results/anim_{file_name}.mp4",
    #     fps=args.fps, stride=args.stride
    # )
