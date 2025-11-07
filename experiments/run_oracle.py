import os
import numpy as np



from core.types import GridSpec
from env.grid import GridMC, OBSTACLE, START, make_env, load_dynamic_config, make_env_from_json, update_map
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import plot_grid, plot_trajectory, plot_distributions, plot_convergence
from viz.animation import animate_trajectory



def _build_P(env, phi_vec, method):
    if method == "mh":
        return build_transition_matrix_MH(env, phi_vec)
    return build_transition_matrix_REMC(env, phi_vec, max_iters=5000, verbose=False)

def run(args):
    os.makedirs("results", exist_ok=True)
    file_name= f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}"

    # 1) Build env
    if args.config is not None:
        env = make_env_from_json(args.config)
        roi_specs = load_dynamic_config(args.config) if args.env_dyn == "dynamic" else []
    else:
        env = make_env(neighbors=args.neighbors, seed=1)
    plot_grid(env, f"results/map_init_grid_{file_name}.png", title="Grid")



    # 2) Freeze true map & build phi_vec
    #    (true_phi shows the real world you want to target)
    env.true_phi[env.grid == OBSTACLE] = 0.0
    phi_vec = env.build_map_vec(env.true_phi)   

    # 3) Build P from oracle map 
    P = _build_P(env, phi_vec, args.method)
    
    # 4) Simulate

    sim = Simulator(env, Sensor(sigma=args.sensor_sigma), ObservationLogger())


    if getattr(args, "env_dyn", "static") == "static":

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
    else:

        T = args.T

        cur_start = (4, 0)
        traj_idx_all, traj_rc_all = [], []
        counts_all = np.zeros(env.nstates, dtype=int)
        time2rois = {}
        for name, r in roi_specs.items():
            rate = int(r.get("rate", 0))
            if rate > 0:
                for t in range(rate, T + 1, rate):
                    time2rois.setdefault(t, []).append(name)

        # Also handy: the sorted time chunks
        T_chunks = sorted(time2rois.keys())


        # initial map
        env.true_phi[env.grid == OBSTACLE] = 0.0
        phi_vec = env.build_map_vec(env.true_phi)
        P = _build_P(env, phi_vec, args.method)
        t = 0
        while t < T:

            next_event = next((tt for tt in T_chunks if tt > t), T)
            steps = min(next_event - t, T - t)
            steps = max(1, steps)

            # simulate chunk
            idx, rc, cnt = sim.simulate(P, steps, start=cur_start)
            traj_idx_all.extend(idx)
            traj_rc_all.extend(rc)
            counts_all += cnt
            t += steps
            cur_start = rc[-1]

            if t in time2rois:
                update_map(env, time2rois[t], roi_specs)
                phi_vec = env.build_map_vec(env.true_phi)
                P = _build_P(env, phi_vec, args.method)

        pi = env.target_distribution()
        emp = counts_all / counts_all.sum()

        error_series = []
        cr = np.zeros_like(counts_all)
        for k, s in enumerate(traj_idx_all, start=1):
            cr[s] += 1
            r_hat = cr / k
            error_series.append(np.linalg.norm(r_hat - pi))

        plot_trajectory(env, traj_rc_all, filename=f"results/trajectory_{file_name}.png")
        plot_distributions(env, emp, pi, f"results/distributions_{file_name}.png")
        plot_convergence(error_series, pi, f"results/convergence_{file_name}.png")

        # np.savez_compressed(
        #     f"results/run_{file_name}.npz",
        #     phi_vec=phi_vec, pi=pi, emp=emp,
        #     counts=counts_all,
        #     traj_idx=np.array(traj_idx_all),
        #     traj_rc=np.array(traj_rc_all),
        #     time2rois=np.array(sorted((t, time2rois[t]) for t in time2rois), dtype=object),
        # )
        print(f"[dynamic] Saved plots + data under results/: {file_name}")