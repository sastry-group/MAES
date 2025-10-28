import os
import numpy as np
import json


from core.types import GridSpec
from env.grid import GridMC, OBSTACLE, START
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import plot_grid, plot_trajectory, plot_distributions, plot_convergence
from viz.animation import animate_trajectory

def make_env(args):
    spec = GridSpec(width=5, height=5, neighbors=args.neighbors, rng_seed=1)
    env = GridMC(spec)
    # baseline world
    env.set_start_rect(3, 1, 4, 1)
    env.set_obstacles([(2, 2)])


    base_dir = os.path.dirname(__file__)
    roi_path = args.rois_json
    if not os.path.isabs(roi_path):
        roi_path = os.path.join(base_dir, "..", roi_path)  # one level up (from experiments/)

    roi_path = os.path.abspath(roi_path)
    print(f"[INFO] Loading ROI config from {roi_path}")
    if getattr(args, "dynamic_env", True) and os.path.exists(roi_path):
        with open(roi_path, "r") as f:
            rois = json.load(f)
        for r in rois:
            r0, c0, r1, c1 = r["rect"]
            w = float(r.get("weight", 1.0))
            rtype = r.get("type", "move")
            if rtype == "move":
                m = r.get("move", {})
                env.add_moving_roi(
                    r0, c0, r1, c1,
                    dr=int(m.get("dr", 0)),
                    dc=int(m.get("dc", 0)),
                    every_k=int(m.get("every_k", 1)),
                    weight=w,
                    bounce=bool(m.get("bounce", True))
                )
            elif rtype == "spread":
                s = r.get("spread", {})
                env.add_spreading_roi(
                    r0, c0, r1, c1,
                    p=float(s.get("p", 0.25)),
                    every_k=int(s.get("every_k", 1)),
                    weight=w
                )
            else:
                env.add_static_roi_rect(r0, c0, r1, c1, weight=w)

        env._rebuild_true_phi_from_rois()
    else:
        # static default
        env.set_roi_rect(0, 0, 1, 1, weight=3.0)
        env.set_roi_rect(0, 3, 2, 4, weight=6.0)

    return env

def _build_P(env, phi_vec, method):
    if method == "mh":
        return build_transition_matrix_MH(env, phi_vec)
    return build_transition_matrix_REMC(env, phi_vec, max_iters=5000, verbose=False)

def run(args):
    os.makedirs("results", exist_ok=True)
    file_name= f"T{args.T}_{args.method}_{args.neighbors}_mode{args.mode}"

    # 1) Build env
    env = make_env(args)

    # 2) Freeze true map & build phi_vec
    #    (true_phi shows the real world you want to target)
    env.true_phi[env.grid == OBSTACLE] = 0.0
    phi_vec = env.build_map_vec(env.true_phi)   

    # 3) Build P from oracle map 
    P = _build_P(env, phi_vec, args.method)

    # 4) Simulate
    if not getattr(args, "dynamic_env", False): 
        #static path
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

    # dynamic map
    file_name += "_dyn"

    # precomputed maps
    s2i = env._state_index_map
    i2s = env._index_state_map

    # start state (prefer START cell)
    start_rc = next(((r, c) for r in range(env.H) for c in range(env.W) if env.grid[r, c] == START), i2s[0])
    s = s2i[start_rc]

    traj_idx, traj_rc = [], []
    counts = np.zeros(env.nstates, dtype=np.int64)
    emp_series, error_series = [], []

    updates = 0
    for t in range(1, args.T + 1):
        # record
        traj_idx.append(s)
        traj_rc.append(i2s[s])
        counts[s] += 1

        # track vs current pi_t
        r_hat = counts / t
        pi_t = env.target_distribution()
        emp_series.append(r_hat.copy())
        error_series.append(np.linalg.norm(r_hat - pi_t))

        # step with current P
        r = P.getrow(s)               # CSR row
        idx = r.indices               # indices of nonzero transitions
        probs = r.data.astype(float) 
        psum = probs.sum()
        if psum > 0:
            probs /= psum
            s = np.random.choice(idx, p=probs)
        else:
            # fallback (if row has no outgoing edges, very rare)
            s = np.random.randint(P.shape[1])

        # periodic env update + P recompute
        if t % args.env_update_every == 0:
            env.step_dynamic_rois(updates)  # uses per-ROI cadences
            updates += 1
            env.true_phi[env.grid == OBSTACLE] = 0.0
            phi_vec = env.build_map_vec(env.true_phi)
            P = _build_P(env, phi_vec, args.method)
            plot_grid(env, f"results/map_{file_name}_{t}.png", title="Grid (dynamic)")
        

    # final summaries vs last pi_T
    pi_T = env.target_distribution()
    emp = counts / counts.sum()

    plot_grid(env, f"results/map_{file_name}.png", title="Grid (dynamic)")
    plot_trajectory(env, traj_rc, filename=f"results/trajectory_{file_name}.png")
    plot_distributions(env, emp, pi_T, f"results/distributions_{file_name}.png")
    plot_convergence(error_series, pi_T, f"results/convergence_{file_name}.png")

    np.savez_compressed(
        f"results/run_{file_name}.npz",
        phi_vec=phi_vec, pi_T=pi_T, emp=emp,
        counts=counts, traj_idx=np.array(traj_idx), traj_rc=np.array(traj_rc),
        error_series=np.array(error_series)
    )
    print(f"Final L2 to last pi_T (dynamic): {error_series[-1]:.4f}")
    print(f"[oracle-dynamic] Saved plots + data under results/ with name: {file_name}")

