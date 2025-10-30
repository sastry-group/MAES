import os
import numpy as np
import json


from core.types import GridSpec
from env.grid import GridMC, OBSTACLE, START
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator, Sensor, ObservationLogger
from viz.plotting import plot_grid, plot_trajectory, plot_distributions, plot_convergence
from viz.animation import animate_trajectory


def load_dynamic_config(path):
    """Returns: list of {'name','mode','rate'} for each ROI that has 'dynamic'."""
    if path is None:
        return []

    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # MAES/
        path = os.path.join(base_dir, path)
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"config file not found: {abs_path}")

    with open(abs_path, "r") as f:
        cfg = json.load(f)

    rois = cfg.get("rois", [])
    roi_dict = {}
    for r in rois:
        d = r.get("dynamic")
        if not d:
            continue
        roi_dict[r.get("name", "roi")] = {
            "mode": d.get("mode", "static"),
            "rate": int(d.get("rate", 0))
        }
    return roi_dict

def make_env_from_json(path):
    if not os.path.isabs(path):
        # otherwise, interpret relative to this script’s parent (MAES/)
        base_dir = os.path.dirname(os.path.dirname(__file__))  # moves up from experiments/ to MAES/
        path = os.path.join(base_dir, path)

    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    
    with open(abs_path, "r") as f:
        cfg = json.load(f)

    g = cfg["grid"]
    spec = GridSpec(width=g["width"], height=g["height"],
                    neighbors=g.get("neighbors", "4"),
                    rng_seed=g.get("rng_seed", 1))
    env = GridMC(spec)

    # ROI setup
    for r in cfg["rois"]:
        if r["shape"]["type"] == "rect":
            x0, y0, x1, y1 = r["shape"]["rect"]
            env.set_roi_rect(x0, y0, x1, y1, weight=r["weight"])

    # Start
    s = cfg["start"]["rect"]
    env.set_start_rect(*s)

    # Obstacles
    obs = [tuple(o) for o in cfg["obstacles"]["cells"]]
    env.set_obstacles(obs)

    return env

def make_env(neighbors="4", seed=1):
    spec = GridSpec(width=5, height=5, neighbors=neighbors, rng_seed=seed)
    env = GridMC(spec)
    # Define the “true world” (oracle)
    env.set_roi_rect(0, 0, 1, 1, weight=3.0)
    env.set_roi_rect(0, 3, 2, 4, weight=6.0)
    env.set_start_rect(3, 1, 4, 1)
    env.set_obstacles([(2, 2)])
    return env

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
    else:
        env = make_env(neighbors=args.neighbors, seed=1)
    plot_grid(env, f"results/map_init_grid_{file_name}.png", title="Grid")


    # 2) Freeze true map & build phi_vec
    #    (true_phi shows the real world you want to target)
    env.true_phi[env.grid == OBSTACLE] = 0.0
    phi_vec = env.build_map_vec(env.true_phi)   

    # 3) Build P from oracle map 
    P = _build_P(env, phi_vec, args.method)
    # roi_dyn = load_dynamic_config(args.dynamic_config) if args.dynamic_config else []
    
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

   