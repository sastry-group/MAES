import os
import numpy as np

import re
import h5py

from core.types import GridSpec
from env.grid import GridMC, OBSTACLE, START,  load_dynamic_config, make_env_from_json, update_map
from transitions.builder import build_transition_matrix_MH, build_transition_matrix_REMC
from sim.runner import Simulator
from viz.plotting import plot_grid, plot_trajectory, plot_distributions, plot_convergence
from viz.animation import animate_trajectory

def parse_starts(starts_str: str) -> list[tuple[int,int]]:
    out = []
    for chunk in starts_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        r_s, c_s = chunk.split(",")
        out.append((int(r_s), int(c_s)))
    return out

def extract_map_id(config_path: str) -> str:
    # Remove trailing slash if present
    path = config_path.rstrip("/")

    # Take filename or folder name
    name = os.path.basename(path)

    # Remove .json if static
    name = name.replace(".json", "")

    # Extract version like v0, v1, v12, etc.
    match = re.search(r"(v\d+)", name)
    return match.group(1) if match else "unknown"

def _build_P(env, phi_vec, method):
    if method == "mh":
        return build_transition_matrix_MH(env, phi_vec)
    return build_transition_matrix_REMC(env, phi_vec, max_iters=5000, verbose=False)

def run(args):
    results_dir = "results"
    stem = f"T{args.T}_{args.planner}_GPupdate{args.update_GP_every}"

    if args.planner == "ergodic":
        stem += f"_{args.method}_Pupdate{args.update_P_every}"



    n_agents = getattr(args, "n_agents", 1)
    map_id = extract_map_id(args.config)
    stem += f"_{map_id}"
    start_tag = args.starts.split(";")[0]  # e.g. "0,0"
    # start_tag = start_tag.replace(",", "")
    stem += f"_gp{args.gp_mode}_{args.env_dyn}_{n_agents}_start{start_tag}x{n_agents}"

    if n_agents > 1:
        stem += f"_commMode{getattr(args, 'comm_mode', 'global')}_comm_radius{getattr(args, 'comm_radius', 'NA')}_concenIter{getattr(args, 'consensus_iters', 'NA')}"

    run_dir = os.path.join(results_dir, stem)
    # 1) Build env
    os.makedirs(run_dir, exist_ok=True)  
    config_path = args.config
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        config_path = os.path.join(base_dir, config_path)
    config_path = os.path.abspath(config_path)

    env = make_env_from_json(config_path, neighbors=args.neighbors)
        # roi_specs = load_dynamic_config(args.config_path) if args.env_dyn == "dynamic" else []

    plot_grid(env, os.path.join(run_dir, "gp_grid_true.png"), title="Grid (true info map)")




    # 2) Freeze true map & build phi_vec
    #    (true_phi shows the real world you want to target)
    # env.true_phi[env.grid == OBSTACLE if ] = 0.0 #FIX THIS BUG GABY
    true_phi_vec = env.build_map_vec(env.true_phi).astype(float)
    pi_vec = true_phi_vec / np.maximum(true_phi_vec.sum(), 1e-12)

    # 3) Build P from oracle map 
    P = _build_P(env, true_phi_vec, args.method)
    
    # 4) Simulate

    # sim = Simulator(env, Sensor(sigma_map=env.noise_sigma), ObservationLogger())
    sim = Simulator(env)

    if getattr(args, "env_dyn", "static") == "static":

        agent_initial_positions = parse_starts(args.starts)[0]
        traj_idx, traj_rc, _ = sim.simulate(P, args.T, agent_initial_positions)

        traj_idx = np.asarray(traj_idx)
        traj_rc  = np.asarray(traj_rc)

        if traj_idx.ndim == 1:
            traj_idx = traj_idx[:, None]
            traj_rc  = traj_rc[:, None, :]

        T, N = traj_idx.shape
        nstates = env.nstates

        belief_phi_time = np.tile(true_phi_vec[None, None, :], (T, N, 1))
        belief_phi_avg  = np.tile(true_phi_vec[None, :], (T, 1))
        true_phi_time   = np.tile(true_phi_vec[None, :], (T, 1))

        h5_path = os.path.join(run_dir, "run.h5")
        h5_file = h5py.File(h5_path, "w")

        h5_file.create_dataset("traj_states", data=traj_idx)
        h5_file.create_dataset("traj_rc_agents_all", data=traj_rc)
        h5_file.create_dataset("s", data=traj_idx)

        h5_file.create_dataset("belief_phi_map", data=belief_phi_time)
        h5_file.create_dataset("belief_phi_map_avg", data=belief_phi_avg)
        h5_file.create_dataset("true_phi_map_time", data=true_phi_time)

        h5_file.create_dataset("pi_vec_final", data=pi_vec)

        h5_file.create_dataset("grid_shape", data=np.array(env.true_phi.shape))
        h5_file.create_dataset("true_phi_map", data=env.true_phi.astype(float))

        h5_file.attrs["T"] = T
        h5_file.attrs["n_agents"] = N
        h5_file.attrs["planner"] = "oracle"
        h5_file.attrs["method"] = args.method
        h5_file.attrs["env_source"] = config_path
        h5_file.attrs["env_dyn"] = args.env_dyn

        roi_rc = []
        roi_name = []

        for roi in getattr(env, "roi_cells", []):
            name = roi["name"]
            r0, c0, r1, c1 = roi["rect"]

            for r in range(r0, r1):
                for c in range(c0, c1):
                    if (r, c) in env._state_index_map:
                        roi_rc.append((r, c))
                        roi_name.append(name)

        # ---- Store datasets ----
        h5_file.create_dataset("roi_rc", data=np.array(roi_rc, dtype=int))
        h5_file.create_dataset("roi_name", data=np.array(roi_name, dtype="S"))



        h5_file.create_dataset(
            "state_index_map_keys",
            data=np.array(list(env._state_index_map.keys()), dtype=int)
        )

        h5_file.create_dataset(
            "state_index_map_vals",
            data=np.array(list(env._state_index_map.values()), dtype=int)
        )




        # ---- Attributes (metadata) ----
        h5_file.attrs["stem"] = run_dir
        h5_file.attrs["neighbors"] = int(args.neighbors)
        h5_file.attrs["mode"] = str(args.mode)
        h5_file.attrs["gp_mode"] = str(args.gp_mode)
        h5_file.attrs["gp_update"] = getattr(args, "update_GP_every", 0)
        h5_file.attrs["gp_update"] = getattr(args, "update_P_every", 0)



        h5_file.attrs["comm_mode"] = str(getattr(args, "comm_mode", "NA"))
        h5_file.attrs["comm_radius"] = float(getattr(args, "comm_radius", np.nan))
        h5_file.attrs["consensus_iters"] = int(getattr(args, "consensus_iters", 0))

        h5_file.attrs["ucb_beta_m"] = float(args.ucb_beta_m)
        h5_file.attrs["ucb_beta_u"] = float(args.ucb_beta_u)

        h5_file.attrs["nstates"] = int(env.nstates)

        h5_file.close()
       