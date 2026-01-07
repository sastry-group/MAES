import os
import numpy as np
import glob
import re

from core.types import GridSpec
from env.grid import GridMC, make_env, update_env, make_env_from_json, load_env_sequence
from viz.plotting import counts_to_field, plot_grid, plot_trajectory, plot_distributions, plot_convergence, vector_to_field, counts_to_field
from experiments.planners.methods import make_planner


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




def run(args):
    results_dir = "results"
    stem = f"T{args.T}_{args.planner}_GPupdate{args.update_GP_every}"

    if args.planner == "ergodic":
        stem += f"_{args.method}_Pupdate{args.update_P_every}"



    n_agents = getattr(args, "n_agents", 1)
    map_id = extract_map_id(args.config)
    stem += f"_{map_id}"
    stem += f"_gp{args.gp_mode}_{args.env_dyn}_{n_agents}_starts{args.starts.replace(',', '').replace(';', '')}"

    if n_agents > 1:
        stem += f"_commMode{getattr(args, 'comm_mode', 'global')}_comm_radius{getattr(args, 'comm_radius', 'NA')}_concenIter{getattr(args, 'consensus_iters', 'NA')}"

    run_dir = os.path.join(results_dir, stem)

    os.makedirs(run_dir, exist_ok=True)  


    print(f"Running with n_agents = {n_agents}")

    # Checking if dynamic or static environment
    config_path = args.config
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(__file__)) 
        config_path = os.path.join(base_dir, config_path)
    config_path = os.path.abspath(config_path)

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
    plot_grid(env, os.path.join(run_dir, "gp_grid_true.png"), title="Grid (true info map)")

    planner = make_planner(args)
    agent_initial_positions = parse_starts(args.starts)

        
    planner.reset(
        env=env,
        args=args,                # optional if planner needs it (but ideally keep in run_gp)
        agents_initial_positions=agent_initial_positions,      # list of (r,c)
        steps_per_cfg=steps_per_cfg,
        cfg_paths=cfg_paths,            # only for dynamic envs (list of json paths
        config_path=config_path,        # only for static envs (single json path)
        stem=run_dir,
    )


    for _ in  range(args.T):
        planner.step()

    results = planner.finalize()

    return results

     
