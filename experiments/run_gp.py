import os
import numpy as np
import glob
import re
import h5py

from core.types import GridSpec
from env.grid import GridMC,  make_env_from_json
from viz.plotting import  plot_grid
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


def parse_attrition(attrition_str):
    """Parse '200:1;400:2' -> {200: 1, 400: 2}"""
    schedule = {}
    if not attrition_str:
        return schedule
    for pair in attrition_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        step_str, agent_str = pair.split(":")
        schedule[int(step_str)] = int(agent_str)
    return schedule

def run(args):
    results_dir = "results"
    stem = f"T{args.T}_{args.planner}_GPupdate{args.update_GP_every}"

    if args.planner == "ergodic":
        stem += f"_{args.method}_Pupdate{args.update_P_every}"



    n_agents = getattr(args, "n_agents", 1)
    map_id = extract_map_id(args.config)
    stem += f"_{map_id}"
    start_tag = args.starts.split(";")[0]  # e.g. "0,0"
    start_tag = start_tag.replace(",", "")
    stem += f"_gp{args.gp_mode}_{args.env_dyn}_{n_agents}_start{start_tag}x{n_agents}"

    if n_agents > 1:
        stem += f"_commMode{getattr(args, 'comm_mode', 'global')}_comm_radius{getattr(args, 'comm_radius', 'NA')}_concenIter{getattr(args, 'consensus_iters', 'NA')}_kr10_ucbbeta3"



    if args.attrition:
        stem += f"_attrition"
        
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
        if args.env_change_steps:
            steps_per_cfg = args.env_change_steps[0]
            # assert len(steps_per_cfg) + 1 == n_cfg 
        else:
            steps_per_cfg = int(np.ceil(total_steps / n_cfg)) 


    else:  # args.env_dyn == "static"
        env = make_env_from_json(config_path)
        cfg_paths = None
        n_cfg = 1
        current_cfg_idx = 0
        steps_per_cfg = total_steps
    plot_grid(env, os.path.join(run_dir, "gp_grid_true.png"), title="Grid (true info map)")

    attrition_schedule = {}
    if args.attrition:
        attrition_schedule = parse_attrition(args.attrition_steps)
        if not attrition_schedule:
            raise ValueError("--attrition is set but --attrition_steps is empty")



    planner = make_planner(args)
    agent_initial_positions = parse_starts(args.starts)
    h5_path = os.path.join(run_dir, "run.h5")
    h5_file = h5py.File(h5_path, "w")
        
    planner.reset(
        env=env,
        args=args,                # optional if planner needs it (but ideally keep in run_gp)
        agents_initial_positions=agent_initial_positions,      # list of (r,c)
        steps_per_cfg=steps_per_cfg,
        cfg_paths=cfg_paths,            # only for dynamic envs (list of json paths
        config_path=config_path,        # only for static envs (single json path)
        stem=run_dir,
        h5_file=h5_file, 
        attrition_schedule=attrition_schedule,
    )


    for _ in  range(args.T):
        planner.step()

    planner.finalize()

    return None

     
