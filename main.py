#!/usr/bin/env python main.py

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oracle", "gp"], default="gp") # known vs unknown map distribution
    p.add_argument("--method", choices=["mh", "remc"], default="remc")
    p.add_argument("--T", type=int, default=2000) # total steps 
    p.add_argument("--neighbors", choices=["4", "8"], default="4")
    p.add_argument("--update_P_every", type=int, default=25) # this is for the P matrix
    p.add_argument("--update_GP_every", type=int, default=25) # this is for the GP update
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--gp_mode", choices=["ucb", "mean", "variance"], default="mean")
    p.add_argument("--ucb_beta_m", type=float, default=1.0)
    p.add_argument("--ucb_beta_u", type=float, default=1.0)
    p.add_argument("--l_space", type=float, default=2.0)
    p.add_argument("--l_time", type=float, default=10.0)



    p.add_argument("--animation", type=bool, default=True)
    p.add_argument("--fps", type=int, default=10, help="Frames per second.")
    p.add_argument("--stride", type=int, default=1, help="Use every k-th frame.")

    # static vs dynamic environment
    p.add_argument(
        "--env_dyn",
        choices=["static", "dynamic"],
        default="dynamic",
        help="Environment type: 'static' (single config) or 'dynamic' (folder of JSON snapshots)",
    )

    p.add_argument(
        "--n_agents",
        type=int,
        default=1,
        help="Number of ergodic agents (1 = original single-agent behavior).",
    )

    p.add_argument(
        "--starts",
        type=str,
        default="3,1",
        help='Semicolon-separated start cells per agent: "r,c;r,c;..." e.g. "3,1;4,1"'
    )

    # For static: this is a single JSON config.
    # For dynamic: this should be a folder containing multiple JSON files.
    # p.add_argument(
    #     "--config",
    #     type=str,
    #     default="configs/simple5x5_static.json",
    #     help="Path to JSON config file for map (static) or folder (dynamic)",
    # )
    p.add_argument(
        "--config",
        type=str,
        default="configs/dynamic_simple5x5_1/",
        help="Path to JSON config file for map (static) or folder (dynamic)",
    )


    p.add_argument(
        "--comm_mode",
        choices=["global", "local"],
        default="local",
        help=(
            "Communication / consensus topology: "
            "'global' = all-to-all (complete graph), "
            "'local' = only neighbors within proximity / comm_radius."
        ),
    )

    p.add_argument(
        "--comm_radius",
        type=float,
        default=2.0,
        help=(
            "Radius (in grid steps) for local communication when comm_mode='local'. "
            "Used to define which agents are 'neighbors' for consensus."
        ),
    )

    p.add_argument(
        "--consensus_iters",
        type=int,
        default=5,
        help=(
            "Number of consensus iterations per GP/P update step. "
            "Set to 1 for cheap, >1 to approximate better mixing."
        ),
    )






    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "oracle":
        from experiments.run_oracle import run as run_oracle
        run_oracle(args)
    else:
        from experiments.run_gp import run as run_gp
        run_gp(args)



