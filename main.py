#!/usr/bin/env python main.py

import argparse


def parse_args(argv=None):
 
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oracle", "gp"], default="gp") # known vs unknown map distribution
    p.add_argument("--planner", choices=["ergodic", "mac_dt"], default="ergodic", help="Which planner to use.")
    p.add_argument("--method", choices=["mh", "remc"], default="mh")
    p.add_argument("--T", type=int, default=4000) # total steps 
    p.add_argument("--neighbors", choices=["4", "8"], default="4")
    p.add_argument("--update_P_every", type=int, default=25) # this is for the P matrix
    p.add_argument("--update_GP_every", type=int, default=25) # this is for the GP update
    p.add_argument("--gp_mode", choices=["ucb", "mean", "variance"], default="ucb")
    p.add_argument("--init_phi_scale", type=float, default=6.0)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--ucb_beta_m", type=float, default=1.0)
    p.add_argument("--ucb_beta_u", type=float, default=3.0)
    p.add_argument("--l_space", type=float, default=2.0)
    p.add_argument("--l_time", type=float, default=10000.0)
    p.add_argument("--kernel", choices=["matern12", "matern32", "matern52", "rbf", "prod_matern_exp"], default="matern32")
    p.add_argument("--kernel_amp", type=float, default=10.0,
                help="prior variance: standardized units if --gp_prior data, phi^2 units if fixed")
    p.add_argument("--gp_prior", choices=["data", "fixed"], default="data")
    p.add_argument("--gp_prior_mean", type=float, default=0.0)
    p.add_argument("--learn_hypers", choices=["none", "once", "every"], default="none")
    p.add_argument("--ucb_delta", type=float, default=0.05)
    p.add_argument("--ucb_beta_schedule", choices=["const", "mac_dt"], default="const")
    p.add_argument("--gp_obs_hist", type=int, default=2000, help="FIFO budget B on GP data per agent")
    p.add_argument("--share_obs_hist", type=int, default=100, help="Steps of history shared on contact")
 
    # sensing noise, Eq. (3)
    p.add_argument("--noise_rel", type=float, default=0.05,
                   help="sigma_eps as a fraction of max phi*_0 (used unless --noise_sigma is set)")
    p.add_argument("--noise_sigma", type=float, default=None,
                   help="absolute sigma_eps in phi units; overrides --noise_rel")
    p.add_argument("--gp_noise_scale", type=float, default=1.0,
                   help="GP assumes (scale*sigma)^2; 1.0 = correctly specified (legacy: 0.7)")
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--env_change_steps", type=int, nargs="+", default=[500, 1000, 1500])
    p.add_argument("--large_scale", action="store_true")
    # p.add_argument("--attrition", action="store_true")
    # p.add_argument("--attrition_steps", type=list, default="")
 
    # p.add_argument("--shared_P", action="store_true")
    p.add_argument("--comm_delay", type=int, default=0, help="Set to 0 if no delay or >0 will set the max number of steps to delay comms")
    # p.add_argument("--comm_delay_prob", type=float, default=1.0,
    #                     help="Fraction of relayed packets that get delayed (0.0-1.0)")
 
 
 
    # static vs dynamic environment
    p.add_argument(
        "--env_dyn",
        choices=["static", "dynamic"],
        default="static",
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
        default="4,4",
        help='Semicolon-separated start cells per agent: "r,c;r,c;..." e.g. "3,1;4,1"'
    )
 
    # For static: this is a single JSON config.
    # For dynamic: this should be a folder containing multiple JSON files.
    # p.add_argument(
    #     "--config",
    #     type=str,
    #     default="configs/dynamic_simple5x5_v0",
    #     help="Path to JSON config file for map (static) or folder (dynamic)",
    # )
    p.add_argument(
        "--config",
        type=str,
        default="configs/simple5x5_static_v2.json",
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
        default=1.0,
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
 
    p.add_argument(
        "--attrition",
        action="store_true",
        help="Enable robot attrition (agents are removed over time).",
    )
 
    p.add_argument(
        "--attrition_steps",
        type=str,
        default="",
        help=(
            "Semicolon-separated 'step:agent_id' pairs. "
            "E.g. '200:1;400:2'. Only used if --attrition is set., if not just leave the default empty"
        ),
    )
 
 
 
 
    p.add_argument("--randomize_starts", action="store_true",
                   help="Ignore --starts; sample n_agents distinct free cells")
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--oracle_radius", type=int, default=0,
                   help="mac_dt: coverage radius of greedy destination oracle")
 
    return p.parse_args(argv)
 
if __name__ == "__main__":
    args = parse_args()
 
 
    if args.mode == "oracle":
        from experiments.run_oracle import run as run_oracle
        run_oracle(args)
    else:
        from experiments.run_gp import run as run_gp
        run_gp(args)