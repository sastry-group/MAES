#!/usr/bin/env python main.py

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oracle", "gp"], default="oracle")
    p.add_argument("--method", choices=["mh", "remc"], default="mh")
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--neighbors", choices=["4", "8"], default="4")
    p.add_argument("--update_every", type=int, default=500) # this is for GP mode
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--sensor_sigma", type=float, default=0.5)
    p.add_argument("--gp_mode", choices=["ucb", "mean", "variance"], default="ucb")
    p.add_argument("--ucb_beta_m", type=float, default=1.0)
    p.add_argument("--ucb_beta_u", type=float, default=1.0)
    p.add_argument("--l_space", type=float, default=2.0)
    p.add_argument("--l_time", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=10, help="Frames per second.")
    p.add_argument("--stride", type=int, default=1, help="Use every k-th frame.")

    ## these are for dynamic envs
    p.add_argument("--dynamic_env", action="store_true",
                help="If set, environment (ROIs) evolves over time and P is recomputed periodically.", default=True)
    p.add_argument("--env_update_every", type=int, default=20,
                help="How often to update environment + recompute P when --dynamic_env is on.") # this is only for oracle
    p.add_argument("--rois_json", type=str, default="configs/simple5x5.json",
                help="Path to JSON file describing ROIs (moving/spreading). Used if --dynamic_env.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "oracle":
        from experiments.run_oracle import run as run_oracle
        run_oracle(args)
    else:
        from experiments.run_gp import run as run_gp
        run_gp(args)



