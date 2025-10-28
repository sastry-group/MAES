#!/usr/bin/env python3

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oracle", "gp"], default="gp")
    p.add_argument("--method", choices=["mh", "remc"], default="remc")
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--neighbors", choices=["4", "8"], default="4")
    p.add_argument("--update_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--sensor_sigma", type=float, default=0.5)
    p.add_argument("--gp_mode", choices=["ucb", "mean", "variance"], default="ucb")
    p.add_argument("--ucb_beta_m", type=float, default=1.0)
    p.add_argument("--ucb_beta_u", type=float, default=1.0)
    p.add_argument("--l_space", type=float, default=2.0)
    p.add_argument("--l_time", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=10, help="Frames per second.")
    p.add_argument("--stride", type=int, default=1, help="Use every k-th frame.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "oracle":
        from experiments.run_oracle import run as run_oracle
        run_oracle(args)
    else:
        from experiments.run_gp import run as run_gp
        run_gp(args)



