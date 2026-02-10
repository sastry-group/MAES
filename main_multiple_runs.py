#!/usr/bin/env python

"""
Batch runner for multiple experiments using the current main.py arguments.
"""

from copy import deepcopy


# -------------------------------------------------
# Base args (shared across all runs)
# -------------------------------------------------


N_AGENTS = 15
START_CELL = "0,0"

STARTS_80 = ";".join([START_CELL] * N_AGENTS)

BASE_ARGS = dict(
    mode="gp",
    method="remc",
    T=3000,
    neighbors="4",
    update_P_every=1000,
    update_GP_every=1000,
    save_every=200,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=1.0,
    l_space=2.0,
    l_time=10.0,
    animation=False,
#     comm_radius=2.0,
    fps=10,
    stride=1,
    env_dyn="static",
    comm_mode="local",
    consensus_iters=5,
    n_agents=N_AGENTS,
)



EXPERIMENTS = [

    dict(
        planner="ergodic",
        env_dyn="static",
        comm_radius=15.0,   
        config="configs/simple20x20_static_v2.json",
        starts=STARTS_80,
    ),

    dict(
        planner="ergodic",
        env_dyn="static",
        comm_radius=3.0,   
        config="configs/simple20x20_static_v2.json",
        starts=STARTS_80,
    ),

    dict(
        planner="ergodic",
        env_dyn="static",
        comm_radius=30.0,   
        config="configs/simple20x20_static_v2.json",
        starts=STARTS_80,
    ),

]


# -------------------------------------------------
# Runner
# -------------------------------------------------

def run_experiments():
    for i, exp in enumerate(EXPERIMENTS):
        args_dict = deepcopy(BASE_ARGS)
        args_dict.update(exp)

        print("\n" + "=" * 60)
        print(f"Running experiment {i+1}/{len(EXPERIMENTS)}")
        for k, v in args_dict.items():
            print(f"{k}: {v}")
        print("=" * 60)

        # Create a lightweight args object
        class Args:
            pass

        args = Args()
        for k, v in args_dict.items():
            setattr(args, k, v)

        # Dispatch (same logic as main.py)
        if args.mode == "oracle":
            from experiments.run_oracle import run as run_oracle
            run_oracle(args)
        else:
            from experiments.run_gp import run as run_gp
            run_gp(args)


if __name__ == "__main__":
    run_experiments()


