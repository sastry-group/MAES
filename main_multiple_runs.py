#!/usr/bin/env python

"""
Batch runner for multiple experiments using the current main.py arguments.
"""

from copy import deepcopy

# -------------------------------------------------
# Base args (shared across all runs)
# -------------------------------------------------


BASE_ARGS = dict(
    mode="gp",
    method="remc",
    T=1000,
    neighbors="4",
    update_P_every=25,
    update_GP_every=25,
    save_every=200,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=1.0,
    l_space=2.0,
    l_time=10.0,
    animation=False,
    # comm_radius=20.0,
    fps=10,
    stride=1,
    comm_mode="local",
    consensus_iters=5,
#     n_agents=3,
)



EXPERIMENTS = [
  
    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=20.0,
         config="configs/simple10x10_static_v1.json",
         starts="0,9;0,9;0,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=20.0,
         config="configs/simple10x10_static_v1.json",
         starts="4,9;4,9;4,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=6,comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="0,7;0,7;0,7;0,7;0,7;0,7"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,0;9,0;9,0;9,0;9,0;9,0"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="5,4;5,4;5,4;5,4;5,4;5,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=20.0,
         config="configs/simple10x10_static_v3.json",
         starts="6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6"),




    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=5.0,
         config="configs/simple10x10_static_v1.json",
         starts="0,9;0,9;0,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=5.0,
         config="configs/simple10x10_static_v1.json",
         starts="4,9;4,9;4,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=6,comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="0,7;0,7;0,7;0,7;0,7;0,7"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,0;9,0;9,0;9,0;9,0;9,0"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="5,4;5,4;5,4;5,4;5,4;5,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0,
         config="configs/simple10x10_static_v3.json",
         starts="6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6"),




    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=2.0,
         config="configs/simple10x10_static_v1.json",
         starts="0,9;0,9;0,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=2.0,
         config="configs/simple10x10_static_v1.json",
         starts="4,9;4,9;4,9"),

    dict(planner="ergodic", env_dyn="static", n_agents=6,comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="0,7;0,7;0,7;0,7;0,7;0,7"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,0;9,0;9,0;9,0;9,0;9,0"),

    dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="5,4;5,4;5,4;5,4;5,4;5,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1"),

    dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=2.0,
         config="configs/simple10x10_static_v3.json",
         starts="6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6"),

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


