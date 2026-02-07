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
    T=2000,
    neighbors="4",
#     update_P_every=25,
#     update_GP_every=25,
    save_every=200,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=1.0,
    l_space=2.0,
    l_time=10.0,
    animation=False,
    comm_radius=2.0,
    fps=10,
    stride=1,
    env_dyn="dynamic",
    comm_mode="local",
    consensus_iters=5,
    n_agents=1,
)


EXPERIMENTS = [
    
#     -------- map v0 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=5, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=5, update_GP_every=10),


    # -------- map v1 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=5, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=5, update_GP_every=10),

    # -------- map v2 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=5, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=5, update_GP_every=10),


    # -------- map v0 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=25, update_GP_every=50),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=25, update_GP_every=50),


    # -------- map v1 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=5, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=5, update_GP_every=10),


    # -------- map v2 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=5, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=5, update_GP_every=10),




#     -------- map v0 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=10, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=10, update_GP_every=10),

    # -------- map v1 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=10, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=10, update_GP_every=10),

    # -------- map v2 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=10, update_GP_every=10),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=10, update_GP_every=10),

         

    # -------- map v0 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=25, update_GP_every=25),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=25, update_GP_every=25),

    # -------- map v1 --------
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=25, update_GP_every=25),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=25, update_GP_every=25),
#     # -------- map v2 --------
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=25, update_GP_every=25),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=25, update_GP_every=25),


     #    -------- map v0 --------
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=50, update_GP_every=50),
    dict(planner="ergodic",
         config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=50, update_GP_every=50),

    # -------- map v1 --------
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=50, update_GP_every=50),
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=50, update_GP_every=50),

#     -------- map v2 --------
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=50, update_GP_every=50),
    dict(planner="ergodic", 
         config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=50, update_GP_every=50),

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


