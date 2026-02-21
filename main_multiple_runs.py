#!/usr/bin/env python

"""
Batch runner for multiple experiments using the current main.py arguments.
"""

from copy import deepcopy


# -------------------------------------------------
# Base args (shared across all runs)
# -------------------------------------------------


N_AGENTS = 3
START_CELL = "0,0"

STARTS_80 = ";".join([START_CELL] * N_AGENTS)

BASE_ARGS = dict(
    mode="gp",
    method="remc",
    T=100000,
    neighbors="4",
    update_P_every=50,
    update_GP_every=50,
    save_every=200,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=1.0,
    l_space=2.0,
    l_time=10.0,
    animation=False,
    comm_radius=5.0,
    fps=10,
    stride=1,
    # env_dyn="dynamic",
    comm_mode="local",
    consensus_iters=5,
    n_agents=N_AGENTS,
    # env_change_steps= [100,1000,1200,1500],
)


EXPERIMENTS = [
    
    


# # #     -------- map v0 --------
    # dict(planner="ergodic", n_agents=1,
    #     config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=10, update_GP_every=10, T=4000),
    # dict(planner="ergodic",n_agents=1,
    #      config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=10, update_GP_every=10, T=4000),

    # # -------- map v1 --------
    # dict(planner="ergodic", n_agents=1,
    #      config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=10, update_GP_every=10, T=4000),
    # dict(planner="ergodic", n_agents=1,
    #      config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=10, update_GP_every=10, T=4000),

#     # -------- map v2 --------
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=10, update_GP_every=10, T=4000),
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=10, update_GP_every=10, T=4000),

         

#      #    -------- map v0 --------
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=50, update_GP_every=50, T=4000),
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=50, update_GP_every=50, T=4000),

#     # -------- map v1 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=50, update_GP_every=50, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=50, update_GP_every=50, T=4000),

# #     -------- map v2 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=50, update_GP_every=50, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=50, update_GP_every=50, T=4000),


     #    -------- map v0 --------
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=500, update_GP_every=500, T=4000),
#     dict(planner="ergodic",n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=500, update_GP_every=500, T=4000),

#     # -------- map v1 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=500, update_GP_every=500, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=500, update_GP_every=500, T=4000),

# #     -------- map v2 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=500, update_GP_every=500, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=500, update_GP_every=500, T=4000),



#      #    -------- map v0 --------
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="2,3",  update_P_every=1000, update_GP_every=1000, T=4000),
#     dict(planner="ergodic", n_agents=1,
#          config="configs/dynamic_simple5x5_v0/", starts="3,1",  update_P_every=1000, update_GP_every=1000, T=4000),

#     # -------- map v1 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,0",  update_P_every=1000, update_GP_every=1000, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v1/", starts="0,2",  update_P_every=1000, update_GP_every=1000, T=4000),

# #     -------- map v2 --------
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,0",  update_P_every=1000, update_GP_every=1000, T=4000),
#     dict(planner="ergodic",  n_agents=1,
#          config="configs/dynamic_simple5x5_v2/", starts="0,2",  update_P_every=1000, update_GP_every=1000, T=4000),


# # ]

#  # -------- map v0 --------
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="2,3;2,3;2,3"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="3,1;3,1;3,1"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="4,0;4,0;4,0"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="4,4;4,4;4,4"),

#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="2,3;2,3;2,3"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="3,1;3,1;3,1"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="4,0;4,0;4,0"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v0.json", starts="4,4;4,4;4,4"),

#     # -------- map v2 --------
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="0,0;0,0;0,0"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="0,2;0,2;0,2"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="4,4;4,4;4,4"),
#     dict(planner="ergodic", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="1,2;1,2;1,2"),


#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="0,0;0,0;0,0"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="0,2;0,2;0,2"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="4,4;4,4;4,4"),
#     dict(planner="mac_dt", env_dyn="static",
#          config="configs/simple5x5_static_v2.json", starts="1,2;1,2;1,2"),




    ## dynamic
#     # -------- map v0 --------
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="2,3;2,3;2,3"),
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="3,1;3,1;3,1"),
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="4,0;4,0;4,0"),
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="4,4;4,4;4,4"),


#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="2,3;2,3;2,3"),
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="3,1;3,1;3,1"),
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="4,0;4,0;4,0"),
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="4,4;4,4;4,4"),

# #     # -------- map v1 --------
#     dict(planner="ergodic",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/",  starts="0,0;0,0;0,0"),
#     dict(planner="ergodic",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/",  starts="0,2;0,2;0,2"),
#     dict(planner="ergodic",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="4,4;4,4;4,4"),


#     dict(planner="mac_dt",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/",  starts="0,0;0,0;0,0"),
#     dict(planner="mac_dt",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/",  starts="0,2;0,2;0,2"),
#     dict(planner="mac_dt",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="4,4;4,4;4,4"),








# #  ##### Statuc 10x10 grid
# #  # 
#     dict(planner="mac_dt", env_dyn="static", n_agents=3, comm_radius=5.0, T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v1.json",
#          starts="0,9;0,9;0,9"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=3, comm_radius=5.0,  T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v1.json",
#          starts="4,9;4,9;4,9"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=3, comm_radius=5.0,  T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v1.json",
#          starts="0,5;0,5;0,5"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=6,comm_radius=5.0, T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v3.json",
#          starts="0,7;0,7;0,7;0,7;0,7;0,7"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=6, comm_radius=5.0, T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v3.json",
#          starts="9,0;9,0;9,0;9,0;9,0;9,0"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=6, comm_radius=5.0, T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v3.json",
#          starts="5,4;5,4;5,4;5,4;5,4;5,4"),

#     dict(planner="mac_dt", env_dyn="static", n_agents=10, comm_radius=5.0, T = 3000,update_GP_every=100,
#          config="configs/simple10x10_static_v3.json",
#          starts="4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4"),

    dict(planner="ergodic", env_dyn="static", n_agents=1, comm_radius=5.0, update_GP_every=100,
         config="configs/simple5x5_static_v0_noObs.json",
         starts="0,0"),

    # dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0, T = 3000,update_GP_every=100,
    #      config="configs/simple10x10_static_v3.json",
    #      starts="6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6"),



#     dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=5.0, T = 3000,
#          config="configs/simple10x10_static_v1.json",
#          starts="0,9;0,9;0,9"),

#     dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=5.0,  T = 3000,
#          config="configs/simple10x10_static_v1.json",
#          starts="4,9;4,9;4,9"),

#     dict(planner="ergodic", env_dyn="static", n_agents=3, comm_radius=5.0,  T = 3000,
#          config="configs/simple10x10_static_v1.json",
#          starts="0,5;0,5;0,5"),

#     dict(planner="ergodic", env_dyn="static", n_agents=6,comm_radius=5.0, T = 3000,
#          config="configs/simple10x10_static_v3.json",
#          starts="0,7;0,7;0,7;0,7;0,7;0,7"),

#     dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=5.0, T = 3000,
#          config="configs/simple10x10_static_v3.json",
#          starts="9,0;9,0;9,0;9,0;9,0;9,0"),

#     dict(planner="ergodic", env_dyn="static", n_agents=6, comm_radius=5.0, T = 3000,
#          config="configs/simple10x10_static_v3.json",
#          starts="5,4;5,4;5,4;5,4;5,4;5,4"),

#     dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0, T = 3000,
#          config="configs/simple10x10_static_v3.json",
#          starts="4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4;4,4"),

    # dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0, T = 3000,
    #      config="configs/simple10x10_static_v3.json",
    #      starts="9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1;9,1"),

    # dict(planner="ergodic", env_dyn="static", n_agents=10, comm_radius=5.0, T = 3000,
    #      config="configs/simple10x10_static_v3.json",
    #      starts="6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6;6,6"),







# ### Dynamic 10x10
#    #    -------- map v0 --------
#     dict(planner="mac_dt",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="2,3;2,3;2,3;2,3;2,3;2,3",  n_agents=6),
#     dict(planner="mac_dt",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="3,1;3,1;3,1;3,1;3,1;3,1;",   n_agents=6),

#     # -------- map v1 --------
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="0,0;0,0;0,0;0,0;0,0;0,0",   n_agents=6),
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="0,2;0,2;0,2;0,2;0,2;0,2",   n_agents=6),

# #     -------- map v2 --------
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v2/", starts="0,0;0,0;0,0;0,0;0,0;0,0",   n_agents=6),
#     dict(planner="mac_dt", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v2/", starts="0,2;0,2;0,2;0,2;0,2;0,2",   n_agents=6),


#    #    -------- map v0 --------
#     dict(planner="ergodic",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="2,3;2,3;2,3;2,3;2,3;2,3",  n_agents=6),
#     dict(planner="ergodic",env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v0/", starts="3,1;3,1;3,1;3,1;3,1;3,1;",   n_agents=6),

#     # -------- map v1 --------
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="0,0;0,0;0,0;0,0;0,0;0,0",   n_agents=6),
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v1/", starts="0,2;0,2;0,2;0,2;0,2;0,2",   n_agents=6),

# #     -------- map v2 --------
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v2/", starts="0,0;0,0;0,0;0,0;0,0;0,0",   n_agents=6),
#     dict(planner="ergodic", env_dyn="dynamic",
#          config="configs/dynamic_simple5x5_v2/", starts="0,2;0,2;0,2;0,2;0,2;0,2",   n_agents=6),


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


