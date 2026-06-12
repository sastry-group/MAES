#!/usr/bin/env python

"""
Batch runner for multiple experiments using the current main.py arguments.
"""

from copy import deepcopy


# -------------------------------------------------
# Base args (shared across all runs)
# -------------------------------------------------


N_AGENTS_LIST = [50]
COMM_RADII    = [5.0]  
DELAY         = 0


### if running scale with multiprocessing do not run on debug


MAPS = [
    # "configs/simple10x10_static_v1.json",
    # "configs/simple10x10_static_v3.json",
    # "configs/simple10x10_static_v4.json",
    # "configs/simple20x20_att_v1.json",
    # "configs/simple20x20_att_v2.json",
    # "configs/simple20x20_att_v3.json",
    # "configs/dynamic_simple10x10_v3/",
    "configs/simple50x50_static_v1.json"
    # "configs/dynamic_simple10x10_v4/",
    # "configs/dynamic_simple5x5_v0/",
    # "configs/dynamic_simple5x5_v1/",
    # "configs/dynamic_simple5x5_v2/"

]



# if attrition:
# STARTS_PER_N = {6: [
#     "0,0;19,0;0,19;19,19;0,8;8,0",   # corners + mid edges
#     "19,0;19,0;19,0;19,0;19,0;19,0",    
#     "0,0;19,19;0,19;7,0;8,10;8,12",   

# ]}

# Spread starts per N — agents at different positions

# if v1:
# STARTS_PER_N = {
# # v0 5x5
#     # 3: [
#     #     "0,4;0,4;0,4",
#     #     "0,4;4,4;4,0",
#     #     "4,4;2,4;0,2",
#     # ],
#     # # v1 5x5
#     # 3: [
#     #     "4,0;4,0;4,0",
#     #     "0,0;4,0;0,4",
#     #     "0,2;2,2;0,2",
#     # ],
#     # # v1 10x10
#     3: [
#         "9,0;4,5;9,5",
#         "9,0;9,0;9,0",
#         "0,5;5,0;0,5",
#     ],
#     6: [
#         "9,0;9,0;9,4;5,0;0,5;5,9",
#         "9,0;9,0;9,0;9,0;9,0;9,0",
#         "0,5;5,0;4,5;9,5;5,9;5,7",
#     ],
#     10: [
#         "9,0;9,4;5,0;0,5;5,9;9,0;9,4;5,0;0,5;5,9",
#         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
#         "0,5;5,0;4,5;9,5;5,9;0,5;5,0;4,5;9,5;5,9",
#     ],
# }

# UPDATE_FREQ = [10,50,100,250,500]


# # # if v3:
STARTS_PER_N = {
    # 3: [
    #     "9,0;0,5;9,5",
    #     "9,0;9,0;9,0",
    #     "0,5;5,0;0,5",
    # ],
    # 6: [
    #     "9,0;9,0;9,4;5,0;0,5;5,9",
    #     "9,0;9,0;9,0;9,0;9,0;9,0",
    #     "0,5;5,0;4,5;9,5;5,9;5,7",
    # ],
    # 10: [
    #     "9,0;9,4;5,0;0,5;5,9;9,0;9,4;5,0;0,5;5,9",
    #     "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
    #     "0,5;5,0;4,5;9,5;5,9;0,5;5,0;4,5;9,5;5,9",
    # ],
    50: ["0,49;0,0;49,9;49,0;49,49;20,25;30,25;9,49;15,15;30,10;40,40;35,45;45,25"]
}


# # # if v4:
# STARTS_PER_N = {
#     3: [
#         "9,9;9,5;9,9",
#         "9,0;9,0;9,0",
#         "0,9;5,9;9,9",
#     ],
#     6: [
#         "9,9;9,5;9,9;9,9;9,5;9,9",
#         "9,0;9,0;9,0;9,0;9,0;9,0",
#         "0,9;5,9;9,9;0,9;5,9;9,9",

#     ],
#     10: [
#         "9,9;9,5;9,9;9,9;9,5;9,9;9,9;9,5;9,9;9,9",
#         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
#         "0,9;5,9;9,9;0,9;5,9;9,9;0,9;5,9;9,9;0,9",
#     ],
# };

BASE_ARGS = dict(
    mode="gp",
    planner="ergodic",  
    method="mh",
    T=40000,
    neighbors="4",
    update_P_every=200,
    update_GP_every=200,
    save_every=500,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=3.0,
    l_space=2.0,
    l_time=10000,
    gp_obs_hist=1000,
    share_obs_hist=50,
    animation=False,
    comm_mode="local",
    consensus_iters=5,
    attrition=False,
    attrition_steps="",
    env_dyn="static",
    # env_dyn="dynamic",
    # env_change_steps=[500,1000,1500],
    # comm_delay_prob=0.5,
    comm_delay=DELAY,
)

# ATTRITION_CONDITIONS = [
#     # (label, attrition_flag, attrition_steps_str)
#     ("none",     False, ""),
#     ("mild",     True,  "20:1"),
#     ("moderate", True,  "20:1;100:3;300:5"),    
#     ("severe",   True,  "20:1;100:3;300:5;350:2"),
# ]


# def _starts(cell, n=N_AGENTS):
#     return ";".join([cell] * n)


# STATIC_MAP_START_COMBOS = [
#     # -------- static v3 --------
#     ("configs/simple10x10_static_v3.json", _starts("3,1")),  # original start
#     ("configs/simple10x10_static_v3.json", _starts("0,0")),
#     ("configs/simple10x10_static_v3.json", _starts("9,0")),
#     ("configs/simple10x10_static_v3.json", _starts("0,9")),
 
#     # -------- static v4 --------
#     ("configs/simple10x10_static_v4.json", _starts("8,0")),  # original start
#     ("configs/simple10x10_static_v4.json", _starts("0,0")),
#     ("configs/simple10x10_static_v4.json", _starts("0,9")),
#     ("configs/simple10x10_static_v4.json", _starts("9,9")),
# ]
# MAP_START_COMBOS = [
#     # map v0
#     ("configs/dynamic_simple5x5_v0/", _starts("2,3")),
#     ("configs/dynamic_simple5x5_v0/", _starts("3,1")),
#     ("configs/dynamic_simple5x5_v0/", _starts("4,0")),
#     ("configs/dynamic_simple5x5_v0/", _starts("4,4")),
#     # map v1
#     ("configs/dynamic_simple5x5_v1/", _starts("0,0")),
#     ("configs/dynamic_simple5x5_v1/", _starts("0,2")),
#     ("configs/dynamic_simple5x5_v1/", _starts("4,4")),
# ]


EXPERIMENTS = []

for n in N_AGENTS_LIST:
    for radius in COMM_RADII:
        for map_cfg in MAPS:
            for starts in STARTS_PER_N[n]:
                # for freq in UPDATE_FREQ:
                EXPERIMENTS.append(dict(
                    **BASE_ARGS,
                    config=map_cfg,
                    starts=starts,
                    n_agents=n,
                    comm_radius=radius,
                    # update_P_every=freq,
                    # update_GP_every=freq,
                ))

print(f"Total: {len(EXPERIMENTS)}") 

# EXPERIMENTS = []

# for att_label, att_flag, att_steps in ATTRITION_CONDITIONS:
#     for n in N_AGENTS_LIST:
#         for radius in COMM_RADII:
#             for map_cfg in MAPS:
#                 for starts in STARTS_PER_N[n]:
#                     EXPERIMENTS.append(dict(
#                         **BASE_ARGS,
#                         config=map_cfg,
#                         starts=starts,
#                         n_agents=n,
#                         comm_radius=radius,
#                         attrition=att_flag,
#                         attrition_steps=att_steps,
#                     ))

print(f"Total: {len(EXPERIMENTS)}")





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


