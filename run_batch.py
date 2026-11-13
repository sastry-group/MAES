#!/usr/bin/env python

"""
Batch runner for multiple experiments using the current main.py arguments.

Starts are resolved per (map, n_agents) via MAP_STARTS. Each VERSION's starts are
defined once (e.g. _S5_V0) and the static AND dynamic maps of that version point
at the same dict, so they stay identical. Entries are either explicit lists of
start-strings, or RAND(...) to generate valid random starts on demand.

This batch: 5 static maps (5x5 v0/v2, 10x10 v1/v3/v4), 1 agent, two planners
(ergodic-mh and mac_dt) on identical maps+starts for a fair comparison.
"""

import os
import sys
import glob
from copy import deepcopy
import numpy as np
from env.grid import make_env_from_json
from main import parse_args


# -------------------------------------------------
# Sweep axes
# -------------------------------------------------

N_AGENTS_LIST = [1,3,6]
COMM_RADII    = [30]
DELAY         = 0
 

# UPDATE_FREQ = [25]  # ,100,250,500]   (used by ergodic; ignored by mac_dt)
UPDATE_FREQ = [25] 
RESULTS_DIR="results/TCNS_synch_5x5"
# # Both planners run on every (map, start). method is unused by mac_dt but harmless.
PLANNERS = [
    # dict(planner="mac_dt",  method="mh"),
    # dict(planner="ergodic", method="mh"),
    # dict(planner="hedac", hedac_block=0),
    dict(planner="ergodic", method="mh", epoch_rule="computed"),
    dict(planner="ergodic", method="mh", epoch_rule="early"),
    *[dict(planner="ergodic", method="mh", epoch_rule="off",
           update_P_every=L, update_GP_every=L) for L in (100, 1000, 10000)],

    # dict(planner="hedac", hedac_block=0),
]
# PLANNERS = [dict(planner="hedac", hedac_w="hard", hedac_ell=1.0)]
# Noise sweep. Each entry overrides BASE_ARGS.
# NOISE_SETTINGS = [
#     # A: noise level (both planners)
#     dict(noise_rel=0.0,  gp_noise_scale=1.0),   # noiseless upper bound
#     dict(noise_rel=0.05, gp_noise_scale=1.0),   # sigma_eps = 0.35 on 10x10 v3
#     dict(noise_rel=0.2,  gp_noise_scale=1.0),   # sigma_eps = 1.4
#     # B: GP told the wrong noise level (ergodic only)
#     dict(noise_rel=0.2,  gp_noise_scale=0.5),
#     dict(noise_rel=0.2,  gp_noise_scale=2.0),
# ]
# A: initial belief scale
# NOISE_SETTINGS = [dict(init_phi_scale=6.0), dict(init_phi_scale=1.0)]
# B: prior / amplitude
# NOISE_SETTINGS = [
#     dict(gp_prior="data",  kernel_amp=10.0),
#     dict(gp_prior="data",  kernel_amp=1.0),
#     dict(gp_prior="fixed", kernel_amp=9.0, gp_prior_mean=0.0),
#     dict(gp_prior="fixed", kernel_amp=9.0, gp_prior_mean=6.0),
# ]
# # C: exploration   (first put the best prior from B into BASE_ARGS)
# NOISE_SETTINGS = [dict(ucb_beta_u=b) for b in (0.0, 1.0, 3.0, 5.0)] + \
#                  [dict(ucb_beta_schedule="mac_dt")]
# # D: kernels × noise   (first put the best beta_u from C into BASE_ARGS)
# NOISE_SETTINGS = [dict(kernel=k)
#                   for k in ("matern12", "matern32", "matern52", "rbf", "prod_matern_exp")
#                   ]
# NOISE_SETTINGS = [dict(kernel= "prod_matern_exp")]
                  
# NOISE_SETTINGS = (
#     [dict(kernel=k, l_time=lt)
#      for k in ("matern32", "prod_matern_exp")          # candidates: full sweep
#      for lt in (200, 10000, 10000000)])
    #  for n in (0.05, 0.2)]
#   + [dict(kernel=k, noise_rel=n, l_time=100)
#      for k in ("matern12", "matern52")                  # trend check: one value
#      for n in (0.05, 0.2)]
# )
# # E: hyperparameter learning
# NOISE_SETTINGS = [dict(kernel=k, learn_hypers=l)
#                   for k in ("matern32", "matern52") for l in ("none", "once")]
# NOISE_SETTINGS = [dict(kernel=k, learn_hypers="once") for k in ("matern32", "matern52")]
# F: learning cost (timing only)
# NOISE_SETTINGS = [dict(learn_hypers="every", T=300)]
# G : Update tau_P
# NOISE_SETTINGS = [dict(kernel="prod_matern_exp", noise_rel=0.05)]

# H: point reduction (learn_hypers="none")

# BIG = 100000   # FIFO effectively off
# # NOISE_SETTINGS = (
# #     [dict(gp_method="exact", gp_obs_hist=B) for B in (500, 1000, 2000)]                  # 1  FIFO (2000 = baseline)
# #   + [dict(gp_method="exact", gp_bin_dt=25, gp_obs_hist=BIG)]                              # 2  averaging
# #   + [dict(gp_method="sgpr", gp_M=M, gp_obs_hist=BIG) for M in (50, 200)]                  # 3  inducing pts
# #   + [dict(gp_method="exact", gp_bin_dt=25, gp_obs_hist=500)]                              # 1+2
# #   + [dict(gp_method="sgpr", gp_M=100, gp_obs_hist=500)]                                   # 1+3
# # )
GP = dict(gp_method="exact", gp_bin_dt=100000000, gp_obs_hist=100000000)
# NOISE_SETTINGS = [dict(**GP, noise_sigma=s) for s in (0.5, 1.0)]                               # memory-bounded
NOISE_SETTINGS = [dict(**GP, noise_sigma=0.5, kernel="matern32"),
    dict(**GP, noise_sigma=0.5, kernel="matern32", target_q=10.0)]
#  [dict(gp_method="sgpr", gp_M=M, gp_obs_hist=BIG) for M in (500, 1000)]         # sparse, full history
#   + [dict(gp_method="exact", gp_bin_dt=25, gp_obs_hist=BIG)]                       # fused, full history
#   + [dict(gp_method="exact", gp_bin_dt=25, gp_obs_hist=1000)]                      # fused + memory-bounded
#   + [dict(gp_method="sgpr", gp_M=500, gp_obs_hist=1000)]                           # sparse + memory-bounded
#   + [dict(gp_method="exact", gp_obs_hist=BIG)]
#    [dict(gp_method="exact", gp_bin_dt=0, gp_obs_hist=1000)]   # exact, memory bound

# GP  = dict(gp_method="exact", gp_bin_dt=0, gp_obs_hist=1000)
# ATT4 = "200:0;500:1;700:2;900:3"     # MUST match the existing attrition4 runs
# ATT2 = "200:0;500:1"                 # first two kills of ATT4
# OFF  = dict(attrition=False, attrition_steps="")

# NOISE_SETTINGS = [
#     dict(**GP, **OFF, comm_radius=0,  comm_delay=0),                          # 1 sweep
#     dict(**GP, **OFF, comm_radius=5,  comm_delay=0),
#     dict(**GP, **OFF, comm_radius=15, comm_delay=0),
#     dict(**GP, **OFF, comm_radius=15, comm_delay=10),                         # 2 delay
#     dict(**GP, attrition=True, attrition_steps=ATT4, comm_radius=15, comm_delay=0),  # 3
#     dict(**GP, attrition=True, attrition_steps=ATT2, comm_radius=0,  comm_delay=0),  # 4 optional
#     dict(**GP, attrition=True, attrition_steps=ATT2, comm_radius=15, comm_delay=0),
# ]



NOISE_SEEDS = [1]   # add 1, 2 if error bars are wide
 

MAPS = [
    "configs/simple5x5_static_v0.json",
    "configs/simple5x5_static_rooms.json",
    # "configs/simple5x5_static_v1.json",
    # "configs/simple5x5_static_v2.json",
    # "configs/simple10x10_static_v1.json",
    # "configs/simple10x10_static_v3.json",
    # "configs/simple10x10_static_v4.json",
    # "configs/simple20x20_att_v1.json",
    # "configs/simple20x20_att_v2.json",
    # "configs/simple20x20_att_v3.json",
    # "configs/dynamic_simple10x10_v3/",
    # "configs/dynamic_simple10x10_v01/",
    # "configs/dynamic_simple10x10_v02/",
    # "configs/dynamic_rooms10x10_d1/",
    # "configs/dynamic_simple10x10_v03/",
    # "configs/dynamic_simple10x10_v04/",
    # "configs/dynamic_simple10x10_v05/",
    # "configs/dynamic_simple20x20_v1/",
    # "configs/dynamic_simple20x20_v2/",
    # "configs/simple50x50_static_v1.json",
    # "configs/dynamic_simple10x10_v4/",
    # "configs/dynamic_simple5x5_v0/",
    # "configs/dynamic_simple5x5_v1/",
    # "configs/dynamic_simple5x5_v2/",
    # "configs/dynamic_simple10x10_v06/",
    # "configs/dynamic_simple20x20_v03/",
    # "configs/dynamic_simple20x20_v04/",
]



# Per-map schedule: horizon T and env change times


# len(env_change_steps) must equal (#json files in the folder) - 1.
_CH_10_V01 = [107, 204, 329, 422, 541, 650, 742, 859, 1063]  # removed 950 
_CH_10_V02 = [107, 204, 329, 422, 541, 650, 742, 859, 1063]   
_CH_10_V03 = [93, 187, 299, 434, 530, 631, 755, 896, 1017, 1128, 1271, 1362, 1498, 1700]   
_CH_10_V04 = [93, 187, 299, 434, 530, 631, 755, 896, 1017, 1128, 1271, 1362, 1498, 1603, 1700, 1830, 1910, 2300, 2415]   
_CH_10_V05 = [93, 187, 299, 434, 530, 631, 755, 896, 1017, 1128, 1271, 1362, 1498, 1603, 1700, 1830, 1910, 2300, 2415] 
_CH_20     = [107, 204, 329, 422, 541, 650, 742, 859, 950, 1063,
              1170, 1267, 1392, 1485, 1604, 1713, 1805, 1922, 2013, 2126]  # same as 10x10 (replace if different)
_CH_39 = [107, 204, 329, 422, 541, 650, 742, 859, 950, 1063,
          1170, 1267, 1392, 1485, 1604, 1713, 1805, 1922, 2013, 2126,
          2233, 2330, 2455, 2548, 2667, 2776, 2868, 2985, 3076, 3189,
          3296, 3393, 3518, 3611, 3730, 3839, 3931, 4048, 4139]

ENV_SCHEDULE = {
    "configs/dynamic_simple10x10_v01/": dict(T=2000, env_change_steps=_CH_10_V01),
    "configs/dynamic_simple10x10_v02/": dict(T=2000, env_change_steps=_CH_10_V02),
    "configs/dynamic_simple10x10_v03/": dict(T=2000, env_change_steps=_CH_10_V03),
    # "configs/dynamic_simple10x10_v04/": dict(T=6000, env_change_steps=_CH_10_V04),
    # "configs/dynamic_simple10x10_v05/": dict(T=6000, env_change_steps=_CH_10_V05),
    "configs/dynamic_simple10x10_v04/": dict(T=6000, env_change_steps=[2 * t for t in _CH_10_V04]),
    "configs/dynamic_simple10x10_v05/": dict(T=6000, env_change_steps=[2 * t for t in _CH_10_V05]),
    "configs/dynamic_simple10x10_v06/": dict(T=6000, env_change_steps=_CH_39),
    "configs/dynamic_simple20x20_v03/": dict(T=6000, env_change_steps=_CH_39),
    "configs/dynamic_simple20x20_v04/": dict(T=6000, env_change_steps=_CH_39),
    "configs/dynamic_rooms10x10_d1/" : dict(T=4000, env_change_steps=[1000, 2000, 3000]),
    "configs/dynamic_simple20x20_v1/":  dict(T=10000, env_change_steps=_CH_20),   
    "configs/dynamic_simple20x20_v2/":  dict(T=10000, env_change_steps=_CH_20),
}


# -------------------------------------------------
# Start-position validation + random generation
# -------------------------------------------------

VALIDATE_STARTS  = True   # check every resolved start is a valid free cell
AVOID_ROI        = True   # keep random starts off ROI (avoids trivial first-visit)

_ENV_CACHE = {}


def _load_env(map_cfg):
    """Load env from a single json path or a dynamic-config folder (first json)."""
    if map_cfg in _ENV_CACHE:
        return _ENV_CACHE[map_cfg]
    if map_cfg.endswith("/") or os.path.isdir(map_cfg):
        path = sorted(glob.glob(os.path.join(map_cfg, "*.json")))[0]
    else:
        path = map_cfg
    env = make_env_from_json(path)
    if hasattr(env, "_indexing"):
        env._indexing()
    _ENV_CACHE[map_cfg] = env
    return env


def _roi_cells(env):
    cells = set()
    for roi in getattr(env, "roi_cells", []):
        r0, c0, r1, c1 = roi["rect"]
        for r in range(r0, r1):
            for c in range(c0, c1):
                cells.add((r, c))
    return cells


def RAND(n_configs=3, seed_base=0, avoid_roi=AVOID_ROI):
    """Marker for the resolver: generate `n_configs` random start-strings."""
    return {"_random": True, "n_configs": n_configs,
            "seed_base": seed_base, "avoid_roi": avoid_roi}


def random_starts(map_cfg, n, n_configs=3, seed_base=0, avoid_roi=AVOID_ROI):
    """`n_configs` start-strings, each with n distinct valid (optionally non-ROI) cells."""
    env = _load_env(map_cfg)
    valid = list(env._state_index_map.keys())          # free cells only (obstacles excluded)
    if avoid_roi:
        roi = _roi_cells(env)
        valid = [rc for rc in valid if rc not in roi]
    assert n <= len(valid), f"{map_cfg}: need {n} starts but only {len(valid)} valid cells"
    out = []
    for k in range(n_configs):
        rng = np.random.default_rng(seed_base + k)     # reproducible per replicate
        idx = rng.choice(len(valid), size=n, replace=False)
        out.append(";".join(f"{valid[i][0]},{valid[i][1]}" for i in idx))
    return out


def _check_starts(map_cfg, starts):
    """Assert every start cell is valid; warn (don't fail) if a start sits on ROI."""
    env = _load_env(map_cfg)
    roi = _roi_cells(env)
    for s in starts:
        for cell in s.split(";"):
            rc = tuple(int(x) for x in cell.split(","))
            assert rc in env._state_index_map, \
                f"{map_cfg}: start {rc} is not a valid free cell"
            if rc in roi:
                print(f"  [WARN] {map_cfg}: start {rc} is on ROI "
                      f"(ROI first-visit will be ~0)")


def _check_schedule(map_cfg):
    """Dynamic folder must hold len(env_change_steps) + 1 json files."""
    if not (map_cfg.endswith("/") or os.path.isdir(map_cfg)):
        return
    n_json = len(glob.glob(os.path.join(map_cfg, "*.json")))
    n_ch = len(ENV_SCHEDULE.get(map_cfg, {}).get("env_change_steps", []))
    assert n_json == n_ch + 1, f"{map_cfg}: {n_json} json files, {n_ch} change steps (need {n_ch + 1} files)"

# -------------------------------------------------
# Starts per version (defined once; shared static <-> dynamic)
# -------------------------------------------------
# Naming: _S<grid>_V<version>. The static and dynamic map of the same version
# point at the SAME dict below, so their starts are guaranteed identical.
#
# NOTE: entries marked RAND(...) for n=1 are auto-generated (valid, non-ROI,
# reproducible) because explicit single-agent starts weren't specified for those
# maps. Replace with explicit lists if you want fixed positions.

# ---- 5x5 ----
_S5_SWEEP_V0 = {1: ["4,4"], 3: ["4,4;4,4;4,4"], 6: ["4,4;4,4;4,4;4,4;4,4;4,4"]}
_S5_ROOMS    = {1: ["4,0"], 3: ["4,0;4,0;4,0"], 6: ["4,0;4,0;4,0;4,0;4,0;4,0"]}
_S5_V0 = {
    1: ["4,4", "0,4", "0,2"],
    # 1: ["4,4", "0,4"],
    2: [ "4,4;4,4","4,0;4,4", "4,4;2,4"], 
    # 3: ["0,4;4,4;4,0", "4,4;2,4;0,2","4,0;4,0;4,0"] # tauGP, remove
    3: [ "4,4;4,4;4,4","0,4;4,4;4,0", "4,4;2,4;0,2"] #3: [ "4,0;4,0;2,0"], # 3: [ "4,4;4,4;4,4","0,4;4,4;4,0", "4,4;2,4;0,2"],#"0,4;0,4;0,4", 
}
_S5_V1 = {
    1: ["4,0", "0,0", "2,0"],
    2: ["4,0;4,0", "0,0;4,0", "0,2;0,0"],
    # 3: ["4,0;4,0;4,0", "0,0;4,0;0,4", "0,2;2,2;2,0"],
    3: ["4,0;4,0;4,0", "0,0;4,0;2,0", "2,0;2,2;4,0"], #3: ["0,0;4,0;1,4"]
}
_S5_V2 = {
    # 1: RAND(n_configs=3, seed_base=0),   
    1: ["4,4", "4,0", "0,4"], #   
    2: ["4,4;4,4", "0,4;4,0"], #    
    3: ["4,4;4,4;4,4", "4,4;4,0;0,4", "0,2;4,4;2,0"],
}

# ---- 10x10 ----  (n=1 auto-generated; n=3/6/10 are your explicit layouts)
_S10_V1 = {
    # 1:  RAND(n_configs=3, seed_base=0),
    1:  ["9,0","4,5","9,5"],
    3:  ["9,0;4,5;9,5", "9,0;9,0;9,0", "0,5;5,0;0,5"],
    6:  [
        "9,0;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;5,7"],
    10: ["9,0;9,4;5,0;0,5;5,9;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;0,5;5,0;4,5;9,5;5,9"],
}
_S10_V3 = {
    # 1:  RAND(n_configs=3, seed_base=0),
    1:  ["9,0","0,5","9,5"],
    3:  ["9,0;0,5;9,5", "9,0;9,0;9,0", "0,5;5,0;0,5"],
    6:  ["9,0;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;5,7"],
    10: ["9,0;9,4;5,0;0,5;5,9;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;0,5;5,0;4,5;9,5;5,9"],
}
_S10_V4 = {
    # 1:  RAND(n_configs=3, seed_base=0),
    1:  ["9,9","9,5","9,0"],
    3:  ["9,9;9,5;9,9", "9,0;9,0;9,0", "0,9;5,9;9,9"],
    6:  ["9,9;9,5;9,9;9,9;9,5;9,9",
         "9,0;9,0;9,0;9,0;9,0;9,0",
         "0,9;5,9;9,9;0,9;5,9;9,9"],
    10: ["9,9;9,5;9,9;9,9;9,5;9,9;9,9;9,5;9,9;9,9",
         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
         "0,9;5,9;9,9;0,9;5,9;9,9;0,9;5,9;9,9;0,9"],
}

# ---- 10x10 dynamic v01/02/03 
_S10_D = {
    1:  ["9,0","4,5","9,5"],
    3:  [
        "9,0;4,5;9,5", 
         "9,0;9,0;9,0", 
         "0,5;5,0;0,5"
         ],
    6:  [
        "9,0;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;5,7"
         ],
    10: ["9,0;9,4;5,0;0,5;5,9;9,0;9,4;5,0;0,5;5,9",
         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0",
         "0,5;5,0;4,5;9,5;5,9;0,5;5,0;4,5;9,5;5,9"],
}

# _S20_D = {
#     12: ["3,2;3,7;3,12;3,16;10,2;10,7;10,12;10,17;16,2;16,7;16,12;16,17",   # spread (3x4 lattice)
#          "19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0",    # all at depot
#          "10,10;11,8;1,2;7,2;13,16;13,3;11,0;8,6;12,18;9,16;15,13;5,19"],  # random (seed 0)
# }



_S20_D = {
    10: [
        "3,2;3,12;3,16;10,2;10,7;10,12;10,17;16,2;16,7;16,12",   # spread (3x4 lattice)
         "19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0",    # all at depot
         "10,10;11,8;1,2;7,2;13,16;13,3;11,0;8,6;12,18;9,16"],  # random (seed 0)
}

_S10_V06 = {
    20: ["1,1;1,3;1,5;1,6;1,9;3,1;3,3;3,5;3,7;3,9;6,1;5,3;6,5;6,7;5,9;8,1;8,3;8,5;8,7;8,9",  # spread 4x5
         "9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0;9,0"],  # depot
}
_S20_V03 = {
    20: [
        "2,2;2,6;2,10;2,14;2,18;7,2;7,6;7,10;7,14;7,18;12,2;11,6;12,10;12,14;12,18;17,2;17,6;17,10;17,14;17,18",  # spread 4x5 ((12,6) moved off obstacle)
         "19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0",
        "0,0;0,1;1,0;1,1;0,18;0,19;1,18;1,19;18,0;18,1;19,0;19,1;18,18;18,19;19,18;19,19;9,9;9,10;10,9;10,10"],  # 4 per corner + 4 at center (2x2 blocks)
}

_S20_V04 = {
    20: [
        "2,2;2,6;2,10;2,14;2,18;7,2;7,6;7,10;7,14;7,18;12,2;11,6;12,10;12,14;12,18;17,2;17,6;17,10;17,14;17,18",  # spread 4x5 (passed on v04)
        #  "19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0;19,0",  # depot (passed on v04)
        #  "0,0;0,1;1,0;1,1;0,18;0,19;1,18;1,19;18,0;18,1;19,0;19,1;18,18;18,19;19,18;19,19;10,10;10,11;11,10;11,11"
         ],  # 4 per corner + 4 at center (center block shifted off (9,9))
}

# ---- 50x50 (large -> random) ----
_S50_V1 = {
    1:   ["0,0", "49,49", "25,25"],
    100: RAND(n_configs=1, seed_base=0),
    200: RAND(n_configs=1, seed_base=0),
    # Explicit large layouts you used previously (kept for reference):
    # 200: ["0,49;0,0;49,9;49,0;49,49;20,25;30,25;9,49;15,15;30,10;40,40;35,45;45,25;4,4;15,0;0,15;0,20"],
    # 100: ["0,49;0,0;49,9;49,0;49,49;20,25;30,25;9,49;15,15;30,10;40,40;35,45;45,25;4,4;15,0;0,15;0,20;0,30;45,0;10,5;10,10;45,45"],
}


# -------------------------------------------------
# Per-(map, n) start registry  (static & dynamic share the version dict)
# -------------------------------------------------

MAP_STARTS = {
    # 5x5
    "configs/simple5x5_static_v0.json":    _S5_SWEEP_V0, 
    "configs/simple5x5_static_rooms.json": _S5_ROOMS,
    # "configs/simple5x5_static_v0.json": _S5_V0,
    "configs/dynamic_simple5x5_v0/":    _S5_V0,
    "configs/simple5x5_static_v1.json": _S5_V1,
    "configs/dynamic_simple5x5_v1/":    _S5_V1,
    "configs/simple5x5_static_v2.json": _S5_V2,
    "configs/dynamic_simple5x5_v2/":    _S5_V2,

    # 10x10
    "configs/simple10x10_static_v1.json": _S10_V1,
    "configs/simple10x10_static_v3.json": _S10_V3,
    "configs/dynamic_simple10x10_v3/":    _S10_V3,
    "configs/simple10x10_static_v4.json": _S10_V4,
    "configs/dynamic_simple10x10_v4/":    _S10_V4,

    "configs/dynamic_simple10x10_v01/": _S10_D,
    "configs/dynamic_simple10x10_v02/": _S10_D,
    "configs/dynamic_simple10x10_v03/": _S10_D,
    "configs/dynamic_simple10x10_v04/": _S10_D,
    "configs/dynamic_simple10x10_v05/": _S10_D,

    "configs/dynamic_rooms10x10_d1/":  _S10_D,
    "configs/dynamic_simple20x20_v1/": _S20_D,
    "configs/dynamic_simple20x20_v2/": _S20_D,

    "configs/dynamic_simple10x10_v06/": _S10_V06,
    "configs/dynamic_simple20x20_v03/": _S20_V03,
    "configs/dynamic_simple20x20_v04/": _S20_V04,


    # 50x50
    "configs/simple50x50_static_v1.json": _S50_V1,
}


def get_starts(map_cfg, n):
    """Resolve start-strings for (map, n): explicit list or RAND generation."""
    if map_cfg not in MAP_STARTS:
        raise KeyError(f"No start spec for map {map_cfg}")
    spec = MAP_STARTS[map_cfg]
    if n not in spec:
        raise KeyError(f"No starts for n={n} on {map_cfg}; defined n: {sorted(spec)}")
    entry = spec[n]
    if isinstance(entry, dict) and entry.get("_random"):
        starts = random_starts(map_cfg, n,
                               n_configs=entry["n_configs"],
                               seed_base=entry["seed_base"],
                               avoid_roi=entry["avoid_roi"])
    else:
        starts = entry
    if VALIDATE_STARTS:
        _check_starts(map_cfg, starts)
    return starts


# -------------------------------------------------
# Base args (shared across all runs)
# -------------------------------------------------
# planner/method here are defaults; the PLANNERS loop overrides them per run.

BASE_ARGS = dict(
    mode="gp",
    planner="ergodic",
    method="mh",
    T=10000000,
    neighbors="4",
    # update_P_every=25,
    # update_GP_every=25,
    # gp_mode="ucb",
    gp_mode="softmax",
    ucb_beta_m=1.0,
    ucb_beta_u=3.0,
    ucb_delta=0.05,
    # ucb_beta_schedule="const",
    ucb_beta_schedule="mac_dt", 
    l_space=1.5,
    l_time=1e9,
    gp_obs_hist=2000,
    share_obs_hist=100,
    noise_rel=0.05,
    noise_sigma=0.5,
    gp_noise_scale=1.0,
    noise_seed=0,
    comm_mode="local",
    consensus_iters=5,
    attrition=False,
    attrition_steps="",

    kernel="rbf", #prod_matern_exp

    # kernel="matern32",
    gp_prior="fixed", 
    gp_prior_mean=0.0, 
    kernel_amp=4.0,
    learn_hypers="none",
    init_phi_scale=6.0,
    results_dir=RESULTS_DIR,
    epoch_alpha=2.0,
    epoch_delta2=0.05,
    target_q=3.0,
    early_check_every=10,
    env_dyn="static",
    # env_dyn="dynamic",
    # env_change_steps=[500,1000,2000],
    # 10 changes (11 files); gaps: 107, 97, 125, 93, 119, 109, 92, 117, 91, 113
    # env_change_steps=[107, 204, 329, 422, 541, 650, 742, 859, 950, 1063],

    # 15 changes (16 files); gaps: 93, 94, 112, 135, 96, 101, 124, 141, 121, 111, 143, 91, 136, 105, 97
    # env_change_steps=[93, 187, 299, 434, 530, 631, 755, 896, 1017, 1128, 1271, 1362, 1498, 1603, 1700],
    # env_change_steps=[107, 204, 329, 422, 541, 650, 742, 859, 950, 1063,
    #               1170, 1267, 1392, 1485, 1604, 1713, 1805, 1922, 2013, 2126],

    # comm_delay_prob=0.5,
    comm_delay=DELAY,
    # randomize_starts=True,
    # start_seed=0,
)

# ATTRITION_CONDITIONS = [
#     # (label, attrition_flag, attrition_steps_str)
#     ("none",     False, ""),
#     ("mild",     True,  "20:1"),
#     ("moderate", True,  "20:1;100:3;300:5"),
#     ("severe",   True,  "20:1;100:3;300:5;350:2"),
# ]


# -------------------------------------------------
# Build experiment list
# -------------------------------------------------

# -------------------------------------------------
# Overnight batches:  python run_batch.py planners | large  [smoke]
# # -------------------------------------------------
# BATCH = sys.argv[1] if len(sys.argv) > 1 else None
# SMOKE_RUN = len(sys.argv) > 2 and sys.argv[2] == "smoke"

# if BATCH == "planners":
#     # v01/v02 stretched x5: same change sequence, changes now reach t = 5315 (late-change / windup test)
#     _CH_10_LONG = [2 * t for t in _CH_10_V04]          # 19 changes, [186, ..., 4830]
#     MAPS = ["configs/dynamic_simple10x10_v04/", "configs/dynamic_simple10x10_v05/"]
#     for m in MAPS:
#         ENV_SCHEDULE[m] = dict(T=6000, env_change_steps=_CH_10_LONG)
#     # MAPS           = list(ENV_SCHEDULE)[:2]
#     N_AGENTS_LIST  = [6]
#     COMM_RADII     = [0]      # keep 0: HEDAC reads neighbours' coverage for free if > 0
#     UPDATE_FREQ    = [25]     # tau_P = 25 for all planners
#     NOISE_SEEDS    = [1]
#     NOISE_SETTINGS = [dict(gp_method="exact", gp_bin_dt=25, gp_obs_hist=2000)]
#     PLANNERS = [              # HEDAC first so it is never the one cut off
#         dict(planner="hedac", hedac_w="hard", hedac_ell=1.0, hedac_block=0),
#         dict(planner="ergodic", method="mh"),
#         dict(planner="mac_dt",  method="mh"),
#         dict(planner="hedac", hedac_gp="hedac_gp", hedac_block=0)
#     ]
#     BASE_ARGS.update(env_dyn="dynamic", kernel="prod_matern_exp", l_time=10000,
#                      results_dir="results/PLANNER_LONG")

# elif BATCH == "large":
#     MAPS           = ["configs/simple50x50_static_v1.json"]
#     N_AGENTS_LIST  = [200]
#     COMM_RADII     = [5]
#     UPDATE_FREQ    = [200]
#     NOISE_SEEDS    = [1]
#     NOISE_SETTINGS = [dict(gp_method="exact", gp_bin_dt=200, gp_obs_hist=2000)]
#     PLANNERS       = [dict(planner="ergodic", method="mh")]
#     BASE_ARGS.update(env_dyn="static", T=40000, kernel="matern32", l_time=10000,
#                      results_dir="results/LARGE")

# elif BATCH is not None:
#     raise SystemExit(f"unknown batch {BATCH!r}; use 'planners' or 'large'")

# if SMOKE_RUN:   # timing check; dynamic maps keep their ENV_SCHEDULE T
#     BASE_ARGS.update(T=400, results_dir="results/SMOKE")


EXPERIMENTS = []

for ns in NOISE_SETTINGS:                      # outermost -> conditions run in priority order
    for n in N_AGENTS_LIST:
        for radius in COMM_RADII:
            for map_cfg in MAPS:
                _check_schedule(map_cfg)
                for starts in get_starts(map_cfg, n):
                    for freq in UPDATE_FREQ:
                        for seed in NOISE_SEEDS:
                            for planner_cfg in PLANNERS:
                                exp = dict(
                                    **BASE_ARGS,
                                    config=map_cfg,
                                    starts=starts,
                                    n_agents=n,
                                    comm_radius=radius,
                                    update_P_every=freq,
                                    update_GP_every=freq,
                                )
                                exp.update(ENV_SCHEDULE.get(map_cfg, {}))
                                exp.update(ns)
                                exp["noise_seed"] = seed
                                exp.update(planner_cfg)
                                EXPERIMENTS.append(exp)

print(f"Total: {len(EXPERIMENTS)}")

# SMOKE = True
# if SMOKE:
#     seen, smoke = set(), []
#     for exp in EXPERIMENTS:
#         key = (exp["planner"], exp.get("hedac_w"), exp.get("hedac_ell"))
#         if key not in seen:
#             seen.add(key)
#             exp.update(T=300, results_dir="results/SMOKE", hedac_debug_every=0)
#             smoke.append(exp)
#     EXPERIMENTS = smoke
#     print(f"SMOKE: {len(EXPERIMENTS)} runs")
# ---- Attrition sweep (alternative build; uncomment to use) ----
# EXPERIMENTS = []
# for att_label, att_flag, att_steps in ATTRITION_CONDITIONS:
#     for n in N_AGENTS_LIST:
#         for radius in COMM_RADII:
#             for map_cfg in MAPS:
#                 for starts in get_starts(map_cfg, n):
#                     EXPERIMENTS.append(dict(
#                         **BASE_ARGS,
#                         config=map_cfg,
#                         starts=starts,
#                         n_agents=n,
#                         comm_radius=radius,
#                         attrition=att_flag,
#                         attrition_steps=att_steps,
#                     ))
# print(f"Total: {len(EXPERIMENTS)}")


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

        # args = Args()
        # for k, v in args_dict.items():
        #     setattr(args, k, v)
        args = parse_args([])
        unknown = set(args_dict) - set(vars(args))
        assert not unknown, f"Not in main.py parser: {unknown}"
        vars(args).update(args_dict)

        # Dispatch (same logic as main.py)
        if args.mode == "oracle":
            from experiments.run_oracle import run as run_oracle
            run_oracle(args)
        else:
            from experiments.run_gp import run as run_gp
            run_gp(args)
        # print("stop here")


if __name__ == "__main__":
    run_experiments()