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
import glob
from copy import deepcopy

import numpy as np

from env.grid import make_env_from_json


# -------------------------------------------------
# Sweep axes
# -------------------------------------------------

N_AGENTS_LIST = [2]
COMM_RADII    = [10.0]
DELAY         = 0

UPDATE_FREQ = [25]  # ,100,250,500]   (used by ergodic; ignored by mac_dt)

# Both planners run on every (map, start). method is unused by mac_dt but harmless.
PLANNERS = [
    dict(planner="ergodic", method="mh"),
    # dict(planner="mac_dt",  method="mh"),
]


MAPS = [
    # "configs/simple5x5_static_v0.json",
    # "configs/simple5x5_static_v2.json",
    # "configs/simple10x10_static_v1.json",
    # "configs/simple10x10_static_v3.json",
    # "configs/simple10x10_static_v4.json",
    # "configs/simple20x20_att_v1.json",
    # "configs/simple20x20_att_v2.json",
    # "configs/simple20x20_att_v3.json",
    # "configs/dynamic_simple10x10_v3/",
    # "configs/simple50x50_static_v1.json",
    # "configs/dynamic_simple10x10_v4/",
    "configs/dynamic_simple5x5_v0/",
    "configs/dynamic_simple5x5_v1/",
    "configs/dynamic_simple5x5_v2/",
]


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
_S5_V0 = {
    1: ["4,4", "0,4", "0,2"],
    2: [ "4,4;4,4","4,0;4,4", "4,4;2,4"], 
    3: [ "4,4;4,4;4,4","0,4;4,4;4,0", "4,4;2,4;0,2"], #"0,4;0,4;0,4", 
}
_S5_V1 = {
    1: ["0,4", "0,0", "2,4"],
    2: ["4,0;4,0", "0,0;4,0", "0,2;0,0"],
    3: ["4,0;4,0;4,0", "0,0;4,0;0,4", "0,2;2,2;0,0"],
}
_S5_V2 = {
    # 1: RAND(n_configs=3, seed_base=0),   
    1: ["4,4", "4,0", "0,4"], #   
    2: ["4,4;4,4", "0,4;4,0"], #    
    3: ["4,4;4,4;4,4", "4,4;4,0;0,4", "0,2;2,4;2,0"],
}

# ---- 10x10 ----  (n=1 auto-generated; n=3/6/10 are your explicit layouts)
_S10_V1 = {
    # 1:  RAND(n_configs=3, seed_base=0),
    1:  ["9,0","4,5","9,5"],
    3:  ["9,0;4,5;9,5", "9,0;9,0;9,0", "0,5;5,0;0,5"],
    6:  ["9,0;9,0;9,4;5,0;0,5;5,9",
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
    "configs/simple5x5_static_v0.json": _S5_V0,
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
    T=6000,
    neighbors="4",
    # update_P_every=25,
    # update_GP_every=25,
    save_every=500,
    gp_mode="ucb",
    ucb_beta_m=1.0,
    ucb_beta_u=3.0,
    ucb_delta=0.05,
    ucb_beta_schedule="mac_dt", 
    l_space=2.0,
    l_time=10000,
    gp_obs_hist=10000,
    share_obs_hist=50,
    animation=False,
    comm_mode="local",
    consensus_iters=5,
    attrition=False,
    attrition_steps="",
    # env_dyn="static",
    env_dyn="dynamic",
    env_change_steps=[500,800,2000],
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

EXPERIMENTS = []

for planner_cfg in PLANNERS:
    for n in N_AGENTS_LIST:
        for radius in COMM_RADII:
            for map_cfg in MAPS:
                for starts in get_starts(map_cfg, n):   # deterministic -> both planners get same starts
                    for freq in UPDATE_FREQ:
                        exp = dict(
                            **BASE_ARGS,
                            config=map_cfg,
                            starts=starts,
                            n_agents=n,
                            comm_radius=radius,
                            update_P_every=freq,
                            update_GP_every=freq,
                        )
                        exp.update(planner_cfg)          # override planner/method
                        EXPERIMENTS.append(exp)

print(f"Total: {len(EXPERIMENTS)}")


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