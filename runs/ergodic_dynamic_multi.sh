#!/bin/bash

python main.py \
  --planner ergodic \
  --mode gp \
  --method remc \
  --gp_mode mean \
  --T 2001 \
  --neighbors 4 \
  --n_agents 3 \
  --starts "3,1;3,1;3,1" \
  --comm_mode local \
  --comm_radius 2.0 \
  --consensus_iters 5 \
  --update_GP_every 25 \
  --update_P_every 25 \
  --env_dyn dynamic \
  --config configs/dynamic_simple5x5_1/