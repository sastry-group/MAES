#!/bin/bash

python main.py \
  --planner ergodic \
  --mode gp \
  --method remc \
  --gp_mode mean \
  --T 2000 \
  --neighbors 4 \
  --n_agents 1 \
  --starts "3,1" \
  --update_GP_every 25 \
  --update_P_every 25 \
  --env_dyn static \
  --config configs/simple5x5_static.json