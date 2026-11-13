#!/bin/bash
OUT=results/TCNS_sim_5x5_v
for map in v0 rooms; do
  if [ "$map" = v0 ]; then s=4,4; else s=4,0; fi
  for M in 1 3 6; do
    for q in 3 10; do
      python test/ergodic_sim_wMatern.py --config configs/simple5x5_static_${map}.json \
        --agents $M --starts $s --q $q \
        --sigma 0.5 --lengthscale 1.5 --kvar 4.0 --kernel matern32 \
        --steps 10000000 --seeds 3 --workers 8 \
        --modes computed early fixed100 fixed1000 fixed10000 \
        --out $OUT/${map}_M${M}_q${q}
    done
  done
done