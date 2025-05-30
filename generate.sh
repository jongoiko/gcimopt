#!/bin/bash

declare -a experiments=(
   "double_integrator"
   "cartpole"
   "drone2d"
   "drone3d"
   "robot_reach"
)

for exp in "${experiments[@]}"; do
   python -m experiments.run +experiment=$exp generate=true
done
