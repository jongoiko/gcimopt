#!/bin/bash

# Double integrator example
python -m experiments.run +experiment=double_integrator +experiment/policy_model=mlp_2_16_tanh evaluate=true

# Cart-pole
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_32 evaluate=true
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_64 evaluate=true
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_128 evaluate=true

# Planar quadrotor
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_32 evaluate=true
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_64 evaluate=true
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_128 evaluate=true

# Three-dimensional quadrotor
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_32 evaluate=true
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_64 evaluate=true
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_128 evaluate=true

# Panda robot arm
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_32 evaluate=true
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_64 evaluate=true
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_128 evaluate=true
