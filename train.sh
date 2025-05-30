#!/bin/bash

# Double integrator example
python -m experiments.run +experiment=double_integrator +experiment/policy_model=mlp_2_16_tanh train=true

# Cart-pole
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_32 train=true
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_64 train=true
python -m experiments.run +experiment=cartpole +experiment/policy_model=mlp_2_128 train=true

# Planar quadrotor
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_32 train=true
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_64 train=true
python -m experiments.run +experiment=drone2d +experiment/policy_model=mlp_2_128 train=true

# Three-dimensional quadrotor
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_32 train=true
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_64 train=true experiment.train.n_epochs=6
python -m experiments.run +experiment=drone3d +experiment/policy_model=mlp_4_128 train=true experiment.train.n_epochs=10

# Panda robot arm
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_32 train=true
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_64 train=true
python -m experiments.run +experiment=robot_reach +experiment/policy_model=mlp_2_128 train=true
