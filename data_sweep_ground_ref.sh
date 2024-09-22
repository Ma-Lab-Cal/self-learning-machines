#!/bin/bash

iterations=100000

# Sweep over dataset scalings for the ground referenced XOR task
for scale in $(seq 0.0 0.1 1.0); do
    python train_nets.py scale_$scale xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_$scale.npz
done
