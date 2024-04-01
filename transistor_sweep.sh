#!/bin/bash

iterations=100000

# Sweep over learning rates from 1e-3 to 1e-1 (finish sweep)
python train_nets.py net xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --learning_rate=1e-3 --num_iterations=$iterations --num_checkpoints=20

# Try new run with different constants to test robustness
python train_nets.py no_scale xor source_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz
python train_nets.py no_scale xor source_reference --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz
python train_nets.py no_scale xor source_reference --learning_rate=1e-3 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz

python train_nets.py no_scale xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz
python train_nets.py no_scale xor ground_reference --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz
python train_nets.py no_scale xor ground_reference --learning_rate=1e-3 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_no_scale.npz