#!/bin/bash

iterations=100000

# Sweep over dataset scalings for the ground referenced XOR task
python train_nets.py scale_0.0 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.0.npz
python train_nets.py scale_0.1 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.1.npz
python train_nets.py scale_0.2 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.2.npz
python train_nets.py scale_0.3 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.3.npz
python train_nets.py scale_0.4 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.4.npz
python train_nets.py scale_0.5 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.5.npz
python train_nets.py scale_0.6 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.6.npz
python train_nets.py scale_0.7 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.7.npz
python train_nets.py scale_0.8 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.8.npz
python train_nets.py scale_0.9 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_0.9.npz
python train_nets.py scale_1.0 xor ground_reference --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20 --dataset=data/xor_train_data_scale_1.0.npz