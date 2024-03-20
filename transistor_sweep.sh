#!/bin/bash

# Sweep over learning rates from 1e-3 to 1e-1
python train_nets.py net xor source_reference --learning_rate=1e-1 --num_iterations=100000 --num_checkpoints=20
python train_nets.py net xor source_reference --learning_rate=1e-2 --num_iterations=100000 --num_checkpoints=20
python train_nets.py net xor source_reference --learning_rate=1e-3 --num_iterations=100000 --num_checkpoints=20

python train_nets.py net xor ground_reference --learning_rate=1e-1 --num_iterations=100000 --num_checkpoints=20
python train_nets.py net xor ground_reference --learning_rate=1e-2 --num_iterations=100000 --num_checkpoints=20
python train_nets.py net xor ground_reference --learning_rate=1e-3 --num_iterations=100000 --num_checkpoints=20