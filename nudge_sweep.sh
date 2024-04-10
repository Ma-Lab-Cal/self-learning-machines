#!/bin/bash

iterations=100000

# Fix learning rate at 1e-1 and sweep over nudge factor values 0.3, 0.1, 0.03
python train_nets.py net xor source_reference --nudge_factor 0.3 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor source_reference --nudge_factor 0.1 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor source_reference --nudge_factor 0.03 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20

python train_nets.py net xor ground_reference --nudge_factor 0.3 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --nudge_factor 0.1 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --nudge_factor 0.03 --learning_rate=1e-1 --num_iterations=$iterations --num_checkpoints=20

# Repeat experiment with learning rate fixed at 1e-2
python train_nets.py net xor source_reference --nudge_factor 0.3 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor source_reference --nudge_factor 0.1 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor source_reference --nudge_factor 0.03 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20

python train_nets.py net xor ground_reference --nudge_factor 0.3 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --nudge_factor 0.1 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20
python train_nets.py net xor ground_reference --nudge_factor 0.03 --learning_rate=1e-2 --num_iterations=$iterations --num_checkpoints=20