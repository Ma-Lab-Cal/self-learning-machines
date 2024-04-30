#!/bin/bash

# Finish aborted runs from last time
python train_nets.py paper_robustness xor ground_reference --learning_rate=0.1 --nudge_factor=1.0 --num_iterations=100000 --num_checkpoints=20

python train_nets.py paper_robustness xor ground_reference --learning_rate=0.66 --nudge_factor=1.0 --num_iterations=100000 --num_checkpoints=20


# Run experiments on source referenced network to see its performance vs. ground-referenced

# Define the values for nudge_factor and learning_rate
nudge_factors=(0.1 0.5 1.0)
learning_rates=(0.066 0.1 0.66)

# Outer loop for nudge_factor
for nudge_factor in "${nudge_factors[@]}"; do
    # Inner loop for learning_rate
    for learning_rate in "${learning_rates[@]}"; do
        echo "Running with nudge_factor=$nudge_factor and learning_rate=$learning_rate"
        python train_nets.py ours_robustness xor source_reference \
            --learning_rate=$learning_rate \
            --nudge_factor=$nudge_factor \
            --num_iterations=100000 \
            --num_checkpoints=20
    done
done
