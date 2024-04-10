import argparse
from PySpice.Unit import *
import PySpice
import numpy as np
import math
import networkx as nx
from helpers import *
from spice_net import *
import pickle
import json
import time
import tqdm
import datetime
from os import path
import os
import wandb

# ngspice
PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = "ngspice-subprocess" 
import PySpice.Spice.NgSpice.Server
PySpice.Spice.NgSpice.Server.SpiceServer.SPICE_COMMAND = 'ngspice'

# Define command-line arguments
parser = argparse.ArgumentParser(description='Run a single experimental run.')
parser.add_argument('name', type=str, help='Name of the experiment')
parser.add_argument('task_name', type=str, help='Name of the task')
parser.add_argument('model_type', type=str, help='Type of model to use')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (default: 0.1)')
parser.add_argument('--nudge_factor', type=float, default=0.5, help='Nudge factor (default: 0.5)')
parser.add_argument('--num_iterations', type=int, default=100000, help='Number of iterations (default: 100000)')
parser.add_argument('--num_checkpoints', type=int, default=20, help='Number of checkpoints (default: 20)')
parser.add_argument('--dataset', type=str, default=None, help='Number of checkpoints. Defaults to dataset from paper')

# Parse command-line arguments
args = parser.parse_args()

# XOR task
def init_model_xor(name, cls, eta=0.5):
    np.random.seed(0)

    grid_graph = nx.grid_graph([4, 4], periodic=True)
    grid_graph.add_node((-1, -1))

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]["weight"] = np.random.uniform(0, 1)  # random init
        # grid_graph[e[0]][e[1]]['weight'] = 5                           # init to max value

    node_cfg = (np.array([[5, 16], [7, 16], [13, 16], [15, 16]]), np.array([[10, 0]]))
    return cls(name, con_graph=grid_graph, node_cfg=node_cfg, epsilon=1e-16)
    # TODO: look for checkpoints and load them if found

# Nonlinear task
def init_model_nonlinear(name, cls):
    np.random.seed(0)

    grid_graph = nx.grid_graph([4, 4], periodic=True)
    grid_graph.add_node((-1, -1))

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]["weight"] = np.random.uniform(0, 5)  # random init
        # grid_graph[e[0]][e[1]]['weight'] = 5                            # init to max value

    # input orderings: I_neg, I_pos, I_1
    # output orderings: O is represented using a single output
    node_cfg = (np.array([[8, 16], [2, 16], [15, 16]]), np.array([[10, 16]]))
    return cls(name, con_graph=grid_graph, node_cfg=node_cfg, epsilon=1e-16)


    
# Load data
if args.dataset is not None:
    data = np.load(args.dataset)
    train_inputs = data["inputs"]
    train_outputs = data["outputs"]
else:
    if args.task_name == "xor":
        xor_data = np.load(path.join("data", "xor_train_data.npz"))
        train_inputs = xor_data["inputs"]
        train_outputs = xor_data["outputs"]
    elif args.task_name == "regression":
        nonlinear_data = np.load(path.join("data", "nonlinear_regression_data.npz"))
        train_inputs = nonlinear_data["inputs"]
        train_outputs = nonlinear_data["outputs"]
    else: 
        raise ValueError(f"Unknown task: {args.task_name}")

# Initialize model
try:
    network_type = {"ground_reference": GroundReferenceNetwork, "source_reference": TransistorNetwork}[args.model_type]
except KeyError:
    raise ValueError(f"Unknown model type: {args.model_type}")

if args.task_name == "xor":
    model = init_model_xor(args.name, network_type)
elif args.task_name == "regression":
    model = init_model_nonlinear(args.name, network_type)

e1, e2 = [], []
for E in model.edges:
    a, b = list(map(int, E.circ.node_names[:2]))
    e1.append(a)
    e2.append(b)

# Run a single experimental run
intermediate_preds = []
total_loss = []
total_updates = []
total_weights = []

# create logging dir
date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
run_name = f'{args.name}_{args.task_name}_{args.model_type}_lr_{args.learning_rate}_eta_{args.nudge_factor}_{date}'
checkpoint_path = path.join("checkpoints", run_name)
os.makedirs(checkpoint_path, exist_ok=True)

# initialize logging
wandb.init(project='transistor-networks', name=run_name, config=args)

# save hyperparameters
print(args)
with open(os.path.join(checkpoint_path,'args.json'), 'w') as fh:
    json.dump(vars(args), fh, indent=4)

start = time.time()

for i in tqdm.trange(args.num_checkpoints):
    this_steps = min(
        args.num_iterations // args.num_checkpoints, args.num_iterations - i * (args.num_iterations // args.num_checkpoints)
    )
    random_samples = np.random.choice(
        len(train_inputs), size=this_steps, replace=True
    )
    preds = np.empty(this_steps)
    updates = np.empty((this_steps, len(model.edges)))

    for j in tqdm.trange(this_steps, leave=False):
        model, pred, update = step_network(
            model,
            train_inputs[random_samples[j]],
            train_outputs[random_samples[j]],
            e1, 
            e2,
            eta=args.learning_rate,
            gamma=args.nudge_factor,
        )
        preds[j] = pred.item()
        updates[j] = update
        wandb.log({"loss": ((train_outputs[random_samples[j]].squeeze() - pred) ** 2).item(), "step": i * (args.num_iterations // args.num_checkpoints) + j})

    intermediate_preds.append(model.predict(train_inputs))
    wandb.log({"intermediate_pred": intermediate_preds[-1]})

    # generate plt plot of intermediate and save to wandb
    fig = plt.figure()
    if args.dataset is not None and "no_scale" in args.dataset:
        plt.imshow(intermediate_preds[-1].reshape(2, 2), vmin=-1, vmax=1)

        for m in range(2):
            for n in range(2):
                plt.text(n, m, f"{intermediate_preds[-1].reshape(2, 2)[m, n]:.2f}", ha='center', va='center', color='white')
    else:
        L_0 = -0.087
        plt.imshow(intermediate_preds[-1].reshape(2, 2) / L_0, vmin=-1, vmax=1)

        for m in range(2):
            for n in range(2):
                plt.text(n, m, f"{intermediate_preds[-1].reshape(2, 2)[m, n]/L_0:.2f} $L_0$", ha='center', va='center', color='white')

    wandb.log({"intermediate_plot": wandb.Image(fig)})


    total_loss.append(((train_outputs[random_samples].squeeze() - preds) ** 2))
    total_updates.append(updates)
    total_weights.append([E.get_val() for E in model.edges])

    with open(path.join(checkpoint_path, f"checkpoint_{i}.pkl"), "wb") as f:
        pickle.dump(
            dict(
                total_loss=total_loss,
                total_updates=total_updates,
                total_weights=total_weights,
                intermediate_preds=intermediate_preds,
                eta=args.learning_rate,
                gamma=args.nudge_factor,
            ),
            f,
        )

end = time.time()
print(f"{args.name} took {end - start} seconds")
