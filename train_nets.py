from PySpice.Unit import *
import PySpice
import numpy as np
import math
import networkx as nx
from helpers import *
from spice_net import *
import pickle
import time
import tqdm

# ngspice
PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = "ngspice-subprocess"


# XOR task
def init_model_xor(name, cls):
    np.random.seed(0)

    grid_graph = nx.grid_graph([4, 4], periodic=True)
    grid_graph.add_node((-1, -1))

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]["weight"] = np.random.uniform(0, 1)  # random init
        # grid_graph[e[0]][e[1]]['weight'] = 5                           # init to max value

    node_cfg = (np.array([[5, 16], [7, 16], [13, 16], [15, 16]]), np.array([[10, 0]]))
    return cls(name, con_graph=grid_graph, node_cfg=node_cfg, epsilon=1e-16)
    # TODO: look for checkpoints and load them if found

experiments = ["ground_reference_xor4", "source_reference_xor4"]
classes = [GroundReferenceNetwork, TransistorNetwork]

# load data
xor_data = np.load("data/xor_train_data.npz")
train_inputs = xor_data["inputs"]
train_outputs = xor_data["outputs"]

# define parameters
eta = 0.5
gamma = 0.01 * 1 / eta
total_steps = 100000
checkpoints = 20

# experiment runs
for name, cls in zip(experiments, classes):
    model = init_model_xor(name, cls)

    e1, e2 = [], []
    for E in model.edges:
        a, b = list(map(int, E.circ.node_names[:2]))
        e1.append(a)
        e2.append(b)

    intermediate_preds = []
    total_loss = []
    total_updates = []
    total_weights = []

    start = time.time()

    # training loop, total_epochs
    for i in tqdm.trange(checkpoints):
        this_steps = min(
            total_steps // checkpoints, total_steps - i * (total_steps // checkpoints)
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
                eta=eta,
                gamma=gamma,
            )
            preds[j] = pred.item()
            updates[j] = update

        intermediate_preds.append(model.predict(train_inputs))

        total_loss.append(((train_outputs[random_samples].squeeze() - preds) ** 2))
        total_updates.append(updates)
        total_weights.append([E.get_val() for E in model.edges])

        with open(f"checkpoints/{name}_{i}.pkl", "wb") as f:
            pickle.dump(
                dict(
                    total_loss=total_loss,
                    total_updates=total_updates,
                    total_weights=total_weights,
                    intermediate_preds=intermediate_preds,
                    eta=eta,
                    gamma=gamma,
                ),
                f,
            )

    end = time.time()
    print(f"{name} took {end - start} seconds")

experiments_2 = ["ground_reference_nonlinear4", "source_reference_nonlinear4"]
classes_2 = [GroundReferenceNetwork, TransistorNetwork]


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


# load data
nonlinear_data = np.load("data/nonlinear_regression_data.npz")
train_inputs = nonlinear_data["inputs"]
train_outputs = nonlinear_data["outputs"]

# define parameters
eta = 1.0
gamma = 0.01 * 1 / eta
total_steps = 100000
checkpoints = 20

for name, cls in zip(experiments_2, classes_2):
    model = init_model_nonlinear(name, cls)

    e1, e2 = [], []
    for E in model.edges:
        a, b = list(map(int, E.circ.node_names[:2]))
        e1.append(a)
        e2.append(b)

    intermediate_preds = []
    total_loss = []
    total_updates = []
    total_weights = []

    start = time.time()

    for i in tqdm.trange(checkpoints):
        this_steps = min(
            total_steps // checkpoints, total_steps - i * (total_steps // checkpoints)
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
                eta=eta,
                gamma=gamma,
            )
            preds[j] = pred.item()
            updates[j] = update

        intermediate_preds.append(model.predict(train_inputs))

        total_loss.append(((train_outputs[random_samples].squeeze() - preds) ** 2))
        total_updates.append(updates)
        total_weights.append([E.get_val() for E in model.edges])

        with open(f"checkpoints/{name}_{i}.pkl", "wb") as f:
            pickle.dump(
                dict(
                    total_loss=total_loss,
                    total_updates=total_updates,
                    total_weights=total_weights,
                    intermediate_preds=intermediate_preds,
                    eta=eta,
                    gamma=gamma,
                ),
                f,
            )

    end = time.time()
    print(f"{name} took {end - start} seconds")
