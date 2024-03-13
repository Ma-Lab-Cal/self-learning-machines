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
PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess' 

# XOR task
def init_model_xor(name, cls):
    np.random.seed(0)

    grid_graph = nx.grid_graph([4, 4], periodic=True)
    grid_graph.add_node((-1, -1))

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]['weight'] = np.random.uniform(0, 1)    # random init
        # grid_graph[e[0]][e[1]]['weight'] = 5                           # init to max value

    node_cfg = (np.array([[5, 16], [7, 16], [13, 16], [15, 16]]), np.array([[10, 0]]))
    return cls(name, con_graph=grid_graph, node_cfg=node_cfg, epsilon=1e-16)

experiments = ['ground_reference_xor', 'source_reference_xor']
classes = [GroundReferenceNetwork, TransistorNetwork]

# load data
xor_data = np.load('data/xor_train_data.npz')
train_inputs = xor_data['inputs']
train_outputs = xor_data['outputs']

# define parameters
eta = 0.5
gamma = 0.1 * 1/eta
total_epochs = 100000
checkpoints = 15
epochs = int(math.ceil(total_epochs / checkpoints))

for name, cls in zip(experiments, classes):
    model = init_model_xor(name, cls)

    intermediate_preds = []
    total_loss = []
    total_updates = []
    total_weights = []

    start = time.time()

    for i in tqdm.trange(checkpoints):
        model, loss, updates, weights = train(model, train_inputs, train_outputs, epochs=epochs, gamma=gamma, eta=eta, log_steps=epochs//10)
        intermediate_preds.append(model.predict(train_inputs))

        total_loss.append(loss)
        total_updates.append(updates)
        total_weights.append(weights)

        with open (f'checkpoints/{name}_{i}.pkl', 'wb') as f:
            pickle.dump(dict(
                total_loss=total_loss,
                total_updates=total_updates,
                total_weights=total_weights,
                intermediate_preds=intermediate_preds,
                eta=eta,
                gamma=gamma,
                epochs=epochs
            ), f)

    end = time.time()
    print(f'{name} took {end - start} seconds')

experiments_2 = ['ground_reference_nonlinear', 'source_reference_nonlinear']
classes_2 = [GroundReferenceNetwork, TransistorNetwork]

# Nonlinear task
def init_model_nonlinear(name, cls):
    np.random.seed(0)

    grid_graph = nx.grid_graph([4, 4], periodic=True)
    grid_graph.add_node((-1, -1))

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]['weight'] = np.random.uniform(0, 5)    # random init
        # grid_graph[e[0]][e[1]]['weight'] = 5                            # init to max value

    # input orderings: I_neg, I_pos, I_1
    # output orderings: O is represented using a single output
    node_cfg = (np.array([[8, 16], [2, 16], [15, 16]]), np.array([[10, 16]]))
    return cls(name, con_graph=grid_graph, node_cfg=node_cfg, epsilon=1e-16)


#load data
nonlinear_data = np.load('data/nonlinear_regression_data.npz')
train_inputs = nonlinear_data['inputs']
train_outputs = nonlinear_data['outputs']

# define parameters
eta = 1.0
gamma = 0.1 * 1/eta
total_iters = 100000
checkpoints = 15
epochs = int(math.ceil(total_iters / checkpoints))

for name, cls in zip(experiments_2, classes_2):
    model = init_model_nonlinear(name, cls)

    intermediate_preds = []
    total_loss = []
    total_updates = []
    total_weights = []

    start = time.time()

    for i in tqdm.trange(checkpoints):
        model, loss, updates, weights = train(model, train_inputs, train_outputs, epochs=epochs, gamma=gamma, eta=eta, log_steps=epochs//10)
        intermediate_preds.append(model.predict(train_inputs))

        total_loss.append(loss)
        total_updates.append(updates)
        total_weights.append(weights)

        with open (f'checkpoints/{name}_{i}.pkl', 'wb') as f:
            pickle.dump(dict(
                total_loss=total_loss,
                total_updates=total_updates,
                total_weights=total_weights,
                intermediate_preds=intermediate_preds,
                eta=eta,
                gamma=gamma,
                epochs=epochs
            ), f)

    end = time.time()
    print(f'{name} took {end - start} seconds')