from spice_net import *
from PySpice.Unit import u_V
import networkx as nx
from typing import Union
import os
import ltspice
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import pickle

def get_content_cocontent(VGS, vmin=-0.5, vmax=5, n=1000):
    test_circuit = f"""
    M1 D G 0 0 NMOS
    VD D 0 0
    * source-reference VGS
    VGS G S {VGS}
    .model NMOS NMOS
    .dc VD {vmin} {vmax} {n}
    """
    with open('test.cir', 'w') as f:
        f.write(test_circuit)
    os.system('/Applications/LTspice.app/Contents/MacOS/LTspice -b test.cir test.cir -o test.raw')

    l = ltspice.Ltspice(f'test.raw')
    l.parse()

    I = l.get_data('Id(M1)')
    V = l.get_data('V(D)')

    cocontent_table = cumtrapz(I, V, initial=0)
    cocontent_table -= np.interp([0], V, cocontent_table)
    content_table = cumtrapz(V, I, initial=0)
    content_table -= np.interp([0], I, content_table)

    return content_table, cocontent_table, I, V

def step_network(net: AbstractNetwork, x, y, e1, e2, gamma = 10, eta = 0.1, l = 0):
    n_nodes = len(net.__nodes__)

    free = net.solve(x)
    # free = np.array([u_V(free_analysis.nodes[str(i)]) for i in net.__nodes__])

    preds = np.zeros((len(net.outputs), 1))

    for k, v in enumerate(net.outputs):
        a, b = v.node_names
        a, b = int(a), int(b)
        preds[k] = u_V(free[a] - free[b,:])
    nudges = eta * y + (1-eta) * preds

    clamped = net.solve(x, nudges.reshape(y.shape))

    free_rep = np.tile(free, [n_nodes, 1])
    clamped_rep = np.tile(clamped, [n_nodes,1])

    delta_free = free_rep - free_rep.T
    delta_clamped = clamped_rep - clamped_rep.T

    update = -gamma * (delta_clamped**2 - delta_free**2)

    trainable_updates = update[e1, e2]

    net.update(trainable_updates)

    return net, preds, trainable_updates

def train(net: AbstractNetwork, xs, ys, epochs, gamma = 10, eta = 0.1, l = 0, log_steps=1, shuffle=True):
    """Training loop

    Args:
        net (Union[LinearNetwork, TransistorNetwork]): The network to be trained. Only one copy is needed. Needs to have
        each edge derived from the template Edge class. 
        xs: Training inputs
        ys: Target outputs
        epochs (Int): Number of epochs to train for
        gamma (float, optional): Edge scaling factor. As a rule of thumb, should be inversely porportional to eta. Defaults to 0.01.
        eta (float, optional): Nudge parameter for contrastive learning. Defaults to 0.1.
        l (float, optional): Regularization parameter. Balances finding solutions that fit the data with those that dissipate less power. 
        Defaults to 0.
        log_steps (int, optional): Log the loss every log_steps epochs. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset before each epoch. Defaults to True.

    Returns:
        _type_: _description_
    """
    n_nodes = len(net.__nodes__)

    n_edges = len(net.edges)
    edges = net.edges

    step_interval = int(np.ceil(epochs / log_steps)) # todo: better name for this? 
    loss = np.zeros(step_interval+1)
    weights = np.empty((step_interval+1, xs.shape[0], n_edges))
    updates = np.empty((step_interval, xs.shape[0], n_edges))

    # Calculate initial accuracy 
    pred = net.predict(xs)
    loss[0] = np.mean((ys - pred)**2)

    weights[0] = np.tile([E.get_val() for E in net.edges], (xs.shape[0],1))

    e1, e2 = [], []
    for E in edges:
        a, b = list(map(int, E.circ.node_names[:2]))
        e1.append(a)
        e2.append(b)

    for i in range(epochs):
        if shuffle:
            perm = np.random.permutation(len(xs))
            xs = xs[perm]
            ys = ys[perm]
        for j, x, y in zip(range(xs.shape[0]), xs, ys):
            # compute nudges from free state without an additional
            # expensive call to solve: saves roughly 33% computation time

            free = net.solve(x)

            preds = np.zeros((len(net.outputs), 1))

            for k, v in enumerate(net.outputs):
                a, b = v.node_names
                print("HHHH")
                print(preds.shape, free.shape)
                preds[k] = u_V(free[a,:] - free[b,:])
            nudges = eta * y + (1-eta) * preds

            clamped = net.solve(x, nudges.reshape(y.shape))

            # TODO: update computation can be optimized since most 
            # values here are unused, worth?
            free_rep = np.tile(free, [n_nodes, 1])
            clamped_rep = np.tile(clamped, [n_nodes,1])

            delta_free = free_rep - free_rep.T
            delta_clamped = clamped_rep - clamped_rep.T

            update = -gamma * (delta_clamped**2 - delta_free**2)

            trainable_updates = update[e1, e2]

            net.update(trainable_updates) 

            if (i % log_steps) == 0:
                updates[i // log_steps, j] = trainable_updates
                weights[(i // log_steps) + 1, j] = [E.get_val() for E in net.edges]

        preds = net.predict(xs)
        loss[(i // log_steps) + 1] = np.mean((ys - preds)**2)

        if (i % log_steps) == 0:
            print(f'Epoch {i+1}: {loss[(i // log_steps) + 1]}')

    return net, loss, updates, weights
    
def visualize(net: Union[LinearNetwork, TransistorNetwork], pos=None):
    G = nx.DiGraph()
    G.add_nodes_from(range(len(net.__nodes__)))

    edge_types = {}
    for E in net.edges:
        a, b = list(map(int, E.circ.node_names[:2]))
        edge_types[type(E)] = str(type(E))
        G.add_edge(a, b, weight=E.get_val(), type=str(type(E)))

    for V in net.inputs:
        # ltspice lists coordinates as v+, v-, so flip the order 
        #   so that arrow points from low -> high potential
        a, b = list(map(int, V.node_names))
        nx.set_node_attributes(G, {a: 'input+', b: 'input-'}, 'type')

    for V in net.outputs:
        # ltspice lists coordinates as v+, v-, so flip the order 
        #   so that arrow points from low -> high potential
        a, b = list(map(int, V.node_names))
        nx.set_node_attributes(G, {a: 'output+', b: 'output-'}, 'type')


    def filter_edges(label):
        return nx.subgraph_view(G, filter_edge=lambda u, v: ('type' in G[u][v]) and G[u][v]['type'] == label)

    def filter_nodes(G, label):
        return nx.subgraph_view(G, filter_node=lambda u: ('type' in G.nodes[u]) and G.nodes[u]['type'] == label)

    if pos is None:
        pos = nx.spring_layout(G)

    node_colors = {'input-': 'tab:purple', 'input+': 'tab:red', 'output-': 'tab:cyan', 'output+': 'tab:green'}

    nx.draw(G, pos=pos, with_labels=True)
    for node_type in set(nx.get_node_attributes(G, 'type').values()):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=filter_nodes(G, node_type).nodes, node_color=node_colors[node_type], label=node_type)

    for edge_type in set(edge_types.values()):
        weights = list(nx.get_edge_attributes(filter_edges(edge_type), 'weight').values())
        nx.draw_networkx_edges(G, pos=pos, edgelist=filter_edges(edge_type).edges, label=edge_type, width=weights)

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels={tuple(map(int, E.circ.node_names[:2])): f'{E.get_val():.2f}' for E in net.edges}, rotate=False)
    plt.legend()
    plt.title(f'Network: {net.name}')

def load_checkpoint(path):
    # sample checkpoint path: 'checkpoints/paper_regression_ground_reference_lr_11.0_eta_1.0_2024-04-10-19-39'

    # split path into directory and file name
    _, file = os.path.split(path)

    i = 0
    loss=[]
    updates=[]
    weights=[]
    intermediate_preds=[]
    eta=None
    gamma=None
    seed=None
    while os.exists(os.path.join(path, f'checkpoint{i}.pkl')):
        with open(os.path.join(path, f'checkpoint{i}.pkl'), 'rb') as f:
            d = pickle.load(f)
            loss.append(d['loss'])
            updates.append(d['updates'])
            weights.append(d['weights'])
            intermediate_preds.append(d['intermediate_preds'])
            eta = d['eta']
            gamma = d['gamma']
            seed = d['seed']
        i += 1

    return loss, updates, weights, intermediate_preds, eta, gamma, seed