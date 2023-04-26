from spice_net import *
import networkx as nx
from typing import Union

def train(net: Union[LinearNetwork, NonLinearNetwork], xs, ys, epochs, gamma = 0.01, eta = 0.1, log_steps=None, shuffle=True):
    n_nodes = len(net.__nodes__)

    if log_steps is None:
        log_steps = list(range(epochs))

    if hasattr(net, 'diodes'):
        n_edges = len(net.edges) + len(net.diodes)
        edges = net.edges + net.diodes
    else:
        n_edges = len(net.edges)
        edges = net.edges

    loss = np.zeros(epochs+1)
    weights = np.empty((len(log_steps)+1, xs.shape[0], n_edges))
    updates = np.empty((len(log_steps), xs.shape[0], n_edges))

    # Calculate initial accuracy 
    pred = net.predict(xs)
    loss[0] = np.mean((ys - pred)**2)

    if hasattr(net, 'diodes'):
        weights[0] = np.tile(
            [R.resistance for R in net.edges] + [X.R1.resistance for X in net.nonlinear_vals]\
            , (xs.shape[0],1))
    else:
        weights[0] = np.tile([R.resistance for R in net.edges], (xs.shape[0],1))

    e1, e2 = [], []
    for R in edges:
        a, b = list(map(int, R.node_names))
        e1.append(a)
        e2.append(b)

    for i in range(epochs):
        if shuffle:
            perm = np.random.permutation(len(xs))
            xs = xs[perm]
            ys = ys[perm]
        for j, x, y in zip(range(xs.shape[0]), xs, ys):
            free = net.solve(x)
            nudges = eta * y + (1-eta) * net.predict(x)
            clamped = net.solve(x, nudges.reshape(y.shape))

            free_rep = np.tile(free, [n_nodes, 1])
            clamped_rep = np.tile(clamped, [n_nodes,1])

            delta_free = free_rep - free_rep.T
            delta_clamped = clamped_rep - clamped_rep.T

            update = -gamma * (delta_clamped**2 - delta_free**2)
            # trainable_updates = np.empty(n_edges)

            trainable_updates = update[e1, e2] #/ (R.resistance**2)

            net.update_y(trainable_updates)
            if i in log_steps:
                step = log_steps
                updates[i, j] = trainable_updates
                weights[i+1, j] = [R.resistance for R in net.edges] + [X.R1.resistance for X in net.nonlinear_vals]

        pred = net.predict(xs)
        loss[i+1] = np.mean((ys - pred)**2)
        print(f'Epoch {i+1}: {loss[i+1]}')

    return net, loss, updates, weights
    
def visualize(net: Union[LinearNetwork, NonLinearNetwork], mode: str='y'):
    G = nx.DiGraph()

    for R in net.edges:
        a, b = R.node_names
        # a, b = list(map(int, R.node_names))
        if mode == 'r':
            G.add_edge(b, a, weight=R.resistance, type='edge')
            G.add_edge(a, b, weight=R.resistance, type='edge')
        else:
            G.add_edge(b, a, weight=1./R.resistance, type='edge')
            G.add_edge(a, b, weight=1./R.resistance, type='edge')

    for V in net.inputs:
        # ltspice lists coordinates as v+, v-, so flip the order 
        #   so that arrow points from low -> high potential
        a, b = V.node_names
        G.add_edge(b, a, weight=-1, type='source', io='input')

    for V in net.outputs:
        # ltspice lists coordinates as v+, v-, so flip the order 
        #   so that arrow points from low -> high potential
        a, b = V.node_names
        G.add_edge(b, a, weight=-1, type='source', io='output')

    # nonlinear networks only
    # create a separate graph for diodes so diode edges don't overwrite voltage sources
    Gprime = nx.DiGraph()
    Gprime.add_nodes_from(G)
    if hasattr(net, 'diodes'):
        for X, ss in zip(net.diodes, net.nonlinear_vals):
            a, b = X.node_names
            if mode == 'r':
                Gprime.add_edge(a, b, weight=ss.R1.resistance, type='diode')
            else:
                Gprime.add_edge(a, b, weight=1./ss.R1.resistance, type='diode')
        
    def filter_type(label):
        return lambda u, v: G.get_edge_data(u, v, default={'type':None})['type'] == label

    pos = nx.shell_layout(nx.compose(Gprime, G))

    inputs_only = nx.subgraph_view(G, filter_edge=lambda u, v: filter_type('source')(u, v) and G.get_edge_data(u, v, default={'type':None})['io'] == 'input')
    outputs_only = nx.subgraph_view(G, filter_edge=lambda u, v: filter_type('source')(u, v) and G.get_edge_data(u, v, default={'type':None})['io'] == 'output')
    edges_only = nx.subgraph_view(G, filter_edge=filter_type('edge')).to_undirected(as_view=True)
    diodes_only = Gprime

    # nodes
    nx.draw_networkx_labels(nx.compose(Gprime, G), pos)
    nx.draw_networkx_nodes(nx.compose(Gprime, G), pos)

    # resistive edges
    nx.draw_networkx_edges(edges_only, pos, edge_color='black',)
    nx.draw_networkx_edge_labels(edges_only, pos, edge_labels={(u, v): round(edges_only[u][v]['weight'], 2) for u, v in edges_only.edges()})

    # diode (ReLu) edges
    nx.draw_networkx_edges(diodes_only, pos, edge_color='blue',)
    nx.draw_networkx_edge_labels(diodes_only, pos, edge_labels={(u, v): round(diodes_only[u][v]['weight'], 2) for u, v in diodes_only.edges()})

    # voltage sources
    nx.draw_networkx_edges(inputs_only, pos, edge_color='red', connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_edges(outputs_only, pos, edge_color='purple', connectionstyle='arc3,rad=0.2')
    # nx.draw_networkx_edge_labels(sources_only, pos, edge_labels={(u, v): sources_only[u][v]['io'] for u, v in sources_only.edges()},)
