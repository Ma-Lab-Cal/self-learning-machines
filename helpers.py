from spice_net import *
import networkx as nx
from typing import Union

def train(net: Union[LinearNetwork, NonLinearNetwork], xs, ys, epochs, gamma = 0.01, eta = 0.1, shuffle=True):
    n_nodes = len(net.__nodes__)

    if hasattr(net, 'diodes'):
        n_edges = len(net.edges) + len(net.diodes)
        edges = net.edges + net.diodes
    else:
        n_edges = len(net.edges)
        edges = net.edges

    loss = np.empty(epochs+1)
    weights = np.empty((epochs+1, xs.shape[0], n_edges))
    updates = np.empty((epochs, xs.shape[0], n_edges))

    # Calculate initial accuracy 
    pred = np.array([net.predict(x) for x in xs])
    loss[0] = np.mean((ys - pred)**2)

    if hasattr(net, 'diodes'):
        weights[0] = np.tile(
            [R.resistance for R in net.edges] + [X.R1.resistance for X in net.nonlinear_vals]\
            , (xs.shape[0],1))
    else:
        weights[0] = np.tile([R.resistance for R in net.edges], (xs.shape[0],1))

    for i in range(epochs):
        if shuffle:
            perm = np.random.permutation(len(xs))
            xs = xs[perm]
            ys = ys[perm]
        for j, x, y in zip(range(xs.shape[0]), xs, ys):
            free = net.solve(x)
            nudges = eta * y + (1-eta) * net.predict(x)
            clamped = net.solve(x, nudges)

            free_rep = np.tile(free, [n_nodes, 1])
            clamped_rep = np.tile(clamped, [n_nodes,1])

            delta_free = free_rep - free_rep.T
            delta_clamped = clamped_rep - clamped_rep.T

            update = -gamma * (delta_clamped**2 - delta_free**2)
            trainable_updates = np.empty(n_edges)

            for k, R in enumerate(edges):
                a, b = list(map(int, R.node_names))
                trainable_updates[k] = update[a, b] #/ (R.resistance**2)

            net.update_y(trainable_updates)
            updates[i, j] = trainable_updates
            weights[i+1, j] = [R.resistance for R in net.edges] + [X.R1.resistance for X in net.nonlinear_vals]

        pred = np.array([net.predict(x) for x in xs])
        loss[i+1] = np.mean((ys - pred)**2)
        print(f'Epoch {i+1}: {loss[i+1]}')

    return net, loss, updates, weights
    
def visualize(net: Union[LinearNetwork, NonLinearNetwork], mode: str='r'):
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
