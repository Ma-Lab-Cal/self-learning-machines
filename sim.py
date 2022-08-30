import numpy as np

def sim_res(nodes: list, edges: list):
    '''
    Returns a list of voltage and current values for each node and edge. 
    Assumes only resistors and sources are used. 
    Returns voltage and current values for all node voltages, then all edge currents:
    [v1, v2, v3, ... i1, i2, ...]
    '''
    # bookkeeping data
    visited = [False] * len(nodes)
    loops = []
    loop_voltages = []
    unseen = set(edges)
    node_inds = {node: i for i, node in enumerate(nodes)}
    edge_inds = {edge: i for i, edge in enumerate(edges)}
    
    # loop detection
    while unseen:
        # logic ...
        pass

    # equation setup
    A = np.zeros((len(loops) + len(nodes), len(nodes) + len(edges)))
    b = np.concatenate((np.array(loop_voltages), np.zeros(len(nodes))))

    # KVL equations
    for loop in loops:
        pass

    # KCL equations 
    for node in nodes:
        for edge in node.neighbors:
            A[node_inds[node] + len(loops), len(nodes) + edge_inds[edge]] = 1

    # Solve system
    return np.linalg.solve(A, b)
