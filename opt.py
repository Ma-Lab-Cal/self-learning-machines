import numpy as np
from scipy.optimize import minimize

def min_fun(mat):
    def pwr_dis(x):
        pwr = 0
        for i in range(len(x)):
            for j in range(len(x)):
                pwr += int(x[i]>x[j])*(x[i]-x[j])**2*mat[i,j]

        return pwr
    return pwr_dis

def better_obj(mat):
    def pwr_dis(x):
        xs = np.repeat(x[:,None], x.shape, 1)
        return np.sum((xs > xs.T).astype(int)*(xs - xs.T)**2*mat)
    return pwr_dis

def solve(mat, bounds):
    obj = better_obj(mat)
    x0 = np.random.random(bounds.shape[0])
    return minimize(obj, x0, bounds=bounds)

def train(mat, xs, ys, node_cfg, epochs, gamma = 0.01, nu = 0.1, ic = None):
    assert np.count_nonzero(node_cfg > 0) == xs.shape[1], "Shape of training examples doesn't match number of inputs."
    assert np.count_nonzero(node_cfg < 0) == ys.shape[1], "Shape of training examples doesn't match number of outputs."

    trainable_mask = np.logical_and(mat == mat.T, mat > 0)
    ic = ic or np.random.rand(*node_cfg.shape)
    n = ic.shape[0]

    bounds = np.array([[None, None] for _ in node_cfg])
    bounds[-1,: ] = [0, 0] # ground node

    in_nodes = np.argwhere(node_cfg > 0).flatten()
    out_nodes = np.argwhere(node_cfg < 0).flatten()

    for _ in range(epochs):
        for x, y in zip(xs, ys):
                # Solve free state
                bounds[in_nodes, 0] = x
                bounds[in_nodes, 1] = x
                bounds[out_nodes, 0] = None
                bounds[out_nodes, 1] = None
                free = solve(mat, bounds)
                # Solve clamped state
                bounds[out_nodes, 0] = y * nu + (1-nu) * free.x[out_nodes]
                bounds[out_nodes, 1] = y * nu + (1-nu) * free.x[out_nodes]
                clamped = solve(mat, bounds)

                if not (free.success and clamped.success):
                    return free.message + ',' + clamped.message

                free_rep = np.repeat(free.x[:,None], n, axis=1)
                clamped_rep = np.repeat(clamped.x[:,None], n, axis=1)

                delta_free = free_rep - free_rep.T
                delta_clamped = clamped_rep - clamped_rep.T

                nudges = gamma * (delta_clamped**2 - delta_free**2)
                mat -= trainable_mask * nudges

    return mat

def inference(mat, xs, node_cfg):
    n = xs.shape[0]
    bounds = np.array([[None, None] for _ in node_cfg])
    bounds[-1,: ] = [0, 0] # ground node

    in_nodes = np.argwhere(node_cfg > 0).flatten()
    out_mask = node_cfg < 0

    outputs = np.empty(shape=(n, out_mask.size))

    for i in range(n):
        bounds[in_nodes, 0] = xs[i]
        bounds[in_nodes, 1] = xs[i]

        sol = solve(mat, bounds)
        if not sol.success:
            return sol.message

        outputs[i] = sol.x

    return outputs
