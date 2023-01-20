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
<<<<<<< Updated upstream
        xs = np.repeat(x[:,None], x.shape, 1)
        return np.sum((xs > xs.T).astype(int)*(xs - xs.T)**2*mat)
=======
        xs = np.repeat(x[:,None], x.shape[0], 1)
        return np.sum((xs > xs.T) * (xs - xs.T)**2 * mat)
>>>>>>> Stashed changes
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

    bounds = np.repeat([[None, None]], n, 0)

    bounds[-1] = [0,0]

    losses = np.empty(epochs)

    for i in range(epochs):
        for x, y in zip(xs, ys):
                # assert (mat >= 0).all(), f'{mat}'
                # Solve free state
                bounds[in_nodes, 0] = x
                bounds[in_nodes, 1] = x
                bounds[out_nodes, 0] = None
                bounds[out_nodes, 1] = None
<<<<<<< Updated upstream
                free = solve(mat, bounds)
                # Solve clamped state
                bounds[out_nodes, 0] = y * nu + (1-nu) * free.x[out_nodes]
                bounds[out_nodes, 1] = y * nu + (1-nu) * free.x[out_nodes]
                clamped = solve(mat, bounds)
=======

                free = solve(mat, bounds)
                retries, cnt = 2, 0
                while not free.success and cnt < retries:
                    free = solve(mat, bounds)
                    retries += 1

                # Solve clamped state
                bounds[out_nodes, 0] = y * nu + (1-nu) * free.x[out_nodes, None].flatten()
                bounds[out_nodes, 1] = y * nu + (1-nu) * free.x[out_nodes, None].flatten()
                clamped = solve(mat, bounds)

                cnt = 0
                while not clamped.success and cnt < retries:
                    clamped = solve(mat, bounds)
                    retries += 1
>>>>>>> Stashed changes

                if not (free.success and clamped.success):
                    return free.message + ',' + clamped.message

<<<<<<< Updated upstream
                free_rep = np.repeat(free.x[:,None], n, axis=1)
                clamped_rep = np.repeat(clamped.x[:,None], n, axis=1)
=======
                free_rep = np.repeat(free.x[:n,None], n, axis=1)
                clamped_rep = np.repeat(clamped.x[:n,None], n, axis=1)
>>>>>>> Stashed changes

                delta_free = free_rep - free_rep.T
                delta_clamped = clamped_rep - clamped_rep.T

                nudges = gamma * (delta_clamped**2 - delta_free**2)
                mat -= trainable_mask * nudges

        pred = inference(mat, xs, node_cfg)
        losses[i] = np.sum(-ys*np.log(np.maximum(pred, 1e-6)) -(1-ys)*np.log(np.maximum(1-pred, 1e-6)))
        if i % (epochs//5)  == 0:
            print(f'Epoch {i}: {losses[i//2]}')         

    return mat, losses

def inference(mat, xs, node_cfg):
    n = xs.shape[0]
<<<<<<< Updated upstream
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
=======

    in_nodes = np.argwhere(node_cfg > 0).flatten()
    out_mask = node_cfg < 0
    outputs = np.empty(shape=(n, np.count_nonzero(out_mask)))

    bounds = np.repeat([[None, None]], mat.shape[0], 0)
>>>>>>> Stashed changes

    for i in range(n):
        bounds[in_nodes, 0] = xs[i]
        bounds[in_nodes, 1] = xs[i]
        
        sol = solve(mat, bounds)

        retries, cnt = 0, 2
        while not sol.success:
            if retries >= cnt:
                raise Exception(sol.message)
            sol = solve(mat, bounds)
            cnt += 1

        outputs[i] = sol.x[out_mask]

    return outputs
