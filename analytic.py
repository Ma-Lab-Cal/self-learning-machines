import networkx as nx
import numpy as np
from spice_net import *
from helpers import *
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import matplotlib.pyplot as plt
import torch.nn as nn

def gen_A(net: LinearNetwork, all_sources):
    '''
    Decomposes a LinearNetwork into the A matrix. 

    parameters:
        net:LinearNetwork = Network to simulate
        all_sources:List = list of source node indices in the form (+, -)

    returns:
        A:np.array = A matrix that describes the LinearNetwork using the MNA
                framework.
    '''
    n = len(net.__nodes__[1:])
    # all_sources = np.concatenate((net.inputs, net.outputs))
    m = len(all_sources)

    # ideally G should be all 0, but add tiny resistance from each
    # node to ground for solver stability
    G = np.eye(n+1) * net.epsilon
    B = np.zeros((n, m))
    C = B.T
    D = np.zeros((m, m))

    # populate G with resistor values
    for R in net.edges:
        i, j = list(map(lambda i: int(i), R.node_names))
        if min(i, j) >= 0:
            G[i, j] = G[j, i] = -1/R.resistance

            G[i, i] += 1/R.resistance
            G[j, j] += 1/R.resistance

    for k, V in enumerate(all_sources):
        i, j = list(map(lambda i: int(i)-1, V.node_names))
        if i >= 0:
            B[i, k] = 1
        if j >= 0:
            B[j, k] = -1

    return np.block([[G[1:, 1:], B], [C, D]])

def decomp_A(net: LinearNetwork, all_sources):
    '''
    Decomposes a LinearNetwork into three components that, when combined,
        results in the A matrix. 

    parameters:
        net:LinearNetwork = Network to simulate
        all_sources:List = list of source node indices in the form (+, -)

    returns:
        r:Torch.Variable = Tunable parameter representing individual admittances. 
                Initialized to all ones. Dimension (r)
        M:Torch.Tensor = Component of A that depends on r values. 
        constant_part:Torch.Tensor = Component of A that does not depend on r values. 
                Dimension (m+n, m+n)
    '''

    # for stability with Numpy arrays
    torch.set_default_dtype(torch.double)

    n = len(net.__nodes__[1:])
    m = len(all_sources)

    B = np.zeros((n, m))
    C = B.T
    D = np.zeros((m, m))
    
    # populate G with resistor values
    M = []
    r = torch.ones(len(net.edges), dtype=torch.double)
    for k, R in enumerate(net.edges):
        i, j = list(map(lambda i: int(i), R.node_names))
        mat = np.zeros((m+n+1, m+n+1), dtype=np.double)
        
        r[k] = 1/R.resistance

        mat[i, j] = mat[j, i] = -1
        mat[i, i] = mat[j, j] = 1
        
        M.append(mat[1:, 1:])
    
    r = Variable(r, requires_grad=True)
    M = torch.DoubleTensor(np.array(M))

    for k, V in enumerate(all_sources):
        i, j = list(map(lambda i: int(i)-1, V.node_names))
        if i >= 0:
            B[i, k] = 1
        if j >= 0:
            B[j, k] = -1

    # ideally should be all zeros, tweak as needed for solver feasability
    # (e.g shunt to ground)
    G_constant = np.eye(n) * net.epsilon
    constant_part = torch.DoubleTensor(np.block([[G_constant, B], [C, D]]))
    return r, M, constant_part

def make_downsampler(net):
    nodes = len(net.__nodes__)
    in_nodes = len(net.inputs)
    out_nodes = len(net.outputs)

    downsampler = np.zeros((out_nodes, nodes + in_nodes - 1))

    for k, V in enumerate(net.outputs):
        i, j = list(map(lambda i: int(i)-1, V.node_names))
        if i >= 0:
            downsampler[k, i] = 1
        if j >= 0:
            downsampler[k, j] = -1
    
    return downsampler

def solve(A, e):
    m = len(e)
    n = len(A) - m

    e = np.reshape(e, (m, -1))
    z = np.concatenate((np.zeros((n, e.shape[1])), e))
    return np.linalg.solve(A, z)

def pad_input(A, e):
    num_pts, m = e.shape # number of sources
    n = len(A) - m # number of nodes

    e = torch.DoubleTensor(e)
    e = e.reshape(m, -1) # FIXME: define valid input and output forms, and robustly handle those cases. 
                                    # maybe check Torch and sklearn for inspiration 

    z = torch.vstack((torch.zeros((n, e.shape[1])), e))
    return z

def solve_torch(A, e):
    z = pad_input(A, e)
    return torch.linalg.solve(A, z)

# def contrastive_update(net, e1, e2, eta, downsampler, x, y):
#     r, M, constant_part = decomp_A(net, net.inputs)
#     A_free = constant_part + np.einsum('a...,a->...', M, r)

#     r, M, constant_part = decomp_A(net, np.concatenate([net.inputs, net.outputs]))
#     A_clamp = constant_part + np.einsum('a...,a->...', M, r)
#     free = solve(A, x)
#     nudges = eta * y + (1-eta) * downsampler @ free
#     clamped = solve(x, nudges.reshape(y.shape))

#     free_rep = np.tile(free, [n_nodes, 1])
#     clamped_rep = np.tile(clamped, [n_nodes,1])

#     delta_free = free_rep - free_rep.T
#     delta_clamped = clamped_rep - clamped_rep.T

#     update = -gamma * (delta_clamped**2 - delta_free**2)
#     # trainable_updates = np.empty(n_edges)

#     trainable_updates = update[e1, e2] #/ (R.resistance**2)

def analytic_solve(net: LinearNetwork, inputs, targets, optimizer, iters=100):
    # for numerical stability
    torch.set_default_dtype(torch.double)

    # construct the model and circuit matrix
    n = len(net.__nodes__[1:])
    m = len(net.inputs)
    
    r, M, constant_part = decomp_A(net, inputs)

    # prepare the data matrices
    n_examples = inputs.shape[0]
    output_dim = targets.shape[1]

    inputs = inputs.T
    targets = targets.T

    downsampler = torch.DoubleTensor(make_downsampler(net))
    
    X = torch.DoubleTensor(np.concatenate((np.zeros((n, n_examples)), inputs)))
    targets = torch.DoubleTensor(targets)

    mse = MSELoss()

    # do the optimization
    optim = optimizer([r], lr=0.01)
    alpha = 1e-5

    losses = []

    for i in range(iters):
        A = constant_part + torch.einsum('a...,a->...', M, r)
        preds = downsampler @ torch.linalg.solve(A, X)

        loss = mse(preds, targets) + torch.sum(torch.maximum(-alpha*r, torch.zeros(r.shape)))
        losses.append(loss.detach())
        optim.zero_grad()
        loss.backward()
        optim.step()

        alpha = min(10*alpha, 1e5)

    return r.detach(), A.detach(), downsampler, losses

class Analytic_net(nn.Module):
    def __init__(self, net: LinearNetwork, sources):
        super().__init__()
        self.r, self.M, self.constant_part = decomp_A(net, sources)
        self.downsampler = make_downsampler(net)
        
        self.n = len(net.__nodes__) - 1 # subtract 1 because ground node is implicit in MNA
        self.m = len(sources)

        assert len(self.M[0]) == self.n + self.m, [self.M.shape, self.n, self.m]

    def forward(self, x):
        # assert x.shape[0] == self.m, f'Expected {self.m} inputs but got {x.shape[0]}'

        A = self.constant_part + torch.einsum('a...,a->...', self.M, self.r)
        return solve_torch(A, x)

class Contrastive_net(nn.Module):
    '''
    A wrapper of the analytic network to allow for the contrastive update rule.
    Implicitly assumes that we will use net.inputs as our inputs and net.outputs
    as our outputs and clamps, as done in the reference paper.
    '''
    def __init__(self, net:LinearNetwork, eta, alpha):
        super().__init__()
        self.free_net = Analytic_net(net, net.inputs)
        self.clamp_net = Analytic_net(net, np.concatenate([net.inputs, net.outputs]))

        self.eta = eta
        self.alpha = alpha
        self.net = net

        self.e1, self.e2 = [], []
        for R in net.edges:
            a, b = list(map(int, R.node_names))
            self.e1.append(a)
            self.e2.append(b)

    def forward(self, x):
        self.cache = x
        return self.free_net(x)
    
    def contrastive_update(self, y):
        y = torch.DoubleTensor(y)
        x = self.cache

        free_out = self.free_net(x)
        nudges = self.eta * y + (1-self.eta) * (torch.DoubleTensor(self.free_net.downsampler) @ free_out.detach())
        clamp_out = self.clamp_net(torch.cat([x, nudges.reshape(y.shape)]))

        free_rep = torch.tile(free_out, [len(self.net.__nodes__), 1])
        clamp_rep = torch.tile(clamp_out, [len(self.net.__nodes__), 1])

        delta_free = free_rep - free_rep.T
        delta_clamp = clamp_rep - clamp_rep.T

        full_update = delta_clamp**2 - delta_free**2
        param_update = full_update[self.e1, self.e2]

        self.free_net.r -= self.alpha * 0.5 / self.eta * param_update
        self.clamp_net.r -= self.alpha * 0.5 / self.eta * param_update

        return param_update

        # TRANSLATED CODE FROM ORIGINAL CONTRASTIVE UPDATE (ABOVE SHOULD MATCH THIS)
        # free = net.solve(x)
        # nudges = eta * y + (1-eta) * net.predict(x)
        # clamped = net.solve(x, nudges.reshape(y.shape))

        # free_rep = np.tile(free, [n_nodes, 1])
        # clamped_rep = np.tile(clamped, [n_nodes,1])

        # delta_free = free_rep - free_rep.T
        # delta_clamped = clamped_rep - clamped_rep.T

        # update = -gamma * (delta_clamped**2 - delta_free**2)
        # # trainable_updates = np.empty(n_edges)

        # trainable_updates = update[e1, e2] #/ (R.resistance**2)

        # net.update_y(trainable_updates)