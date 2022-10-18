from multiprocessing.sharedctypes import Value
from PySpice.Spice.Netlist import Circuit, SubCircuit
import ltspice
import numpy as np
from os import system

class LinearNetwork(Circuit):
    def __init__(self, name, con_mat, node_cfg, epsilon=1e-9):
        self.name = name
        super().__init__(name)
        self.__nodes__ = np.array([i for i in range(con_mat.shape[0])])
        self.con_mat = con_mat
        self.node_cfg = node_cfg
        res_mat = np.minimum(np.triu(con_mat, k=1), con_mat.T) > 0
        self.epsilon = epsilon

        for i, coord in enumerate(np.argwhere(res_mat > 0)):
            self.R(i, self.__nodes__[coord[0]], self.__nodes__[coord[1]], 1/con_mat[coord[0], coord[1]])

    def update(self, updates):
        for i, n in enumerate(updates):
            self[f'R{i}'].resistance = max(1./(1./self[f'R{i}'].resistance + n), self.epsilon)

    def solve(self, inputs, output_clamps = None):
        circ = self
        in_nodes, all_nodes= np.sum(self.node_cfg > 0), np.sum(self.node_cfg != 0)
        if output_clamps is not None:
            inputs = np.hstack((inputs, output_clamps))
        if inputs.size == in_nodes:
            node_inds = np.argwhere(self.node_cfg > 0).flatten()
        elif inputs.size == all_nodes:
            node_inds = np.concatenate((np.argwhere(self.node_cfg > 0).flatten(), np.argwhere(self.node_cfg < 0).flatten()))
        else:
            raise ValueError(f'Expected {all_nodes if output_clamps is not None else in_nodes} nodes but got {inputs.size}')
        for v, ind in zip(inputs.flatten(), node_inds):
            circ.V(ind, circ.__nodes__[ind], circ.gnd, v)
        netlist = str(circ)
        with open(f'{self.name}.cir', 'wt') as f:
            f.write(netlist + '.op\n')
        system(f'/Applications/LTspice.app/Contents/MacOS/LTspice -b {self.name}.cir')
        for i in node_inds:
            circ[f'V{i}'].detach()

        l = ltspice.Ltspice(f'{self.name}.raw')
        l.parse()
        return np.concatenate([l.get_data(f'V({i})') or [0] for i in self.__nodes__])

    def predict(self, inputs):
        return self.solve(inputs)[self.node_cfg < 0]

class NonLinearNetwork(LinearNetwork):
    def __init__(self, name, con_mat, node_cfg):
        super().__init__(name, con_mat, node_cfg)

    def solve(self, inputs):
        assert inputs.size == np.sum(self.node_cfg > 0), 'invalid input size'
        circ = self.clone()
        for v, ind in zip(inputs.flatten(), np.argwhere(self.node_cfg > 0).flatten()):
            circ.V(ind, circ.__nodes__[ind], circuit.gnd, v)
        netlist = str(circ)
        with open(f'{self.name}.cir', 'wt') as f:
            f.write(netlist + '.op\n')
        system(f'/Applications/LTspice.app/Contents/MacOS/LTspice -b {self.name}.cir')