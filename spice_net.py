from multiprocessing.sharedctypes import Value
from PySpice.Spice.Netlist import Circuit, SubCircuit
import ltspice
import numpy as np
from os import system

class LinearNetwork(Circuit):
    def __init__(self, name, con_mat, node_cfg, epsilon=1e-9):
        assert (con_mat == con_mat.T).all(), 'connectivity matrix is not symmetric!'
        self.name = name
        self.epsilon = epsilon
        super().__init__(name)
        self.__nodes__ = np.array([i for i in range(con_mat.shape[0])])
        self.inputs = [self.V(n+1, *inds) \
         for n, inds in enumerate(node_cfg[0])]
        self.outputs = [self.V(n+1 + len(self.inputs), *inds) \
         for n, inds in enumerate(node_cfg[1])]
        for source in self.inputs:
            source.enabled = False
        for source in self.outputs:
            source.enabled = False

        res_mat = np.minimum(np.triu(con_mat, k=1), con_mat.T) > 0
        self.edges = [self.R(n+1, *coord, con_mat[coord[0], coord[1]]) \
        for n, coord in enumerate(np.argwhere(res_mat > 0))]

    def update_r(self, updates):
        '''updates internal resistances given a list of resistance deltas'''
        assert len(self.edges) == len(updates), \
            f'Have {len(self.edges)} resistors but {len(updates)} updates'
        for R, delta in zip(self.edges, updates):
            R.resistance = min(R.resistance - delta, self.epsilon)

    def update_y(self, updates):
        '''updates internal resistances given a list of admittance deltas'''
        assert len(self.edges) == len(updates), \
            f'Have {len(self.edges)} resistors but {len(updates)} updates'
        for R, delta in zip(self.edges, updates):
            R.resistance = 1./max(1./R.resistance + delta, self.epsilon)

    def _solve(self, inputs, outputs = None):
        circ = self
        assert len(inputs) == len(self.inputs), \
            f'Expected {len(self.inputs)} but got {len(inputs)} inputs'

        for source, V in zip(self.inputs, inputs):
            source.enabled = True
            source.dc_value = V
        
        if outputs is not None:
            assert len(outputs) == len(self.outputs), \
                f'Expected {len(self.outputs)} but got {len(outputs)} inputs'

            for source, V in zip(self.outputs, outputs):
                source.enabled = True
                source.dc_value = V

        else:
            for source in self.outputs:
                source.enabled = False

        netlist = str(circ)
        with open(f'{self.name}.cir', 'wt') as f:
            f.write(netlist + '.op\n')
        system(f'/Applications/LTspice.app/Contents/MacOS/LTspice -b {self.name}.cir')
        
        l = ltspice.Ltspice(f'{self.name}.raw')
        l.parse()
        return l

    def solve(self, inputs, outputs = None):
        l = self._solve(inputs, outputs)
        return np.concatenate([l.get_data(f'V({i})') or [0] for i in self.__nodes__])

    def predict(self, inputs):
        l = self._solve(inputs)
        out = np.zeros(len(self.outputs))
        for i, v in enumerate(self.outputs):
            a, b = v.node_names
            v1 = l.get_data(f'V({a})') or [0]
            v2 = l.get_data(f'V({b})') or [0]
            out[i] = v1[0] - v2[0]
        return out

class NonLinearNetwork(LinearNetwork):
    def __init__(self, name, con_mat, node_cfg):
        super().__init__(name, con_mat, node_cfg)
        # TODO: different constructor which allows for nonlinear components
        # everything else should be the same as in the linear case