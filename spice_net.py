from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
import ltspice
import numpy as np
from os import system
import networkx as nx

class LinearNetwork(Circuit):
    def __init__(self, name: str, con_graph: nx.Graph, node_cfg, epsilon=1e-9):
        self.name = name
        self.epsilon = epsilon
        super().__init__(name)
        self.__nodes__ = np.array([i for i in range(con_graph.number_of_nodes())])
        self.inputs = [self.V(n+1, *inds) \
         for n, inds in enumerate(node_cfg[0])]
        self.outputs = [self.V(n+1 + len(self.inputs), *inds) \
         for n, inds in enumerate(node_cfg[1])]
        for source in self.inputs:
            source.enabled = False
        for source in self.outputs:
            source.enabled = False

        self.edges = [self.R(n+1, u, v, r) \
            for n, (u, v, r) in enumerate(con_graph.edges(data='weight'))]

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
            # R.resistance = 1./(1./R.resistance + delta)

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
            # f.write(netlist + '.tran 1\n')
        system(f'/Applications/LTspice.app/Contents/MacOS/LTspice -b {self.name}.cir')
        
        l = ltspice.Ltspice(f'{self.name}.raw')
        l.parse()
        return l

    def solve(self, inputs, outputs = None):
        l = self._solve(inputs, outputs)
        return np.array([0 if l.get_data(f'V({i})') is None\
            else l.get_data(f'V({i})')[-1] for i in self.__nodes__])

    def predict(self, inputs):
        l = self._solve(inputs)
        out = np.zeros(len(self.outputs))
        for i, v in enumerate(self.outputs):
            a, b = v.node_names
            v1 = l.get_data(f'V({a})')
            v2 = l.get_data(f'V({b})')
            if v1 is None: v1 = [0]
            if v2 is None: v2 = [0]
            out[i] = v1[0] - v2[0]
        return out

    def copy(self, name):
        copy = LinearNetwork(name, nx.Graph(), [[], []], self.epsilon)
        copy.__nodes__ = self.__nodes__.copy()

        copy.inputs = []
        for n, V in enumerate(self.inputs):
            inds = V.node_names
            copy.inputs.append(copy.V(n+1, *inds))

        copy.outputs = []
        for n, V in enumerate(self.outputs):
            inds = V.node_names
            copy.outputs.append(copy.V(n+1 + len(self.inputs), *inds))

        for source in copy.inputs:
            source.enabled = False
        for source in copy.outputs:
            source.enabled = False

        copy.edges = []
        for n, R in enumerate(self.edges):
            inds = R.node_names
            r = R.resistance
            copy.edges.append(copy.R(n+1, *inds, r))

        return copy

class ReLu_edge(SubCircuit):
    __nodes__ = ('t_in', 't_out')
    def __init__(self, name, r):

        SubCircuit.__init__(self, name, *self.__nodes__)

        self.R(1, 't_in', 'dummy', r)
        self.D(1, 'dummy', 't_out', model='ReLu')

class NonLinearNetwork(LinearNetwork):
    def __init__(self, name: str, con_graph: nx.DiGraph, node_cfg, epsilon=1e-9):

        res_filter = lambda u, v: con_graph[u][v]['type'] == 'resistor'
        diode_filter = lambda u, v: con_graph[u][v]['type'] == 'diode'
        resistor_net = nx.subgraph_view(con_graph, filter_edge=res_filter).to_undirected(as_view=True)
        super().__init__(name, resistor_net, node_cfg, epsilon)

        # self.diodes = [self.D(n+1, u if d == v else v, d, model='ReLu')\

        self.model('ReLu', 'D', Ron=0, Roff=1@u_GOhm, Vfwd=0, Vrev=10, Epsilon=0.001)
        self.diodes = []
        self.nonlinear_vals = []

        for n, (u, v, r) in enumerate(nx.subgraph_view(con_graph, filter_edge=diode_filter).edges(data='weight')):
            edge = ReLu_edge(f'e{n+1}', r)
            self.nonlinear_vals.append(edge)
            self.subcircuit(edge)
            self.diodes.append(self.X(n+1, f'e{n+1}', u, v))

    def update_r(self, updates):
        '''updates internal resistances given a list of resistance deltas'''
        assert len(self.edges) + len(self.diodes) == len(updates), \
            f'Have {len(self.edges) + len(self.diodes)} resistors but {len(updates)} updates'
        u_e = updates[:len(self.edges)]
        u_d = updates[len(self.edges):]
        for R, delta in zip(self.edges, u_e):
            R.resistance = min(R.resistance - delta, self.epsilon)
        for X, delta in zip(self.nonlinear_vals, u_d):
            X.R1.resistance = min(X.R1.resistance - delta, self.epsilon)

    def update_y(self, updates):
        '''updates internal resistances given a list of admittance deltas'''
        assert len(self.edges) + len(self.diodes) == len(updates), \
            f'Have {len(self.edges) + len(self.diodes)} resistors but {len(updates)} updates'
        u_e = updates[:len(self.edges)]
        u_d = updates[len(self.edges):]
        for R, delta in zip(self.edges, u_e):
            R.resistance = 1./max(1./R.resistance + delta, self.epsilon)
        for X, delta in zip(self.nonlinear_vals, u_d):
            X.R1.resistance = 1./max(1./X.R1.resistance + delta, self.epsilon)

    def copy(self, name):
        copy = NonLinearNetwork(name, nx.Graph(), [[], []], self.epsilon)

        copy.__nodes__ = self.__nodes__.copy()

        copy.inputs = []
        for n, V in enumerate(self.inputs):
            inds = V.node_names
            copy.inputs.append(copy.V(n+1, *inds))

        copy.outputs = []
        for n, V in enumerate(self.outputs):
            inds = V.node_names
            copy.outputs.append(copy.V(n+1 + len(self.inputs), *inds))

        for source in copy.inputs:
            source.enabled = False
        for source in copy.outputs:
            source.enabled = False

        copy.edges = []
        for n, R in enumerate(self.edges):
            inds = R.node_names
            r = R.resistance
            copy.edges.append(copy.R(n+1, *inds, r))

        copy.nonlinear_vals = []
        copy.diodes = []
        for n, (D, X) in enumerate(zip(self.nonlinear_vals, self.diodes)):
            r = D.R1.resistance
            edge = ReLu_edge(f'e{n+1}', r)
            copy.nonlinear_vals.append(edge)
            copy.subcircuit(edge)
            inds = X.node_names
            copy.diodes.append(copy.X(n+1, f'e{n+1}', *inds))

        return copy