from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Unit import *
import numpy as np
import networkx as nx

class LinearNetwork(Circuit):
    def __init__(self, name: str, con_graph: nx.Graph, node_cfg, epsilon=1e-9):
        self.name = name
        self.epsilon = epsilon
        super().__init__(name)
        self.__nodes__ = np.array([str(i) for i in range(con_graph.number_of_nodes())])
        self.inputs = [self.B(n+1, *inds) \
         for n, inds in enumerate(node_cfg[0])]
        self.outputs = [self.B(n+1 + len(self.inputs), *inds) \
         for n, inds in enumerate(node_cfg[1])]

        # "hack" - add "index" voltage source to allow us to pass indexed lists of inputs
        self.V('index', 'index', 0, 1)

        self.edges = [self.R(n+1, u, v, r) \
            for n, (u, v, r) in enumerate(con_graph.edges(data='weight'))]

    def update_r(self, updates):
        '''updates internal resistances given a list of resistance deltas'''
        assert len(self.edges) == len(updates), \
            f'Have {len(self.edges)} resistors but {len(updates)} updates'
        for R, delta in zip(self.edges, updates):
            R.resistance = max(R.resistance - delta, self.epsilon)

    def update_y(self, updates):
        '''updates internal resistances given a list of admittance deltas'''
        assert len(self.edges) == len(updates), \
            f'Have {len(self.edges)} resistors but {len(updates)} updates'
        for R, delta in zip(self.edges, updates):
            R.resistance = 1./max(1./R.resistance + delta, self.epsilon)
            # R.resistance = 1./(1./R.resistance + delta)

    def _solve(self, inputs, outputs = None):
        """Solves for all node voltages given differential input voltages
        and optional differential output voltages. No post-processing of outputs

        Args:
            inputs: Input differential voltages. Should either be a vector with length
                equal to number of inputs, or a 2d matrix with shape (num_inputs, num_examples)
            outputs (optional): Optional output differential voltages. Should be a 
                vector with length equal to number of outputs, a 2d matrix with shape 
                (num_outputs, num_examples), or None. 

        Returns:
            _type_: _description_
        """
        # transpose inputs and outputs for simplicity of operations
        inputs = np.array(inputs).T
        if outputs is not None:
            outputs = np.array(outputs).T

        # input validity checks
        assert len(inputs) == len(self.inputs), \
            f'Expected {len(self.inputs)} but got {len(inputs)} inputs'

        if outputs is not None:
            assert len(outputs.shape) == len(inputs.shape), \
                f'given {len(inputs.shape)}-dimensional input but {len(outputs.shape)}-dimensional output'

            assert len(outputs) == len(self.outputs), \
                f'Expected {len(self.outputs)} but got {len(outputs)} inputs'

        # single example (vector inputs)
        if len(inputs.shape) == 1:
            n_examples = 1

            # pass values into model for simulation
            for source, V in zip(self.inputs, inputs):
                source.enabled = True
                source.v = V

            if outputs is not None:
                for source, V in zip(self.outputs, outputs):
                    source.enabled = True
                    source.v = V
            else:
                for source in self.outputs:
                    source.enabled = False

        # multiple examples (matrix input)
        elif len(inputs.shape) == 2:
            if outputs is not None:
                assert inputs.shape[1] == outputs.shape[1], \
                    f'Got {inputs.shape[1]} input examples but {outputs.shape[1]} output examples'

            n_examples = inputs.shape[1]

            for source, V in zip(self.inputs, inputs):
                source.enabled = True
                indexed_V = [str(val) for pair in zip(range(1, n_examples+1), V) for val in pair]
                V_string = ', '.join(indexed_V)
                values_expr = f'pwl(V(index), {V_string})'
                source.v = values_expr

            if outputs is not None:
                for source, V in zip(self.outputs, outputs):
                    source.enabled = True
                    indexed_V = [str(val) for pair in zip(range(1, n_examples+1), V) for val in pair]
                    V_string = ', '.join(indexed_V)
                    values_expr = f'pwl(V(index), {V_string})'
                    source.v = values_expr

            else:
                for source in self.outputs:
                    source.enabled = False

        else:
            raise ValueError(f'input should be vector or 2d matrix, but got input of shape {inputs.shape}')

        simulator = self.simulator()
        analysis = simulator.dc(Vindex=slice(1, n_examples, 1))

        # populate ground reading with zeros for downstream convenience 
        analysis.nodes['0'] = WaveForm.from_unit_values('0', u_V(np.zeros(n_examples)))

        return analysis


    def solve(self, inputs, outputs = None):
        analysis = self._solve(inputs, outputs)
        return np.array([u_V(analysis.nodes[i]) for i in self.__nodes__])

    def predict(self, inputs):
        analysis = self._solve(inputs)
        n_examples = len(analysis.nodes[self.__nodes__[0]])
        out = np.zeros((len(self.outputs), n_examples))

        for i, v in enumerate(self.outputs):
            a, b = v.node_names
            out[i] = u_V(analysis.nodes[a] - analysis.nodes[b])

        return out.T

    def copy(self, name):
        copy = LinearNetwork(name, nx.Graph(), [[], []], self.epsilon)
        copy.__nodes__ = self.__nodes__.copy()

        copy.inputs = []
        for n, B in enumerate(self.inputs):
            inds = B.node_names
            copy.inputs.append(copy.B(n+1, *inds))

        copy.outputs = []
        for n, B in enumerate(self.outputs):
            inds = B.node_names
            copy.outputs.append(copy.B(n+1 + len(self.inputs), *inds))

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

        self.model('ReLu', 'D', js=0, n=0.001)
        # self.model('ReLu', 'sw', ron=0, roff=1@u_GOhm, vt=0, vh=0.001)
        # .model diosw sw vt=0 vh=0.001 ron=0.001 roff=10e9

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
            R.resistance = max(R.resistance - delta, self.epsilon)
        for X, delta in zip(self.nonlinear_vals, u_d):
            X.R1.resistance = max(X.R1.resistance - delta, self.epsilon)

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
        for n, B in enumerate(self.inputs):
            inds = B.node_names
            copy.inputs.append(copy.B(n+1, *inds))

        copy.outputs = []
        for n, B in enumerate(self.outputs):
            inds = B.node_names
            copy.outputs.append(copy.B(n+1 + len(self.inputs), *inds))

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