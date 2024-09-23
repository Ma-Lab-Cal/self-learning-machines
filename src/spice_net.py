import os
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.BasicElement import SubCircuitElement
from PySpice.Spice.Parser import SpiceParser
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Unit import u_V
import numpy as np
import networkx as nx
import itertools


class AbstractNetwork(Circuit):
    def __init__(self, name: str, con_graph: nx.Graph, node_cfg, solver, epsilon=1e-10):
        self.name = name
        self.epsilon = epsilon
        super().__init__(name)
        self.__nodes__ = np.array([str(i) for i in range(con_graph.number_of_nodes())])
        self.inputs = [self.B(n + 1, *inds) for n, inds in enumerate(node_cfg[0])]
        self.outputs = [
            self.B(n + 1 + len(self.inputs), *inds)
            for n, inds in enumerate(node_cfg[1])
        ]

        self.solver = solver
        self.prev = None

        # "hack" - add "index" voltage source to allow us to pass indexed lists of inputs
        self.V("index", "index", 0, 1)

    def _prepare_simulation(self, inputs, outputs=None):
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
        inputs = np.transpose(inputs)
        if outputs is not None:
            outputs = np.transpose(outputs)

        # input validity checks
        assert len(inputs) == len(
            self.inputs
        ), f"Expected {len(self.inputs)} but got {len(inputs)} inputs"

        if outputs is not None:
            assert len(outputs.shape) == len(
                inputs.shape
            ), f"given {len(inputs.shape)}-dimensional input but {len(outputs.shape)}-dimensional output"

            assert len(outputs) == len(
                self.outputs
            ), f"Expected {len(self.outputs)} but got {len(outputs)} inputs"

        # single example (vector inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape((-1, 1))

            if outputs is not None:
                outputs = outputs.reshape((-1, 1))

        # multiple examples (matrix input)
        if len(inputs.shape) == 2:
            if outputs is not None:
                assert (
                    inputs.shape[1] == outputs.shape[1]
                ), f"Got {inputs.shape[1]} input examples but {outputs.shape[1]} output examples"

            n_examples = inputs.shape[1]

            if outputs is not None:
                source_iter = itertools.chain(
                    zip(self.inputs, inputs), zip(self.outputs, outputs)
                )
            else:
                source_iter = zip(self.inputs, inputs)

            for source, v in source_iter:
                source.enabled = True
                if n_examples > 1:
                    indexed_v = [
                        str(val)
                        for pair in zip(range(1, n_examples + 1), v)
                        for val in pair
                    ]
                    v_string = ", ".join(indexed_v)
                    values_expr = f"{{pwl(V(index), {v_string})}}"
                else:
                    values_expr = v[0]
                source.v = values_expr

            if outputs is None:
                for source in self.outputs:
                    source.enabled = False

        else:
            raise ValueError(
                f"input should be vector or 2d matrix, but got input of shape {inputs.shape}"
            )
        self.n_examples = n_examples

    def _run_simulation(self):
        simulator = self.simulator(ngspice_shared=self.solver)
        result = simulator.dc(Vindex=slice(1, self.n_examples, 1))

        return result

    def solve(self, inputs, outputs=None):
        self._prepare_simulation(inputs, outputs)
        result = self._run_simulation()

        # populate floating readings with zeros for downstream convenience
        return np.array(
            [
                u_V(
                    result.nodes.get(
                        str(i),
                        WaveForm.from_unit_values(
                            str(i), u_V(np.zeros(self.n_examples))
                        ),
                    )
                )
                for i in self.__nodes__
            ]
        )

    def predict(self, inputs):
        self._prepare_simulation(inputs)
        result = self._run_simulation()
        n_examples = len(result.nodes[str(self.__nodes__[0])])
        out = np.zeros((len(self.outputs), n_examples))

        for i, v in enumerate(self.outputs):
            a, b = v.node_names
            out[i] = u_V(result.nodes[a] - result.nodes[b])

        return out.T

    # def copy(self, name):
    #     copy = LinearNetwork(name, nx.Graph(), [[], []], self.epsilon)
    #     copy.__nodes__ = self.__nodes__.copy()

    #     copy.inputs = []
    #     for n, B in enumerate(self.inputs):
    #         inds = B.node_names
    #         copy.inputs.append(copy.B(n + 1, *inds))

    #     copy.outputs = []
    #     for n, B in enumerate(self.outputs):
    #         inds = B.node_names
    #         copy.outputs.append(copy.B(n + 1 + len(self.inputs), *inds))

    #     copy.edges = []
    #     for n, R in enumerate(self.edges):
    #         inds = R.node_names
    #         r = R.resistance
    #         copy.edges.append(copy.R(n + 1, *inds, r))

    #     return copy


# class Linear_edge(SubCircuit):
#     """Linear resistive edge parameterized by conductance"""
#     # TODO: there's really no reason to keep the code like this, 
#     # it's inefficient vs. just wrapping the returned component

#     __nodes__ = ("t_D", "t_S")

#     def __init__(self, name, circ, r_init, epsilon=1e-6):

#         SubCircuit.__init__(self, name, *self.__nodes__)

#         self.R(1, "t_D", "t_S", 1.0 / r_init)
#         self.epsilon = epsilon

#     def update(self, delta):
#         self.R1.resistance = 1.0 / max(1.0 / self.R1.resistance + delta, self.epsilon)

#     def get_val(self):
#         """Returns edge conductance"""
#         return 1.0 / self.R1.resistance


class Transistor_edge(SubCircuitFactory):
    """Custom edge class for PySpice. Used to create a parameterized subcircuit with a
     variable VGS parameter that can be implemented repeatedly."""
    NAME = "Transistor_edge"
    NODES = ("t_D", "t_S")

    def __init__(self, vgs=0.5, r_shunt=1e16, epsilon=1e-6):
        super().__init__(vgs=vgs)

        self.alpha = 1

        self.V(1, "t_G", "t_S", r"{vgs}")
        self.R(1, "t_D", "t_S", r_shunt)

        # assume models are already defined globally
        self.MOSFET(1, "t_D", "t_G", "t_S", "t_S", model="Ideal")    
    # def __repr__(self):
    #     return f"Transistor_edge({self.name}, vgs={self.V1.dc_value}, RShunt={self.R1.resistance})"
    
class WrappedTransistorEdge:
    # TODO: if this works, fix the screwed up names of everything
    def __init__(self, edge, alpha=1):
        self.alpha = 1
        self.edge = edge

    def update(self, delta):
        self.edge.parameters["vgs"] += self.alpha * delta

    def get_val(self):
        return self.edge.parameters["vgs"]


# class Scaled_Transistor_edge(SubCircuit):
#     __nodes__ = ("t_D", "t_S")

#     def __init__(self, name, circ, v_gs, r_shunt=0, epsilon=1e-6):

#         SubCircuit.__init__(self, name, *self.__nodes__)
#         # constant of porportionality represenitng slope of I-V curve
#         # intended to make transistor network update match linear network update
#         self.alpha = 1.0 / 1.7716667740993823e-05

#         self.V(1, "t_G", "t_S", self.alpha * v_gs)

#         self.R(1, "t_D", "dummy", r_shunt)
#         self.MOSFET(1, "dummy", "t_G", "t_S", "t_S", model="Ideal")

#         self.model("Ideal", "NMOS", level=1)

#     def update(self, delta):
#         self.V1.dc_value += self.alpha * delta

#     def get_val(self):
#         return self.V1.dc_value


# class Ground_reference_edge(SubCircuit):
#     __nodes__ = ("t_D", "t_S", "gnd")

#     def __init__(self, name, circ, v_gs, r_shunt=1e16, epsilon=1e-6):

#         SubCircuit.__init__(self, name, *self.__nodes__)
#         self.alpha = 1

#         self.V(1, "t_G", "gnd", self.alpha * v_gs)

#         self.R(1, "t_D", "t_S", r_shunt)
#         self.MOSFET(1, "t_D", "t_G", "t_S", "t_S", model="Ideal")

#         self.model("Ideal", "NMOS", level=1)

#     def update(self, delta):
#         self.V1.dc_value += self.alpha * delta

#     def get_val(self):
#         return self.V1.dc_value


class EdgeNetwork(AbstractNetwork):
    def __init__(
        self,
        name: str,
        edge_class,
        con_graph: nx.DiGraph,
        node_cfg,
        solver,
        epsilon=1e-9,
    ):
        resistor_net = con_graph.to_undirected(as_view=True)
        super().__init__(
            name=name,
            con_graph=resistor_net,
            node_cfg=node_cfg,
            solver=solver,
            epsilon=epsilon,
        )

        self.edges = []

        nodes_map = {n: i for i, n in enumerate(con_graph.nodes())}
        edge_subckt = edge_class(epsilon=epsilon)
        self.subcircuit(edge_subckt)
        for n, (u, v, r) in enumerate(con_graph.edges(data="weight")):
            edge = Transistor_edge(self.X(n + 1, edge_subckt.name, nodes_map[u], nodes_map[v], vgs=r))
            self.edges.append(edge)

    def update(self, updates):
        for edge, delta in zip(self.edges, updates):
            edge.update(delta)


class LinearNetwork(EdgeNetwork):
    pass
#     def __init__(self, name: str, con_graph: nx.DiGraph, node_cfg, epsilon=1e-9):
#         super().__init__(name, Linear_edge, con_graph, node_cfg, epsilon)


class TransistorNetwork(EdgeNetwork):
    def __init__(
        self, name: str, con_graph: nx.DiGraph, node_cfg, solver, epsilon=1e-9
    ):
        super().__init__(
            name=name,
            edge_class=Transistor_edge,
            con_graph=con_graph,
            node_cfg=node_cfg,
            solver=solver,
            epsilon=epsilon,
        )
        self.model("Ideal", "NMOS", level=1)

# class ScaledTransistorNetwork(EdgeNetwork):
#     def __init__(self, name: str, con_graph: nx.DiGraph, node_cfg, epsilon=1e-9):
#         super().__init__(name, Scaled_Transistor_edge, con_graph, node_cfg, epsilon)


# class GroundReferenceNetwork(AbstractNetwork):
#     def __init__(self, name: str, con_graph: nx.DiGraph, node_cfg, epsilon=1e-9):
#         super().__init__(name, con_graph, node_cfg, epsilon)

#         self.edges = []

#         nodes_map = {n: i for i, n in enumerate(con_graph.nodes())}
#         for n, (u, v, r) in enumerate(con_graph.edges(data="weight")):
#             edge = Ground_reference_edge(f"e{n+1}", self, r, epsilon=epsilon)
#             self.subcircuit(edge)
#             self.edges.append(edge)
#             edge.circ = self.X(n + 1, f"e{n+1}", nodes_map[u], nodes_map[v], 0)

#     def update(self, updates):
#         for edge, delta in zip(self.edges, updates):
#             edge.update(delta)
