import os
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.BasicElement import SubCircuitElement
from PySpice.Spice.Parser import SpiceParser
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Unit import *
import numpy as np
import networkx as nx
import itertools


class AbstractNetwork(Circuit):
    def __init__(self, name: str, con_graph: nx.Graph, node_cfg, solver, epsilon=1e-10):
        self.name = name
        self.epsilon = epsilon
        super().__init__(name)
        self.__nodes__ = [str(i) for i in range(con_graph.number_of_nodes())]
        self.inputs = [self.B(n + 1, *inds) for n, inds in enumerate(node_cfg[0])]
        self.outputs = [
            self.B(n + 1 + len(self.inputs), *inds)
            for n, inds in enumerate(node_cfg[1])
        ]

        self.solver = solver
        self.prev = None

        # "hack" - add "index" voltage source to allow us to pass indexed lists of inputs
        self.V("index", "index", 0, 1)
        self.cached_simulator = self.simulator(simulator=self.solver) # TODO: need to fix if allowing Xyce again

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
        # simulator = self.simulator(ngspice_shared=self.solver)
        # result = simulator.dc(Vindex=slice(1, self.n_examples, 1))
        result = self.cached_simulator.dc(Vindex=slice(1, self.n_examples, 1))
        # result = self.cached_simulator.operating_point()

        self.prev = result

        return result

    def solve(self, inputs, outputs=None):
        self._prepare_simulation(inputs, outputs)

        # init_conds = {}
        # for node in self.__nodes__:
            

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

class LinearEdge(SubCircuitFactory):
    """Custom edge class for PySpice. Used to create a parameterized subcircuit with a
     variable R parameter that can be implemented repeatedly."""
    NAME = "linear_edge"
    NODES = ("t_D", "t_S")

    def __init__(self, r=0.5, epsilon=1e-6):
        super().__init__(r=r)

        self.alpha = 1

        self.R(1, "t_D", "t_S", r"{r}")  

class TransistorEdge(SubCircuitFactory):
    """Custom edge class for PySpice. Used to create a parameterized subcircuit with a
     variable VGS parameter that can be implemented repeatedly."""
    NAME = "transistor_edge"
    NODES = ("t_D", "t_S")

    def __init__(self, vgs=0.5, r_shunt=1e16, epsilon=1e-6):
        super().__init__(vgs=vgs)

        self.alpha = 1

        self.V(1, "t_G", "t_S", "{vgs}")
        self.R(1, "t_D", "t_S", r_shunt)

        # assume models are already defined globally
        self.MOSFET(1, "t_D", "t_G", "t_S", "t_S", model="Ideal")    
    # def __repr__(self):
    #     return f"Transistor_edge({self.name}, vgs={self.V1.dc_value}, RShunt={self.R1.resistance})"

class TransistorEdgeTeacher(SubCircuitFactory):
    """Custom edge class for PySpice. Used to create a parameterized subcircuit with a
     variable VGS parameter that can be implemented repeatedly."""
    NAME = "transistor_edge_teacher"
    NODES = ("t_D", "t_S", "VGS")

    def __init__(self, r_shunt=1e16, epsilon=1e-6):
        super().__init__()

        self.alpha = 1

        self.B(1, "t_G", "t_S", voltage_expression="V(VGS)")
        self.R(1, "t_D", "t_S", r_shunt)

        # assume models are already defined globally
        self.MOSFET(1, "t_D", "t_G", "t_S", "t_S", model="Ideal")    
    # def __repr__(self):
    #     return f"Transistor_edge({self.name}, vgs={self.V1.dc_value}, RShunt={self.R1.resistance})"

class Teacher(SubCircuitFactory):
    """'Teacher' that takes in a clock, and the D, G, S nodes for two transistor edges and updates
    VGS values according to the contrastive update rule"""
    NAME = "teacher"
    NODES = ("D_FREE", "S_FREE", "D_CLAMPED", "S_CLAMPED", "VGS", "CLK")

    def __init__(self, c_learn=u_uF(22)):
        # subckt will not be parameterized. All parameters are either hardcoded or defined 
        # globally since ideally all teachers have the same parameters
        super().__init__()

        # self.B('UPDATE', "nudge", 0, voltage_expression="V(VGS)+((V(S_FREE)-V(D_FREE))**2-(V(S_CLAMPED)-V(D_CLAMPED))**2)")
        # replace 
        self.B('UPDATE', "nudge", 0, current_expression=f"((V(S_FREE)-V(D_FREE))**2-(V(S_CLAMPED)-V(D_CLAMPED))**2)")
        self.S(1, "nudge", "VGS", "CLK", 0, model="MYSW", initial_state="on")
        self.C(1, "VGS", 0, c_learn)

        # control edge VGS value
        # self.V(1, "G_FREE", "S_FREE", "V(VGS)")
        # self.V(2, "G_CLAMPED", "S_CLAMPED", "V(VGS)")
    
class WrappedTeacherEdge:
    class_subckt = TransistorEdgeTeacher

    # TODO: if this works, fix the screwed up names of everything
    def __init__(self, edge, alpha=1):
        self.alpha = 1
        self.edge = edge

    def update(self, delta):
        raise NotImplementedError("Teacher edge should not be updated manually!")

    def get_val(self):
        raise NotImplementedError("Teacher edge VGS must be read from simulation results!")
    
class WrappedTransistorEdge:
    class_subckt = TransistorEdge

    # TODO: if this works, fix the screwed up names of everything
    def __init__(self, edge, alpha=1):
        self.alpha = 1
        self.edge = edge

    def update(self, delta):
        self.edge.parameters["vgs"] += self.alpha * delta

    def get_val(self):
        return self.edge.parameters["vgs"]
    
class WrappedLinearEdge:
    class_subckt = LinearEdge

    # TODO: if this works, fix the screwed up names of everything
    def __init__(self, edge, alpha=1):
        self.alpha = 1
        self.edge = edge

    def update(self, delta):
        self.edge.parameters["r"] += self.alpha * delta

    def get_val(self):
        return self.edge.parameters["r"]

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
        edge_subckt = edge_class.class_subckt(epsilon=epsilon)
        self.subcircuit(edge_subckt)
        for n, (u, v, r) in enumerate(con_graph.edges(data="weight")):
            edge = edge_class(self.X(n + 1, edge_subckt.name, nodes_map[u], nodes_map[v], vgs=r))
            self.edges.append(edge)

    def update(self, updates):
        for edge, delta in zip(self.edges, updates):
            edge.update(delta)


class LinearNetwork(EdgeNetwork):
    def __init__(
        self, name: str, con_graph: nx.DiGraph, node_cfg, solver, epsilon=1e-9
    ):
        super().__init__(
            name=name,
            edge_class=WrappedLinearEdge,
            con_graph=con_graph,
            node_cfg=node_cfg,
            solver=solver,
            epsilon=epsilon,
        )


class TransistorNetwork(EdgeNetwork):
    def __init__(
        self, name: str, con_graph: nx.DiGraph, node_cfg, solver, epsilon=1e-9
    ):
        super().__init__(
            name=name,
            edge_class=WrappedTransistorEdge,
            con_graph=con_graph,
            node_cfg=node_cfg,
            solver=solver,
            epsilon=epsilon,
        )
        self.model("Ideal", "NMOS", level=1)
        self.model("NMOS", "NMOS", level=1)
        self.model("MYSW", "SW", Ron=100, Roff=u_mOhm(1), Vt=0.5,)

class TwinNetwork(AbstractNetwork):
    def __init__(self, name: str, con_graph: nx.Graph, node_cfg, solver, R_ON=u_Ohm(100), C_LEARN=u_uF(22), t_h=u_us(100), period=u_us(200)):
        assert t_h < period, "Pulse width must be less than period"

        # duplicate graph but preserve ground node
        cg_copy = con_graph.copy()
        cg_copy = nx.relabel_nodes(cg_copy, {x: x + con_graph.number_of_nodes() for x in cg_copy.nodes})
        cg_combined = nx.compose(con_graph, cg_copy)

        cg_combined = nx.contracted_nodes(cg_combined, 0, con_graph.number_of_nodes())

        # let the original graph be the clamped network and the copy be the free network
        # need to copy inputs to the free network, but outputs stay only in the clamped network
        free_net_offset = lambda x: x + con_graph.number_of_nodes() * (x != 0)

        inputs, outputs = node_cfg
        inputs_free = free_net_offset(inputs)
        inputs = np.concatenate([inputs, inputs_free])

        self.outputs_free = [[str(node) for node in of] for of in free_net_offset(outputs)] # for evaluation purposes only

        node_cfg = (inputs, outputs)

        # Set up nodes and underlying methods
        super().__init__(name, cg_combined, node_cfg, solver)
        

        # add edges and teachers
        self.model('nmos', 'nmos', level=1)
        self.model('Ideal', 'nmos', level=1)
        self.model("MYSW", "SW", Ron=R_ON, Roff=u_mOhm(1), Vt=0.5,)

        self.free_edges = []
        self.clamped_edges = []
        self.teachers = []

        self.subcircuit(TransistorEdgeTeacher(r_shunt=1))
        self.subcircuit(Teacher(c_learn=C_LEARN))
        nodes_map = {n: i for i, n in enumerate(cg_combined.nodes())} # shared ground, all other nodes duplicated

        for n, (u, v, r) in enumerate(con_graph.edges(data="weight")):
            aux_node = "VGS" + str(n)
            self.__nodes__.append(aux_node)
            clamped_edge = self.X(f"C{n + 1}", TransistorEdgeTeacher.NAME, nodes_map[u], nodes_map[v], aux_node)
            free_edge = self.X(f"F{n + 1}", TransistorEdgeTeacher.NAME, free_net_offset(nodes_map[u]), free_net_offset(nodes_map[v]), aux_node)
            teacher = self.X(f"T{n+1}", Teacher.NAME, free_net_offset(nodes_map[u]), free_net_offset(nodes_map[v]), nodes_map[u], nodes_map[v], aux_node, "CLK")

            self.cached_simulator.initial_condition(**{aux_node: r})

            self.clamped_edges.append(clamped_edge)
            self.free_edges.append(free_edge)
            self.teachers.append(teacher)

        self.edges = self.clamped_edges + self.free_edges

        # add clock
        self.t_h = t_h
        self.period = period
        self.PulseVoltageSource("VCLK", "CLK", 0, initial_value=0, pulsed_value=1, pulse_width=self.t_h, period=self.t_h + self.period)

        # add support for random indexing
        self.RandomVoltageSource("rand", "rand", 0, "uniform", self.period, 0, 0.5, 0.5)
        self.B("index", "index", 0)

    def _prepare_simulation(self, inputs, outputs, nudge_factor):
        inputs = np.hstack([inputs, inputs])

        super()._prepare_simulation(inputs, outputs)

        # make clamps use nudge factor
        for of, oc in zip(self.outputs_free, self.outputs):
            a, b = of
            if (nudge_factor > 0) and (nudge_factor < 1):
                oc.voltage_expression = f"{1-nudge_factor} * (V({a}) - V({b})) + {nudge_factor}*" + oc.voltage_expression
            elif nudge_factor == 0:
                oc.voltage_expression = f"V({a}) - V({b})"
            elif nudge_factor == 1:
                pass
            else:
                raise ValueError("Nudge factor must be between 0 and 1 (inclusive)")

        # experiment with random indexing
        self.Vindex.enabled = False
        self.Bindex.enabled = True
        self.Bindex.voltage_expression = f"1 + FLOOR(V(rand)*{self.n_examples})"

    def _run_simulation(self):
        raise NotImplementedError("User should call call transient simulation themselves")
    def solve(self, inputs, outputs=None):
        raise NotImplementedError("User should call call transient simulation themselves")
    def predict(self, inputs):
        raise NotImplementedError("User should call call transient simulation themselves")