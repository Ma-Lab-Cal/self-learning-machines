################################################################################
# Solver tolerance
#
# Authored 10/16
#
# Experiments to determine if tolerances can be relaxed for faster simulation
# while maintaining a level of accuracy comparable to existing lab-based
# results
# This script performs the (slow) repeated calls to the simulator to gather
# data, analysis will happen in a separate notebook
################################################################################
import PySpice.Spice.NgSpice.Shared
import PySpice.Spice.Xyce
import PySpice.Spice.Xyce.Server
from spice_net import *

import numpy as np
import networkx as nxs
import tqdm
import pandas as pd
import json

import PySpice

PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = "ngspice-shared"
import PySpice.Spice.NgSpice.Server

PySpice.Spice.NgSpice.Server.SpiceServer.SPICE_COMMAND = "ngspice"
PySpice.Spice.Xyce.Server.XyceServer.XYCE_COMMAND = (
    "/Users/lancemathias/XyceInstall/Serial/bin/Xyce"
)

# Fixed setup params
N_ITERS = 100
NET_SIZES = [4, 5, 10, 20, 30]

# Sweep values
SWEEP_VALS = dict(
    abstol=[1e-12, 1e-9, 1e-6, 1e-3, 1e-2],
    reltol=[1e-3, 1e-2, 1e-2, 1e-1],
    vntol=[1e-6, 1e-3, 1e-2],
    trtol=[1, 7, 10, 50, 100],
    chgtol=[1e-14, 1e-12, 1e-9, 1e-6, 1e-3],
)

# Fixed network parameters
NUDGE_FACTOR = .5 # disable nudging
R_LEARN = u_Ohm(0)
C_LEARN = u_uF(220)
LEARN_TIME = u_us(100)
PERIOD = u_us(200)

T_END = N_ITERS * PERIOD


def make_grid_net(grid_size):
    seed = 0
    np.random.seed(seed)

    grid_graph = nx.grid_graph([grid_size, grid_size], periodic=True)
    # relabel nodes to be ints
    grid_graph = nx.convert_node_labels_to_integers(
        grid_graph, first_label=0, ordering="sorted"
    )

    for e in grid_graph.edges:
        grid_graph[e[0]][e[1]]["weight"] = np.random.uniform(0.1, 0.9)  # random value
    node_cfg = (np.array([[5, 16], [7, 16], [13, 16], [15, 16]]), np.array([[10, 0]]))

    return TwinNetwork(
        "tol_experiments",
        grid_graph,
        node_cfg,
        "ngspice-shared",
        t_h=LEARN_TIME,
        period=PERIOD,
    )

def calc_error(res, baseline_res):
    # compute errors - assume both results have the exact same network topology
    # can only compare final timestep because auto-stepping means timesteps may differ
    max_err = max(
        np.abs(float(res.nodes[nn][-1] - baseline_res.nodes[nn][-1])) for nn in res.nodes.keys()
    )
    l1_err = sum(
        abs(float(res.nodes[nn][-1] - baseline_res.nodes[nn][-1])) for nn in res.nodes.keys()
    )
    l2_err = np.sqrt(
        sum(
            (float(res.nodes[nn][-1] - baseline_res.nodes[nn][-1])) ** 2
            for nn in res.nodes
        )
    )
    return [max_err, l1_err, l2_err]


def run_tol_experiment(
    instance,
    net_size,
    inputs,
    outputs,
    abstol,
    reltol,
    vntol,
    trtol,
    chgtol,
):
    net = make_grid_net(net_size)
    net._prepare_simulation(inputs, outputs, NUDGE_FACTOR)
    net.cached_simulator.options("klu")
    net.cached_simulator.options(seed=1)

    # Set tolerances
    net.cached_simulator.options(
        abstol=abstol, reltol=reltol, vntol=vntol, trtol=trtol, chgtol=chgtol
    )

    instance.resource_usage() # reset clock

    # Run simulation
    res = net.cached_simulator.transient(
        step_time=LEARN_TIME, end_time=T_END
    )
    ru = instance.resource_usage()

    success = np.isclose(float(res.time[-1]), float(T_END))
    return res, success, ru


if __name__ == "__main__":
    instance = PySpice.Spice.NgSpice.Shared.NgSpiceShared.new_instance()
    test_data = np.load("../data/xor_train_data_scale_1.0.npz")
    inputs, outputs = test_data["inputs"], test_data["outputs"]

    pbar = tqdm.tqdm(
        total=len(NET_SIZES)
        * (1 + sum(len(vals) - 1 for vals in SWEEP_VALS.values()) + max(len(vals) - 1 for vals in SWEEP_VALS.values()))
    )
    results = []
    for net_size in NET_SIZES:
        params = {name.lower(): vals[0] for name, vals in SWEEP_VALS.items()}
        baseline_res, baseline_success, baseline_stats = run_tol_experiment(instance, net_size, inputs, outputs, **params)
        results.append(
            [net_size, 'baseline'] + list(params.values()) + calc_error(baseline_res, baseline_res) + list(baseline_stats.values()) + [baseline_success]
        )
        pbar.update(1)

        for param, vals in SWEEP_VALS.items():
            for pv in vals[1:]:  # can avoid redundant baseline runs
                params[param] = pv
                res, success, stats = run_tol_experiment(instance, net_size, inputs, outputs, **params)
                results.append(
                    [net_size, param]
                    + list(params.values())
                    + (calc_error(res, baseline_res) if success else [999, 999, 999])
                    + list(stats.values())
                    + [success]
                )
                pbar.update(1)

            # reset to baseline
            params[param] = vals[0]

        # special experiment where all tolerances are loosened together
        params = {name.lower(): vals[0] for name, vals in SWEEP_VALS.items()}
        for j in range(1, max(len(vals) for vals in SWEEP_VALS.values())):
            for param, vals in SWEEP_VALS.items():
                if j < len(vals):
                    params[param] = vals[j]
            res, success, stats = run_tol_experiment(instance, net_size, inputs, outputs, **params)
            results.append(
                [net_size, "all"]
                + list(params.values())
                + (calc_error(res, baseline_res) if success else [999, 999, 999])
                + list(stats.values())
                + [success]
            )
            pbar.update(1)

    # Save results
    df = pd.DataFrame(
        results,
        columns=["net_size", "experiment"]
        + list(SWEEP_VALS.keys())
        + ["max_err", "l1_err", "l2_err"]
        + list(baseline_stats.keys())
        + ["success"]
    )
    df.to_csv("tol_experiments.csv", index=False)
