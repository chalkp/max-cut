"""arXiv:2009.10095."""
from typing import Optional, List

import numpy as np
import cudaq
from scipy.optimize import minimize
import networkx as nx

from maxcut.core.kernels import problem
from maxcut.core.hamiltonian import max_cut_hamiltonian
from maxcut.instrumentation.budget import BudgetTracker, standard_qaoa_layer_two_qubit_gates
from maxcut.baselines.warm_start.classical import (
    goemans_williamson_relax, random_hyperplane_rounding, regularize,
)


@cudaq.kernel
def warm_start_qaoa_kernel(
    qubit_count: int,
    layer_count: int,
    edges_source: List[int],
    edges_target: List[int],
    edge_weights: List[float],
    thetas: List[float],
    parameters: List[float],
):
    qubits = cudaq.qvector(qubit_count)

    for q in range(qubit_count):
        ry(thetas[q], qubits[q])

    for i in range(layer_count):
        for edge in range(len(edges_source)):
            qu = edges_source[edge]
            qv = edges_target[edge]
            w = edge_weights[edge]
            problem(qubits[qu], qubits[qv], parameters[i] * w)

        for q in range(qubit_count):
            ry(thetas[q], qubits[q])
            rz(-2.0 * parameters[i + layer_count], qubits[q])
            ry(-thetas[q], qubits[q])


def optimize_warm_start_qaoa(
    G: nx.Graph,
    layer_count: int,
    epsilon: float = 0.25,
    seed: int = 42,
    method: str = 'COBYLA',
    maxiter: int = 100,
    init_params: Optional[np.ndarray] = None,
    tracker: Optional[BudgetTracker] = None,
    n_hyperplane_trials: int = 50,
):
    if tracker is not None:
        tracker.start()

    nodes = list(G.nodes())
    qubit_count = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    qubit_source, qubit_target, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))
    n_edges = len(qubit_source)

    Y = goemans_williamson_relax(G, seed=seed)
    c = random_hyperplane_rounding(
        Y, qubit_source, qubit_target, edge_weights,
        seed=seed, n_trials=n_hyperplane_trials)
    c_reg = regularize(c, epsilon)
    thetas = (2 * np.arcsin(np.sqrt(c_reg))).tolist()

    hamil = max_cut_hamiltonian(qubit_source, qubit_target, edge_weights)

    np.random.seed(seed)
    parameter_count = 2 * layer_count
    initial_parameters = (
        init_params if init_params is not None
        else np.random.uniform(0, np.pi, parameter_count)
    )

    losses = []
    gates_per_call = standard_qaoa_layer_two_qubit_gates(n_edges) * layer_count

    def cost(theta):
        exp_val = cudaq.observe(
            warm_start_qaoa_kernel, hamil, qubit_count, layer_count,
            qubit_source, qubit_target, edge_weights, thetas, theta,
        ).expectation()
        if tracker is not None:
            tracker.record_observe(gates_per_call)
        return exp_val

    def callback(xk):
        losses.append(cost(xk))

    optimal_parameters = minimize(
        cost, initial_parameters, method=method, callback=callback,
        options={'maxiter': maxiter},
    ).x

    if tracker is not None:
        tracker.stop()

    return optimal_parameters, node_map, losses, thetas


def solve_warm_start_qaoa_coloring(
    G, layer_count, epsilon=0.25, shots=1000, seed=42,
    method='COBYLA', maxiter=100, init_params=None, tracker=None
):
    nodes = list(G.nodes())
    n_qubits = len(nodes)
    optimal_parameters, node_map, _, thetas = optimize_warm_start_qaoa(
        G, layer_count, epsilon, seed, method, maxiter, init_params, tracker,
    )

    qubit_source, qubit_target, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))

    n_edges = len(qubit_source)
    gates = standard_qaoa_layer_two_qubit_gates(n_edges) * layer_count
    counts = cudaq.sample(
        warm_start_qaoa_kernel, n_qubits, layer_count, qubit_source, qubit_target,
        edge_weights, thetas, optimal_parameters, shots_count=shots,
    )
    if tracker is not None:
        tracker.record_sample(gates, shots=shots)

    best_bitstring = counts.most_probable()
    coloring = {}
    for node in nodes:
        idx = node_map[node]
        bit_val = best_bitstring[idx]
        coloring[node] = 1 if bit_val == '1' else 0
    return coloring
