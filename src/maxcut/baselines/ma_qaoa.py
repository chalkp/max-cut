"""arXiv:2109.11455"""
from typing import List, Optional

import numpy as np
import cudaq
from cudaq import spin
from scipy.optimize import minimize
import networkx as nx

from maxcut.core.kernels import problem, mixer
from maxcut.core.hamiltonian import max_cut_hamiltonian
from maxcut.instrumentation.budget import BudgetTracker, standard_qaoa_layer_two_qubit_gates


@cudaq.kernel
def ma_qaoa_kernel(
    qubit_count: int,
    layer_count: int,
    edges_source: List[int],
    edges_target: List[int],
    edge_weights: List[float],
    parameters: List[float]
):
    qubits = cudaq.qvector(qubit_count)
    h(qubits)

    n_edges = len(edges_source)
    block = n_edges + qubit_count

    for i in range(layer_count):
        for edge in range(n_edges):
            qu = edges_source[edge]
            qv = edges_target[edge]
            w = edge_weights[edge]
            problem(qubits[qu], qubits[qv], parameters[i * block + edge] * w)

        for q in range(qubit_count):
            mixer(qubits[q], parameters[i * block + n_edges + q])


def optimize_ma_qaoa(
    G: nx.Graph,
    layer_count: int,
    seed: int = 42,
    method: str = 'COBYLA',
    maxiter: int = 100,
    init_params: Optional[np.ndarray] = None,
    tracker: Optional[BudgetTracker] = None
):
    nodes = list(G.nodes())
    qubit_count = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    qubit_source, qubit_target, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))

    n_edges = len(qubit_source)
    parameter_count = layer_count * (n_edges + qubit_count)
    hamil = max_cut_hamiltonian(qubit_source, qubit_target, edge_weights)

    np.random.seed(seed)
    initial_parameters = (
        init_params if init_params is not None
        else np.random.uniform(0, np.pi, parameter_count)
    )

    losses = []
    gates_per_call = standard_qaoa_layer_two_qubit_gates(n_edges) * layer_count

    def cost(theta):
        exp_val = cudaq.observe(
            ma_qaoa_kernel, hamil, qubit_count, layer_count,
            qubit_source, qubit_target, edge_weights, theta,
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

    return optimal_parameters, node_map, losses


def solve_ma_qaoa_coloring(
    G, layer_count, shots=1000, seed=42, method='COBYLA',
    maxiter=100, init_params=None, tracker=None
):
    nodes = list(G.nodes())
    n_qubits = len(nodes)
    optimal_parameters, node_map, _ = optimize_ma_qaoa(
        G, layer_count, seed, method, maxiter, init_params, tracker,
    )

    qubit_source, qubit_target, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))

    n_edges = len(qubit_source)
    gates = standard_qaoa_layer_two_qubit_gates(n_edges) * layer_count
    counts = cudaq.sample(
        ma_qaoa_kernel, n_qubits, layer_count, qubit_source, qubit_target,
        edge_weights, optimal_parameters, shots_count=shots,
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
