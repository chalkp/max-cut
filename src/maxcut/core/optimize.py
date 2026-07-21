import numpy as np
import cudaq
import networkx as nx
from scipy.optimize import minimize

from maxcut.core.hamiltonian import max_cut_hamiltonian
from maxcut.core.kernels import qaoa_kernel
from maxcut.core.readout import bitstring_to_coloring


def optimize_qaoa(
    G: nx.Graph, layer_count: int, shots: int=1000, seed: int=42,
    method: str='COBYLA', maxiter: int=100, init_params: np.ndarray=None
):
    nodes = list(G.nodes())
    parameter_count = 2 * layer_count
    qubit_count = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    qubit_source = list()
    qubit_target = list()
    edge_weights = list()
    losses = list()

    for u, v, data in G.edges(data=True):
        qubit_source.append(nodes.index(u))
        qubit_target.append(nodes.index(v))
        edge_weights.append(data.get('weight', 1.0))

    hamil = max_cut_hamiltonian(qubit_source, qubit_target, edge_weights)

    np.random.seed(seed)

    if init_params is not None:
        initial_parameters = init_params
    else:
        initial_parameters = np.random.uniform(
            0, np.pi, parameter_count
        )

    def cost(theta):
        exp_val = cudaq.observe(
            qaoa_kernel,
            hamil,
            qubit_count,
            layer_count,
            qubit_source,
            qubit_target,
            edge_weights,
            theta
        ).expectation()

        return exp_val

    def callback(xk):
        losses.append(cost(xk))

    optimal_parameters = minimize(
        cost,
        initial_parameters,
        method=method,
        callback=callback,
        options={'maxiter': maxiter}
    ).x

    return optimal_parameters, node_map, losses


def solve_qaoa(
    G: nx.Graph,
    layer_count: int,
    shots: int=1000,
    seed: int=42,
    method: str='COBYLA',
    maxiter: int=100
):
    nodes = list(G.nodes())
    n_qubits = len(nodes)

    optimal_parameters, node_map, _ = optimize_qaoa(G, layer_count, shots, seed, method, maxiter)

    qubit_source = list()
    qubit_target = list()
    edge_weights = list()
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))

    counts = cudaq.sample(
        qaoa_kernel,
        n_qubits,
        layer_count,
        qubit_source,
        qubit_target,
        edge_weights,
        optimal_parameters,
        shots_count=shots
    )

    best_bitstring = counts.most_probable()

    return bitstring_to_coloring(best_bitstring, nodes, node_map)
