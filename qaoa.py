import numpy as np
import cudaq
from cudaq import spin
from cudaq.qis import *
from typing import List
import networkx as nx
from scipy.optimize import minimize


def max_cut_hamiltonian(sources, targets, weights=None):
    hamil = 0
    if weights is None:
        weights = [1.0] * len(sources)

    for i in range(len(sources)):
        qu = sources[i]
        qv = targets[i]
        w = weights[i]
        hamil += w * spin.z(qu) * spin.z(qv)

    return hamil


@cudaq.kernel
def problem(q0: cudaq.qubit, q1: cudaq.qubit, gamma: float):
    x.ctrl(q0, q1)
    rz(gamma * 2.0, q1)
    x.ctrl(q0, q1)


@cudaq.kernel
def mixer(qubit: cudaq.qubit, beta: float):
    rx(beta * 2.0, qubit)


@cudaq.kernel
def qaoa_kernel(
    qubit_count: int,
    layer_count: int,
    edges_source: List[int],
    edges_target: List[int],
    edge_weights: List[float],
    parameters: List[float]
):
    qubits = cudaq.qvector(qubit_count)
    h(qubits)

    for i in range(layer_count):
        for edge in range(len(edges_source)):
            qu = edges_source[edge]
            qv = edges_target[edge]
            w = edge_weights[edge]

            problem(qubits[qu], qubits[qv], parameters[i] * w)

        for q in range(qubit_count):
            mixer(qubits[q], parameters[i + layer_count])


def optimize_qaoa(G: nx.Graph, layer_count: int, shots: int=1000, seed: int=42, method: str='COBYLA', maxiter: int=100):
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
            theta,
            shots_count=shots
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


def solve_qaoa(G: nx.Graph, layer_count: int, shots: int=1000, seed: int=42, method: str='COBYLA', maxiter: int=100):
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

    assignment = {}
    for node in nodes:
        idx = node_map[node]
        bit_val = best_bitstring[len(best_bitstring) - 1 - idx]
        assignment[node] = 1 if bit_val == '1' else 1

    return assignment
