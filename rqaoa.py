import cudaq
from cudaq import spin
from typing import List
import networkx as nx
import numpy as np

from qaoa import (
    qaoa_kernel,
    optimize_qaoa
)
from utils import brute_force



def compute_correlations(G: nx.Graph, params: List[float], layer_count, node_map, shots: int=1000):
    nodes = list(G.nodes())
    n_qubits = len(nodes)
    
    qubit_source = list()
    qubit_target = list()
    edge_weights = list()
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))

    correlations = {}
    
    for u, v in G.edges():
        idx_u = node_map[u]
        idx_v = node_map[v]
        
        # Z_u * Z_v
        hamil = spin.z(idx_u) * spin.z(idx_v)
        
        exp_val = cudaq.observe(
            qaoa_kernel,
            hamil,
            n_qubits, 
            layer_count, 
            qubit_source, 
            qubit_target,
            edge_weights,
            params,
            shots_count=shots
        ).expectation()
        
        correlations[(u, v)] = exp_val

    return correlations


def contract_graph(graph: nx.Graph, u, v, correlation_sign: float):
    G_new = graph.copy()

    for edge in G_new.edges():
        if 'weight' not in G_new.edges[edge]:
            G_new.edges[edge]['weight'] = 1.0

    for neighbor in list(G_new.neighbors(v)):
        if neighbor == u:
            continue

        weight_v_neighbor = G_new.edges[v, neighbor]['weight']
        
        effective_weight = correlation_sign * weight_v_neighbor

        if G_new.has_edge(u, neighbor):
            G_new.edges[u, neighbor]['weight'] += effective_weight
        else:
            G_new.add_edge(u, neighbor, weight=effective_weight)

    G_new.remove_node(v)
    return G_new


def reconstruct_solution(base_assignment: dict, elimination_history: list):
    solution = base_assignment.copy()

    for u, v, _, sign in reversed(elimination_history):
        solution[v] = int(sign * solution[u])

    return solution


def solve_rqaoa(
    G: nx.Graph,
    layer_count: int,
    shots: int=1000,
    seed: int=42,
    method: str='COBYLA',
    cutoff: int=3,
    maxiter: int=100
):
    current_graph = G.copy()
    elimination_history = list() # [(u, v, cor, +-)]
    losses = list()

    step = 1

    while current_graph.number_of_nodes() > cutoff:
        params, node_map, loss = optimize_qaoa(current_graph, layer_count, shots, seed, method, maxiter)
        correlations = compute_correlations(current_graph, params, layer_count, node_map)
        losses.append((current_graph.number_of_nodes(), loss))
        
        max_edge = max(correlations, key=lambda e: abs(correlations[e]))
        max_corr = correlations[max_edge]
        u, v = max_edge

        correlation_sign = np.sign(max_corr)

        elimination_history.append((u, v, max_corr, correlation_sign))
        current_graph = contract_graph(current_graph, u, v, correlation_sign)
        step += 1
    
    _, _, max_subset = brute_force(current_graph)
    
    max_subset = max_subset or ()
    
    base_solution = {}
    for node in current_graph.nodes():
        base_solution[node] = 1 if node in max_subset else -1
        
    final_solution = reconstruct_solution(base_solution, elimination_history)

    return final_solution, current_graph, losses
