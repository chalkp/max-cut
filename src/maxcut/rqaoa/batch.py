import networkx as nx
import numpy as np

from maxcut.core.optimize import optimize_qaoa
from maxcut.rqaoa.correlations import compute_correlations
from maxcut.rqaoa.contraction import contract_graph, reconstruct_solution
from maxcut.solvers.brute_force import brute_force


def solve_batch_rqaoa(
    G: nx.Graph,
    layer_count: int,
    shots: int=1000,
    seed: int=42,
    method: str='COBYLA',
    cutoff: int=3,
    maxiter: int=100,
    batch_size: int=1,
    threshold: float=None
):
    current_graph = G.copy()
    elimination_history = list() # [(u, v, corr, +-)]
    losses = list()

    step = 1
    last_params = None

    while current_graph.number_of_nodes() > cutoff:
        params, node_map, loss = optimize_qaoa(current_graph, layer_count, shots, seed, method, maxiter, init_params=last_params)
        last_params = params
        correlations = compute_correlations(current_graph, params, layer_count, node_map)
        losses.append((current_graph.number_of_nodes(), loss))

        sorted_edges = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)

        batch_edges = []
        used_nodes = set()

        for (u, v), corr in sorted_edges:
            if u not in used_nodes and v not in used_nodes:
                if threshold is not None:
                    if abs(corr) >= threshold:
                        batch_edges.append(((u, v), corr))
                        used_nodes.add(u)
                        used_nodes.add(v)
                else:
                    batch_edges.append(((u, v), corr))
                    used_nodes.add(u)
                    used_nodes.add(v)
                    if len(batch_edges) >= batch_size:
                        break

        # Fallback
        if not batch_edges:
            (u, v), corr = sorted_edges[0]
            batch_edges.append(((u, v), corr))

        for (u, v), corr in batch_edges:
            correlation_sign = np.sign(corr)
            elimination_history.append((u, v, corr, correlation_sign))
            current_graph = contract_graph(current_graph, u, v, correlation_sign)

        step += 1

    _, _, max_subset = brute_force(current_graph)

    max_subset = max_subset or ()

    base_solution = {}
    for node in current_graph.nodes():
        base_solution[node] = 1 if node in max_subset else -1

    final_solution = reconstruct_solution(base_solution, elimination_history)

    return final_solution, current_graph, losses
