from typing import Union

import numpy as np
import networkx as nx

from maxcut.core.optimize import optimize_qaoa
from maxcut.rqaoa.correlations import compute_correlations
from maxcut.rqaoa.contraction import contract_graph, reconstruct_solution
from maxcut.rqaoa.exact import _solve_residual_exactly
from maxcut.rqaoa.schedule import select_depth


def solve_rqaoa(
    G: nx.Graph,
    layer_count: Union[int, dict],
    shots: int=1000,
    seed: int=42,
    method: str='COBYLA',
    cutoff: int=3,
    maxiter: int=100,
    long_range: bool=False,
    exact_solver: str='qbf_cuda'
):
    current_graph = G.copy()
    elimination_history = list() # [(u, v, corr, +-)]
    losses = list()

    step = 1
    last_params = None

    while current_graph.number_of_nodes() > cutoff:
        n_current = current_graph.number_of_nodes()

        # Adaptive layer count
        current_p = select_depth(layer_count, n_current)
        if isinstance(layer_count, dict) and last_params is not None and len(last_params) != 2 * current_p:
            last_params = None

        params, node_map, loss = optimize_qaoa(current_graph, current_p, shots, seed, method, maxiter, init_params=last_params)
        last_params = params
        correlations = compute_correlations(current_graph, params, current_p, node_map, long_range=long_range)
        losses.append((n_current, loss))

        max_edge = max(correlations, key=lambda e: abs(correlations[e]))
        max_corr = correlations[max_edge]
        u, v = max_edge

        correlation_sign = np.sign(max_corr)

        elimination_history.append((u, v, max_corr, correlation_sign))
        current_graph = contract_graph(current_graph, u, v, correlation_sign)
        step += 1

    _, _, max_subset = _solve_residual_exactly(current_graph, exact_solver)

    max_subset = max_subset or ()

    base_solution = {}
    for node in current_graph.nodes():
        base_solution[node] = 1 if node in max_subset else -1

    final_solution = reconstruct_solution(base_solution, elimination_history)

    return final_solution, current_graph, losses
