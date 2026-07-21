from typing import Union

import numpy as np
import networkx as nx

from maxcut.core.optimize import optimize_qaoa
from maxcut.rqaoa.correlations import compute_correlations
from maxcut.rqaoa.contraction import contract_graph, reconstruct_solution
from maxcut.rqaoa.exact import _solve_residual_exactly
from maxcut.rqaoa.schedule import select_depth
from maxcut.graphs.evaluation import process_max_cut


def solve_beam_rqaoa(
    G: nx.Graph,
    layer_count: Union[int, dict],
    beam_width: int=2,
    shots: int=1000,
    seed: int=42,
    method: str='COBYLA',
    cutoff: int=3,
    maxiter: int=100,
    long_range: bool=False,
    exact_solver: str='qbf_cuda'
):
    beam = [(G.copy(), [], 0.0, None)]

    while beam[0][0].number_of_nodes() > cutoff:
        new_candidates = []

        for graph, history, total_loss, last_params in beam:
            n_current = graph.number_of_nodes()
            current_p = select_depth(layer_count, n_current)
            if isinstance(layer_count, dict) and last_params is not None and len(last_params) != 2 * current_p:
                last_params = None

            params, node_map, loss = optimize_qaoa(graph, current_p, shots, seed, method, maxiter, init_params=last_params)
            correlations = compute_correlations(graph, params, current_p, node_map, long_range=long_range)

            # Pick top B edges to branch
            sorted_pairs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pairs = sorted_pairs[:beam_width]

            for (u, v), corr in top_pairs:
                sign = np.sign(corr)
                new_graph = contract_graph(graph, u, v, sign)
                new_history = history + [(u, v, corr, sign)]
                new_candidates.append((new_graph, new_history, total_loss + (loss[-1] if loss else 0.0), params))

        # Keep top B unique graphs based on total exp value
        beam = sorted(new_candidates, key=lambda x: x[2])[:beam_width]

    # Resolve all beams
    best_final_cut = -1
    best_solution = None

    for graph, history, _, _ in beam:
        _, _, max_subset = _solve_residual_exactly(graph, exact_solver)
        max_subset = max_subset or ()
        base_solution = {node: 1 if node in max_subset else -1 for node in graph.nodes()}
        final_sol = reconstruct_solution(base_solution, history)

        # Verify cut size on OG graph
        for node, color in final_sol.items():
            G.nodes[node]['color'] = 1 if color == 1 else 0
        cut_val, _ = process_max_cut(G)

        if cut_val > best_final_cut:
            best_final_cut = cut_val
            best_solution = final_sol

    return best_solution, best_final_cut
