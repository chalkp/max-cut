"""pure numpy vectorized max-cut brute force"""
from itertools import chain, combinations
from typing import Tuple, List

import networkx as nx
import numpy as np


def powerset(G: nx.Graph) -> chain:
    s = list(G.nodes())
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def brute_force(G: nx.Graph) -> Tuple[int, List[Tuple[int, int]], Tuple[int, ...]]:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0, [], ()

    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    assignments = ((np.arange(2**n)[:, None] & (1 << np.arange(n))) > 0).astype(int)
    X = assignments
    not_X = 1 - X
    cut_values = np.sum((X @ adj_matrix) * not_X, axis=1)

    best_idx = np.argmax(cut_values)
    max_cut_value = cut_values[best_idx]
    best_assignment = X[best_idx]

    max_subset = tuple(nodes[i] for i in range(n) if best_assignment[i] == 1)

    max_cut_edges = []
    subset_set = set(max_subset)
    for u, v in G.edges():
        if (u in subset_set) != (v in subset_set):
            max_cut_edges.append((u, v))

    return int(max_cut_value), max_cut_edges, max_subset
