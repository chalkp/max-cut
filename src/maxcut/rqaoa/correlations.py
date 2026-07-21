import itertools
from typing import List

import cudaq
from cudaq import spin
import networkx as nx

from maxcut.core.kernels import qaoa_kernel


def compute_correlations(
    G: nx.Graph,
    params: List[float],
    layer_count: int,
    node_map: dict,
    shots: int = 1000,
    long_range: bool = False
):
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

    pairs = itertools.combinations(nodes, 2) if long_range else G.edges()

    for u, v in pairs:
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
            params
        ).expectation()

        correlations[(u, v)] = exp_val

    return correlations
