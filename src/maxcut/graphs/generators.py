import networkx as nx
import numpy as np


def generate_graph_legacy(
    n: int, p: float,
    rng: np.random._generator.Generator = None,
    seed: int = 42
) -> nx.Graph:
    if rng == None:
        rng = np.random.default_rng(seed)
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng)
        if nx.is_connected(G):
            nx.set_node_attributes(G, values=0, name='color')
            nx.set_edge_attributes(G, values=1.0, name='weight')
            return G


def generate_graph(n: int, m: int, seed: int = 42) -> nx.Graph: 
    """Barabasi-Albert graph"""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=rng)
    nx.set_node_attributes(G, values=0, name='color')
    nx.set_edge_attributes(G, values=1.0, name='weight')
    return G


def _assign_integer_weights(G: nx.Graph, low: int = -10, high: int = 10, seed: int = 42) -> nx.Graph:
    """qbf_cuda/maxcut_adapter.py"""
    rng = np.random.default_rng(seed)
    weights = rng.integers(low, high + 1, size=G.number_of_edges())
    for (u, v), w in zip(G.edges(), weights):
        G.edges[u, v]['weight'] = float(w)
    return G


def generate_bipartite_graph(n1: int, n2: int, seed: int = 42) -> nx.Graph:
    """unweighted complete K(n1, n2) graph.
    RQAOA merges nodes across the two sides, see 2408.13207
    """
    G = nx.complete_bipartite_graph(n1, n2)
    nx.set_node_attributes(G, values=0, name='color')
    nx.set_edge_attributes(G, values=1.0, name='weight')
    return G


def generate_complete_weighted_graph(n: int, seed: int = 42, low: int = -10, high: int = 10) -> nx.Graph:
    """Complete K(n) with integer edge weights uniform in [low, high] -- the same
    construction as Egger, Mareček, Woerner's warm-start QAOA Figure 7 benchmark
    (arXiv:2009.10095), parameterized by n so it serves both that validation (n=30,
    classical-only) and this project's own dense/complete comparison family (smaller n)."""
    G = nx.complete_graph(n)
    nx.set_node_attributes(G, values=0, name='color')
    return _assign_integer_weights(G, low, high, seed)


def generate_weighted_graph(n: int, m: int, seed: int = 42, low: int = -10, high: int = 10) -> nx.Graph:
    """Weighted Barabasi-Albert graph"""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=rng)
    nx.set_node_attributes(G, values=0, name='color')
    return _assign_integer_weights(G, low, high, seed)


def generate_irregular_graph(n: int, seed: int = 42, n_blocks: int = 4) -> nx.Graph:
    if n < 3 * n_blocks:
        raise ValueError(
            f"generate_irregular_graph needs n >= 3*n_blocks (each block >= 3 nodes); "
            f"got n={n}, n_blocks={n_blocks} (min n = {3 * n_blocks})."
        )
    rng = np.random.default_rng(seed)

    def motif_graph(motif_idx: int, size: int) -> nx.Graph:
        kind = motif_idx % 4
        if kind == 0:
            return nx.complete_graph(size)
        elif kind == 1:
            return nx.cycle_graph(size)
        elif kind == 2:
            return nx.star_graph(size - 1)
        else:
            return nx.path_graph(size)

    remaining = n
    block_sizes = []
    for i in range(n_blocks):
        blocks_left = n_blocks - i
        min_size = 3
        max_size = remaining - min_size * (blocks_left - 1)
        if i == n_blocks - 1:
            size = remaining
        else:
            size = int(rng.integers(min_size, max(min_size, max_size) + 1))
        block_sizes.append(size)
        remaining -= size

    blocks = [motif_graph(i, size) for i, size in enumerate(block_sizes)]
    G = blocks[0]
    offsets = [0]
    for block in blocks[1:]:
        offsets.append(G.number_of_nodes())  # where this block's nodes will start
        G = nx.disjoint_union(G, block)

    for i in range(len(blocks) - 1):
        u = rng.integers(0, block_sizes[i]) + offsets[i]
        v = rng.integers(0, block_sizes[i + 1]) + offsets[i + 1]
        G.add_edge(int(u), int(v))

    nx.set_node_attributes(G, values=0, name='color')
    nx.set_edge_attributes(G, values=1.0, name='weight')
    return G
