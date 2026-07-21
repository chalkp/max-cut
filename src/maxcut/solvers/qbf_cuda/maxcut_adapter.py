"""
networkx Max-Cut <-> Ising <-> QUBO adapter, plus brute_force_qbf_cuda(G): a drop-in
replacement for utils.brute_force(G) (same signature, same (cut_value, cut_edges,
max_subset) return) backed by either the pure-NumPy reference solver (always available,
used for testing and for graphs too small to bother with a GPU) or the CUDA kernel
(requires cupy + a GPU; see solver.py and README.md for what has and hasn't been run).

Sign convention: Max-Cut maximizes sum of cut-edge weights; the QBF solver minimizes a
QUBO. reference.ising_to_qubo maps Ising spins s_i = 1 - 2*x_i, so x_i = 0 <-> s_i = +1.
solve_rqaoa's own final step (rqaoa.py) does exactly this same mapping in the opposite
direction (`base_solution[node] = 1 if node in max_subset else -1`), i.e. "in
max_subset" <-> spin +1 <-> QUBO bit 0. brute_force_qbf_cuda reuses that same
convention so it can replace utils.brute_force in that call site without changing
solve_rqaoa's reconstruct_solution logic at all.
"""
import numpy as np
import networkx as nx

from . import reference as ref


def graph_to_ising(G, nodelist=None):
    """Dense symmetric coupling matrix J (zero diagonal) from a networkx graph's
    'weight' edge attribute (missing weights default to 1.0, matching
    utils.brute_force/rqaoa.contract_graph's own convention). No linear field: RQAOA's
    contract_graph never introduces a per-node bias, only edge-weight updates, so every
    graph this adapter will ever see is a pure edge-weighted Max-Cut instance."""
    nodes = nodelist if nodelist is not None else list(G.nodes())
    n = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}
    J = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = node_map[u], node_map[v]
        J[i, j] += w
        J[j, i] += w
    return J, nodes


def _validate_integer_weights(J, tol=1e-6):
    rounded = np.round(J)
    if not np.allclose(J, rounded, atol=tol):
        max_err = np.max(np.abs(J - rounded))
        raise ValueError(
            f"qbf_cuda requires integer edge weights (kernel accumulates energies in "
            f"int32 -- see README.md); largest deviation from an integer is {max_err:.3g}. "
            f"This holds for RQAOA's own contract_graph output (weights are sums of "
            f"+-1-scaled 1.0-initialized weights, always integral), but not for "
            f"arbitrary weighted graphs. Round/rescale weights before calling, or pass "
            f"already-integer weights."
        )
    return rounded.astype(np.int64)


def cut_value_and_edges(G, subset):
    """Cut value and cut edges for a given subset of nodes -- the same quantity
    utils.brute_force and utils.process_max_cut compute, recomputed directly from the
    graph rather than derived from the solver's returned energy (a returned energy
    only matches the cut value up to the exact sign/constant bookkeeping of the
    Ising<->QUBO conversion; recomputing directly from the assignment sidesteps that
    entirely and is a strictly stronger check that the solver actually optimized the
    right objective)."""
    subset_set = set(subset)
    cut_value = 0.0
    cut_edges = []
    for u, v, data in G.edges(data=True):
        if (u in subset_set) != (v in subset_set):
            w = data.get('weight', 1.0)
            cut_value += w
            cut_edges.append((u, v))
    return cut_value, cut_edges


def brute_force_qbf_cuda(G, backend='reference', A=None, plan=None, gpu=None):
    """Drop-in replacement for utils.brute_force(G): returns (max_cut_value,
    max_cut_edges, max_subset) with the identical semantics (max_subset is the tuple of
    node IDs on the "1" side of the exact optimal cut).

    backend='reference' (default): pure NumPy, no GPU, exact for any N your machine has
        the patience for (the O(2^A)-trip Python loop is the bottleneck, not the O(2^N)
        total work -- see the `A` default chosen below). Always available; this is what
        tests/test_maxcut_adapter.py exercises.
    backend='cuda': the CUDA kernel via solver.py. Requires cupy and a GPU; NOT run as
        part of writing this module (see README.md) -- validate on the target 3080
        before trusting it in place of 'reference' for anything that matters.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0, [], ()

    J, nodes = graph_to_ising(G, nodelist=nodes)
    J_int = _validate_integer_weights(J)
    Q, const = ref.ising_to_qubo(J_int.astype(np.float64))

    if backend == 'reference':
        if A is None:
            # Minimize Python-loop trip count (2**A), not total work (2**N, handled by
            # vectorized NumPy regardless of the split) -- see module docstring.
            A = min(n, 6)
        qubo_energy, bits = ref.qbf_solve(Q, A)
    elif backend == 'cuda':
        from . import solver as cuda_solver  # local import: keep cupy optional (see solver.py)
        qubo_energy, bits = cuda_solver.solve_qubo(Q, plan=plan, gpu=gpu)
    else:
        raise ValueError(f"unknown backend {backend!r}, expected 'reference' or 'cuda'")

    spins = ref.qubo_bits_to_ising_spins(bits)
    max_subset = tuple(nodes[i] for i in range(n) if spins[i] == 1)

    cut_value, cut_edges = cut_value_and_edges(G, max_subset)

    return int(cut_value) if float(cut_value).is_integer() else cut_value, cut_edges, max_subset
