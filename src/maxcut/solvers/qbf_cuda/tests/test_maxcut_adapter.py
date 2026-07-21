"""
Correctness gate for qbf_cuda/maxcut_adapter.py's 'reference' backend -- run with plain
`python3 test_maxcut_adapter.py` (no pytest, no GPU/cupy required). Cross-checks
brute_force_qbf_cuda(G, backend='reference') against the ACTUAL utils.brute_force from
the parent max-cut project (imported directly, not reimplemented -- if utils.py's
formula ever changes, this test changes with it instead of silently comparing against a
stale copy).
"""
import numpy as np
import networkx as nx

# maxcut is editable-installed (pip install -e .), so these resolve from anywhere.
from maxcut.solvers.qbf_cuda import maxcut_adapter as adapter
from maxcut.solvers.brute_force import brute_force as _brute_force


class utils:  # thin namespace so the historical `utils.brute_force(...)` call sites read unchanged
    brute_force = staticmethod(_brute_force)


RNG = np.random.default_rng(0)


def contract_graph(graph, u, v, correlation_sign):
    """Verbatim copy of rqaoa.py's contract_graph -- see test_reference.py for why this
    is duplicated rather than imported (avoids a hard dependency on the cudaq
    package)."""
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


def test_matches_utils_brute_force_unweighted():
    checks = 0
    for n, m, seed in [(1, 1, 0), (2, 1, 0), (5, 2, 1), (7, 2, 2), (8, 3, 3), (10, 3, 4)]:
        if n == 1:
            G = nx.Graph()
            G.add_node(0)
        else:
            G = nx.barabasi_albert_graph(n, m, seed=seed)
        nx.set_edge_attributes(G, 1.0, 'weight')

        ref_cut, ref_edges, ref_subset = utils.brute_force(G)
        for A in range(0, G.number_of_nodes() + 1):
            checks += 1
            cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='reference', A=A)
            # Max-Cut can have multiple optimal partitions; qbf_solve's Gray-code scan
            # order differs from utils.brute_force's lexicographic bit-shift order, so
            # on a tie they can land on different (equally valid) optima. The cut
            # *value* is the correctness criterion, not which optimum was returned --
            # but the returned (subset, edges) must still be mutually consistent and
            # actually achieve the claimed value on THIS graph.
            assert cut == ref_cut, f"n={n} seed={seed} A={A}: cut {cut} != utils.brute_force {ref_cut}"
            recomputed_cut, recomputed_edges = adapter.cut_value_and_edges(G, subset)
            assert recomputed_cut == cut, (
                f"n={n} seed={seed} A={A}: returned subset {subset} actually cuts "
                f"{recomputed_cut}, not the claimed {cut}"
            )
            assert set(recomputed_edges) == set(edges), (
                f"n={n} seed={seed} A={A}: returned edges don't match the subset's actual cut edges"
            )
    print(f"  ({checks} (graph, A) configurations checked against utils.brute_force)")


def test_matches_utils_brute_force_contracted_residual():
    checks = 0
    for n0, m, seed, n_contractions in [(10, 2, 1, 2), (12, 3, 2, 4), (9, 2, 3, 3), (14, 3, 5, 6)]:
        rng = np.random.default_rng(seed)
        G = nx.barabasi_albert_graph(n0, m, seed=seed)
        nx.set_edge_attributes(G, 1.0, 'weight')
        for _ in range(n_contractions):
            if G.number_of_nodes() <= 2:
                break
            edges = list(G.edges())
            u, v = edges[rng.integers(len(edges))]
            sign = 1 if rng.random() < 0.5 else -1
            G = contract_graph(G, u, v, sign)

        ref_cut, ref_edges, ref_subset = utils.brute_force(G)
        n = G.number_of_nodes()
        for A in range(0, n + 1):
            checks += 1
            cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='reference', A=A)
            assert np.isclose(cut, ref_cut, atol=1e-6), (
                f"n0={n0} seed={seed} A={A}: cut {cut} != utils.brute_force {ref_cut} "
                f"(weights={[d['weight'] for _, _, d in G.edges(data=True)]})"
            )
    print(f"  ({checks} (graph, A) configurations checked, including negative weights)")


def test_empty_and_trivial_graphs():
    G = nx.Graph()
    cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='reference')
    assert (cut, edges, subset) == (0, [], ())

    G = nx.Graph()
    G.add_node('only')
    cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='reference')
    assert cut == 0 and edges == []

    G = nx.Graph()
    G.add_edge('a', 'b', weight=3.0)
    cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='reference')
    assert cut == 3.0
    assert set(subset) in ({'a'}, {'b'})


def test_rejects_non_integer_weights():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.5)
    G.add_edge(1, 2, weight=1.0)
    try:
        adapter.brute_force_qbf_cuda(G, backend='reference')
        raise AssertionError("expected ValueError for non-integer weights")
    except ValueError as e:
        assert "integer" in str(e)


def test_default_A_matches_explicit_A():
    G = nx.barabasi_albert_graph(9, 2, seed=7)
    nx.set_edge_attributes(G, 1.0, 'weight')
    default_cut, default_edges, default_subset = adapter.brute_force_qbf_cuda(G, backend='reference')
    explicit_cut, _, _ = adapter.brute_force_qbf_cuda(G, backend='reference', A=4)
    assert default_cut == explicit_cut


if __name__ == "__main__":
    tests = [
        test_matches_utils_brute_force_unweighted,
        test_matches_utils_brute_force_contracted_residual,
        test_empty_and_trivial_graphs,
        test_rejects_non_integer_weights,
        test_default_A_matches_explicit_A,
    ]
    failures = []
    for t in tests:
        print(f"{t.__name__} ... ", end="", flush=True)
        try:
            t()
            print("PASS")
        except AssertionError as e:
            print("FAIL")
            print(f"  {e}")
            failures.append(t.__name__)
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} test(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
