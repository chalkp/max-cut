"""
Correctness gate for qbf_cuda/reference.py -- run with plain `python3 test_reference.py`
(no pytest, no GPU, no cupy/cudaq required). kernel.cu is a hardware-aware port of
reference.qbf_solve; this file is what earns that port the right to be trusted, so it
checks the algorithm (prefix-suffix decomposition + Gray code) against brute force
across many random instances, including the negative/growing-magnitude weight patterns
that RQAOA's contract_graph produces on a residual graph -- exactly the regime the
qbf_cuda solver is meant to run on.
"""
import itertools
import sys
import os

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference as ref  # noqa: E402


RNG = np.random.default_rng(0)


def _random_qubo(n, rng, integer=True, scale=5):
    if integer:
        Q = rng.integers(-scale, scale + 1, size=(n, n)).astype(np.float64)
    else:
        Q = rng.uniform(-scale, scale, size=(n, n))
    return np.triu(Q)


def _random_ising(n, rng, integer=True, scale=5, with_field=False):
    if integer:
        J = rng.integers(-scale, scale + 1, size=(n, n)).astype(np.float64)
    else:
        J = rng.uniform(-scale, scale, size=(n, n))
    J = np.triu(J, k=1)
    J = J + J.T
    h = None
    if with_field:
        h = rng.integers(-scale, scale + 1, size=n).astype(np.float64) if integer \
            else rng.uniform(-scale, scale, size=n)
    return J, h


def contract_graph(graph, u, v, correlation_sign):
    """Verbatim copy of rqaoa.py's contract_graph (duplicated, not imported, to keep
    this test importable without the cudaq package -- same pattern already used by
    fixstars_test/rqaoa_amplify.py in this project)."""
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


def brute_force_maxcut(G):
    """Same formula as utils.brute_force, reproduced here so this test file has no
    dependency on the parent project's module layout."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0, ()
    adj = nx.to_numpy_array(G, nodelist=nodes)
    idx = np.arange(2 ** n)
    X = ((idx[:, None] & (1 << np.arange(n))) > 0).astype(np.int64)
    not_X = 1 - X
    cut_values = np.sum((X @ adj) * not_X, axis=1)
    best = int(np.argmax(cut_values))
    subset = tuple(nodes[i] for i in range(n) if X[best, i] == 1)
    return float(cut_values[best]), subset


def cut_value_of_spins(G, nodes, spins):
    """spins[i] in {-1,+1} for nodes[i]; cut value = sum of weights of edges whose
    endpoints disagree, i.e. the same objective utils.brute_force maximizes."""
    idx = {node: i for i, node in enumerate(nodes)}
    total = 0.0
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        if spins[idx[u]] != spins[idx[v]]:
            total += w
    return total


# ----------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------

def test_gray_code_bijection():
    for k in range(0, 8):
        n = 1 << k
        codes = ref.gray_code(np.arange(n))
        assert sorted(codes.tolist()) == list(range(n)), f"gray_code not a bijection for k={k}"
        if n > 1:
            diffs = codes[1:] ^ codes[:-1]
            popcounts = [bin(int(d)).count('1') for d in diffs]
            assert all(p == 1 for p in popcounts), f"non-single-bit gray transition at k={k}"


def test_flip_bit_and_sign_consistency():
    for A in range(1, 10):
        total = 1 << A
        codes = ref.gray_code(np.arange(total))
        for step in range(total - 1):
            k, sign = ref.flip_bit_and_sign(step)
            expected_next = int(codes[step]) ^ (1 << k)
            assert expected_next == int(codes[step + 1]), (
                f"A={A} step={step}: flip bit {k} on {codes[step]:0{A}b} "
                f"gives {expected_next:0{A}b}, expected {codes[step+1]:0{A}b}"
            )
            bit_turned_on = (int(codes[step + 1]) >> k) & 1
            expected_sign = 1 if bit_turned_on else -1
            assert sign == expected_sign, f"A={A} step={step}: sign {sign} != expected {expected_sign}"


def test_bits_index_roundtrip():
    for width in range(1, 12):
        idx = np.arange(2 ** width)
        bits = ref.bits_from_index(idx, width)
        back = ref.index_from_bits(bits)
        assert np.array_equal(back, idx), f"roundtrip failed for width={width}"


def test_qbf_solve_matches_bruteforce_random_qubo():
    trials = 0
    for n in range(1, 11):
        for integer in (True, False):
            Q = _random_qubo(n, RNG, integer=integer)
            ref_energy, ref_bits = ref.brute_force_qubo(Q)
            for A in range(0, n + 1):
                trials += 1
                energy, bits = ref.qbf_solve(Q, A)
                assert np.isclose(energy, ref_energy, atol=1e-6), (
                    f"n={n} A={A} integer={integer}: qbf_solve energy {energy} "
                    f"!= brute force {ref_energy}"
                )
                recomputed = ref.energy_qubo(Q, bits[None, :])[0]
                assert np.isclose(recomputed, energy, atol=1e-6), (
                    f"n={n} A={A}: returned bits evaluate to {recomputed}, claimed {energy}"
                )
    assert trials > 0
    print(f"  ({trials} (n, A) configurations checked)")


def test_qbf_solve_matches_bruteforce_ising():
    trials = 0
    for n in range(1, 10):
        for with_field in (False, True):
            J, h = _random_ising(n, RNG, integer=True, with_field=with_field)
            Q, const = ref.ising_to_qubo(J, h)

            # ground truth: enumerate all spin assignments directly
            best_ising = None
            for bits_tuple in itertools.product([0, 1], repeat=n):
                s = 1 - 2 * np.array(bits_tuple)
                # np.triu(J, k=1) already sums each i<j pair once -- do not also add its
                # transpose, that would double every coupling.
                e = float(s @ np.triu(J, k=1) @ s + (h @ s if h is not None else 0.0))
                if best_ising is None or e < best_ising:
                    best_ising = e

            for A in range(0, n + 1):
                trials += 1
                qubo_energy, bits = ref.qbf_solve(Q, A)
                ising_energy = qubo_energy + const
                assert np.isclose(ising_energy, best_ising, atol=1e-6), (
                    f"n={n} A={A} with_field={with_field}: ising energy {ising_energy} "
                    f"!= brute force {best_ising}"
                )
                spins = ref.qubo_bits_to_ising_spins(bits)
                direct = float(spins @ np.triu(J, k=1) @ spins + (h @ spins if h is not None else 0.0))
                assert np.isclose(direct, ising_energy, atol=1e-6)
    print(f"  ({trials} (n, A) configurations checked)")


def test_contracted_residual_like_weights():
    """Build small Barabasi-Albert graphs, run a couple of contract_graph steps (the
    exact operation RQAOA performs), and confirm qbf_solve's result matches the same
    max-cut brute force utils.brute_force uses -- on the negative/growing-magnitude
    weights this operation actually produces, not just clean +-1 test weights."""
    configs = 0
    for n0, m, seed, n_contractions in [(8, 2, 1, 2), (10, 3, 2, 3), (9, 2, 3, 4), (12, 3, 4, 5)]:
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

        nodes = list(G.nodes())
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(nodes)}
        J = np.zeros((n, n))
        for a, b, data in G.edges(data=True):
            w = data.get('weight', 1.0)
            J[node_map[a], node_map[b]] += w
            J[node_map[b], node_map[a]] += w

        Q, const = ref.ising_to_qubo(J)

        bound = ref.energy_magnitude_bound(Q)
        assert bound < 2 ** 30, "test fixture accidentally produced an enormous QUBO"

        bf_cut, bf_subset = brute_force_maxcut(G)

        for A in range(0, n + 1):
            configs += 1
            qubo_energy, bits = ref.qbf_solve(Q, A)
            spins = ref.qubo_bits_to_ising_spins(bits)
            qbf_cut = cut_value_of_spins(G, nodes, spins)
            assert np.isclose(qbf_cut, bf_cut, atol=1e-6), (
                f"n={n} A={A} seed={seed}: qbf cut {qbf_cut} != brute-force cut {bf_cut} "
                f"(edge weights={[d['weight'] for _,_,d in G.edges(data=True)]})"
            )
    print(f"  ({configs} (graph, A) configurations checked, subset sizes up to 12 nodes)")


def test_energy_magnitude_bound():
    for _ in range(20):
        n = RNG.integers(1, 15)
        Q = _random_qubo(n, RNG, integer=False, scale=7)
        bound = ref.energy_magnitude_bound(Q)
        # spot-check against a sample of random states rather than full enumeration
        for _ in range(50):
            x = RNG.integers(0, 2, size=n)
            e = ref.energy_qubo(Q, x[None, :])[0]
            assert abs(e) <= bound + 1e-9, f"bound {bound} violated by energy {e}"


if __name__ == "__main__":
    tests = [
        test_gray_code_bijection,
        test_flip_bit_and_sign_consistency,
        test_bits_index_roundtrip,
        test_qbf_solve_matches_bruteforce_random_qubo,
        test_qbf_solve_matches_bruteforce_ising,
        test_contracted_residual_like_weights,
        test_energy_magnitude_bound,
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
