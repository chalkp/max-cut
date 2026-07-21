from typing import List, Optional

import numpy as np
import cudaq
from scipy.optimize import minimize
import networkx as nx

from maxcut.core.hamiltonian import max_cut_hamiltonian
from maxcut.instrumentation.budget import BudgetTracker, standard_qaoa_layer_two_qubit_gates
from maxcut.baselines.adapt.kernels import (
    _adapt_initial_state,
    _adapt_echo_kernel,
    _adapt_grad_kernel,
    adapt_qaoa_kernel
)
from maxcut.baselines.adapt.pool import (
    build_operator_pool,
    _term_coefficients,
    _term_words,
    _commutator,
    _pauli_word_two_qubit_gates
)


def optimize_adapt_qaoa(
    G: nx.Graph,
    max_layers: int = 5,
    gradient_tol: float = 1e-3,
    energy_tol: float = 1e-4, # nvidia's tutorial uses 1e-7
    restrict_pool_to_edges: bool = True,
    bfgs_tol: float = 1e-3,
    bfgs_maxiter: int = 100,
    init_gamma: float = 0.01,
    seed: int = 42,
    tracker: Optional[BudgetTracker] = None,
):
    if tracker is not None:
        tracker.start()

    nodes = list(G.nodes())
    qubits_num = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    qubit_source, qubit_target, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        qubit_source.append(node_map[u])
        qubit_target.append(node_map[v])
        edge_weights.append(data.get('weight', 1.0))
    n_edges = len(qubit_source)
    edges = list(zip(qubit_source, qubit_target))

    ham = max_cut_hamiltonian(qubit_source, qubit_target, edge_weights)
    ham_word = _term_words(ham, qubits_num)
    ham_coef = _term_coefficients(ham)
    cost_layer_gates = standard_qaoa_layer_two_qubit_gates(n_edges)

    pools = build_operator_pool(qubits_num, edges=edges, restrict_to_edges=restrict_pool_to_edges)
    grad_op = [op * (-1j) for op in _commutator(pools, ham)]

    state = cudaq.get_state(_adapt_initial_state, qubits_num)
    E_prev = cudaq.observe(_adapt_echo_kernel, ham, state).expectation()

    beta, gamma = [], []
    mixer_pool: List[List] = []
    losses = [E_prev]
    rng = np.random.default_rng(seed)

    result_energy = E_prev

    for istep in range(1, max_layers + 1):
        gradient_vec = []
        for op in grad_op:
            grad_gates = cost_layer_gates
            val = cudaq.observe(
                _adapt_grad_kernel,
                op, state, ham_word,
                ham_coef, init_gamma
            ).expectation()
            if tracker is not None:
                tracker.record_observe(grad_gates)
            gradient_vec.append(val)

        norm = float(np.linalg.norm(np.array(gradient_vec)))
        if norm <= gradient_tol:
            break

        max_grad = np.max(np.abs(gradient_vec))
        tied = [pools[i] for i in range(len(pools)) if np.abs(gradient_vec[i]) >= max_grad]
        chosen = tied[int(rng.integers(0, len(tied)))]

        pool_added = [cudaq.pauli_word(term.get_pauli_word(qubits_num)) for term in chosen]
        mixer_pool.append(pool_added)

        num_layer = len(mixer_pool)
        beta = beta + [0.0]
        gamma = gamma + [init_gamma]
        theta0 = np.array(gamma + beta)

        layer_gate_cost = cost_layer_gates + sum(
            _pauli_word_two_qubit_gates(w) for w in pool_added
        )
        full_circuit_gates = sum(
            cost_layer_gates + sum(_pauli_word_two_qubit_gates(w) for w in layer)
            for layer in mixer_pool
        )

        def cost(theta):
            g = theta[:num_layer].tolist()
            b = theta[num_layer:].tolist()
            energy = cudaq.observe(
                adapt_qaoa_kernel, ham, qubits_num, ham_word, ham_coef,
                mixer_pool, g, b, num_layer,
            ).expectation()
            if tracker is not None:
                tracker.record_observe(full_circuit_gates)
            return energy

        def manual_central_diff_jac(theta, h=1e-3):
            grad = np.zeros_like(theta)
            for i in range(len(theta)):
                tp, tm = theta.copy(), theta.copy()
                tp[i] += h
                tm[i] -= h
                grad[i] = (cost(tp) - cost(tm)) / (2 * h)
            return grad

        result = minimize(cost, theta0, method='BFGS', jac=manual_central_diff_jac,
                           tol=bfgs_tol, options={'maxiter': bfgs_maxiter})

        dE = abs(result.fun - E_prev)
        E_prev = result.fun
        result_energy = result.fun
        losses.append(result_energy)
        gamma = result.x[:num_layer].tolist()
        beta = result.x[num_layer:].tolist()

        if dE <= energy_tol:
            break

        state = cudaq.get_state(
            adapt_qaoa_kernel, qubits_num, ham_word,
            ham_coef, mixer_pool, gamma, beta, num_layer,
        )

    if tracker is not None:
        tracker.stop()

    return dict(
        energy=result_energy,
        gamma=gamma,
        beta=beta,
        mixer_pool=mixer_pool,
        num_layers=len(mixer_pool),
        node_map=node_map,
        losses=losses,
        ham_word=ham_word,
        ham_coef=ham_coef
    )


def solve_adapt_qaoa_coloring(
    G, max_layers=5, shots=1000,
    restrict_pool_to_edges=True,
    seed=42, tracker=None, **kwargs
):
    result = optimize_adapt_qaoa(
        G, max_layers=max_layers,
        restrict_pool_to_edges=restrict_pool_to_edges,
        seed=seed, tracker=tracker, **kwargs,
    )
    node_map = result['node_map']
    nodes = list(G.nodes())
    n_qubits = len(nodes)
    num_layer = result['num_layers']

    if num_layer == 0:
        counts = cudaq.sample(_adapt_initial_state, n_qubits, shots_count=shots)
        if tracker is not None:
            tracker.record_sample(0, shots=shots)
    else:
        counts = cudaq.sample(
            adapt_qaoa_kernel, n_qubits, result['ham_word'], result['ham_coef'],
            result['mixer_pool'], result['gamma'], result['beta'], num_layer,
            shots_count=shots,
        )
        if tracker is not None:
            gates = sum(
                standard_qaoa_layer_two_qubit_gates(len(list(G.edges())))
                + sum(_pauli_word_two_qubit_gates(w) for w in layer)
                for layer in result['mixer_pool']
            )
            tracker.record_sample(gates, shots=shots)

    best_bitstring = counts.most_probable()
    coloring = {}
    for node in nodes:
        idx = node_map[node]
        bit_val = best_bitstring[idx]
        coloring[node] = 1 if bit_val == '1' else 0
    return coloring
