import cudaq
from cudaq import spin


def build_operator_pool(n_qubits: int, edges=None, restrict_to_edges: bool = True):
    pool = []

    term = spin.x(0)
    for i in range(1, n_qubits):
        term = term + spin.x(i)
    pool.append(term)

    for i in range(n_qubits):
        pool.append(cudaq.SpinOperator(spin.x(i)))

    if restrict_to_edges and edges is not None:
        pairs = list(edges)
    else:
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    for i, j in pairs:
        pool.append(cudaq.SpinOperator(spin.x(i)) * cudaq.SpinOperator(spin.x(j)))
        pool.append(cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.y(j)))
        pool.append(cudaq.SpinOperator(spin.y(i)) * cudaq.SpinOperator(spin.z(j)))
        pool.append(cudaq.SpinOperator(spin.z(i)) * cudaq.SpinOperator(spin.y(j)))

    return pool


def _term_coefficients(ham) -> list:
    return [term.evaluate_coefficient() for term in ham]


def _term_words(ham, qubits_num) -> list:
    return [term.get_pauli_word(qubits_num) for term in ham]


def _commutator(pools, ham):
    return [ham * op - op * ham for op in pools]


def _pauli_word_two_qubit_gates(word: str) -> int:
    from maxcut.instrumentation.budget import pool_operator_two_qubit_gates
    weight = sum(1 for c in str(word) if c not in ('I', 'i'))
    return pool_operator_two_qubit_gates(weight)
