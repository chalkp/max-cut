import numpy as np


def gray_code(i):
    i = np.asarray(i)
    return i ^ (i >> 1)


def bits_from_index(idx, width):
    idx = np.asarray(idx)
    shifts = np.arange(width - 1, -1, -1)
    return ((idx[..., None] >> shifts) & 1).astype(np.int64)


def index_from_bits(bits):
    bits = np.asarray(bits)
    width = bits.shape[-1]
    weights = 1 << np.arange(width - 1, -1, -1)
    return (bits * weights).sum(axis=-1)


def energy_qubo(Q, X):
    return (X @ Q * X).sum(axis=-1)


def energy_magnitude_bound(Q):
    return float(np.abs(Q).sum())


def flip_bit_and_sign(step):
    flip_source = step + 1
    k = (flip_source & (-flip_source)).bit_length() - 1
    gray_next = flip_source ^ (flip_source >> 1)
    sign = 1 if (gray_next >> k) & 1 else -1
    return k, sign


def _chunked_energies(Q_sub, gray_order, chunk_bits=20):
    M = Q_sub.shape[0]
    total = 1 << M
    chunk_size = min(total, 1 << chunk_bits)
    out = np.empty(total, dtype=np.result_type(Q_sub.dtype, np.float64))
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        idx = np.arange(start, end)
        bits = bits_from_index(gray_code(idx) if gray_order else idx, M)
        out[start:end] = energy_qubo(Q_sub, bits)
    return out


def _chunked_interaction_matrix(Q_ps, B, chunk_bits=20):
    A = Q_ps.shape[0]
    total = 1 << B
    chunk_size = min(total, 1 << chunk_bits)
    out = np.empty((A, total), dtype=np.result_type(Q_ps.dtype, np.float64))
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        idx = np.arange(start, end)
        bits = bits_from_index(idx, B)
        out[:, start:end] = Q_ps @ bits.T
    return out


def precompute_tables(Q, A, chunk_bits=20):
    N = Q.shape[0]
    B = N - A
    assert 0 <= A <= N, f"A={A} out of range for N={N}"

    Q_pp = Q[:A, :A]
    Q_ps = Q[:A, A:]
    Q_ss = Q[A:, A:]

    E_A_gray = _chunked_energies(Q_pp, gray_order=True, chunk_bits=chunk_bits)
    E_B = _chunked_energies(Q_ss, gray_order=False, chunk_bits=chunk_bits)
    M_int = _chunked_interaction_matrix(Q_ps, B, chunk_bits=chunk_bits)

    return E_A_gray, E_B, M_int


def qbf_solve(Q, A):
    N = Q.shape[0]
    B = N - A
    E_A_gray, E_B, M_int = precompute_tables(Q, A)

    total_steps = 1 << A
    E_sd = E_B.copy()

    best_energy = None
    best_step = 0
    best_suffix = 0

    for step in range(total_steps):
        candidate = E_sd + E_A_gray[step]
        local_idx = int(np.argmin(candidate))
        local_best = candidate[local_idx]
        if best_energy is None or local_best < best_energy:
            best_energy = local_best
            best_step = step
            best_suffix = local_idx

        if step + 1 < total_steps:
            k, sign = flip_bit_and_sign(step)
            k_var = A - 1 - k
            E_sd = E_sd + sign * M_int[k_var]

    prefix_bits = bits_from_index(np.array([gray_code(best_step)]), A)[0]
    suffix_bits = bits_from_index(np.array([best_suffix]), B)[0]
    best_bits = np.concatenate([prefix_bits, suffix_bits])

    return best_energy.item() if hasattr(best_energy, "item") else best_energy, best_bits


def brute_force_qubo(Q):
    N = Q.shape[0]
    idx = np.arange(2 ** N)
    X = bits_from_index(idx, N)
    E = energy_qubo(Q, X)
    best = int(np.argmin(E))
    return E[best].item(), X[best]


def ising_to_qubo(J, h=None):
    N = J.shape[0]
    if h is None:
        h = np.zeros(N, dtype=J.dtype)

    row_sums = J.sum(axis=1)
    Q = np.zeros((N, N), dtype=np.result_type(J.dtype, h.dtype, np.float64))
    for i in range(N):
        Q[i, i] = -2 * h[i] - 2 * row_sums[i]
    iu = np.triu_indices(N, k=1)
    Q[iu] = 4 * J[iu]

    const = float(h.sum() + np.triu(J, k=1).sum())
    return Q, const


def qubo_bits_to_ising_spins(x):
    return 1 - 2 * np.asarray(x)
