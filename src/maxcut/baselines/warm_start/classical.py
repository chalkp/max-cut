import numpy as np
import cvxpy as cp
import networkx as nx


def goemans_williamson_relax(G: nx.Graph, seed: int = 42) -> np.ndarray:
    nodes = list(G.nodes())
    n = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_map[u], node_map[v]
        w = data.get('weight', 1.0)
        W[i, j] = W[j, i] = w

    Y = cp.Variable((n, n), PSD=True)
    objective = cp.Maximize(cp.sum(cp.multiply(W, (1 - Y))) / 4)
    constraints = [cp.diag(Y) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f'GW SDP relaxation did not solve to optimality: {prob.status}')

    return Y.value


def _extract_unit_vectors(Y: np.ndarray) -> np.ndarray:
    Y_sym = (Y + Y.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Y_sym)
    eigvals = np.clip(eigvals, 0, None)
    V = eigvecs @ np.diag(np.sqrt(eigvals))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def random_hyperplane_rounding(
    Y: np.ndarray, edges_source, edges_target, edge_weights,
    seed: int = 42, n_trials: int = 50
) -> np.ndarray:
    V = _extract_unit_vectors(Y)
    n = V.shape[0]
    rng = np.random.default_rng(seed)

    best_cut = -1.0
    best_x = None
    src, tgt, w = np.array(edges_source), np.array(edges_target), np.array(edge_weights)

    for _ in range(n_trials):
        r = rng.normal(size=n)
        r /= np.linalg.norm(r)
        signs = np.sign(V @ r)
        signs[signs == 0] = 1.0
        x = (signs > 0).astype(float)  # {0, 1}

        cut = np.sum(w[x[src] != x[tgt]])
        if cut > best_cut:
            best_cut = cut
            best_x = x

    return best_x


def regularize(c: np.ndarray, epsilon: float = 0.25) -> np.ndarray:
    return np.clip(c, epsilon, 1 - epsilon)
