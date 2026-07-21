from maxcut.solvers.brute_force import brute_force
from maxcut.solvers.qbf_cuda.maxcut_adapter import brute_force_qbf_cuda


def _solve_residual_exactly(graph, exact_solver):
    if exact_solver == 'numpy':
        return brute_force(graph)
    elif exact_solver == 'qbf_cuda':
        return brute_force_qbf_cuda(graph, backend='cuda')
    elif exact_solver == 'qbf_cuda_reference':
        return brute_force_qbf_cuda(graph, backend='reference')
    else:
        raise ValueError(
            f"unknown exact_solver {exact_solver!r}, expected 'numpy', 'qbf_cuda', "
            f"or 'qbf_cuda_reference'"
        )
