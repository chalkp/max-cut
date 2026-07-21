"""
GPU correctness gate: solver.solve_qubo (the actual CUDA kernel, launched on this
machine's MX550 -- see tune.py's module docstring on why MX550, not the RTX 3080 this
package was originally written for) vs. reference.qbf_solve (the validated NumPy
spec), across random QUBOs and RQAOA-realistic contracted-residual graphs. Requires
cupy -- run via `mamba run -n cudaq python3 test_solver_gpu.py`.

Unlike tests/test_reference.py and tests/test_maxcut_adapter.py (which check the
algorithm), this file checks the actual on-device kernel: it is the one thing that
compiling with `nvcc -Xptxas -v` and reasoning about the reference implementation
cannot substitute for. This package's kernel.cu source is identical to the sibling
../../../sim/qbf_cuda/ copy (already validated there); this file exists to confirm the
same kernel, compiled and launched through *this* package's tune.py/solver.py (now
MX550-tuned too), behaves identically when invoked the way rqaoa.py actually calls it.
"""
import sys
import os

import numpy as np
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # for the sibling test_maxcut_adapter.py's contract_graph

# maxcut is editable-installed (pip install -e .), so these resolve from anywhere.
from maxcut.solvers.qbf_cuda import reference as ref  # noqa: E402
from maxcut.solvers.qbf_cuda import solver  # noqa: E402
from maxcut.solvers.qbf_cuda import tune  # noqa: E402
from maxcut.solvers.qbf_cuda import maxcut_adapter as adapter  # noqa: E402
from maxcut.solvers.brute_force import brute_force as _brute_force  # noqa: E402
from test_maxcut_adapter import contract_graph  # noqa: E402


class utils:  # thin namespace so the historical `utils.brute_force(...)` call sites read unchanged
    brute_force = staticmethod(_brute_force)


def test_gpu_matches_reference_random_qubo():
    """Every N from 1 to 24 (crossing the tiny-N threads_per_block-search boundary,
    the A=0 / A>0 boundary, and the shared-mem opt-in boundary), 3 random trials each."""
    rng = np.random.default_rng(0)
    checks = 0
    for n in [1, 2, 3, 4, 5, 8, 10, 13, 16, 18, 20, 22, 24]:
        for _ in range(3):
            Q = np.triu(rng.integers(-4, 5, size=(n, n)).astype(np.float64))
            ref_energy, _ = ref.qbf_solve(Q, A=min(n, 6))
            plan = tune.plan_launch(n)
            gpu_energy, gpu_bits = solver.solve_qubo(Q, plan=plan)
            checks += 1
            assert gpu_energy == ref_energy, f"n={n}: kernel {gpu_energy} != reference {ref_energy}"
            recomputed = ref.energy_qubo(Q, gpu_bits[None, :])[0]
            assert np.isclose(recomputed, gpu_energy), (
                f"n={n}: returned bits evaluate to {recomputed}, kernel claimed {gpu_energy}"
            )
    print(f"  ({checks} (N, trial) configurations checked)")


def _minimal_plan(N, A, gpu=None):
    """A valid LaunchPlan forcing a *specific* A -- see the sim copy's identical
    helper for the full rationale (it reuses tune._candidates_for's feasibility
    checks rather than re-deriving them, after an earlier hand-rolled version
    overflowed the shared-mem budget and surfaced as a raw CUDA driver error)."""
    gpu = gpu or tune.MX550
    shared_budget = gpu.max_shared_mem_per_block_optin
    vram_budget = gpu.total_global_mem_bytes  # generous: this helper is for correctness, not realistic sizing
    for threads in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
        for items in [8, 4, 2, 1]:
            for c in tune._candidates_for(N, gpu, threads, items, min_gray_steps=0,
                                           max_prefix_bits=31, shared_budget=shared_budget,
                                           vram_budget=vram_budget):
                cA, cB, num_blocks, shared_mem_bytes, tpb, ipt, vram_bytes = c
                if cA == A:
                    return tune.LaunchPlan(
                        N=N, A=A, B=N - A, threads_per_block=tpb, items_per_thread=ipt,
                        num_blocks=num_blocks, block_items=tpb * ipt,
                        shared_mem_bytes=shared_mem_bytes, total_gray_steps=1 << A,
                        needs_shared_mem_optin=shared_mem_bytes > gpu.max_shared_mem_per_block_default,
                        vram_bytes=vram_bytes,
                    )
    raise ValueError(f"no valid plan found for N={N}, A={A}")


def test_gpu_matches_reference_multiple_A_choices():
    """Same N, every possible A split -- confirms the kernel doesn't depend on
    tune.py's specific choice of A being the one that happens to work, and exercises
    the k_var = A - 1 - k conversion (kernel.cu's header comment) across a range of A."""
    rng = np.random.default_rng(1)
    N = 16
    Q = np.triu(rng.integers(-4, 5, size=(N, N)).astype(np.float64))
    ref_energy, _ = ref.qbf_solve(Q, A=5)

    for A in range(0, N + 1):
        plan = _minimal_plan(N, A)
        gpu_energy, gpu_bits = solver.solve_qubo(Q, plan=plan)
        assert gpu_energy == ref_energy, f"A={A}: kernel {gpu_energy} != reference {ref_energy}"
    print(f"  ({N + 1} distinct A values (0..{N}) checked at N={N})")


def test_gpu_matches_utils_brute_force_contracted_residual():
    """The RQAOA-realistic case: negative and growing-magnitude weights from repeated
    contract_graph elimination, run through the full graph -> Ising -> QUBO -> kernel
    -> cut-value pipeline (maxcut_adapter), checked against the ACTUAL utils.brute_force."""
    checks = 0
    for n0, m, seed, n_contractions in [(10, 2, 1, 2), (14, 3, 2, 5), (16, 3, 3, 6), (18, 2, 4, 7)]:
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

        ref_cut, _, _ = utils.brute_force(G)
        cut, edges, subset = adapter.brute_force_qbf_cuda(G, backend='cuda')
        checks += 1
        assert np.isclose(cut, ref_cut, atol=1e-6), (
            f"n0={n0} seed={seed}: kernel cut {cut} != utils.brute_force {ref_cut}"
        )
    print(f"  ({checks} contracted-residual graphs checked end to end on the GPU)")


if __name__ == "__main__":
    tests = [
        test_gpu_matches_reference_random_qubo,
        test_gpu_matches_reference_multiple_A_choices,
        test_gpu_matches_utils_brute_force_contracted_residual,
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
