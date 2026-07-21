import os
import time

import numpy as np

from . import reference as ref
from . import tune

_KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu")
_kernel_cache = {}
_MAX_SAFE_ENERGY_BOUND = 2 ** 30


def _get_kernel(items_per_thread):
    import cupy

    if items_per_thread not in _kernel_cache:
        with open(_KERNEL_SOURCE_PATH) as f:
            source = f.read()
        kernel = cupy.RawKernel(
            source,
            "qbf_search",
            options=(f"-DITEMS_PER_THREAD={items_per_thread}",),
            backend="nvrtc",
        )
        _kernel_cache[items_per_thread] = kernel
    return _kernel_cache[items_per_thread]


def solve_qubo(Q, plan=None, gpu=None, return_timing=False):
    import cupy

    N = Q.shape[0]
    bound = ref.energy_magnitude_bound(Q)
    if bound >= _MAX_SAFE_ENERGY_BOUND:
        raise ValueError(
            f"energy_magnitude_bound(Q) = {bound:.3g} is too close to int32 range "
            f"(kernel accumulates energies in int32 -- see README.md on why int16, the "
            f"paper's default, is not safe here). Scale down Q, or extend the kernel to "
            f"int64 for this input."
        )

    gpu = gpu or tune.DEFAULT_GPU
    if plan is None:
        plan = tune.plan_launch(N, gpu=gpu)
    elif plan.N != N:
        raise ValueError(f"plan was computed for N={plan.N}, but Q has N={N}")
    elif plan.shared_mem_bytes > gpu.max_shared_mem_per_block_optin:
        raise ValueError(
            f"plan.shared_mem_bytes={plan.shared_mem_bytes} exceeds {gpu.name}'s "
            f"{gpu.max_shared_mem_per_block_optin}-byte shared-memory-per-block limit."
        )

    A, B = plan.A, plan.B

    t0 = time.perf_counter()
    E_A_gray, E_B, M_int = ref.precompute_tables(Q, A)
    t1 = time.perf_counter()

    E_A_gray_d = cupy.asarray(E_A_gray, dtype=cupy.int32)
    E_B_d = cupy.asarray(E_B, dtype=cupy.int32)
    M_int_d = cupy.asarray(M_int, dtype=cupy.int32)

    num_blocks = plan.num_blocks
    out_energy = cupy.empty(num_blocks, dtype=cupy.int32)
    out_step = cupy.empty(num_blocks, dtype=cupy.int32)
    out_local = cupy.empty(num_blocks, dtype=cupy.int32)

    kernel = _get_kernel(plan.items_per_thread)
    if plan.needs_shared_mem_optin:
        kernel.max_dynamic_shared_size_bytes = plan.shared_mem_bytes

    cupy.cuda.Device().synchronize()
    t2 = time.perf_counter()

    kernel(
        (num_blocks,), (plan.threads_per_block,),
        (
            E_A_gray_d, M_int_d, E_B_d,
            np.int32(A), np.int64(1 << B),
            out_energy, out_step, out_local,
        ),
        shared_mem=plan.shared_mem_bytes,
    )
    cupy.cuda.Device().synchronize()
    t3 = time.perf_counter()

    out_energy_h = cupy.asnumpy(out_energy)
    out_step_h = cupy.asnumpy(out_step)
    out_local_h = cupy.asnumpy(out_local)
    t4 = time.perf_counter()

    best_block = int(np.argmin(out_energy_h))
    best_energy = int(out_energy_h[best_block])
    best_step = int(out_step_h[best_block])
    best_local = int(out_local_h[best_block])

    block_base = best_block * plan.block_items
    global_suffix_index = block_base + best_local

    prefix_bits = ref.bits_from_index(np.array([ref.gray_code(best_step)]), A)[0]
    suffix_bits = ref.bits_from_index(np.array([global_suffix_index]), B)[0]
    best_bits = np.concatenate([prefix_bits, suffix_bits])

    if return_timing:
        timing = {
            "precompute_s": t1 - t0,
            "h2d_s": t2 - t1,
            "kernel_s": t3 - t2,
            "d2h_s": t4 - t3,
            "states_per_sec": (1 << N) / (t3 - t2) if t3 > t2 else float("inf"),
        }
        return best_energy, best_bits, timing
    return best_energy, best_bits


def benchmark(N, gpu=None, warmup=True):
    rng = np.random.default_rng(0)
    Q = np.triu(rng.integers(-3, 4, size=(N, N)).astype(np.float64))
    plan = tune.plan_launch(N, gpu=gpu or tune.DEFAULT_GPU)

    if warmup:
        solve_qubo(Q, plan=plan)

    energy, bits, timing = solve_qubo(Q, plan=plan, return_timing=True)
    return plan, energy, bits, timing
