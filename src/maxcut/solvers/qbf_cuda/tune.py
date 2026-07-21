from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    compute_capability: tuple
    sm_count: int
    max_threads_per_block: int
    max_threads_per_sm: int
    max_shared_mem_per_block_default: int
    max_shared_mem_per_block_optin: int
    max_shared_mem_per_sm: int
    regs_per_sm: int
    total_global_mem_bytes: int
    max_grid_dim_x: int = 2_147_483_647
    warp_size: int = 32


MX550 = GPUSpec(
    name="NVIDIA GeForce MX550",
    compute_capability=(7, 5),
    sm_count=16,
    max_threads_per_block=1024,
    max_threads_per_sm=1024,
    max_shared_mem_per_block_default=48 * 1024,
    max_shared_mem_per_block_optin=64 * 1024,
    max_shared_mem_per_sm=64 * 1024,
    regs_per_sm=65536,
    total_global_mem_bytes=1_773_142_016,
)
RTX_3080_10GB = GPUSpec(
    name="NVIDIA GeForce RTX 3080 10GB (GA102-200)",
    compute_capability=(8, 6),
    sm_count=68,
    max_threads_per_block=1024,
    max_threads_per_sm=1536,
    max_shared_mem_per_block_default=48 * 1024,
    max_shared_mem_per_block_optin=99 * 1024,
    max_shared_mem_per_sm=100 * 1024,
    regs_per_sm=65536,
    total_global_mem_bytes=10 * 1024 ** 3,
)

RTX_3080_12GB = GPUSpec(
    name="NVIDIA GeForce RTX 3080 12GB (GA102-220)",
    compute_capability=(8, 6),
    sm_count=70,
    max_threads_per_block=1024,
    max_threads_per_sm=1536,
    max_shared_mem_per_block_default=48 * 1024,
    max_shared_mem_per_block_optin=99 * 1024,
    max_shared_mem_per_sm=100 * 1024,
    regs_per_sm=65536,
    total_global_mem_bytes=12 * 1024 ** 3,
)

DEFAULT_GPU = MX550

ENERGY_DTYPE_BYTES = 4
VRAM_SAFETY_FRACTION = 0.6


@dataclass(frozen=True)
class LaunchPlan:
    N: int
    A: int
    B: int
    threads_per_block: int
    items_per_thread: int
    num_blocks: int
    block_items: int
    shared_mem_bytes: int
    total_gray_steps: int  # 2**A -- the sequential critical path length
    needs_shared_mem_optin: bool
    vram_bytes: int        # E_A_gray + E_B + M_int + output arrays, all on-device

    def summary(self) -> str:
        return (
            f"N={self.N}: A={self.A} (2^A={self.total_gray_steps:,} sequential Gray steps), "
            f"B={self.B} (2^B={1 << self.B:,} suffix states), "
            f"{self.num_blocks:,} blocks x {self.threads_per_block} threads x "
            f"{self.items_per_thread} items/thread, "
            f"{self.shared_mem_bytes / 1024:.1f} KB shared mem/block"
            + (" (opt-in required)" if self.needs_shared_mem_optin else "")
            + f", {self.vram_bytes / 1e6:.1f} MB VRAM"
        )


def _vram_bytes(A, B, num_blocks):
    """On-device memory precompute_tables + solver.py must hold at once: E_A_gray
    (2**A), E_B (2**B), M_int (A * 2**B), plus the tiny per-block output arrays. All
    int32 (ENERGY_DTYPE_BYTES)."""
    e_a = (1 << A) * ENERGY_DTYPE_BYTES
    e_b = (1 << B) * ENERGY_DTYPE_BYTES
    m_int = A * (1 << B) * ENERGY_DTYPE_BYTES
    out_arrays = num_blocks * 3 * ENERGY_DTYPE_BYTES
    return e_a + e_b + m_int + out_arrays


def _candidates_for(N, gpu, threads_per_block, items_per_thread, min_gray_steps,
                     max_prefix_bits, shared_budget, vram_budget):
    block_items = threads_per_block * items_per_thread
    log2_block_items = block_items.bit_length() - 1
    if log2_block_items > N:
        return []

    out = []
    for B in range(log2_block_items, N + 1):
        A = N - B
        if A > max_prefix_bits:
            continue
        # A=0 (single Gray step, i.e. B == N) is exempt from the min_gray_steps floor:
        # with no prefix bits at all there is no interaction matrix to amortize a load
        # over in the first place (M_int has zero rows), so the floor's rationale
        # doesn't apply. Any A strictly between 0 and the floor is still excluded.
        if A > 0 and (1 << A) < min_gray_steps:
            continue
        # Scratch region sized for whichever of its two reused purposes is bigger: the
        # E_B staging load (block_items ints) or the (energy, step, local_index)
        # reduction buffers (3 * threads_per_block ints) -- see kernel.cu's header
        # comment. Only equal when items_per_thread == 3.
        scratch_items = max(block_items, 3 * threads_per_block)
        shared_mem_bytes = A * block_items * ENERGY_DTYPE_BYTES
        shared_mem_bytes += scratch_items * ENERGY_DTYPE_BYTES
        if shared_mem_bytes > shared_budget:
            continue
        num_blocks = (1 << B) // block_items
        vram_bytes = _vram_bytes(A, B, num_blocks)
        if vram_bytes > vram_budget:
            continue
        out.append((A, B, num_blocks, shared_mem_bytes, threads_per_block, items_per_thread, vram_bytes))
    return out


def plan_launch(
    N: int,
    gpu: GPUSpec = None,
    threads_per_block: int = 256,
    items_per_thread: int = 8,
    min_gray_steps: int = 1024,
    max_blocks_per_sm_target: int = 16,
    max_prefix_bits: int = 31,
    allow_shared_mem_optin: bool = True,
    vram_safety_fraction: float = VRAM_SAFETY_FRACTION,
) -> LaunchPlan:
    """Choose (A, B, num_blocks, items_per_thread) for a size-N problem on `gpu`
    (defaults to MX550, the GPU actually present on this machine).

    `threads_per_block` and `items_per_thread` set block_items = threads_per_block *
    items_per_thread, the number of suffix states one block owns. Both are treated as
    ceilings, not fixed values -- `items_per_thread`'s halvings (4, 2, 1, ...) are tried
    because the single value that keeps register pressure low is not always the one
    that produces enough blocks to cover every SM (small N + a fixed large block_items
    can starve the grid down to a handful of blocks; MX550 has only 16 SMs, so this
    matters at smaller N than it did when this module assumed a 68-SM 3080);
    `threads_per_block`'s halvings are tried for the same reason at the opposite
    extreme -- below N=8, 256 threads alone already exceeds 2**N, and no
    items_per_thread can fix that.

    For each items_per_thread candidate, every B for which block_items divides 2**B
    evenly is tried; A = N - B survives if:
      - A <= max_prefix_bits (int32 prefix representation, matching the paper's own
        A <= 31 limit, for the same reason: fast native int32 arithmetic for the
        Gray-step counter and prefix bit-position math),
      - A * block_items * ENERGY_DTYPE_BYTES (+ scratch) fits the shared-memory budget
        (opt-in budget if `allow_shared_mem_optin`, else the no-opt-in 48 KB default;
        MX550's opt-in ceiling is only 64 KB, vs. 99 KB on the 3080),
      - 2**A >= min_gray_steps, a documented rule-of-thumb floor (not a measured one --
        see module docstring) meant to keep the one-time cooperative load of M_int into
        shared memory a small fraction of a block's total work,
      - E_A_gray + E_B + M_int + output arrays fit within
        `vram_safety_fraction * gpu.total_global_mem_bytes` -- essentially never binding
        on a 10-12 GB 3080, but a real constraint on MX550's ~1.77 GB, especially with
        cudaq's own "nvidia" QAOA simulation target sharing the same device budget.

    Scoring, across all surviving (items_per_thread, A, B) candidates, in priority
    order: (1) avoid under-utilization -- num_blocks below gpu.sm_count leaves whole
    SMs idle for the entire kernel and dominates any other consideration; (2) avoid
    wildly over-subscribing (num_blocks past max_blocks_per_sm_target * sm_count just
    queues extra waves without shortening the critical path); (3) minimize A, i.e.
    prefer the shortest sequential Gray-code critical path among what's left.
    """
    gpu = gpu or DEFAULT_GPU
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if items_per_thread <= 0 or (items_per_thread & (items_per_thread - 1)) != 0:
        raise ValueError("items_per_thread must be a power of two")

    shared_budget = (
        gpu.max_shared_mem_per_block_optin if allow_shared_mem_optin
        else gpu.max_shared_mem_per_block_default
    )
    vram_budget = int(gpu.total_global_mem_bytes * vram_safety_fraction)
    max_blocks_target = gpu.sm_count * max_blocks_per_sm_target

    # Both threads_per_block and items_per_thread are searched, not just the latter:
    # block_items = threads_per_block * items_per_thread must not exceed 2**N (each
    # thread needs at least 1 item). Below N=8, threads_per_block=256 makes that
    # impossible regardless of items_per_thread (256 * 1 already exceeds 2**N) -- a
    # real bug this module shipped with, caught only once the MX550 sibling copy
    # (../../../sim/qbf_cuda/tune.py) was actually run against N < 8. Goes all the way
    # to 1 rather than stopping at gpu.warp_size: at the N this small, total work is a
    # handful of states and kernel-launch overhead dominates regardless of occupancy,
    # so there's no real tradeoff being given up by under-filling a warp.
    candidates = []
    tpb = threads_per_block
    while tpb >= 1:
        ipt = items_per_thread
        while ipt >= 1:
            candidates.extend(_candidates_for(
                N, gpu, tpb, ipt, min_gray_steps, max_prefix_bits,
                shared_budget, vram_budget,
            ))
            ipt //= 2
        tpb //= 2

    if not candidates:
        raise ValueError(
            f"No feasible (A, B) split for N={N} within a {shared_budget/1024:.0f} KB "
            f"shared-mem budget and {vram_budget/1e6:.0f} MB VRAM budget, even after "
            f"shrinking threads_per_block and items_per_thread down to 1. On {gpu.name}, "
            f"check whether raising vram_safety_fraction or allow_shared_mem_optin is "
            f"safe before assuming this N is out of reach."
        )

    def score(c):
        A, B, num_blocks, shared_mem_bytes, tpb, ipt, vram_bytes = c
        under_util = max(0, gpu.sm_count - num_blocks)
        over_sub = max(0, num_blocks - max_blocks_target)
        return (under_util > 0, under_util, over_sub > 0, over_sub, A)

    candidates.sort(key=score)
    A, B, num_blocks, shared_mem_bytes, chosen_tpb, chosen_ipt, vram_bytes = candidates[0]

    if num_blocks > gpu.max_grid_dim_x:
        raise ValueError(f"num_blocks={num_blocks} exceeds grid dimension limit {gpu.max_grid_dim_x}")

    return LaunchPlan(
        N=N,
        A=A,
        B=B,
        threads_per_block=chosen_tpb,
        items_per_thread=chosen_ipt,
        num_blocks=num_blocks,
        block_items=chosen_tpb * chosen_ipt,
        shared_mem_bytes=shared_mem_bytes,
        total_gray_steps=1 << A,
        needs_shared_mem_optin=shared_mem_bytes > gpu.max_shared_mem_per_block_default,
        vram_bytes=vram_bytes,
    )


def query_device_gpu_spec(device_id: int = 0) -> GPUSpec:
    """Build a GPUSpec from the actual device cupy sees at runtime -- used in
    tests/test_tune.py to confirm the MX550 constant above matches what cupy itself
    reports, rather than trusting the hand-transcribed devquery.cu dump alone."""
    import cupy  # local import: tune.py itself has no hard cupy dependency

    props = cupy.cuda.runtime.getDeviceProperties(device_id)
    major, minor = props["major"], props["minor"]
    return GPUSpec(
        name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
        compute_capability=(major, minor),
        sm_count=props["multiProcessorCount"],
        max_threads_per_block=props["maxThreadsPerBlock"],
        max_threads_per_sm=props["maxThreadsPerMultiProcessor"],
        max_shared_mem_per_block_default=props["sharedMemPerBlock"],
        max_shared_mem_per_block_optin=props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"]),
        max_shared_mem_per_sm=props["sharedMemPerMultiprocessor"],
        regs_per_sm=props["regsPerMultiprocessor"],
        total_global_mem_bytes=props["totalGlobalMem"],
    )


if __name__ == "__main__":
    for n in [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]:
        try:
            plan = plan_launch(n)
            print(plan.summary())
        except ValueError as e:
            print(f"N={n}: infeasible -- {e}")
            break
