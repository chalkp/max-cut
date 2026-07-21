// qbf_cuda/kernel.cu
//
// CUDA kernel for the QBF (QUBO Brute-Force) algorithm: prefix-suffix decomposition +
// Gray-code traversal, from Maltsev et al., "Terastate-per-second QUBO Brute-Force on a
// Single GPU: A Matrix Prefix-Suffix Decomposition" (arXiv:2607.04857v1; source at
// ../../sim/2607.04857v1/qbf_paper.tex). The paper's own optimized CUDA kernel is not
// public -- only their NumPy/CuPy baseline is, per the paper's own text ("the optimized
// solver is provided for experimental access via the Cloud.ru platform") -- so this is
// a from-scratch reimplementation, tuned for the RTX 3080 (Ampere, compute capability
// 8.6) instead of the paper's V100/H100, and using int32 energies instead of the
// paper's int16.
//
// This file is a direct, hardware-aware port of reference.qbf_solve (see
// ../reference.py); every index computation below has an executable, brute-force-
// checked twin there. tests/test_reference.py is the thing that caught the bugs a port
// like this tends to have (a bit-order mismatch between two different MSB/LSB
// conventions, in this case) -- if kernel.cu and reference.py ever disagree, assume
// reference.py is right.
//
// Why int32, not the paper's int16 (see README.md for the full argument): the intended
// caller is RQAOA's contract_graph, which accumulates +-1x-scaled neighbor weights onto
// existing edges every elimination step. Residual-graph edge magnitudes grow with graph
// degree over the course of the recursion, and total energy (a sum of up to ~N^2/2 such
// terms) can exceed int16's +-32767 range well within the N this solver targets. Here
// exactness is the entire point of dropping down to a brute-force solver in the first
// place, so int32 (safe to the edge of any realistic RQAOA residual, checked cheaply on
// the host via reference.energy_magnitude_bound before launch) is the only sane
// default; int16 is left as a possible future fast path for inputs known to be small
// enough, not implemented here.
//
// Memory-hierarchy placement (the one place this port cannot just copy the paper's V100
// design -- see README.md for the full derivation):
//   - E_sd (the running suffix-dependent energy vector) lives in REGISTERS, as a
//     per-thread array of ITEMS_PER_THREAD elements. This works only because the array
//     is indexed exclusively by the fully-unrolled `item` loop below (#pragma unroll on
//     a compile-time-constant trip count) -- every index is then a compile-time
//     constant after unrolling, which is what lets the compiler keep the array in
//     registers instead of spilling it to local memory.
//   - M_int (the prefix-suffix interaction matrix) lives in SHARED memory: one block's
//     column-slice of it, loaded cooperatively once at kernel start. Unlike E_sd, M_int
//     is indexed by `k_var`, the Gray-flip bit position, which changes every Gray step
//     at *runtime* -- a register array can't be indexed by a value the compiler can't
//     prove constant at compile time, so registers were never on the table for M_int.
//     This is the paper's own "the compiler moved data from registers to shared memory"
//     observation (their sec. on GPU profiling) taken as the *starting* design instead
//     of a discovered limitation.
//   - E_A_gray (the prefix self-energy vector) lives in global memory. Every thread in
//     a block reads the same E_A_gray[step] address on every iteration (a uniform,
//     broadcast read), so it costs about one cache-line fill per step regardless of
//     memory tier; what actually matters is *temporal* locality across steps (step,
//     step+1, step+2, ... should stream through DRAM/L2, not scatter), which is exactly
//     what precomputing E_A in Gray-sequence order buys (see reference.precompute_tables
//     and paper sec. 3.3.3).
//
// Bit convention: identical to reference.py -- an integer of width w unpacks MSB-first
// (bit position 0 / column 0 is the *most* significant bit). flip_bit_and_sign's `k`
// (computed here with __ffsll) is an integer-bit position in the *opposite*, LSB=0
// convention, so it is converted to a variable/row index for M_int via `A - 1 - k`
// below -- exactly like reference.qbf_solve. Forgetting that conversion was the first
// bug this project's own reference implementation caught (see tests/test_reference.py):
// it silently indexes the wrong M_int row and produces a wrong but perfectly
// plausible-looking answer, with nothing but a brute-force comparison to catch it.
//
// A is a runtime kernel parameter (0 <= A <= 31, matching the paper's own int32-prefix
// limit), not a compile-time template parameter -- the 2^A Gray-code loop is NOT
// unrolled. Fully unrolling it would make `k_var` compile-time too and could in
// principle let M_int live in registers as well, but only for small A (code size is
// O(2^A) instructions), so that is a possible future fast path, not the default here.
//
// ITEMS_PER_THREAD is a compile-time macro (default 8), overridable via
// `-DITEMS_PER_THREAD=n` at compile time -- see tune.py, which searches over a small
// set of values (8, 4, 2, 1) and reports whichever it picked; solver.py must compile
// (or select an already-compiled variant of) this kernel with the *same* value
// tune.py's LaunchPlan.items_per_thread reports, or the shared-memory layout below and
// the host-side precompute in solver.py disagree about block_items.
//
// Required dynamic shared memory, in ints, that the launcher (solver.py) MUST request:
//   A * block_items + max(block_items, 3 * threads_per_block)
// where block_items = threads_per_block * ITEMS_PER_THREAD. The scratch region (the
// second term) is sized to fit whichever of its two uses is bigger: block_items ints
// for the E_B staging load, or 3 * threads_per_block ints for the (energy, step,
// local_index) reduction buffers. These are equal only when ITEMS_PER_THREAD == 3;
// tune.py's shared_mem_bytes accounts for the max(), not just the E_B staging size --
// under-allocating this is a real out-of-bounds shared-memory write, not just a
// performance bug.

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 8
#endif

extern "C" __global__
void qbf_search(
    const int* __restrict__ E_A_gray,   // [2^A], Gray-sequence order (global memory)
    const int* __restrict__ M_int_g,    // [A][2^B], lexicographic suffix order, row-major (global memory)
    const int* __restrict__ E_B_g,      // [2^B], lexicographic suffix order (global memory)
    int A,
    long long total_suffix_states,      // 2^B -- row stride of M_int_g
    int* __restrict__ out_energy,       // [num_blocks] output: this block's best energy
    int* __restrict__ out_step,         // [num_blocks] output: winning Gray-code step (0..2^A-1)
    int* __restrict__ out_local_index   // [num_blocks] output: winning item*blockDim.x + threadIdx.x
) {
    extern __shared__ int smem[];
    const int block_items = blockDim.x * ITEMS_PER_THREAD;
    const long long block_base = (long long)blockIdx.x * block_items;

    // Shared-memory layout: M_int slice [A][block_items], then a `block_items`-sized
    // scratch region reused for two different, non-overlapping-in-time purposes: first
    // as the E_B staging buffer (read once, at kernel start), later as the block-level
    // argmin reduction buffers (written once, at kernel end). A careful trace shows
    // every thread only ever touches its *own* threadIdx.x-indexed slot in both uses
    // (never another thread's), so the reuse can't actually race -- but the
    // __syncthreads() right after initialization is kept anyway, both as a cheap
    // (one-time) safety margin and so a future edit to the access pattern doesn't
    // silently reintroduce a race that argument no longer covers.
    int* M_int_s = smem;                                    // [A][block_items]
    int* scratch_s = smem + (size_t)A * block_items;         // [block_items]

    for (int k = 0; k < A; ++k) {
        const int* src_row = M_int_g + (size_t)k * total_suffix_states + block_base;
        int* dst_row = M_int_s + (size_t)k * block_items;
        for (int i = threadIdx.x; i < block_items; i += blockDim.x) {
            dst_row[i] = src_row[i];
        }
    }
    for (int i = threadIdx.x; i < block_items; i += blockDim.x) {
        scratch_s[i] = E_B_g[block_base + i];
    }
    __syncthreads();

    int E_sd[ITEMS_PER_THREAD];
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        E_sd[item] = scratch_s[item * blockDim.x + threadIdx.x];
    }
    __syncthreads();  // last read of scratch_s-as-E_B before it is reused for the reduction

    int best_energy = E_sd[0] + E_A_gray[0];
    int best_step = 0;
    int best_item = 0;
    #pragma unroll
    for (int item = 1; item < ITEMS_PER_THREAD; ++item) {
        int candidate = E_sd[item] + E_A_gray[0];
        if (candidate < best_energy) {
            best_energy = candidate;
            best_item = item;
        }
    }

    const long long total_steps = 1LL << A;
    for (long long step = 0; step + 1 < total_steps; ++step) {
        long long flip_source = step + 1;
        int k = __ffsll(flip_source) - 1;                 // trailing-zero-count -> integer-bit position (LSB=0)
        int gray_next = (int)(flip_source ^ (flip_source >> 1));
        int sign = ((gray_next >> k) & 1) ? 1 : -1;
        int k_var = A - 1 - k;                             // integer-bit position -> variable/row index (MSB=0); see file header

        const int* row = M_int_s + (size_t)k_var * block_items;
        int e_a_next = E_A_gray[step + 1];
        int next_step = (int)(step + 1);

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            E_sd[item] += sign * row[item * blockDim.x + threadIdx.x];
            int candidate = E_sd[item] + e_a_next;
            if (candidate < best_energy) {
                best_energy = candidate;
                best_step = next_step;
                best_item = item;
            }
        }
    }

    // Block-level argmin reduction over (best_energy, best_step, local_index). Standard
    // halving-stride shared-memory reduction; local_index (rather than best_item plus a
    // separately-carried thread id) is what rides along, since it already encodes
    // everything the host needs to reconstruct the winning suffix bit pattern and
    // survives the reduction (unlike thread id, which position in the array implicitly
    // gives you only *before* the first reduction round collapses it away).
    int* red_energy = scratch_s;                              // [blockDim.x]
    int* red_step = scratch_s + blockDim.x;                   // [blockDim.x]
    int* red_local = scratch_s + 2 * blockDim.x;              // [blockDim.x]

    red_energy[threadIdx.x] = best_energy;
    red_step[threadIdx.x] = best_step;
    red_local[threadIdx.x] = best_item * blockDim.x + threadIdx.x;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int other = threadIdx.x + stride;
            if (red_energy[other] < red_energy[threadIdx.x]) {
                red_energy[threadIdx.x] = red_energy[other];
                red_step[threadIdx.x] = red_step[other];
                red_local[threadIdx.x] = red_local[other];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_energy[blockIdx.x] = red_energy[0];
        out_step[blockIdx.x] = red_step[0];
        out_local_index[blockIdx.x] = red_local[0];
    }
}
