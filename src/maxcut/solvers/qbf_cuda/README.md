# qbf_cuda

A CUDA-accelerated exact QUBO/Ising solver, based on Maltsev et al., ["Terastate-per-
second QUBO Brute-Force on a Single GPU: A Matrix Prefix-Suffix
Decomposition"](../../sim/2607.04857v1/qbf_paper.tex) (arXiv:2607.04857v1). It exists to
replace `utils.brute_force` as the exact solver for RQAOA's terminal (residual) graph,
so `cutoff` can be raised from the current default of 3 to somewhere in the region of
30-45 nodes.

**Status: implemented, and now running on real hardware.** Originally designed and
tuned for an RTX 3080 with no GPU available at all to test against ("Status" note at
the time: "implemented and tested on CPU/NumPy; not yet run on a GPU" -- see git
history). `tune.py`/`solver.py` have since been updated with a second `GPUSpec` for the
**MX550** actually present on this machine (measured directly via
`cudaGetDeviceProperties`, not assumed -- see `../../sim/qbf_cuda/README.md` for the
full device-query dump), which is now `DEFAULT_GPU`; `RTX_3080_10GB`/`RTX_3080_12GB`
remain defined for whenever that hardware is available. The kernel has been compiled,
launched, and validated end to end on the MX550 (`tests/test_solver_gpu.py`), and a
real RQAOA benchmark comparing it against the numpy baseline is below ("Benchmark:
qbf_cuda vs. the numpy baseline"). See "What has and hasn't been validated" for the
full matrix of what's been run vs. only reasoned about.

## Why this exists

`utils.brute_force(G)` materializes all `2^n` assignments as a NumPy array
(`(np.arange(2**n)[:, None] & (1 << np.arange(n))) > 0`) -- `O(2^n · n)` memory and
compute. That's fine at RQAOA's default `cutoff=3` (8 states), but it cannot scale much
past `n≈20-25` (2^25 × 25 × 8 bytes ≈ 6.7 GB; 2^30 is already ~257 GB). That ceiling is
almost certainly *why* `cutoff` defaults to 3 rather than something that would let RQAOA
stop its recursive QAOA loop earlier.

Every extra elimination step RQAOA takes past a size where an exact solver could just
finish the job is: one more expensive QAOA optimization round (statevector simulation +
COBYLA), and one more greedy correlation-based decision made on a smaller, noisier
graph -- exactly where `strategy.txt`'s own notes on "Adaptive Depth Scaling" flag that
"errors propagate the most." Raising `cutoff` to whatever an exact solver can still
handle quickly trades both of those costs for one GPU brute-force pass with a
correctness *guarantee*, not a QAOA approximation, over the residual graph.

The paper's prefix-suffix decomposition + Gray-code algorithm reaches `O(1)` amortized
work per state and (per the paper) 7.5 × 10¹² states/sec on an H100 and 2.33 × 10¹²
states/sec on a V100 -- either would make `n≈40-45` a sub-minute exact solve. The
paper's own optimized CUDA kernel isn't public (only their NumPy/CuPy baseline is, per
the paper's text), so `kernel.cu` here is a from-scratch reimplementation of the
algorithm described in the paper's sections 2-4, tuned for a different, more accessible
GPU (RTX 3080, Ampere) instead of the paper's V100/H100.

## Two deliberate departures from the paper

### int32 energies, not int16

The paper uses int16 energy accumulators: it halves register/memory footprint and
(they claim) roughly doubles throughput, at the cost of needing an external
simulated-annealing pass to bound the energy spectrum and guarantee no overflow.

That trade only makes sense on "clean" instances. RQAOA's `contract_graph`
(`rqaoa.py`) accumulates `correlation_sign * weight_v_neighbor` onto existing (or new)
edges every elimination step -- residual-graph edge magnitudes grow roughly with node
degree over the course of the recursion, and total energy (a sum of up to `~n²/2` such
terms, and the Ising→QUBO conversion itself multiplies couplings by 4x, see
`reference.ising_to_qubo`) can exceed int16's ±32767 range well within the `n` this
solver targets. Since the entire point of dropping to a brute-force solver here is
*exactness*, not peak throughput, silently overflowing would defeat the purpose.
int32 (range ±2.1 billion) is the default; overflow is checked cheaply before every
solve via `reference.energy_magnitude_bound(Q)` (a `sum(|Q_ij|)` bound -- loose, but
correct and O(n²)), not via an SA pass. An int16 fast path for inputs known to be small
enough is a plausible future addition, not implemented here.

### M_int in shared memory, not registers

The paper's V100/H100 kernel keeps the entire interaction matrix `M_int` register-
resident, indexed by `k`, the Gray-code flip-bit position, which changes every step.
That doesn't port: **a register array indexed by a runtime-varying value cannot stay in
registers** -- the compiler has no choice but to demote it to local (i.e. off-chip)
memory the moment it can't prove every index is a compile-time constant. This is, in
fact, exactly what the paper's own profiling section reports happening on their
hardware ("the compiler moved data from registers to shared memory, making the number
of registers per warp the limiting factor") -- described there as an unexplained
shortfall from theoretical peak, taken here as the starting design instead:

- **`E_sd`** (the running suffix-dependent energy vector) lives in **registers**, as a
  per-thread `int[ITEMS_PER_THREAD]` array. This works because it's indexed *only* by
  the fully-unrolled `item` loop (`#pragma unroll` over a compile-time constant trip
  count) -- every access is a compile-time-constant index after unrolling.
- **`M_int`** lives in **shared memory**: one block's column-slice, loaded
  cooperatively once at kernel start. Shared memory has no compile-time-index
  restriction, so the runtime-varying `k_var` is a non-issue there.
- **`E_A`** (prefix self-energy) lives in **global memory**, read once per Gray step as
  a uniform/broadcast value across the whole block -- cheap regardless of memory tier,
  provided access is temporally sequential (hence precomputing it in Gray-sequence
  order, matching the paper's own sec. 3.3.3 reasoning, just applied to a different
  tier of the memory hierarchy than the paper needed it for).

`kernel.cu`'s header comment has the full derivation. Practical upshot, confirmed by
compiling the kernel (`nvcc -Xptxas -v`, see below): **0 bytes of register spill** at
every `ITEMS_PER_THREAD` the tuner can select, and only 32-40 registers/thread --
occupancy here is shared-memory-limited, not register-limited, which is a very
different (and much less register-starved) regime than the paper's deliberately
register-packed, occupancy-sacrificing V100 configuration (16384 threads for 5120 FP32
cores, "3.2 threads per core").

### Chunked host-side precompute (not in the paper -- an artifact of reusing NumPy at this scale)

`precompute_tables` (in `reference.py`, called by both the pure-Python path and
`solver.py`'s CUDA path) needs `E_A_gray`, a `2^A`-length array. The straightforward way
to build it -- unpack all `2^A` Gray codewords into a `(2^A, A)` bit-matrix, then one
`energy_qubo` call -- costs `O(2^A · A)` *memory*, not just time. That's under a GB for
`A` up to ~20, but `tune.py` picks `A` as large as 26 for `N` in the low 40s (register/
shared-memory feasibility, not this concern, drives that choice), and `2^26 · 26 · 8`
bytes is ~14 GB: a real host-memory crash, not a slowdown, at exactly the `N` this
module documents as supported. The fix (`_chunked_energies` /
`_chunked_interaction_matrix` in `reference.py`) processes at most `2^20` states at a
time, bounding peak memory to a few hundred MB regardless of `A`, while still producing
the same full `2^A`-length output (unavoidably ~500 MB at `A=26`, which is fine) via a
simple loop -- no change in what's computed, only how much is held at once. `chunk_bits`
defaults to 20 (a no-op single chunk for every `A`, `B` ≤ 20, which covers the entire
existing test suite unchanged) and is a keyword on `precompute_tables` if a different
budget is ever needed. `energy_qubo` itself was also switched from `np.einsum` to an
explicit `(X @ Q * X).sum(-1)`, ~2.6x faster in practice for this contraction shape,
which matters once it's called per-chunk at this scale.

Verified directly (not merely argued): with both fixes applied,
`precompute_tables(Q, A)` for `tune.plan_launch`'s own choice of `A` completes in 1.2s /
575 MB peak at N=40 (A=20), and 37.4s / 1.4 GB peak at N=45 (A=26) -- the top of the
documented range, and comfortably far from the 14 GB the naive approach would have
needed there. This is a one-time host-side cost paid once per exact solve (not part of
the kernel's own per-state work), so it's a reasonable price next to the QAOA rounds it
replaces, but it is real wall-clock time worth knowing about before a `cutoff` sweep.

## What's explicitly out of scope for this pass

Per explicit scoping direction, the following were cut from v1 as unnecessary
complexity/risk for code that can't be run against real hardware yet:

- **The paper's `N > 49`-ish embarrassingly-parallel outer split** (their sec. 4.4,
  needed because their int16/register design hit a hard ceiling around N=49-55). RQAOA
  cutoffs in the range this is meant for (see below) don't approach a single launch's
  capacity, so this isn't implemented; `tune.plan_launch` raises a clear error if no
  single-launch configuration fits.
- **int16 energies** (see above) -- int32 only.
- **Any change to RQAOA's algorithm itself** (beam search, adaptive-`p`, batch
  elimination, etc.). This only adds a terminal-solver hook; `strategy.txt`'s other
  proposed strategies are untouched.

## Architecture

```
reference.py       Pure-NumPy algorithm reference: prefix-suffix decomposition + Gray
                    code, exactly mirroring kernel.cu's index math (same Gray-sequence
                    reordering, same MSB-first bit convention, same k -> A-1-k row
                    conversion for M_int). This is the correctness spec -- if kernel.cu
                    and reference.py ever disagree, reference.py is assumed right.
                    No GPU required; tests/test_reference.py runs it against brute
                    force on hundreds of configurations.

tune.py             RTX 3080 launch-configuration heuristic: chooses the A/B
                    prefix/suffix split, items_per_thread, and shared-memory budget for
                    a given N. Pure host-side arithmetic, no GPU required to run.
                    Explicitly NOT an empirically-tuned model (see its own docstring) --
                    a documented starting point for `nsys`/`ncu`-guided tuning on real
                    hardware, the way the paper's own V100/H100 parameters were reached.

kernel.cu           The CUDA kernel (qbf_search). Compiles cleanly for sm_86 via nvcc
                    (see below) with 0 register spill across every ITEMS_PER_THREAD the
                    tuner can select -- not yet executed on any GPU.

solver.py           CuPy RawKernel launcher: precomputes E_A/E_B/M_int (same math as
                    reference.precompute_tables), compiles+launches kernel.cu, reduces
                    per-block results on the host, reconstructs the winning bitstring.
                    Importable without cupy installed (only calling solve_qubo()
                    requires it) -- not executed against a GPU as part of writing it.

maxcut_adapter.py   networkx Max-Cut <-> Ising <-> QUBO conversion, and
                    brute_force_qbf_cuda(G, backend=...) -- a drop-in replacement for
                    utils.brute_force(G) with the same (cut_value, cut_edges,
                    max_subset) return. backend='reference' needs no GPU; backend='cuda'
                    routes through solver.py.

tests/              Runnable now, with `python3 <file>.py` (no pytest, no GPU):
  test_reference.py       267 (n, A) configurations: random QUBO, random Ising (with/
                           without field), and graphs built by literally running
                           rqaoa.contract_graph a few times (duplicated locally to avoid
                           a hard cudaq dependency, same pattern already used by
                           fixstars_test/rqaoa_amplify.py) -- the negative/growing-
                           magnitude weight regime this solver is actually meant for.
  test_maxcut_adapter.py  Cross-checks brute_force_qbf_cuda(backend='reference')
                           against the ACTUAL utils.brute_force (imported directly from
                           the parent project, not reimplemented), including
                           non-integer-weight rejection and empty/trivial graphs.
  test_tune.py             Added after a real bug surfaced: plan_launch raised for
                           every N < 8 (threads_per_block was fixed at 256, so
                           block_items could never shrink below it; caught only once
                           the MX550 sibling copy at ../../../sim/qbf_cuda/ was
                           actually run against small N on real hardware -- see that
                           copy's README). Fixed here too; this file is the regression
                           test for it plus basic shared-mem-budget/utilization sanity
                           checks, all GPU-free.
```

## What has and hasn't been validated

**Run and passing, on the actual MX550** (`mamba run -n cudaq python3 tests/test_solver_gpu.py`):
- The CUDA kernel itself, launched via this package's own `solver.py`/`tune.py` (not
  just the sibling `../../sim/qbf_cuda/` copy): matches `reference.qbf_solve` exactly
  across N=1-24 (39 configurations), every possible A split 0..16 at a fixed N=16 (the
  `k_var = A - 1 - k` conversion's full range), and the full graph → Ising → QUBO →
  kernel → cut-value pipeline against the real `utils.brute_force` on 4
  `contract_graph`-shaped residuals (negative, growing-magnitude weights).
- The actual RQAOA integration: `rqaoa._solve_residual_exactly(G, 'qbf_cuda')` called
  directly and compared against `'numpy'` on a real graph -- identical cut value.
- A real RQAOA benchmark (`experiment_qbf_cuda_vs_numpy.py`, N=20, 5 seeds x layer
  count 1/2/3): see "Benchmark: qbf_cuda vs. the numpy baseline" below.

**Run and passing** (pure NumPy/Python, no GPU, no cupy):
- `tests/test_reference.py` -- 7 test groups, 267 (n, A) configurations, exact energy
  match against brute force for both QUBO and Ising inputs, including graphs shaped by
  real `contract_graph` calls.
- `tests/test_tune.py` -- 6 test groups, confirms `plan_launch` succeeds for every N
  from 1 to 45 on both `MX550` and `RTX_3080_10GB` (regression test for the tiny-N bug
  above), shared-mem/VRAM-budget and utilization sanity checks, and (when cupy is
  importable) that the `MX550` constant matches a live `cudaGetDeviceProperties` query.
- `tests/test_maxcut_adapter.py` -- 5 test groups, 73 (graph, A) configurations, cut
  value matches the real `utils.brute_force` exactly (tie-breaking may pick a different
  but equally-optimal partition -- checked for internally too, see the test file).
- `kernel.cu` compiles cleanly for both `sm_86` and `sm_75` (`nvcc -Xptxas -v -c
  kernel.cu`) at every `ITEMS_PER_THREAD` in {1, 2, 4, 8, 16, 32}: 0 bytes stack frame,
  0 bytes spill stores/loads, no warnings.
- `precompute_tables`'s host-side memory footprint at `tune.plan_launch`'s own choice of
  `A`, measured (not estimated) end to end: 575 MB peak / 1.2s at N=40 (A=20), 1.4 GB
  peak / 37.4s at N=45 (A=26) -- see "Chunked host-side precompute" above. This is
  standalone NumPy, no cupy/GPU involved.

**Still not run** (needs an actual RTX 3080 -- `RTX_3080_10GB`/`RTX_3080_12GB` remain
defined in `tune.py` and are exercised by `tests/test_tune.py`'s launch-config search,
but never by the kernel itself):
- No RTX 3080-specific throughput numbers exist; everything measured so far is MX550.
- `tune.py`'s heuristic is unvalidated against real occupancy/timing data on *that*
  card specifically (it has been validated this way on MX550 -- see
  `../../sim/qbf_cuda/README.md`'s measured-throughput table). It encodes
  provably-correct constraints (shared-memory capacity, power-of-two divisibility, CUDA
  grid limits) plus one honestly-labeled rule of thumb for how many Gray-code steps
  should amortize the one-time shared-memory load -- exactly the kind of thing the
  paper says took "extensive empirical experimentation" on their own hardware.

## Benchmark: qbf_cuda vs. the numpy baseline

`experiment_qbf_cuda_vs_numpy.py` (project root) runs the actual `solve_rqaoa` two
ways on the same graphs and compares wall-clock time and solution quality:
**baseline** = today's default (`cutoff=3`, `exact_solver='numpy'`) vs. **qbf_cuda** =
`cutoff=10`, `exact_solver='qbf_cuda'` -- letting the GPU kernel replace the last 7 of
17 elimination rounds. Same graph family and layer-count sweep (`p=1,2,3`) as
`experiment_layer_count.py`, at N=20 (5 seeds), `maxiter=30`. Full results, chart, and
raw CSV: `experiment_qbf_cuda_vs_numpy.csv` in the project root.

**Headline, across all 15 (seed, layer count) combinations:**
- **qbf_cuda was faster in 15/15** -- averaging ~8% less wall-clock time per full
  `solve_rqaoa` call (5.4s→4.8s at p=1, 8.2s→7.6s at p=2, 10.8s→10.1s at p=3).
- **qbf_cuda matched or beat baseline's cut value in 15/15** (tied in 11, strictly
  better in 4) -- **never worse.**

**The time savings are real but modest, not dramatic, and that's expected, not a
disappointment:** a quick timing check (one RQAOA step in isolation) found
`cudaq`'s `"nvidia"` statevector-simulation cost per elimination step scales with
graph size -- roughly 1.7s at n=20 down to 0.07s at n=8 (`maxiter=30`). Since RQAOA
always starts from the *same* largest graph regardless of where it stops, raising
`cutoff` only skips the *cheapest*, last few rounds; the expensive early rounds happen
either way. The naive expectation ("fewer rounds ⇒ proportionally less time") doesn't
hold here -- the real lever this benchmark demonstrates is solution *quality*, not
raw speed.

**Where the quality gain comes from is worth looking at directly, not just averaging
over:** seed 2 scored 67/68 (0.985) under baseline at *every* layer count (p=1, 2, and
3 alike) -- more QAOA depth didn't fix it, because the mistake was locked in by a
greedy correlation decision made somewhere between graph size 10 and 3, and extra
circuit depth earlier in the recursion can't undo a bad choice made later. qbf_cuda
(`cutoff=10`) scored 68/68 (1.000) at every layer count for that same seed, because it
never let RQAOA make that decision in the first place -- it handed the last 7 nodes to
an exact solver instead. This is exactly the "errors propagate most in the late,
small-graph steps" failure mode `strategy.txt` names as motivation for smarter
reduction strategies (adaptive depth, beam search) -- raising `cutoff` attacks the
same failure mode more directly, and for free (it's also the faster configuration).

**Caveats on generalizing this:** 5 seeds is a small sample (one "seed 2"-style trap
is doing a lot of the average-quality-gain work above); N=20 was chosen for safety
margin under this machine's VRAM limits (cudaq's `"nvidia"` target measured working at
26 qubits, failing at 28), not because it's the size where this comparison matters
most -- the effect this benchmark is actually measuring (greedy elimination traps in
the late, small-graph steps) should, if anything, be *more* pronounced at larger N
where RQAOA takes many more elimination steps before reaching a given `cutoff`. A
`cutoff` sweep at larger N (once available) is the natural follow-up, per "Before
running on the 3080" below.

## Before running on the 3080

1. **Correctness first.** Run `tests/test_reference.py` and `tests/test_maxcut_adapter.py`
   there too (they're GPU-free, but confirming the same Python/NumPy versions behave
   identically costs nothing). Then compare `solver.solve_qubo` against
   `reference.qbf_solve` on small `N` (≤16, so brute force is still instant) across a
   range of `A` and both signs of a few negative-weight instances, before trusting it
   on anything RQAOA actually produces.
2. **Then profile, don't assume.** Use `nsys`/`ncu` the way the paper describes doing
   for V100/H100: check achieved occupancy, shared-memory bank conflicts, and whether
   `tune.py`'s block-count targeting (currently a flat "2×-16× SM count" band) actually
   reflects the real amortization tradeoff for the shared-mem load vs. Gray-step count.
   Adjust `plan_launch`'s `min_gray_steps` / `max_blocks_per_sm_target` from measurement,
   not guesswork.
3. **Then wire into RQAOA experiments** via `exact_solver='qbf_cuda'` (see below),
   probably starting with a `cutoff` sweep (10, 20, 30, 40) to see where the QAOA-round
   savings and the (now much cheaper) exact-solve cost actually trade off in wall-clock
   terms for `experiment_batch.py`/`experiment_advanced.py`-style runs.

## Usage

Direct:
```python
from qbf_cuda import reference as ref
from qbf_cuda.maxcut_adapter import brute_force_qbf_cuda

# Exact, GPU-free (fine up to n~20-25 depending on patience):
cut_value, cut_edges, max_subset = brute_force_qbf_cuda(G, backend='reference')

# GPU-accelerated (needs cupy + the actual 3080 -- unexercised so far, see above):
cut_value, cut_edges, max_subset = brute_force_qbf_cuda(G, backend='cuda')
```

Via RQAOA (new `exact_solver` parameter, defaults to `'numpy'` -- existing behavior is
unchanged unless you opt in):
```python
from rqaoa import solve_rqaoa

# unchanged default behavior:
solve_rqaoa(G, layer_count=2, cutoff=3)

# raise cutoff way past what utils.brute_force could ever handle, GPU-free for now:
solve_rqaoa(G, layer_count=2, cutoff=30, exact_solver='qbf_cuda_reference')

# same, GPU-accelerated, once validated on the 3080:
solve_rqaoa(G, layer_count=2, cutoff=40, exact_solver='qbf_cuda')
```

`solve_beam_rqaoa` takes the same `exact_solver` parameter.

## Assumptions

- Designed and tested for `N` (RQAOA residual size) roughly up to 40-45. Nothing
  hard-stops higher `N` other than `tune.plan_launch`'s `max_prefix_bits=31` and
  shared-memory feasibility, but the `N > 49`-ish regime is where the paper needed its
  outer split (not implemented here -- see "explicitly out of scope" above).
- Edge weights must be integer-valued (`maxcut_adapter._validate_integer_weights`
  raises otherwise). True for every graph RQAOA's `contract_graph` actually produces
  (sums of ±1.0-scaled, 1.0-initialized weights), not for arbitrary weighted graphs.
- No linear field (`h`): RQAOA never introduces a per-node bias, only edge-weight
  updates, so every graph this sees is a pure edge-weighted Max-Cut instance. `h` is
  supported in `reference.ising_to_qubo` for generality but always `0` from
  `maxcut_adapter.graph_to_ising`.
