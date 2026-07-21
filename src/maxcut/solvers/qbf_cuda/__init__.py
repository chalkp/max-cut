"""
QBF exact solver, targeting RTX 3080.



Based on Maltsev et al., "Terastate-per-second QUBO Brute-Force on a Single GPU: A
Matrix Prefix-Suffix Decomposition" (arXiv:v1; source at
../../sim/2607.04857v1/qbf_paper.tex). See README.md in this directory for the design,
the RTX 3080-specific tuning rationale, what has been validated so far and how, and
what remains before running on real hardware.

Submodules:
  reference.py       -- pure-NumPy algorithm reference (no GPU); the correctness spec.
  tune.py             -- RTX 3080 launch-configuration heuristic (no GPU needed to run).
  kernel.cu           -- the CUDA kernel itself (compiled via solver.py, or standalone
                         with nvcc for a register/shared-mem sanity check).
  solver.py           -- CuPy RawKernel launcher (requires cupy + a GPU to *run*, but
                         importable without cupy installed; only calling solve_qubo()
                         requires it).
  maxcut_adapter.py   -- networkx Max-Cut <-> Ising <-> QUBO conversion, plus
                         brute_force_qbf_cuda(G), a drop-in replacement for
                         utils.brute_force(G) with the 'reference' or 'cuda' backend.

Only reference.py, tune.py, and maxcut_adapter.py's 'reference' backend have actually
been run as part of building this (see tests/); the CUDA path has been compiled and
register/shared-memory-checked with nvcc, but not executed on a GPU -- this development
machine's only GPU is a 2GB MX550 (compute capability 7.5), not representative of the
target RTX 3080, and running it here would not validate anything the compile-check
+ NumPy reference haven't already covered.
"""
from . import reference
from . import tune

__all__ = ["reference", "tune"]
