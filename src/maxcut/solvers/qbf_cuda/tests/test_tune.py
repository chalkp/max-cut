"""
tune.py correctness gate -- run with plain `python3 test_tune.py` (no GPU required;
pure host-side arithmetic).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tune  # noqa: E402


def test_plan_launch_succeeds_for_every_n_from_1_to_45():
    """Regression test for a real bug: threads_per_block=256 was fixed (only
    items_per_thread was searched), so block_items = threads_per_block *
    items_per_thread could never shrink below 256, and plan_launch raised for every
    N < 8 -- caught only once the MX550 sibling copy (../../../sim/qbf_cuda/) was
    actually run against real hardware at small N, since this project's own tests
    never exercised N < 8 (the whole point of this copy was large N). Now both
    threads_per_block and items_per_thread are searched, so plan_launch should never
    fail purely for N being small. Checked against both GPUSpecs this module defines
    -- MX550 (the actual default, this being the only hardware available to test
    against) and RTX_3080_10GB (unexecuted, but the search logic doesn't care which
    GPUSpec it's given)."""
    for gpu in [tune.MX550, tune.RTX_3080_10GB]:
        for n in range(1, 46):
            plan = tune.plan_launch(n, gpu=gpu)
            assert plan.A + plan.B == n
            assert plan.num_blocks >= 1
            assert plan.block_items == plan.threads_per_block * plan.items_per_thread
            assert (1 << plan.B) % plan.block_items == 0


def test_shared_mem_within_budget():
    for gpu in [tune.MX550, tune.RTX_3080_10GB]:
        for n in [8, 16, 24, 28, 32, 36]:
            plan = tune.plan_launch(n, gpu=gpu)
            assert plan.shared_mem_bytes <= gpu.max_shared_mem_per_block_optin
            if plan.needs_shared_mem_optin:
                assert plan.shared_mem_bytes > gpu.max_shared_mem_per_block_default


def test_vram_within_budget():
    for gpu in [tune.MX550, tune.RTX_3080_10GB]:
        for n in [8, 16, 24, 28, 32, 36]:
            plan = tune.plan_launch(n, gpu=gpu)
            assert plan.vram_bytes <= gpu.total_global_mem_bytes * tune.VRAM_SAFETY_FRACTION


def test_utilization_reasonable_once_n_allows_it():
    # Below a GPU-dependent N there just aren't enough states to fill every SM
    # regardless of tuning -- only check utilization once the problem is big enough
    # that failing to hit it would indicate a real tuning bug, not an inherent lack of
    # parallelism at small N.
    for gpu, sizes in [(tune.MX550, [20, 24, 28, 32, 36]), (tune.RTX_3080_10GB, [24, 28, 32, 36, 40, 45])]:
        for n in sizes:
            plan = tune.plan_launch(n, gpu=gpu)
            assert plan.num_blocks >= gpu.sm_count, (
                f"{gpu.name} N={n}: only {plan.num_blocks} blocks, fewer than {gpu.sm_count} SMs"
            )


def test_default_gpu_is_mx550():
    """DEFAULT_GPU should be the hardware actually present on this machine, not the
    RTX 3080 this module was originally written against -- see tune.py's module
    docstring for why the default changed."""
    assert tune.DEFAULT_GPU is tune.MX550


def test_matches_live_device_query():
    """Requires cupy (the `cudaq` mamba env on this machine) -- confirms the
    hand-recorded MX550 GPUSpec constant still matches what cupy sees on the actual
    device, rather than trusting the transcription forever."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        print("  (skipped: cupy not installed in this interpreter)")
        return
    queried = tune.query_device_gpu_spec()
    fields = ['compute_capability', 'sm_count', 'max_threads_per_block', 'max_threads_per_sm',
              'max_shared_mem_per_block_default', 'max_shared_mem_per_block_optin',
              'max_shared_mem_per_sm', 'regs_per_sm', 'total_global_mem_bytes']
    mismatches = [f for f in fields if getattr(queried, f) != getattr(tune.MX550, f)]
    assert not mismatches, f"MX550 spec drifted from live device query: {mismatches}"


if __name__ == "__main__":
    tests = [
        test_plan_launch_succeeds_for_every_n_from_1_to_45,
        test_shared_mem_within_budget,
        test_vram_within_budget,
        test_utilization_reasonable_once_n_allows_it,
        test_default_gpu_is_mx550,
        test_matches_live_device_query,
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
