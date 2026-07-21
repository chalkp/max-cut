from dataclasses import dataclass, field
from typing import Optional
import time
import cudaq


@dataclass
class BudgetTracker:
    circuit_executions: int = 0
    two_qubit_gates: int = 0
    shots: int = 0
    wall_clock_s: float = 0.0
    _t0: Optional[float] = field(default=None, repr=False)

    def start(self) -> 'BudgetTracker':
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> 'BudgetTracker':
        if self._t0 is not None:
            self.wall_clock_s += time.perf_counter() - self._t0
            self._t0 = None
        return self

    def record_observe(self, two_qubit_gates: int, shots: int = 0) -> None:
        self.circuit_executions += 1
        self.two_qubit_gates += two_qubit_gates
        self.shots += shots

    def record_sample(self, two_qubit_gates: int, shots: int) -> None:
        self.circuit_executions += 1
        self.two_qubit_gates += two_qubit_gates
        self.shots += shots

    def as_dict(self, prefix: str = '') -> dict:
        return {
            f'{prefix}circuit_executions': self.circuit_executions,
            f'{prefix}two_qubit_gates': self.two_qubit_gates,
            f'{prefix}shots': self.shots,
            f'{prefix}wall_clock_s': self.wall_clock_s,
        }


def standard_qaoa_layer_two_qubit_gates(n_edges: int) -> int:
    return 2 * n_edges


def pool_operator_two_qubit_gates(pauli_weight: int) -> int:
    return 2 if pauli_weight >= 2 else 0


_RQAOA_OBSERVE_ARG_ORDER = (
    'kernel', 'hamil', 'qubit_count', 'layer_count',
    'edges_source', 'edges_target', 'edge_weights', 'parameters',
)


def instrumented_rqaoa_run(fn, *args, **kwargs):
    tracker = BudgetTracker().start()
    orig_observe, orig_sample = cudaq.observe, cudaq.sample

    def wrapped_observe(*a, **kw):
        gates = 2 * a[3] * len(a[4])
        tracker.record_observe(gates, shots=kw.get('shots_count', 0))
        return orig_observe(*a, **kw)

    def wrapped_sample(*a, **kw):
        gates = 2 * a[2] * len(a[3])
        tracker.record_sample(gates, shots=kw.get('shots_count', 1000))
        return orig_sample(*a, **kw)

    cudaq.observe, cudaq.sample = wrapped_observe, wrapped_sample
    try:
        result = fn(*args, **kwargs)
    finally:
        cudaq.observe, cudaq.sample = orig_observe, orig_sample
        tracker.stop()
    return result, tracker
