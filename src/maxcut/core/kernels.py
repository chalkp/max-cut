import cudaq
from cudaq.qis import *
from typing import List


@cudaq.kernel
def problem(q0: cudaq.qubit, q1: cudaq.qubit, gamma: float):
    x.ctrl(q0, q1)
    rz(gamma * 2.0, q1)
    x.ctrl(q0, q1)


@cudaq.kernel
def mixer(qubit: cudaq.qubit, beta: float):
    rz(beta * 2.0, qubit)


@cudaq.kernel
def qaoa_kernel(
    qubit_count: int,
    layer_count: int,
    edges_source: List[int],
    edges_target: List[int],
    edge_weights: List[float],
    parameters: List[float]
):
    qubits = cudaq.qvector(qubit_count)
    h(qubits)

    for i in range(layer_count):
        for edge in range(len(edges_source)):
            qu = edges_source[edge]
            qv = edges_target[edge]
            w = edge_weights[edge]

            problem(qubits[qu], qubits[qv], parameters[i] * w)

        for q in range(qubit_count):
            mixer(qubits[q], parameters[i + layer_count])
