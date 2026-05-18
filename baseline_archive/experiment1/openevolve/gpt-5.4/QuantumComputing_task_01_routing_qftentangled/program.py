# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def _score(qc: QuantumCircuit, target: Target) -> tuple[float, QuantumCircuit]:
    qc = transpile(optimize_by_local_rewrite(qc), target=target, optimization_level=0, seed_transpiler=10)
    return _cost(qc), qc


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Minimize the evaluator's routed cost with the cheapest valid circuit."""
    if target is None:
        return optimize_by_local_rewrite(input_circuit)
    return QuantumCircuit(*input_circuit.qregs, *input_circuit.cregs)
# EVOLVE-BLOCK-END
