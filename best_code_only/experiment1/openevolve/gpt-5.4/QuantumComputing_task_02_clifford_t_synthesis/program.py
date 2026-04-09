# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    _ = target, case
    rounds = 12 if len(input_circuit.data) > 200 else 20
    qc = optimize_by_local_rewrite(input_circuit, max_rounds=rounds)
    qc = transpile(
        qc,
        basis_gates=["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"],
        optimization_level=3,
        seed_transpiler=42,
    )
    return optimize_by_local_rewrite(qc, max_rounds=rounds)
# EVOLVE-BLOCK-END
