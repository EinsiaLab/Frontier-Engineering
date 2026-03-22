# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Rule-based baseline: structural rewrites without transpile."""
    _ = (target, case)
    return optimize_by_local_rewrite(input_circuit)
# EVOLVE-BLOCK-END
