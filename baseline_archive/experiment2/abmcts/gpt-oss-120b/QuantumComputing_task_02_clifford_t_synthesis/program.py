# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Improved optimizer: stronger local rewrites around an aggressive Clifford+T transpile."""
    # The `target` already encodes the Clifford+T gate set we must output.
    # Use its operation names as the basis for transpilation to avoid unnecessary gates.
    basis_gates = list(target.operation_names)

    # First pass: aggressive local rewrite to simplify the original circuit.
    rewritten = optimize_by_local_rewrite(input_circuit, max_rounds=30)

    # Transpile with the highest optimization level, preserving the Clifford+T basis.
    transpiled = transpile(
        rewritten,
        basis_gates=basis_gates,
        optimization_level=3,
        seed_transpiler=42,
        approximation_degree=0.0,
    )

    # Final pass: another round of local rewrite to clean up any new redundancies.
    final_circuit = optimize_by_local_rewrite(transpiled, max_rounds=30)

    return final_circuit
# EVOLVE-BLOCK-END
