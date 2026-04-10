# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _circuit_cost(qc: QuantumCircuit) -> int:
    """Estimate circuit cost: weighted sum of gates (2-qubit gates cost more)."""
    cost = 0
    for instruction in qc.data:
        n_qubits = instruction.operation.num_qubits
        if n_qubits >= 2:
            cost += 10 * n_qubits
        else:
            cost += 1
    return cost


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Multi-seed transpilation with iterative local rewrite refinement."""
    _ = (target, case)

    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]

    # Initial local rewrite optimization
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=20)

    # Try multiple seeds and keep the best
    best = None
    best_cost = float("inf")

    for seed in range(8):
        transpiled = transpile(
            optimized,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )
        refined = optimize_by_local_rewrite(transpiled, max_rounds=15)
        cost = _circuit_cost(refined)
        if cost < best_cost:
            best_cost = cost
            best = refined

    # Second round: transpile the best result again with multiple seeds
    best2 = best
    best2_cost = best_cost

    for seed in range(4):
        transpiled2 = transpile(
            best,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed + 100,
        )
        refined2 = optimize_by_local_rewrite(transpiled2, max_rounds=15)
        cost2 = _circuit_cost(refined2)
        if cost2 < best2_cost:
            best2_cost = cost2
            best2 = refined2

    return best2
# EVOLVE-BLOCK-END
