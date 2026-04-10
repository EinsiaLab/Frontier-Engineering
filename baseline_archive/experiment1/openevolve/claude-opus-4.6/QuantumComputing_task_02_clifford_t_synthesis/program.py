# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _circuit_cost(qc: QuantumCircuit) -> float:
    """Estimate cost matching the actual evaluation metric:
    cost = (T + Tdg) + 0.2 * two_qubit_count + 0.05 * depth
    """
    t_count = 0
    tdg_count = 0
    two_qubit_count = 0
    for instruction in qc.data:
        name = instruction.operation.name
        if name == "t":
            t_count += 1
        elif name == "tdg":
            tdg_count += 1
        if instruction.operation.num_qubits >= 2:
            two_qubit_count += 1
    depth = qc.depth()
    return (t_count + tdg_count) + 0.2 * two_qubit_count + 0.05 * depth


def _try_transpile(circuit, basis_gates, opt_level, seed):
    """Transpile and polish, return (circuit, cost)."""
    transpiled = transpile(
        circuit, basis_gates=basis_gates,
        optimization_level=opt_level, seed_transpiler=seed,
    )
    polished = optimize_by_local_rewrite(transpiled, max_rounds=20)
    return polished, _circuit_cost(polished)


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Multi-seed iterative optimization combining local rewrites with transpilation."""
    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]

    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=30)

    best = None
    best_cost = float("inf")

    seeds = [42, 0, 1, 7, 13, 99, 23, 55]

    # Try opt_level=3 with multiple seeds
    for seed in seeds:
        polished, cost = _try_transpile(optimized, basis_gates, 3, seed)
        if cost < best_cost:
            best, best_cost = polished, cost

    # Try opt_level=2 with fewer seeds
    for seed in seeds[:4]:
        polished, cost = _try_transpile(optimized, basis_gates, 2, seed)
        if cost < best_cost:
            best, best_cost = polished, cost

    # Also try from raw input (different decomposition path)
    for seed in [42, 0, 1]:
        polished, cost = _try_transpile(input_circuit, basis_gates, 3, seed)
        if cost < best_cost:
            best, best_cost = polished, cost

    # Iterative refinement
    for _ in range(5):
        improved = False
        for seed in seeds:
            polished, cost = _try_transpile(best, basis_gates, 3, seed)
            if cost < best_cost:
                best, best_cost = polished, cost
                improved = True
        if not improved:
            break

    return best
# EVOLVE-BLOCK-END
