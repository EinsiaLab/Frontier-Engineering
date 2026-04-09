# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Combines local rewrites with aggressive basis-aware transpilation; selects best via approx. cost."""
    _ = (target, case)  # unused

    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=20)
    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]

    best_circuit = None
    best_cost = float("inf")
    for seed in [42, 123, 0, 7]:
        transpiled = transpile(
            optimized,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )
        after = optimize_by_local_rewrite(transpiled, max_rounds=20)
        ops = after.count_ops()
        t_total = ops.get("t", 0) + ops.get("tdg", 0)
        twoq = after.num_nonlocal_gates()
        dep = after.depth()
        approx_cost = t_total + 0.2 * twoq + 0.05 * dep
        if approx_cost < best_cost:
            best_cost = approx_cost
            best_circuit = after
    return best_circuit
# EVOLVE-BLOCK-END
