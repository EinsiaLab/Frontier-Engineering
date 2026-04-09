# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile baseline for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=32)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)

    if "ionq" in target_name:
        transpiled = transpile(
            optimized,
            target=target,
            optimization_level=3,
            basis_gates=["rz", "sx", "x", "rzz", "measure"],
            seed_transpiler=42,
        )
        return optimize_by_local_rewrite(transpiled, max_rounds=32)

    # IBM: multi-seed SABRE search (more trials) for better mapping/routing
    # Increases chance of finding lower-swap layouts on heavy-hex topology
    best_circuit = None
    best_cost = float("inf")
    for seed in [42, 123, 456, 789, 1011, 7, 13, 19, 27, 31]:
        transpiled = transpile(
            optimized,
            target=target,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            approximation_degree=0.9,
            seed_transpiler=seed,
        )
        post_opt = optimize_by_local_rewrite(transpiled, max_rounds=32)
        # Exact cost used by evaluator
        twoq = sum(1 for inst in post_opt.data if len(inst.qubits) == 2)
        depth = post_opt.depth()
        cost = twoq + 0.2 * depth
        if cost < best_cost:
            best_cost = cost
            best_circuit = post_opt
    return best_circuit
# EVOLVE-BLOCK-END
