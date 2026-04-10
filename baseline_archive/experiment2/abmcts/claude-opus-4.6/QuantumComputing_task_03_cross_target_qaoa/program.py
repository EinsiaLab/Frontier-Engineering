# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile baseline for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=32)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    # Try multiple seeds and optimization strategies, pick the best
    best_circuit = None
    best_cost = float('inf')

    def compute_cost(circ):
        two_q = 0
        for inst in circ.data:
            if inst.operation.num_qubits == 2:
                two_q += 1
        depth = circ.depth()
        return two_q + 0.2 * depth

    seeds = [42, 0, 1, 7, 13, 21, 37, 53, 97, 123, 200, 314, 500, 777, 999]

    for seed in seeds:
        for opt_level in [3, 2]:
            transpile_kwargs = {
                "circuits": optimized,
                "target": target,
                "optimization_level": opt_level,
                "seed_transpiler": seed,
            }
            if "ionq" in target_name:
                transpile_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
            if "ibm" in target_name or "rigetti" in target_name:
                transpile_kwargs.update(
                    {
                        "layout_method": "sabre",
                        "routing_method": "sabre",
                    }
                )

            try:
                transpiled = transpile(**transpile_kwargs)
                transpiled = optimize_by_local_rewrite(transpiled, max_rounds=32)
                cost = compute_cost(transpiled)
                if cost < best_cost:
                    best_cost = cost
                    best_circuit = transpiled
            except Exception:
                continue

    # Also try with approximation_degree for IBM/Rigetti
    if "ibm" in target_name or "rigetti" in target_name:
        for seed in [42, 0, 7, 13, 53, 97, 314]:
            for approx in [0.99, 0.95, 0.9]:
                transpile_kwargs = {
                    "circuits": optimized,
                    "target": target,
                    "optimization_level": 3,
                    "seed_transpiler": seed,
                    "layout_method": "sabre",
                    "routing_method": "sabre",
                    "approximation_degree": approx,
                }
                try:
                    transpiled = transpile(**transpile_kwargs)
                    transpiled = optimize_by_local_rewrite(transpiled, max_rounds=32)
                    cost = compute_cost(transpiled)
                    if cost < best_cost:
                        best_cost = cost
                        best_circuit = transpiled
                except Exception:
                    continue

    # Try starting from original input too (without pre-optimization)
    for seed in [42, 0, 7, 13, 97]:
        transpile_kwargs = {
            "circuits": input_circuit,
            "target": target,
            "optimization_level": 3,
            "seed_transpiler": seed,
        }
        if "ionq" in target_name:
            transpile_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
        if "ibm" in target_name or "rigetti" in target_name:
            transpile_kwargs.update(
                {
                    "layout_method": "sabre",
                    "routing_method": "sabre",
                }
            )
        try:
            transpiled = transpile(**transpile_kwargs)
            transpiled = optimize_by_local_rewrite(transpiled, max_rounds=32)
            cost = compute_cost(transpiled)
            if cost < best_cost:
                best_cost = cost
                best_circuit = transpiled
        except Exception:
            continue

    if best_circuit is None:
        # Fallback
        transpiled = transpile(
            circuits=optimized,
            target=target,
            optimization_level=3,
            seed_transpiler=42,
        )
        best_circuit = optimize_by_local_rewrite(transpiled, max_rounds=32)

    return best_circuit
# EVOLVE-BLOCK-END
