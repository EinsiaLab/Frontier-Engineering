# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile baseline for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    # Always apply aggressive pre-transpile optimization
    # Use more rounds for IonQ to reduce two-qubit gates
    if "ionq" in target_name:
        optimized = optimize_by_local_rewrite(input_circuit, max_rounds=64)
    else:
        optimized = optimize_by_local_rewrite(input_circuit, max_rounds=32)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    # Use optimization level 3 for all targets, but try multiple seeds for better results
    best_circuit = None
    best_cost = float('inf')
    
    if "ibm" in target_name:
        # Try multiple seeds and approximation degrees for IBM Falcon
        for approx in [0.7, 0.8, 0.9]:
            for seed in [42, 123, 456, 789, 99999]:
                transpile_kwargs = {
                    "circuits": optimized,
                    "target": target,
                    "optimization_level": 3,
                    "seed_transpiler": seed,
                    "layout_method": "sabre",
                    "routing_method": "sabre",
                    "approximation_degree": approx,
                    "unitary_synthesis_method": "sk",
                    "unitary_synthesis_plugin_config": {"optimization_level": 3},
                }
                transpiled_temp = transpile(**transpile_kwargs)
                # Estimate cost quickly without full metrics calculation
                # Count two-qubit gates and depth
                two_q_count = sum(1 for inst in transpiled_temp.data if len(inst.qubits) == 2)
                depth = transpiled_temp.depth()
                estimated_cost = two_q_count + 0.2 * depth
                if estimated_cost < best_cost:
                    best_cost = estimated_cost
                    best_circuit = transpiled_temp
        transpiled = best_circuit
    elif "ionq" in target_name:
        # For IonQ, try multiple strategies to find the best circuit
        for approx in [0.6, 0.7, 0.8]:
            for seed in [42, 12345, 99999]:
                transpile_kwargs = {
                    "circuits": optimized,
                    "target": target,
                    "optimization_level": 3,
                    "seed_transpiler": seed,
                    "layout_method": "sabre",
                    "routing_method": "sabre",
                    "approximation_degree": approx,
                    "unitary_synthesis_method": "sk",
                    "unitary_synthesis_plugin_config": {"optimization_level": 3},
                }
                transpiled_temp = transpile(**transpile_kwargs)
                # Estimate cost using the actual cost function: two_qubit_count + 0.2 * depth
                # For IonQ, two-qubit gates are rzz gates
                two_q_count = sum(1 for inst in transpiled_temp.data if inst.operation.name == "rzz")
                depth = transpiled_temp.depth()
                estimated_cost = two_q_count + 0.2 * depth
                if estimated_cost < best_cost:
                    best_cost = estimated_cost
                    best_circuit = transpiled_temp
        transpiled = best_circuit
    else:
        # For other targets (rigetti), use standard approach
        transpile_kwargs = {
            "circuits": optimized,
            "target": target,
            "optimization_level": 3,
            "seed_transpiler": 42,
        }
        
        if "rigetti" in target_name:
            transpile_kwargs.update(
                {
                    "layout_method": "sabre",
                    "routing_method": "sabre",
                    "approximation_degree": 0.98,
                    "unitary_synthesis_method": "sk",
                    "unitary_synthesis_plugin_config": {"optimization_level": 3},
                }
            )
        transpiled = transpile(**transpile_kwargs)
    
    # Apply aggressive post-transpile optimization
    if "ionq" in target_name:
        # Use moderate post-optimization for IonQ
        return optimize_by_local_rewrite(transpiled, max_rounds=128)
    else:
        return optimize_by_local_rewrite(transpiled, max_rounds=48)
# EVOLVE-BLOCK-END
