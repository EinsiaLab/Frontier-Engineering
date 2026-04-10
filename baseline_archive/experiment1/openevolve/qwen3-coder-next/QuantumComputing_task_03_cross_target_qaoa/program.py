# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import CircuitInstruction, SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile baseline for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    
    # More conservative optimization for IonQ to avoid performance degradation
    num_qubits = case.get("num_qubits", 10)
    repetitions = case.get("repetitions", 1)
    
    # Dynamic optimization rounds based on circuit size and target
    if "ionq" in str(case.get("target_name", "")).lower():
        # Conservative approach for IonQ to maintain performance
        initial_rounds = 24 if num_qubits >= 12 else 12
    else:
        # More aggressive for IBM targets
        initial_rounds = 48 if num_qubits >= 14 else (32 if num_qubits >= 12 else 16)
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=initial_rounds)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    transpile_kwargs = {
        "circuits": optimized,
        "target": target,
        "optimization_level": case.get("optimization_level", 3),
        "seed_transpiler": 42,
    }
    if "ionq" in target_name:
        # Better optimization level for IonQ native gates
        transpile_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
        # Use optimization level 1 for IonQ as it often provides better native gate handling
        transpile_kwargs["optimization_level"] = 1  # Level 1 often works better for native gates
        # Use consistent approximation for IonQ
        transpile_kwargs["approximation_degree"] = 0.95
    if "ibm" in target_name or "rigetti" in target_name:
        # Better settings for IBM-like targets
        transpile_kwargs.update(
            {
                "layout_method": "sabre",
                "routing_method": "sabre",
                "approximation_degree": 0.95,
                "unitary_synthesis_method": "sk",
                "optimization_level": 2,  # Better balance for IBM targets
            }
        )
        # For IBM targets, use better approximation for larger circuits
        if num_qubits >= 12 and repetitions >= 2:
            transpile_kwargs["approximation_degree"] = 0.97

    transpiled = transpile(**transpile_kwargs)
    
    # Apply additional optimizations based on target
    if "ionq" in target_name:
        # For IonQ, apply moderate rounds of structural optimization to reduce gate count
        # but not too aggressive to avoid performance degradation
        post_opt_rounds = 16 if num_qubits >= 12 else 8
        transpiled = optimize_by_local_rewrite(transpiled, max_rounds=post_opt_rounds)
    elif "ibm" in target_name:
        # IBM-specific optimization pass with more rounds for larger circuits
        post_opt_rounds = 24 if num_qubits >= 12 else 16
        transpiled = optimize_by_local_rewrite(transpiled, max_rounds=post_opt_rounds)
    
    return transpiled
# EVOLVE-BLOCK-END
