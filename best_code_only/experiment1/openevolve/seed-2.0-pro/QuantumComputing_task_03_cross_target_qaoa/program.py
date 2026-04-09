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
    # Load target-specific rules before first optimization pass
    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    # Aggressive rewrite for IonQ all-to-all circuits to maximize gate merges
    max_rounds_pre = 128 if "ionq" in target_name else 64
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=max_rounds_pre)

    transpile_kwargs = {
        "circuits": optimized,
        "target": target,
        "optimization_level": case.get("optimization_level", 3),
        "seed_transpiler": 42,
    }
    if "ionq" in target_name:
        # IonQ has full all-to-all connectivity, skip routing overhead and enable exact optimizations
        transpile_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
        transpile_kwargs.update({
            "coupling_map": None,
            "approximation_degree": 1.0,
        })
    if "ibm" in target_name or "rigetti" in target_name:
        transpile_kwargs.update(
            {
                "layout_method": "sabre",
                "routing_method": "sabre",
                "approximation_degree": 0.85,
                "unitary_synthesis_method": "sk",
                "unitary_synthesis_plugin_config": {"optimization_level": 3},
            }
        )

    transpiled = transpile(**transpile_kwargs)
    # Aggressive post-transpilation optimization for target-compliant circuits
    max_rounds_post = 128 if "ionq" in target_name else 64
    return optimize_by_local_rewrite(transpiled, max_rounds=max_rounds_post)
# EVOLVE-BLOCK-END
