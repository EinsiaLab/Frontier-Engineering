# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary, QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target‑aware transpile baseline for cross‑platform QAOA circuits with enhanced optimisations."""
    target_name = str(case.get("target_name", "")).lower()

    # Aggressive pre‑transpile local rewrite
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=64)

    # Register target‑specific equivalences
    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    # Base transpile arguments
    transpile_kwargs = {
        "circuits": optimized,
        "target": target,
        "optimization_level": 3,  # force maximal optimisation
        "seed_transpiler": case.get("seed", 42),
    }

    # Target‑specific basis gates and routing/approximation settings
    if "ionq" in target_name:
        transpile_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
    if "ibm" in target_name or "rigetti" in target_name:
        transpile_kwargs.update(
            {
                "layout_method": "sabre",
                "routing_method": "sabre",
                # more aggressive approximation to reduce depth and two‑qubit count
                "approximation_degree": 0.5,
                "unitary_synthesis_method": "sk",
                "unitary_synthesis_plugin_config": {"optimization_level": 3},
            }
        )

    # Perform transpilation
    transpiled = transpile(**transpile_kwargs)

    # Post‑transpile aggressive local rewrite
    return optimize_by_local_rewrite(transpiled, max_rounds=64)
# EVOLVE-BLOCK-END
