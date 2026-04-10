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
    # Choose pre‑transpile rewrite depth per backend.
    if "ibm" in target_name:
        # Use a larger budget for IBM back‑ends to expose more cancellations before transpilation.
        init_max_rounds = 512
    elif "ionq" in target_name:
        # Give IonQ a larger pre‑transpile rewrite budget to expose more cancellations.
        init_max_rounds = 512
    else:
        # Use a larger budget for other back‑ends as well to expose more cancellations.
        init_max_rounds = 256
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=init_max_rounds)

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
    # For IonQ we let the transpiler decide the most efficient native basis.
    # Removing an explicit basis_gates list avoids unnecessary rzz insertion.
    if "ibm" in target_name or "rigetti" in target_name:
        # Use the native basis gates provided by the Target (e.g. u3, cx, measure)
        basis = getattr(target, "basis_gates", None) or getattr(target, "operation_names", None)
        if basis:
            transpile_kwargs["basis_gates"] = list(basis)
    # For IonQ we let the transpiler decide the optimal native basis by not setting basis_gates.
    if "ibm" in target_name or "rigetti" in target_name:
        transpile_kwargs.update(
            {
                "layout_method": "sabre",
                "routing_method": "sabre",
                "approximation_degree": 0.5,
                "unitary_synthesis_method": "sk",
                "unitary_synthesis_plugin_config": {"optimization_level": 3},
            }
        )

    transpiled = transpile(**transpile_kwargs)
    # Determine post‑transpile rewrite depth per backend.
    if "ibm" in target_name:
        final_max_rounds = 512
    elif "ionq" in target_name:
        # Increase post‑transpile rewrite rounds for IonQ to further reduce cost.
        final_max_rounds = 512
    else:
        final_max_rounds = 256
    return optimize_by_local_rewrite(transpiled, max_rounds=final_max_rounds)
# EVOLVE-BLOCK-END