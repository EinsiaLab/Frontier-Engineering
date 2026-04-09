# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Optimize QAOA circuit for target platform."""
    name = str(case.get("target_name", "")).lower()
    num_qubits = input_circuit.num_qubits
    
    # More aggressive adaptive rounds for larger circuits
    adaptive_rounds = min(256, max(64, len(input_circuit.data) // 5))
    
    opt = optimize_by_local_rewrite(input_circuit, max_rounds=adaptive_rounds)

    if "ionq" in name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    kwargs = {"circuits": opt, "target": target, "optimization_level": 3, "seed_transpiler": 12345}
    
    if "ionq" in name:
        # IonQ: Use native gates with moderate approximation
        kwargs["basis_gates"] = ["rz", "rx", "ry", "rxx", "rzz", "measure"]
        kwargs["approximation_degree"] = 0.9
        kwargs["layout_method"] = "dense"
        kwargs["routing_method"] = "sabre"
    elif "ibm" in name:
        # IBM: Aggressive optimization for better two-qubit reduction
        kwargs.update({
            "layout_method": "dense",
            "routing_method": "sabre", 
            "approximation_degree": 0.7,
            "unitary_synthesis_method": "sk",
        })
    elif "rigetti" in name:
        kwargs.update({
            "layout_method": "dense",
            "routing_method": "sabre",
            "approximation_degree": 0.9,
        })

    transpiled = transpile(**kwargs)
    # Second pass with different seed for additional optimization
    result = optimize_by_local_rewrite(transpiled, max_rounds=adaptive_rounds)
    return result
# EVOLVE-BLOCK-END
