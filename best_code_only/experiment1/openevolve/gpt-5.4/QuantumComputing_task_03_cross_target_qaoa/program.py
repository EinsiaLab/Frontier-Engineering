# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, SessionEquivalenceLibrary
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    n = str(case.get("target_name", "")).lower()
    if "ionq" in n:
        ionq.add_equivalences(SessionEquivalenceLibrary)
        return optimize_by_local_rewrite(input_circuit, max_rounds=8)
    if "rigetti" in n:
        rigetti.add_equivalences(SessionEquivalenceLibrary)
    qc = optimize_by_local_rewrite(input_circuit, max_rounds=16)
    kw = {"circuits": qc, "target": target, "optimization_level": 3, "seed_transpiler": 42}
    if "ibm" in n or "rigetti" in n:
        kw.update({"layout_method": "sabre", "routing_method": "sabre", "approximation_degree": 0.95})
    return optimize_by_local_rewrite(transpile(**kw), max_rounds=16)
# EVOLVE-BLOCK-END
