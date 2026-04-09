# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search baseline for routing-heavy circuits."""
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc

    num_qubits = case.get("num_qubits", input_circuit.num_qubits)
    best = qc
    best_score = _cost(qc)
    for seed in (0, 1, 2, 3, 5, 8, 13, 21, 34, 42, 123, 1337, num_qubits):
        for kwargs in (
            {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
            {"optimization_level": 3, "layout_method": "sabre", "routing_method": "stochastic"},
            {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
            {"optimization_level": 3},
            {"optimization_level": 2},
        ):
            try:
                cand = transpile(qc, target=target, seed_transpiler=seed, **kwargs)
                score = _cost(cand)
                if score < best_score:
                    best, best_score = cand, score
            except Exception:
                pass
    return best
# EVOLVE-BLOCK-END
