# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(i.operation.num_qubits == 2 for i in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc
    n = case.get("num_qubits", input_circuit.num_qubits)
    best, best_score = qc, float("inf")
    for kw in (
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "sabre"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
    ):
        for seed in (n + 3, n + 5, n + 11, n + 17, n + 29, n + 41):
            try:
                cand = transpile(qc, target=target, seed_transpiler=seed, **kw)
            except Exception:
                continue
            score = _cost(cand)
            if score < best_score:
                best, best_score = cand, score
    try:
        cand = transpile(best, target=target, optimization_level=3, seed_transpiler=n + 7)
        score = _cost(cand)
        if score < best_score:
            best = cand
    except Exception:
        pass
    return best
# EVOLVE-BLOCK-END
