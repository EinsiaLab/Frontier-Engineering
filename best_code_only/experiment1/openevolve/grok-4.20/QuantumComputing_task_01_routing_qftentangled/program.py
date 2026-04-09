# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _c(qc):
    return sum(i.operation.num_qubits > 1 for i in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit, target, case):
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc
    n = case.get("num_qubits", input_circuit.num_qubits)
    best, bs = qc, float("inf")
    seeds = [n + x for x in (0, 1, 5, 13, 29, 42, 43, 67, 100, 123)]
    for s in seeds:
        for o in (3, 2):
            try:
                cand = transpile(
                    qc,
                    target=target,
                    optimization_level=o,
                    layout_method="sabre",
                    routing_method="sabre",
                    seed_transpiler=s,
                )
                sc = _c(cand)
                if sc < bs:
                    best, bs = cand, sc
            except Exception:
                continue
    return optimize_by_local_rewrite(best)
# EVOLVE-BLOCK-END
