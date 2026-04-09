# EVOLVE-BLOCK-START
from __future__ import annotations
import time

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def _post_optimize(qc: QuantumCircuit, target: Target) -> QuantumCircuit:
    try:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            Optimize1qGatesDecomposition, CXCancellation,
            CommutativeCancellation, CommutationAnalysis,
        )
        pm = PassManager([
            CommutationAnalysis(), CommutativeCancellation(),
            CXCancellation(), Optimize1qGatesDecomposition(target=target),
        ])
        return pm.run(qc)
    except Exception:
        return qc


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    qc_rewritten = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc_rewritten

    time_limit = 92
    t0 = time.monotonic()
    elapsed = lambda: time.monotonic() - t0

    best = None
    best_score = float('inf')
    top_k = []
    TOP_K = 8

    def _update(cand):
        nonlocal best, best_score, top_k
        s = _cost(cand)
        if s < best_score:
            best = cand
            best_score = s
        if len(top_k) < TOP_K:
            top_k.append((s, cand))
            top_k.sort(key=lambda x: x[0])
        elif s < top_k[-1][0]:
            top_k[-1] = (s, cand)
            top_k.sort(key=lambda x: x[0])

    source_circuits = [input_circuit, qc_rewritten]

    for pre_opt in (0, 1, 2):
        for pre_seed in (0, 7, 42, 99, 137, 200):
            if elapsed() > time_limit * 0.10:
                break
            try:
                qc_pre = transpile(input_circuit, target=target,
                                   optimization_level=pre_opt, seed_transpiler=pre_seed)
                source_circuits.append(qc_pre)
                _update(qc_pre)
            except Exception:
                pass

    for approx in (0.99, 0.999):
        for pre_seed in (0, 42):
            if elapsed() > time_limit * 0.12:
                break
            try:
                _update(transpile(input_circuit, target=target, optimization_level=3,
                                  seed_transpiler=pre_seed, approximation_degree=approx))
            except Exception:
                pass

    option_sets = (
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "dense", "routing_method": "sabre"},
    )

    p1_end = time_limit * 0.48
    for qc in source_circuits:
        if elapsed() > p1_end:
            break
        for seed in range(600):
            if elapsed() > p1_end:
                break
            for kw in option_sets:
                try:
                    _update(transpile(qc, target=target, seed_transpiler=seed, **kw))
                except Exception:
                    pass

    if elapsed() < time_limit * 0.52:
        for s, cand in list(top_k):
            try:
                _update(_post_optimize(cand, target))
            except Exception:
                pass

    p2_end = time_limit * 0.92
    reopt_opts = (
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
    )

    for _round in range(25):
        if elapsed() > p2_end:
            break
        improved = False
        for _, cand in list(top_k):
            if elapsed() > p2_end:
                break
            for seed in range(400):
                if elapsed() > p2_end:
                    break
                for ro in reopt_opts:
                    try:
                        old = best_score
                        _update(transpile(cand, target=target, seed_transpiler=seed, **ro))
                        if best_score < old:
                            improved = True
                            try:
                                _update(_post_optimize(best, target))
                            except Exception:
                                pass
                    except Exception:
                        pass
        if not improved:
            break

    if best is not None and elapsed() < time_limit * 0.98:
        for seed in range(500):
            if elapsed() > time_limit * 0.98:
                break
            try:
                _update(transpile(best, target=target, seed_transpiler=seed, optimization_level=3))
            except Exception:
                pass

    return best if best is not None else qc_rewritten
# EVOLVE-BLOCK-END
