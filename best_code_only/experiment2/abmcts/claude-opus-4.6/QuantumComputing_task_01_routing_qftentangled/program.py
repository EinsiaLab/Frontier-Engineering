# EVOLVE-BLOCK-START
from __future__ import annotations

import time
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search with extensive seed/option exploration."""
    TIME_BUDGET = 85.0  # seconds per case
    t0 = time.time()
    
    qc_rewritten = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc_rewritten

    num_qubits = case.get("num_qubits", input_circuit.num_qubits)
    best = None
    best_score = float('inf')

    # Try both the rewritten and original circuit as starting points
    starting_circuits = [qc_rewritten, input_circuit]

    # Option sets ordered by expected quality
    option_sets = [
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 2},
    ]

    seeds = list(range(0, 500, 3))

    def elapsed():
        return time.time() - t0

    # Phase 1: broad search over starting circuits, options, and seeds
    for src_qc in starting_circuits:
        for transpile_kwargs in option_sets:
            for seed in seeds:
                if elapsed() > TIME_BUDGET * 0.55:
                    break
                try:
                    candidate = transpile(src_qc, target=target, seed_transpiler=seed, **transpile_kwargs)
                except Exception:
                    continue
                score = _cost(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
            if elapsed() > TIME_BUDGET * 0.55:
                break
        if elapsed() > TIME_BUDGET * 0.55:
            break

    if best is None:
        best = qc_rewritten

    # Phase 2: iterative re-optimization of best circuit
    iteration = 0
    while elapsed() < TIME_BUDGET * 0.85:
        improved = False
        prev_best = best
        prev_score = best_score
        for seed in seeds:
            if elapsed() > TIME_BUDGET * 0.85:
                break
            for opt_level in [3, 2]:
                if elapsed() > TIME_BUDGET * 0.85:
                    break
                try:
                    candidate = transpile(best, target=target, seed_transpiler=seed,
                                          optimization_level=opt_level)
                except Exception:
                    continue
                score = _cost(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
        iteration += 1
        if not improved:
            break

    # Phase 3: final polish - try generating fresh circuits and re-transpiling
    while elapsed() < TIME_BUDGET * 0.95:
        seed = int((elapsed() * 1000) % 100000)
        try:
            candidate = transpile(input_circuit, target=target, seed_transpiler=seed,
                                  optimization_level=3, layout_method="sabre", routing_method="sabre")
        except Exception:
            continue
        score = _cost(candidate)
        if score < best_score:
            best = candidate
            best_score = score
            # Try to further optimize this new best
            for s2 in range(0, 50, 3):
                if elapsed() > TIME_BUDGET * 0.95:
                    break
                try:
                    c2 = transpile(best, target=target, seed_transpiler=s2, optimization_level=3)
                except Exception:
                    continue
                s = _cost(c2)
                if s < best_score:
                    best = c2
                    best_score = s

    return best
# EVOLVE-BLOCK-END
