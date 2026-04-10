# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    # Exactly match official evaluation cost metric for optimal candidate selection
    return qc.num_nonlocal_gates() + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search baseline for routing-heavy circuits."""
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc

    num_qubits = case.get("num_qubits", input_circuit.num_qubits)
    # Initialize best to infinity - only compare fully mapped transpiled circuits
    best = None
    best_score = float('inf')
    option_sets = (
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "stochastic_swap"},
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "basic"},
        {"optimization_level": 3, "layout_method": "lookahead", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "lookahead", "routing_method": "stochastic_swap"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "stochastic_swap"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "noise_adaptive", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "noise_adaptive", "routing_method": "stochastic_swap"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "noise_adaptive", "routing_method": "sabre"},
        {"optimization_level": 2},
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre", "unitary_synthesis_method": "kak"},
        {"optimization_level": 3, "layout_method": "noise_adaptive", "routing_method": "stochastic_swap", "unitary_synthesis_method": "kak"},
        {"optimization_level": 3, "layout_method": "lookahead", "routing_method": "basic"},
        {"optimization_level": 3, "layout_method": "noise_adaptive", "routing_method": "basic"},
        {"optimization_level": 1},
    )
    # Try both original input and rewritten circuit to cover all possible starting points
    for source_qc in (qc, input_circuit):
        for seed in (num_qubits + 5, num_qubits + 17, num_qubits + 29, num_qubits + 41, num_qubits + 53, 42, 99, 123, 777, 2024, 1001, 2048):
            for transpile_kwargs in option_sets:
                try:
                    candidate = transpile(source_qc, target=target, seed_transpiler=seed, **transpile_kwargs)
                    # Remove redundant operations added during transpilation/routing
                    candidate = optimize_by_local_rewrite(candidate)
                except Exception:
                    continue
                score = _cost(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
    # Fallback if no valid candidates found from search
    if best is None:
        best = transpile(input_circuit, target=target, optimization_level=3)
    # Final polish: try multiple seeds to re-optimize best candidate for minimal cost
    polish_scores = {_cost(best): best}
    for polish_seed in (42, 99, 123, 777, 2024, 1001, 2048):
        try:
            polished = transpile(best, target=target, optimization_level=3, seed_transpiler=polish_seed)
            polished = optimize_by_local_rewrite(polished)
            polish_scores[_cost(polished)] = polished
        except Exception:
            continue
    # Select the lowest cost polished candidate
    best = polish_scores[min(polish_scores.keys())]
    # Final cleanup pass on the best candidate
    best = optimize_by_local_rewrite(best)
    return best
# EVOLVE-BLOCK-END
