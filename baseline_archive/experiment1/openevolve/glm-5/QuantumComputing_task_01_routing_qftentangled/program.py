# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    """Cost function matching evaluation: two_qubit_count + 0.2 * depth"""
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search with proven configuration space."""
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc

    num_qubits = case.get("num_qubits", input_circuit.num_qubits)
    best = None
    best_score = float('inf')
    
    # Expanded option sets with all optimization levels and routing methods
    option_sets = [
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "stochastic"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "stochastic"},
        {"optimization_level": 3},
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 2},
        {"optimization_level": 1, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 1, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 1},
    ]
    
    circuits = [qc, input_circuit]
    # Expanded seed range for more thorough search
    seeds = list(range(0, 50, 2)) + list(range(num_qubits, num_qubits + 40, 2))
    
    for circuit in circuits:
        for seed in seeds:
            for opts in option_sets:
                try:
                    candidate = transpile(circuit, target=target, seed_transpiler=seed, **opts)
                except Exception:
                    continue
                score = _cost(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
    
    # Iterative refinement: re-transpile best circuit to find additional optimizations
    if best is not None:
        for _ in range(3):
            improved = False
            for seed in range(0, 30, 3):
                try:
                    candidate = transpile(best, target=target, seed_transpiler=seed, optimization_level=3)
                    score = _cost(candidate)
                    if score < best_score:
                        best = candidate
                        best_score = score
                        improved = True
                except Exception:
                    continue
            if not improved:
                break
    
    return best if best is not None else qc
# EVOLVE-BLOCK-END
