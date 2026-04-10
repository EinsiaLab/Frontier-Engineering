# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _compute_cost(circuit: QuantumCircuit) -> float:
    """Compute synthesis cost matching evaluation metric exactly."""
    ops = circuit.count_ops()
    t_count = ops.get('t', 0)
    tdg_count = ops.get('tdg', 0)
    two_qubit = ops.get('cx', 0) + ops.get('swap', 0) + ops.get('ecr', 0)
    return (t_count + tdg_count) + 0.2 * two_qubit + 0.05 * circuit.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Multi-strategy optimization exploring target-based and approximation approaches."""
    basis = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
    best = None
    best_cost = float('inf')
    
    # Strategy 1: Use target parameter directly with optimization_level=3
    for seed in [42, 123, 456]:
        try:
            trans = transpile(input_circuit, target=target, optimization_level=3, seed_transpiler=seed)
            result = optimize_by_local_rewrite(trans, max_rounds=20)
            cost = _compute_cost(result)
            if cost < best_cost:
                best_cost, best = cost, result
        except Exception:
            pass
    
    # Strategy 2: Basis gates with approximation for T-gate reduction
    for approx in [0.9, 0.8, 0.7]:
        for seed in [42, 123]:
            try:
                trans = transpile(
                    input_circuit, basis_gates=basis, optimization_level=3,
                    approximation_degree=approx, seed_transpiler=seed
                )
                result = optimize_by_local_rewrite(trans, max_rounds=20)
                cost = _compute_cost(result)
                if cost < best_cost:
                    best_cost, best = cost, result
            except Exception:
                pass
    
    # Strategy 3: Pre-rewrite + transpile + post-rewrite pipeline
    for seed in [42, 456, 789]:
        pre_opt = optimize_by_local_rewrite(input_circuit, max_rounds=20)
        trans = transpile(pre_opt, basis_gates=basis, optimization_level=3, seed_transpiler=seed)
        result = optimize_by_local_rewrite(trans, max_rounds=20)
        cost = _compute_cost(result)
        if cost < best_cost:
            best_cost, best = cost, result
    
    # Strategy 4: Use target with approximation
    for approx in [0.85, 0.75]:
        try:
            trans = transpile(
                input_circuit, target=target, optimization_level=3,
                approximation_degree=approx, seed_transpiler=42
            )
            result = optimize_by_local_rewrite(trans, max_rounds=20)
            cost = _compute_cost(result)
            if cost < best_cost:
                best_cost, best = cost, result
        except Exception:
            pass
    
    return best if best is not None else input_circuit
# EVOLVE-BLOCK-END
