# EVOLVE-BLOCK-START
from __future__ import annotations

import time
import random
import sys
import os
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager

# Add baseline directory to path so structural_optimizer can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baseline'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from structural_optimizer import optimize_by_local_rewrite
except ImportError:
    # Fallback: define a no-op if not available
    def optimize_by_local_rewrite(qc, max_rounds=20):
        return qc


def _compute_cost(qc: QuantumCircuit) -> float:
    """Cost = (T + Tdg) + 0.2 * two_qubit_count + 0.05 * depth"""
    t_count = 0
    two_qubit_count = 0
    for inst in qc.data:
        name = inst.operation.name
        if name in ('t', 'tdg'):
            t_count += 1
        if len(inst.qubits) >= 2:
            two_qubit_count += 1
    depth = qc.depth()
    return t_count + 0.2 * two_qubit_count + 0.05 * depth


def _try_transpile(circuit, basis_gates, opt_level, seed, target=None):
    """Transpile with given settings and return result."""
    kwargs = {
        'basis_gates': basis_gates,
        'optimization_level': opt_level,
        'seed_transpiler': seed,
    }
    if target is not None:
        kwargs['target'] = target
        del kwargs['basis_gates']
    return transpile(circuit, **kwargs)


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Optimized Clifford+T synthesis with multiple strategies and seed search."""
    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
    
    start_time = time.time()
    time_budget = 90.0  # seconds per case
    
    best_circuit = None
    best_cost = float('inf')
    
    def update_best(qc):
        nonlocal best_circuit, best_cost
        try:
            cost = _compute_cost(qc)
            if cost < best_cost:
                best_cost = cost
                best_circuit = qc.copy()
            return cost
        except Exception:
            return float('inf')
    
    # Strategy 1: Local rewrite first, then transpile
    try:
        optimized_local = optimize_by_local_rewrite(input_circuit, max_rounds=30)
    except Exception:
        optimized_local = input_circuit
    
    # Try multiple optimization levels and seeds on the locally-optimized circuit
    for opt_level in [3, 2, 1]:
        for seed in range(25):
            if time.time() - start_time > time_budget * 0.35:
                break
            try:
                transpiled = _try_transpile(optimized_local, basis_gates, opt_level, seed)
                update_best(transpiled)
                post = optimize_by_local_rewrite(transpiled, max_rounds=30)
                update_best(post)
            except Exception:
                pass
    
    # Try transpiling the raw input directly with multiple seeds
    for opt_level in [3, 2, 1]:
        for seed in range(20):
            if time.time() - start_time > time_budget * 0.6:
                break
            try:
                transpiled = _try_transpile(input_circuit, basis_gates, opt_level, seed)
                update_best(transpiled)
                post = optimize_by_local_rewrite(transpiled, max_rounds=30)
                update_best(post)
            except Exception:
                pass
    
    # If we have time left, try re-optimizing the best circuit found so far
    if best_circuit is not None and time.time() - start_time < time_budget * 0.8:
        current_best = best_circuit.copy()
        for seed in range(40):
            if time.time() - start_time > time_budget * 0.9:
                break
            try:
                re_transpiled = _try_transpile(current_best, basis_gates, 3, seed)
                update_best(re_transpiled)
                post = optimize_by_local_rewrite(re_transpiled, max_rounds=30)
                update_best(post)
            except Exception:
                pass
    
    # Final local rewrite pass
    if best_circuit is not None:
        try:
            final = optimize_by_local_rewrite(best_circuit, max_rounds=50)
            update_best(final)
        except Exception:
            pass
        return best_circuit
    
    # Fallback
    try:
        transpiled = transpile(
            input_circuit,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=42,
        )
        result = optimize_by_local_rewrite(transpiled, max_rounds=20)
        return result
    except Exception:
        return input_circuit
# EVOLVE-BLOCK-END
