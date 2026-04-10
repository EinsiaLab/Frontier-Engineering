# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def compute_raw_cost(circuit: QuantumCircuit) -> float:
    """Compute approximate cost from raw circuit (before canonicalization)."""
    t_count = 0
    tdg_count = 0
    two_qubit_count = 0
    depth = circuit.depth()
    
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == 't':
            t_count += 1
        elif name == 'tdg':
            tdg_count += 1
        elif len(instruction.qubits) == 2:
            two_qubit_count += 1
    
    return (t_count + tdg_count) + 0.2 * two_qubit_count + 0.05 * depth


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Optimization combining local rewrites with aggressive transpilation and multi-seed selection."""
    # Use fixed basis gates that match clifford+t gateset exactly, as in top-performing programs
    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
    
    # Apply local rewrites with fixed rounds (as in best performing program)
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=20)
    
    # Try multiple seeds to find the best transpilation result, focusing on optimization_level=3
    best_circuit = None
    best_cost = float('inf')
    
    # Use seeds that worked well in previous top programs
    for seed in [42, 123, 789, 1001, 2024]:
        transpiled = transpile(
            optimized,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )
        # Compute raw cost to approximate final cost
        cost = compute_raw_cost(transpiled)
        if cost < best_cost:
            best_cost = cost
            best_circuit = transpiled
    
    # Apply local rewrites on the best circuit with moderate rounds
    return optimize_by_local_rewrite(best_circuit, max_rounds=20)
# EVOLVE-BLOCK-END
