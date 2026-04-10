# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Baseline that combines local rewrites with aggressive phase-aware transpilation."""
    _ = (target, case)

    # First pass: optimize with more rounds for better cleanup
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=15)
    
    # Use standard basis gates that worked well in previous successful attempts
    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
    
    # Phase-aware transpilation: use phase tracking for better optimization
    transpiled = transpile(
        optimized,
        basis_gates=basis_gates,
        optimization_level=3,
        seed_transpiler=42,
        layout_method="sabre",
        routing_method="sabre",
        coupling_map=None,
    )
    
    # Final pass: more aggressive local optimization to clean up any remaining patterns
    return optimize_by_local_rewrite(transpiled, max_rounds=12)
# EVOLVE-BLOCK-END
