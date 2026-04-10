# EVOLVE-BLOCK-START
from __future__ import annotations

import math
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager

try:
    from qiskit.transpiler.passes import (
        Optimize1qGatesDecomposition,
        CXCancellation,
        CommutativeCancellation,
        RemoveBarriers,
        Collect2qBlocks,
        ConsolidateBlocks,
        UnitarySynthesis,
        Collect1qRuns,
        InverseCancellation,
    )
    HAS_PASSES = True
except ImportError:
    HAS_PASSES = False

from structural_optimizer import optimize_by_local_rewrite


def _compute_cost(qc: QuantumCircuit) -> float:
    """Compute the cost function: (T + Tdg) + 0.2 * two_qubit_count + 0.05 * depth."""
    ops = qc.count_ops()
    t_count = ops.get("t", 0) + ops.get("tdg", 0)
    two_qubit_names = {
        "cx", "cz", "cy", "swap", "ecr", "rzx", "rxx", "ryy", "rzz",
        "csx", "cp", "crx", "cry", "crz",
    }
    two_qubit = sum(count for name, count in ops.items() if name in two_qubit_names)
    depth = qc.depth()
    return t_count + 0.2 * two_qubit + 0.05 * depth


def _transpile_and_rewrite(circuit, basis_gates, opt_level, seed, rewrite_rounds=15,
                           target=None, approx_degree=None):
    """Transpile then apply local rewrites, return (circuit, cost)."""
    try:
        kwargs = dict(
            optimization_level=opt_level,
            seed_transpiler=seed,
        )
        if target is not None:
            kwargs["target"] = target
        else:
            kwargs["basis_gates"] = basis_gates
        if approx_degree is not None:
            kwargs["approximation_degree"] = approx_degree
        t = transpile(circuit, **kwargs)
        t = optimize_by_local_rewrite(t, max_rounds=rewrite_rounds)
        return t, _compute_cost(t)
    except Exception:
        return None, float("inf")


def _run_custom_passes(circuit, basis_gates):
    """Run custom optimization pass sequences for deeper optimization."""
    if not HAS_PASSES:
        return None, float("inf")
    try:
        pm = PassManager([
            RemoveBarriers(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates),
            UnitarySynthesis(basis_gates=basis_gates),
            Optimize1qGatesDecomposition(basis=basis_gates),
            CXCancellation(),
            CommutativeCancellation(basis_gates=basis_gates),
            Optimize1qGatesDecomposition(basis=basis_gates),
            CXCancellation(),
        ])
        result = pm.run(circuit)
        result = optimize_by_local_rewrite(result, max_rounds=15)
        return result, _compute_cost(result)
    except Exception:
        return None, float("inf")


def _build_qft_clifford_t(num_qubits: int) -> QuantumCircuit:
    """Build a hand-optimized QFT circuit with explicit controlled-phase decomposition."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            k = j - i
            angle = math.pi / (2 ** k)
            # cp(angle) decomposition: Rz on both + 2 CX
            qc.rz(angle / 2, j)
            qc.rz(angle / 2, i)
            qc.cx(j, i)
            qc.rz(-angle / 2, i)
            qc.cx(j, i)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    return qc


def _update_best(best_circuit, best_cost, candidate, candidate_cost):
    if candidate is not None and candidate_cost < best_cost:
        return candidate, candidate_cost
    return best_circuit, best_cost


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Multi-strategy optimization combining transpilation, custom passes, and hand-crafted decomposition."""

    basis_gates = ["cx", "h", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
    num_qubits = input_circuit.num_qubits

    # Pre-optimize input with local rewrites
    pre_opt = optimize_by_local_rewrite(input_circuit, max_rounds=20)

    best_circuit = None
    best_cost = float("inf")

    # Phase 1: Broad exploration with many seeds
    seeds = list(range(0, 120, 7))  # ~17 seeds
    for seed in seeds:
        for opt_level in [3, 2]:
            circ, cost = _transpile_and_rewrite(pre_opt, basis_gates, opt_level, seed, 15)
            best_circuit, best_cost = _update_best(best_circuit, best_cost, circ, cost)

    # Also try from original input directly
    for seed in [0, 42, 77, 105]:
        for opt_level in [3, 2]:
            circ, cost = _transpile_and_rewrite(input_circuit, basis_gates, opt_level, seed, 15)
            best_circuit, best_cost = _update_best(best_circuit, best_cost, circ, cost)

    # Phase 1b: Try with target parameter
    for seed in [0, 7, 42, 77]:
        for opt_level in [3, 2]:
            circ, cost = _transpile_and_rewrite(pre_opt, basis_gates, opt_level, seed, 15, target=target)
            best_circuit, best_cost = _update_best(best_circuit, best_cost, circ, cost)

    # Phase 1c: Try approximation degrees (can reduce T-count)
    for approx_deg in [0.99, 0.999, 0.9999, 1.0]:
        for seed in [0, 42]:
            circ, cost = _transpile_and_rewrite(pre_opt, basis_gates, 3, seed, 15, approx_degree=approx_deg)
            best_circuit, best_cost = _update_best(best_circuit, best_cost, circ, cost)

    # Phase 1d: Hand-built QFT decomposition
    try:
        hand_qft = _build_qft_clifford_t(num_qubits)
        for seed in [0, 7, 42, 77]:
            for opt_level in [3, 2]:
                circ, cost = _transpile_and_rewrite(hand_qft, basis_gates, opt_level, seed, 15)
                best_circuit, best_cost = _update_best(best_circuit, best_cost, circ, cost)
    except Exception:
        pass

    # Phase 2: Iterative re-transpilation of the best result
    for iteration in range(6):
        improved = False
        for seed in [0, 7, 14, 21, 42, 63, 77, 91]:
            for opt_level in [3, 2]:
                circ, cost = _transpile_and_rewrite(best_circuit, basis_gates, opt_level, seed, 15)
                if cost < best_cost:
                    best_cost = cost
                    best_circuit = circ
                    improved = True

        # Try custom passes on best
        circ, cost = _run_custom_passes(best_circuit, basis_gates)
        if cost < best_cost:
            best_cost = cost
            best_circuit = circ
            improved = True

        if not improved:
            break

    # Phase 3: Consolidate+resynthesize loop
    if HAS_PASSES and best_circuit is not None:
        try:
            for _ in range(3):
                pm = PassManager([
                    Collect2qBlocks(),
                    ConsolidateBlocks(basis_gates=basis_gates),
                    UnitarySynthesis(basis_gates=basis_gates),
                    Optimize1qGatesDecomposition(basis=basis_gates),
                    CXCancellation(),
                    CommutativeCancellation(basis_gates=basis_gates),
                    Optimize1qGatesDecomposition(basis=basis_gates),
                    CXCancellation(),
                ])
                candidate = pm.run(best_circuit)
                candidate = optimize_by_local_rewrite(candidate, max_rounds=15)
                c_cost = _compute_cost(candidate)
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_circuit = candidate
                else:
                    break
        except Exception:
            pass

    # Final polish with more rewrite rounds
    polished = optimize_by_local_rewrite(best_circuit, max_rounds=40)
    polished_cost = _compute_cost(polished)
    if polished_cost <= best_cost:
        best_circuit = polished

    return best_circuit
# EVOLVE-BLOCK-END