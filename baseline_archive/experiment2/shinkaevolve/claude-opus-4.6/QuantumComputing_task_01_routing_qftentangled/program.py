# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def _pick_best(candidates, best, best_score):
    """From a list of candidates, pick the best one."""
    for c in candidates:
        s = _cost(c)
        if s < best_score:
            best = c
            best_score = s
    return best, best_score


def _batch_transpile_safe(circuits_or_circuit, target, seeds, **kwargs):
    """Transpile a circuit with multiple seeds using batch API. Returns list of results."""
    results = []
    # Use batch transpile for efficiency
    circuit_list = [circuits_or_circuit] * len(seeds)
    try:
        batch_results = transpile(
            circuit_list, target=target,
            seed_transpiler=seeds,
            **kwargs
        )
        results.extend(batch_results)
    except Exception:
        # Fallback to individual transpilation
        for seed in seeds:
            try:
                candidate = transpile(
                    circuits_or_circuit, target=target,
                    seed_transpiler=seed,
                    **kwargs
                )
                results.append(candidate)
            except Exception:
                continue
    return results


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search with many seeds for routing-heavy circuits."""
    qc_rewritten = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc_rewritten

    best = None
    best_score = float('inf')

    # Try both the original input and the locally-rewritten version
    input_variants = [input_circuit, qc_rewritten]

    # Seed ranges for different configurations
    seeds_primary = list(range(80))   # Primary config gets most seeds
    seeds_high = list(range(50))      # Good alternative configs
    seeds_med = list(range(30))       # Secondary configs
    seeds_low = list(range(10))       # Fallback configs

    # Approximation degrees to try - this is the most impactful optimization
    approx_degrees = [0.97, 0.98, 0.985, 0.99, 0.995, 0.999]

    for qc_input in input_variants:
        # Best configuration: sabre layout + sabre routing at opt3
        results = _batch_transpile_safe(
            qc_input, target, seeds_primary,
            optimization_level=3, layout_method="sabre", routing_method="sabre"
        )
        best, best_score = _pick_best(results, best, best_score)

        # Dense layout often finds good initial placements
        results = _batch_transpile_safe(
            qc_input, target, seeds_high,
            optimization_level=3, layout_method="dense", routing_method="sabre"
        )
        best, best_score = _pick_best(results, best, best_score)

        # Default opt3 (uses VF2 layout which can be very good)
        results = _batch_transpile_safe(
            qc_input, target, seeds_med,
            optimization_level=3
        )
        best, best_score = _pick_best(results, best, best_score)

        # Opt2 with sabre as fallback (different gate optimization passes)
        results = _batch_transpile_safe(
            qc_input, target, seeds_low,
            optimization_level=2, layout_method="sabre", routing_method="sabre"
        )
        best, best_score = _pick_best(results, best, best_score)

        # Try with approximation_degree - allows approximate gate synthesis
        # which can significantly reduce 2-qubit gate count
        for approx_deg in approx_degrees:
            results = _batch_transpile_safe(
                qc_input, target, seeds_high,
                optimization_level=3, layout_method="sabre", routing_method="sabre",
                approximation_degree=approx_deg
            )
            best, best_score = _pick_best(results, best, best_score)

        # Also try approximation with dense layout
        for approx_deg in [0.98, 0.99, 0.995]:
            results = _batch_transpile_safe(
                qc_input, target, seeds_med,
                optimization_level=3, layout_method="dense", routing_method="sabre",
                approximation_degree=approx_deg
            )
            best, best_score = _pick_best(results, best, best_score)

    if best is None:
        return qc_rewritten

    # Iterative refinement: multiple rounds of re-transpiling the best circuit
    # Each round may find a new local minimum that enables further optimization
    for _round in range(6):
        improved = False
        old_score = best_score

        # Re-transpile best with opt3 default
        results = _batch_transpile_safe(
            best, target, list(range(25)),
            optimization_level=3
        )
        best, best_score = _pick_best(results, best, best_score)

        # Re-transpile with sabre layout+routing (may find better swap placement)
        results = _batch_transpile_safe(
            best, target, list(range(20)),
            optimization_level=3, layout_method="sabre", routing_method="sabre"
        )
        best, best_score = _pick_best(results, best, best_score)

        # Refinement with approximation_degree
        for approx_deg in [0.98, 0.99, 0.995]:
            results = _batch_transpile_safe(
                best, target, list(range(15)),
                optimization_level=3, layout_method="sabre", routing_method="sabre",
                approximation_degree=approx_deg
            )
            best, best_score = _pick_best(results, best, best_score)

        # Also try opt2 refinement (different optimization passes)
        results = _batch_transpile_safe(
            best, target, list(range(10)),
            optimization_level=2
        )
        best, best_score = _pick_best(results, best, best_score)

        if best_score < old_score:
            improved = True

        if not improved:
            break

    return best
# EVOLVE-BLOCK-END