# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    return sum(inst.operation.num_qubits == 2 for inst in qc.data) + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile search baseline for routing-heavy circuits."""
    qc = optimize_by_local_rewrite(input_circuit)
    if target is None:
        return qc

    num_qubits = case.get("num_qubits", input_circuit.num_qubits)
    best = qc
    best_score = _cost(qc)
    # Expanded search space: more layout/routing combos, additional optimization levels,
    # and a larger set of random seeds.
    option_sets = (
        # High‑level optimizations with various layout/routing strategies
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "lookahead", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "basic"},
        # Additional promising combos
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "basic"},
        {"optimization_level": 2, "layout_method": "trivial", "routing_method": "basic"},
        # Lower optimization levels with the same layout/routing combos
        {"optimization_level": 2, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "lookahead", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 2, "layout_method": "trivial", "routing_method": "sabre"},
        {"optimization_level": 1, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 1, "layout_method": "lookahead", "routing_method": "sabre"},
        # Simple fallback with only optimization level set
        {"optimization_level": 3},
        {"optimization_level": 2},
        {"optimization_level": 1},
    )
    # Use a broader range of seeds to diversify the random search.
    # Expanded set of seed offsets for a richer random search space
    seed_offsets = (5, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
    for seed in (num_qubits + offset for offset in seed_offsets):
        for transpile_kwargs in option_sets:
            # If the option set uses SABRE routing, add a moderate approximation degree
            # to reduce SWAP overhead without sacrificing aggressive optimisation.
            if transpile_kwargs.get("routing_method") == "sabre":
                transpile_kwargs = dict(transpile_kwargs)  # copy to avoid mutating the tuple entry
                transpile_kwargs["approximation_degree"] = 0.5
            try:
                candidate = transpile(
                    qc,
                    target=target,
                    seed_transpiler=seed,
                    basis_gates=target.operation_names,
                    **transpile_kwargs,
                )
                # Run the lightweight structural optimizer again to capture any new local
                # simplifications introduced by the transpilation step.
                candidate = optimize_by_local_rewrite(candidate)
            except Exception:
                continue
            score = _cost(candidate)
            if score < best_score:
                best = candidate
                best_score = score
    # Apply a final local rewrite to the best circuit found
    best = optimize_by_local_rewrite(best)

    # -----------------------------------------------------------------
    # Final deterministic aggressive optimization pass (restored from previous high‑score version)
    # -----------------------------------------------------------------
    try:
        final_candidate = transpile(
            best,
            target=target,
            seed_transpiler=0,
            basis_gates=target.operation_names,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            approximation_degree=0.5,
        )
        # One more lightweight structural rewrite to capture any new simplifications
        best = optimize_by_local_rewrite(final_candidate)
    except Exception:
        # If the final aggressive pass fails (e.g., unsupported layout), keep the best found so far.
        pass

    # -----------------------------------------------------------------
    # Additional aggressive passes: try alternative layouts and a few extra seeds
    # -----------------------------------------------------------------
    # First, explore a couple of different layout methods on the current best circuit.
    for _layout in ("dense", "lookahead"):
        try:
            _candidate = transpile(
                best,
                target=target,
                seed_transpiler=0,
                basis_gates=target.operation_names,
                optimization_level=3,
                layout_method=_layout,
                routing_method="sabre",
                approximation_degree=0.5,
            )
            _candidate = optimize_by_local_rewrite(_candidate)
            if _cost(_candidate) < _cost(best):
                best = _candidate
        except Exception:
            continue

    # Then, run a few extra seeds with the standard sabre layout.
    for _seed in (1, 2, 3, 4, 5, 7, 13):
        try:
            _candidate = transpile(
                best,
                target=target,
                seed_transpiler=_seed,
                basis_gates=target.operation_names,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                approximation_degree=0.5,
            )
            _candidate = optimize_by_local_rewrite(_candidate)
            if _cost(_candidate) < _cost(best):
                best = _candidate
        except Exception:
            continue

    # Additional deterministic refinement passes: re‑apply the aggressive
    # sabre layout/routing with seed 0 up to two times, stopping early if
    # no further improvement is observed.
    for _ in range(2):
        try:
            refined = transpile(
                best,
                target=target,
                seed_transpiler=0,
                basis_gates=target.operation_names,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                approximation_degree=0.5,
            )
            refined = optimize_by_local_rewrite(refined)
            if _cost(refined) < _cost(best):
                best = refined
            else:
                break
        except Exception:
            break

    return best
# EVOLVE-BLOCK-END