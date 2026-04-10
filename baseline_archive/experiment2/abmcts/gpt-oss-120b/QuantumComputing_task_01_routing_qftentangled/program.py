# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from structural_optimizer import optimize_by_local_rewrite


def _cost(qc: QuantumCircuit) -> float:
    """Cost used for comparison: two‑qubit gate count + 0.2 × depth."""
    two_qubit_cnt = sum(inst.operation.num_qubits == 2 for inst in qc.data)
    return two_qubit_cnt + 0.2 * qc.depth()


def _transpile_candidates(
    base_circuit: QuantumCircuit, target: Target, num_qubits: int
) -> list[QuantumCircuit]:
    """Generate a list of candidate circuits via extensive transpile search."""
    candidates: list[QuantumCircuit] = []

    # Different layout and routing configurations to explore.
    option_sets = (
        {"optimization_level": 3, "layout_method": "sabre", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "lookahead", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "sabre"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "basic"},
        {"optimization_level": 3, "layout_method": "trivial", "routing_method": "stochastic"},
        {"optimization_level": 3, "layout_method": "dense", "routing_method": "stochastic"},
        {"optimization_level": 2},
        {"optimization_level": 1},
        {},  # default (level 0)
    )

    # A few diverse seeds to break deterministic tie‑breaks.
    seeds = (
        num_qubits + 5,
        num_qubits + 11,
        num_qubits + 17,
        num_qubits + 23,
        num_qubits + 29,
    )

    for seed in seeds:
        for opts in option_sets:
            try:
                cand = transpile(
                    base_circuit,
                    target=target,
                    seed_transpiler=seed,
                    **opts,
                )
                candidates.append(cand)
            except Exception:
                # Silently ignore configurations that fail for this target.
                continue
    return candidates


def _final_polish(circuit: QuantumCircuit, target: Target) -> QuantumCircuit:
    """Apply a final high‑level transpilation followed by a local rewrite."""
    try:
        polished = transpile(
            circuit,
            target=target,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=42,
        )
    except Exception:
        polished = circuit
    polished = optimize_by_local_rewrite(polished)
    return polished


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target‑aware optimisation: local rewrite + extensive transpile search + per‑candidate polish."""
    # Initial cheap clean‑up.
    base_circuit = optimize_by_local_rewrite(input_circuit)

    if target is None:
        return base_circuit

    num_qubits = case.get("num_qubits", base_circuit.num_qubits)

    # Gather raw candidates from the extensive search.
    raw_candidates = _transpile_candidates(base_circuit, target, num_qubits)

    # Ensure the cleaned circuit itself is also considered.
    raw_candidates.append(base_circuit)

    # Polish each candidate individually and keep the best according to cost.
    best_circuit = None
    best_cost = float("inf")
    for cand in raw_candidates:
        polished = _final_polish(cand, target)
        cand_cost = _cost(polished)
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_circuit = polished

    # Fallback (should never happen) – return the base circuit.
    return best_circuit if best_circuit is not None else base_circuit
# EVOLVE-BLOCK-END
