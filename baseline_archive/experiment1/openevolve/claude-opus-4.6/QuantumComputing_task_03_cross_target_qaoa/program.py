# EVOLVE-BLOCK-START
from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite


def _circuit_cost(qc: QuantumCircuit) -> float:
    """Match the actual scoring metric: two_qubit_count + 0.2 * depth."""
    two_q = sum(1 for inst in qc.data if inst.operation.num_qubits >= 2)
    return two_q + 0.2 * qc.depth()


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile with multi-seed search for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=40)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    base_kwargs = {
        "target": target,
        "optimization_level": 3,
    }
    if "ionq" in target_name:
        base_kwargs["basis_gates"] = ["rz", "sx", "x", "rzz", "measure"]
    if "ibm" in target_name or "rigetti" in target_name:
        base_kwargs.update({
            "layout_method": "sabre",
            "routing_method": "sabre",
            "approximation_degree": 0.97,
        })

    # For IonQ, try multiple seeds and optimization strategies
    if "ionq" in target_name:
        import time
        t0_ionq = time.time()
        best_ionq = None
        best_ionq_cost = float("inf")
        
        # Try multiple optimization levels and seeds
        for opt_lvl in [3, 2, 1]:
            kw = dict(base_kwargs)
            kw["optimization_level"] = opt_lvl
            for seed in range(30):
                if time.time() - t0_ionq > 15:
                    break
                try:
                    t = transpile(circuits=optimized, seed_transpiler=seed, **kw)
                    t = optimize_by_local_rewrite(t, max_rounds=40)
                    c = _circuit_cost(t)
                    if c < best_ionq_cost:
                        best_ionq_cost = c
                        best_ionq = t
                except Exception:
                    continue
        
        # Try with unitary synthesis method variations
        for approx in [0.9, 0.95, 0.99, 1.0]:
            kw = dict(base_kwargs)
            kw["approximation_degree"] = approx
            for seed in range(10):
                if time.time() - t0_ionq > 20:
                    break
                try:
                    t = transpile(circuits=optimized, seed_transpiler=seed, **kw)
                    t = optimize_by_local_rewrite(t, max_rounds=40)
                    c = _circuit_cost(t)
                    if c < best_ionq_cost:
                        best_ionq_cost = c
                        best_ionq = t
                except Exception:
                    continue
        
        if best_ionq is None:
            best_ionq = transpile(circuits=optimized, seed_transpiler=42, **base_kwargs)
            best_ionq = optimize_by_local_rewrite(best_ionq, max_rounds=40)
        
        # Iterative re-transpilation for IonQ
        for _pass in range(10):
            if time.time() - t0_ionq > 25:
                break
            improved = False
            for seed in range(15):
                if time.time() - t0_ionq > 25:
                    break
                try:
                    t2 = transpile(circuits=best_ionq, target=target, optimization_level=3,
                                   seed_transpiler=seed)
                    t2 = optimize_by_local_rewrite(t2, max_rounds=40)
                    c2 = _circuit_cost(t2)
                    if c2 < best_ionq_cost:
                        best_ionq_cost = c2
                        best_ionq = t2
                        improved = True
                except Exception:
                    continue
            if not improved:
                break
        
        return best_ionq

    import time
    t0 = time.time()
    t_budget = 80

    best = None
    best_cost = float("inf")

    def _try(qc, **kw):
        nonlocal best, best_cost
        try:
            t = transpile(circuits=qc, **kw)
            t = optimize_by_local_rewrite(t, max_rounds=50)
            c = _circuit_cost(t)
            if c < best_cost:
                best_cost = c
                best = t
        except Exception:
            pass

    # Phase 1: broad opt_level=3 search with sabre
    for seed in range(300):
        if time.time() - t0 > t_budget * 0.3:
            break
        _try(optimized, seed_transpiler=seed, **base_kwargs)

    # Phase 2: vary opt levels, approx degrees, and layout methods
    for layout in ["sabre", "dense", "trivial"]:
        for opt_lvl in [3, 2]:
            for approx in [0.95, 0.97, 0.99, 1.0]:
                kw = dict(base_kwargs)
                kw["optimization_level"] = opt_lvl
                kw["approximation_degree"] = approx
                kw["layout_method"] = layout
                if layout != "sabre":
                    kw["routing_method"] = "sabre"
                for seed in range(15):
                    if time.time() - t0 > t_budget * 0.5:
                        break
                    _try(optimized, seed_transpiler=seed, **kw)

    if best is None:
        best = transpile(circuits=optimized, seed_transpiler=42, **base_kwargs)
        best = optimize_by_local_rewrite(best, max_rounds=50)
        best_cost = _circuit_cost(best)

    # Phase 3: iterative re-transpilation from best with more budget
    for _pass in range(60):
        if time.time() - t0 > t_budget * 0.95:
            break
        improved = False
        for opt_lvl in [3, 2, 1]:
            for approx in [0.95, 0.97, 1.0]:
                for seed in range(20):
                    if time.time() - t0 > t_budget * 0.95:
                        break
                    try:
                        t2 = transpile(circuits=best, target=target, optimization_level=opt_lvl,
                                       seed_transpiler=seed, approximation_degree=approx)
                        t2 = optimize_by_local_rewrite(t2, max_rounds=50)
                        c2 = _circuit_cost(t2)
                        if c2 < best_cost:
                            best_cost = c2
                            best = t2
                            improved = True
                    except Exception:
                        continue
        if not improved:
            break

    return best
# EVOLVE-BLOCK-END
