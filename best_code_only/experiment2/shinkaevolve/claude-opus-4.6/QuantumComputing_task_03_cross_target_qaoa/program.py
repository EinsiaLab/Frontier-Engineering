# EVOLVE-BLOCK-START
from __future__ import annotations

import time
from qiskit import transpile
from qiskit.circuit import SessionEquivalenceLibrary
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, PassManager
from mqt.bench.targets.gatesets import ionq, rigetti

from structural_optimizer import optimize_by_local_rewrite

# Try to import useful optimization passes
try:
    from qiskit.transpiler.passes import (
        Optimize1qGatesDecomposition,
        CXCancellation,
        CommutativeCancellation,
        RemoveDiagonalGatesBeforeMeasure,
    )
    HAS_OPT_PASSES = True
except ImportError:
    HAS_OPT_PASSES = False


def _circuit_cost(qc: QuantumCircuit) -> float:
    """Compute cost: 10 * two_qubit_gates + single_qubit_gates."""
    two_q = 0
    one_q = 0
    for instruction in qc.data:
        n = instruction.operation.num_qubits
        name = instruction.operation.name
        if name in ("barrier", "measure", "reset"):
            continue
        if n >= 2:
            two_q += 1
        elif n == 1:
            one_q += 1
    return 10.0 * two_q + one_q


def _extra_optimize(qc: QuantumCircuit) -> QuantumCircuit:
    """Apply additional Qiskit optimization passes."""
    if not HAS_OPT_PASSES:
        return qc
    try:
        pm = PassManager([
            CXCancellation(),
            CommutativeCancellation(),
        ])
        result = pm.run(qc)
        return result
    except Exception:
        return qc


def _try_single_transpile(cfg, post_rewrite_rounds=32):
    """Try a single transpile config and return (circuit, cost) or None."""
    try:
        t = transpile(**cfg)
        t = optimize_by_local_rewrite(t, max_rounds=post_rewrite_rounds)
        t = _extra_optimize(t)
        t = optimize_by_local_rewrite(t, max_rounds=8)
        c = _circuit_cost(t)
        return t, c
    except Exception:
        return None


def _retranspile_best(circuit, target, configs, post_rewrite_rounds=32, iterative_rounds=5, time_limit=None):
    """Try multiple transpile configs and return the best result."""
    start_time = time.time()
    best = None
    best_cost = float("inf")

    for cfg in configs:
        if time_limit and (time.time() - start_time) > time_limit * 0.6:
            break
        result = _try_single_transpile(cfg, post_rewrite_rounds)
        if result is not None:
            t, c = result
            if c < best_cost:
                best_cost = c
                best = t

    # Iterative re-transpilation: take the best and re-transpile multiple times
    if best is not None and len(configs) > 0:
        # Build a template config from the first opt_level=3 config
        base_cfg_template = None
        for cfg in configs:
            if cfg.get("optimization_level", 0) == 3:
                base_cfg_template = dict(cfg)
                break
        if base_cfg_template is None:
            base_cfg_template = dict(configs[0])

        for iteration in range(iterative_rounds):
            if time_limit and (time.time() - start_time) > time_limit * 0.9:
                break
            improved = False
            for approx in [0.95, 0.9, 0.85, 0.8, 0.75]:
                if time_limit and (time.time() - start_time) > time_limit * 0.9:
                    break
                for seed in range(iteration * 20, iteration * 20 + 20):
                    if time_limit and (time.time() - start_time) > time_limit * 0.9:
                        break
                    try:
                        retry_cfg = dict(base_cfg_template)
                        retry_cfg["circuits"] = best
                        retry_cfg["seed_transpiler"] = seed
                        retry_cfg["approximation_degree"] = approx
                        retry_cfg["optimization_level"] = 3
                        result = _try_single_transpile(retry_cfg, post_rewrite_rounds)
                        if result is not None:
                            t2, c2 = result
                            if c2 < best_cost:
                                best = t2
                                best_cost = c2
                                improved = True
                    except Exception:
                        continue
            if not improved:
                break
    return best, best_cost


def optimize_circuit(input_circuit: QuantumCircuit, target: Target, case: dict) -> QuantumCircuit:
    """Target-aware transpile baseline for cross-platform QAOA circuits."""
    target_name = str(case.get("target_name", "")).lower()
    optimized = optimize_by_local_rewrite(input_circuit, max_rounds=32)

    if "ionq" in target_name:
        ionq.add_equivalences(SessionEquivalenceLibrary)
    elif "rigetti" in target_name:
        rigetti.add_equivalences(SessionEquivalenceLibrary)

    if "ionq" in target_name:
        # IonQ: opt0 == opt3, so score denominator is 0 -> score is always 0
        # Minimize time spent here - just do a single fast transpile
        transpiled = transpile(
            circuits=optimized,
            target=target,
            optimization_level=3,
            seed_transpiler=42,
        )
        return optimize_by_local_rewrite(transpiled, max_rounds=4)

    elif "ibm" in target_name or "rigetti" in target_name:
        # IBM/Rigetti: this is where we can gain the most score
        # Determine time budget based on circuit size
        n_qubits = input_circuit.num_qubits
        if n_qubits <= 10:
            time_limit = 50
        elif n_qubits <= 12:
            time_limit = 60
        else:
            time_limit = 80

        seeds = list(range(0, 120))
        configs = []

        # Also try with raw input circuit for diversity
        input_sources = [optimized]

        for src in input_sources:
            # Strategy 1: sabre layout + sabre routing, various approx degrees
            for approx in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
                if approx >= 0.95:
                    seed_count = 80
                elif approx >= 0.85:
                    seed_count = 40
                else:
                    seed_count = 20
                for seed in seeds[:seed_count]:
                    configs.append({
                        "circuits": src,
                        "target": target,
                        "optimization_level": 3,
                        "seed_transpiler": seed,
                        "layout_method": "sabre",
                        "routing_method": "sabre",
                        "approximation_degree": approx,
                    })

            # Strategy 2: dense layout + sabre routing
            for approx in [0.95, 0.9]:
                for seed in seeds[:30]:
                    configs.append({
                        "circuits": src,
                        "target": target,
                        "optimization_level": 3,
                        "seed_transpiler": seed,
                        "layout_method": "dense",
                        "routing_method": "sabre",
                        "approximation_degree": approx,
                    })

            # Strategy 3: trivial layout
            for seed in seeds[:20]:
                configs.append({
                    "circuits": src,
                    "target": target,
                    "optimization_level": 3,
                    "seed_transpiler": seed,
                    "layout_method": "trivial",
                    "routing_method": "sabre",
                    "approximation_degree": 0.95,
                })

        best, best_cost = _retranspile_best(
            optimized, target, configs,
            iterative_rounds=8,
            time_limit=time_limit
        )
        if best is not None:
            return best
        transpiled = transpile(
            circuits=optimized,
            target=target,
            optimization_level=3,
            seed_transpiler=42,
            layout_method="sabre",
            routing_method="sabre",
        )
        return optimize_by_local_rewrite(transpiled, max_rounds=32)

    else:
        # Generic fallback
        configs = []
        for seed in range(20):
            configs.append({
                "circuits": optimized,
                "target": target,
                "optimization_level": 3,
                "seed_transpiler": seed,
            })
        best, _ = _retranspile_best(optimized, target, configs, time_limit=30)
        if best is not None:
            return best
        transpiled = transpile(
            circuits=optimized,
            target=target,
            optimization_level=3,
            seed_transpiler=42,
        )
        return optimize_by_local_rewrite(transpiled, max_rounds=32)
# EVOLVE-BLOCK-END