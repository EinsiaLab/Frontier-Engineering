# EVOLVE-BLOCK-START
"""
pyMOTO SIMP compliance minimization candidate program.

Baseline implementation is intentionally faithful to pyMOTO official examples:
- examples/topology_optimization/ex_compliance.py
- examples/topology_optimization/ex_compliance_69line.py

Allowed to modify:
- solve()
- custom_update_step()
- filter and step-size strategy choices used by solve()

Do not modify:
- output schema in main()
- benchmark constants and IO contract
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pymoto as pym


BENCHMARK_ID = "pymoto_simp_compliance"
NELX = 120
NELY = 40
VOLFRAC = 0.5
FILTER_RADIUS = 2.0
PENAL = 3.0
POISSON_RATIO = 0.3
XMIN = 1e-9
RHO_MIN = 1e-9
MOVE_LIMIT = 0.1
MAX_ITER = 40
FORCE_MAGNITUDE = 1.0
FORCE_DIRECTION = 1  # 0: x, 1: y
VOL_TOL = 1e-3


def _problem() -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "nelx": NELX,
        "nely": NELY,
        "volfrac": VOLFRAC,
        "filter_radius": FILTER_RADIUS,
        "penal": PENAL,
        "poisson_ratio": POISSON_RATIO,
        "xmin": XMIN,
        "rho_min": RHO_MIN,
        "move_limit": MOVE_LIMIT,
        "max_iter": MAX_ITER,
        "force_magnitude": FORCE_MAGNITUDE,
        "force_direction": FORCE_DIRECTION,
    }


def _build_domain_boundary_force(problem: dict[str, Any]) -> tuple[pym.VoxelDomain, np.ndarray, np.ndarray]:
    domain = pym.VoxelDomain(int(problem["nelx"]), int(problem["nely"]))

    # Hard-coded cantilever setup: left edge clamped, point load at right-middle node.
    left_nodes = domain.nodes[0, :].flatten()
    boundary_dofs = np.concatenate([left_nodes * 2, left_nodes * 2 + 1]).astype(int)

    force = np.zeros(domain.nnodes * 2, dtype=float)
    load_node = int(domain.nodes[int(problem["nelx"]), int(problem["nely"] // 2), 0])
    load_dof = 2 * load_node + int(problem["force_direction"])
    force[load_dof] = float(problem["force_magnitude"])

    return domain, boundary_dofs, force


def _build_compliance_network(
    *,
    problem: dict[str, Any],
    x_signal: pym.Signal,
    filter_radius: float,
    penal: float,
    apply_filter: bool,
) -> dict[str, Any]:
    domain, boundary_dofs, force = _build_domain_boundary_force(problem)

    with pym.Network() as function:
        if apply_filter:
            x_analysis = pym.DensityFilter(domain=domain, radius=float(filter_radius))(x_signal)
        else:
            x_analysis = x_signal

        simp = pym.MathExpression(
            expression=f"{problem['xmin']} + {1.0 - problem['xmin']}*inp0^{float(penal)}"
        )(x_analysis)
        stiffness = pym.AssembleStiffness(
            domain=domain,
            bc=boundary_dofs,
            e_modulus=1.0,
            poisson_ratio=float(problem["poisson_ratio"]),
        )(simp)
        displacement = pym.LinSolve(symmetric=True, positive_definite=True)(stiffness, force)
        compliance = pym.EinSum(expression="i,i->")(displacement, force)

        objective_scaled = pym.Scaling(scaling=100.0)(compliance)
        volume_sum = pym.EinSum(expression="i->")(x_analysis)
        volume_constraint = pym.MathExpression(
            expression=f"(inp0/{domain.nel})-{problem['volfrac']}"
        )(volume_sum)

    return {
        "function": function,
        "domain": domain,
        "x_analysis": x_analysis,
        "compliance": compliance,
        "objective_scaled": objective_scaled,
        "volume_constraint": volume_constraint,
    }


def baseline_solve(max_iter: int | None = None, optimizer: str = "oc") -> np.ndarray:
    """Faithful pyMOTO baseline: SIMP + density filter + OC/MMA."""
    problem = _problem()
    iters = int(problem["max_iter"] if max_iter is None else max_iter)

    x0 = np.full(int(problem["nelx"] * problem["nely"]), float(problem["volfrac"]), dtype=float)
    sx = pym.Signal("x", state=x0)

    model = _build_compliance_network(
        problem=problem,
        x_signal=sx,
        filter_radius=float(problem["filter_radius"]),
        penal=float(problem["penal"]),
        apply_filter=True,
    )

    optimizer_key = str(optimizer).strip().lower()
    if optimizer_key == "mma":
        pym.minimize_mma(
            sx,
            [model["objective_scaled"], model["volume_constraint"]],
            function=model["function"],
            maxit=iters,
            tolx=1e-3,
            tolf=1e-4,
            xmin=float(problem["rho_min"]),
            xmax=1.0,
        )
    else:
        pym.minimize_oc(
            sx,
            model["objective_scaled"],
            function=model["function"],
            maxit=iters,
            tolx=1e-3,
            tolf=1e-4,
            move=float(problem["move_limit"]),
            xmin=float(problem["rho_min"]),
            xmax=1.0,
            maxvol=float(problem["volfrac"]),
        )

    density = np.asarray(sx.state, dtype=float).reshape((int(problem["nely"]), int(problem["nelx"])))
    return np.clip(density, float(problem["rho_min"]), 1.0)


def custom_update_step(density: np.ndarray, step_id: int, problem: dict[str, Any]) -> np.ndarray:
    """
    Optional hook for agent-modified update logic.

    Baseline keeps pyMOTO's native optimizer behavior and returns input as-is.
    """
    _ = (step_id, problem)
    return density


def solve(max_iter: int | None = None, strategy: str = "baseline_oc") -> np.ndarray:
    """
    Agent-editable entrypoint.

    You may replace this with custom updates / filters / step-size rules,
    but keep return type and shape unchanged.
    """
    key = str(strategy).strip().lower()

    if key == "baseline_mma":
        density = baseline_solve(max_iter=max_iter, optimizer="mma")
    else:
        density = baseline_solve(max_iter=max_iter, optimizer="oc")

    return custom_update_step(density, step_id=0, problem=_problem())


def evaluate(density_field: np.ndarray | list[float]) -> dict[str, Any]:
    """Evaluate compliance + volume fraction for a provided density field."""
    problem = _problem()
    nelx = int(problem["nelx"])
    nely = int(problem["nely"])

    arr = np.asarray(density_field, dtype=float)
    if arr.size != nelx * nely:
        raise ValueError(f"Expected {nelx * nely} densities, got {arr.size}")

    density = arr.reshape((nely, nelx))
    density = np.clip(density, float(problem["rho_min"]), 1.0)

    sx_eval = pym.Signal("x_eval", state=density.flatten())
    model = _build_compliance_network(
        problem=problem,
        x_signal=sx_eval,
        filter_radius=float(problem["filter_radius"]),
        penal=float(problem["penal"]),
        apply_filter=False,
    )
    model["function"].response()

    compliance = float(model["compliance"].state)
    volume_fraction = float(np.mean(density))
    feasible = bool(volume_fraction <= float(problem["volfrac"]) + float(VOL_TOL))

    return {
        "compliance": compliance,
        "volume_fraction": volume_fraction,
        "feasible": feasible,
    }


def main() -> None:
    problem = _problem()
    density = solve(max_iter=int(problem["max_iter"]), strategy="baseline_oc")
    result = evaluate(density)

    submission = {
        "benchmark_id": problem["benchmark_id"],
        "nelx": int(problem["nelx"]),
        "nely": int(problem["nely"]),
        "density_vector": density.flatten().tolist(),
        "compliance": float(result["compliance"]),
        "volume_fraction": float(result["volume_fraction"]),
        "feasible": bool(result["feasible"]),
    }

    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / "submission.json"
    out_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")

    print(f"benchmark_id: {submission['benchmark_id']}")
    print(f"elements: {submission['nelx'] * submission['nely']}")
    print(f"compliance: {submission['compliance']:.6f}")
    print(f"volume_fraction: {submission['volume_fraction']:.6f}")
    print(f"feasible: {submission['feasible']}")
    print(f"submission: {out_path}")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
