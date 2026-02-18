# EVOLVE-BLOCK-START
"""
ISCSO 2015 — 45-Bar 2D Truss Size + Shape Optimization

Objective: Minimize structural weight subject to stress and displacement constraints.
Design variables: 45 cross-sectional areas + 9 node y-coordinates = 54 variables.

Outputs submission.json in the working directory.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


# ===================== Inline 2D Truss FEM =====================

def fem_solve_2d(nodes, elements, areas, E, supports, force_vec):
    """Solve 2D truss FEM. Returns (displacements, stresses, lengths)."""
    n_nodes = len(nodes)
    n_elements = len(elements)
    n_dofs = 2 * n_nodes

    fixed_dofs = set()
    for sup in supports:
        nid = sup["node"]
        if sup.get("fix_x", False):
            fixed_dofs.add(2 * nid)
        if sup.get("fix_y", False):
            fixed_dofs.add(2 * nid + 1)

    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    free_map = {dof: idx for idx, dof in enumerate(free_dofs)}
    n_free = len(free_dofs)

    K = np.zeros((n_free, n_free))
    lengths = np.zeros(n_elements)

    for e in range(n_elements):
        ni, nj = elements[e]
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        dx, dy = xj - xi, yj - yi
        L = np.sqrt(dx * dx + dy * dy)
        lengths[e] = L
        c, s = dx / L, dy / L
        coeff = E * areas[e] / L
        ke = coeff * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s],
        ])
        dofs_e = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
        for i_l, di in enumerate(dofs_e):
            if di not in free_map:
                continue
            ii = free_map[di]
            for j_l, dj in enumerate(dofs_e):
                if dj not in free_map:
                    continue
                jj = free_map[dj]
                K[ii, jj] += ke[i_l, j_l]

    F = force_vec[free_dofs]
    u_red = np.linalg.solve(K, F)

    disp = np.zeros(n_dofs)
    for idx, dof in enumerate(free_dofs):
        disp[dof] = u_red[idx]

    stresses = np.zeros(n_elements)
    for e in range(n_elements):
        ni, nj = elements[e]
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        dx, dy = xj - xi, yj - yi
        L = lengths[e]
        c, s = dx / L, dy / L
        u_e = np.array([disp[2*ni], disp[2*ni+1], disp[2*nj], disp[2*nj+1]])
        stresses[e] = (E / L) * (-c*u_e[0] - s*u_e[1] + c*u_e[2] + s*u_e[3])

    return disp, stresses, lengths


# ===================== Load Problem Data =====================

def load_problem():
    """Load problem_data.json from references/ directory."""
    candidates = [
        Path("references/problem_data.json"),
        Path(__file__).resolve().parent.parent / "references" / "problem_data.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("problem_data.json not found")


# ===================== Evaluation =====================

def evaluate_design(x, problem):
    """Evaluate a design vector. Returns (weight, max_constraint_violation)."""
    num_bars = problem["num_bars"]
    areas = x[:num_bars]
    shape_vars = x[num_bars:]

    E = problem["material"]["E"]
    rho = problem["material"]["rho"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]

    nodes = np.zeros((problem["num_nodes"], 2))
    for nd in problem["nodes"]:
        nodes[nd["id"], 0] = nd["x"]
        nodes[nd["id"], 1] = nd["y"]
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        nodes[nid, 1] = shape_vars[idx]

    elements = np.array([[b["node_i"], b["node_j"]] for b in problem["bars"]], dtype=int)
    supports = problem["supports"]

    max_vio = 0.0
    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * problem["num_nodes"])
        for load in lc["loads"]:
            fvec[2 * load["node"]] += load["fx"]
            fvec[2 * load["node"] + 1] += load["fy"]

        try:
            disp, stresses, lengths = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        except Exception:
            return 1e18, 1e18

        stress_vio = float(np.max(np.abs(stresses) - sigma_limit))
        disp_vio = float(np.max(np.abs(disp) - disp_limit))
        max_vio = max(max_vio, stress_vio, disp_vio)

    weight = 0.0
    for e in range(num_bars):
        ni, nj = elements[e]
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        L = np.sqrt(dx*dx + dy*dy)
        weight += rho * L * areas[e]

    return weight, max(max_vio, 0.0)


def penalized_objective(x, problem):
    """Objective with quadratic penalty for constraint violations."""
    weight, max_vio = evaluate_design(x, problem)
    penalty = 1e6 * max(max_vio, 0.0) ** 2
    return weight + penalty


# ===================== Main (top-level execution) =====================

problem = load_problem()
bounds_cfg = problem["variable_bounds"]

num_bars = problem["num_bars"]
num_shape = len(problem["shape_variable_node_ids"])
dim = num_bars + num_shape

bounds = (
    [(bounds_cfg["area_min"], bounds_cfg["area_max"])] * num_bars
    + [(bounds_cfg["y_min"], bounds_cfg["y_max"])] * num_shape
)

print(f"ISCSO 2015 Baseline Optimization")
print(f"  Design variables: {dim}")
print(f"  Area bounds: [{bounds_cfg['area_min']}, {bounds_cfg['area_max']}] mm^2")
print(f"  Shape bounds: [{bounds_cfg['y_min']}, {bounds_cfg['y_max']}] mm")
print(f"  Stress limit: {problem['constraints']['stress_limit']} MPa")
print(f"  Displacement limit: {problem['constraints']['displacement_limit']} mm")
print()

best_feasible = None
best_weight = float("inf")
iteration_count = 0
start_time = time.time()
last_print_time = start_time
MAX_RUNTIME = 300  # 5 minutes maximum
PRINT_INTERVAL = 10  # Print progress every 10 iterations or 30 seconds


def callback(xk, convergence=0):
    global best_feasible, best_weight, iteration_count, last_print_time
    iteration_count += 1
    current_time = time.time()
    elapsed = current_time - start_time
    
    w, vio = evaluate_design(xk, problem)
    should_print = False
    
    if vio <= 1e-6 and w < best_weight:
        best_weight = w
        best_feasible = xk.copy()
        should_print = True
        print(f"  [{iteration_count:4d}] New best feasible: weight = {w:.4f} kg (time: {elapsed:.1f}s)")
    elif iteration_count % PRINT_INTERVAL == 0 or (current_time - last_print_time) >= 30:
        should_print = True
        status = "feasible" if vio <= 1e-6 else f"violation={vio:.2e}"
        print(f"  [{iteration_count:4d}] Current: weight = {w:.4f} kg ({status}), time: {elapsed:.1f}s")
    
    if should_print:
        last_print_time = current_time
    
    # Note: Time limit is enforced by maxiter, which is set conservatively


print("Running Differential Evolution (maxiter=100, popsize=30, max_time=5min)...")
print("  Progress will be printed every 10 iterations or 30 seconds")
result = differential_evolution(
    penalized_objective,
    bounds=bounds,
    args=(problem,),
    maxiter=100,  # Reduced from 200 for faster baseline
    popsize=30,
    tol=1e-8,
    seed=42,
    callback=callback,
    disp=False,
    workers=1,
)

elapsed_time = time.time() - start_time
print(f"\nDE finished: {result.message}")
print(f"  Total iterations: {iteration_count}")
print(f"  Total time: {elapsed_time:.1f}s")
print(f"  Final objective (penalized): {result.fun:.4f}")

if best_feasible is not None:
    x_best = best_feasible
    w, vio = evaluate_design(x_best, problem)
    print(f"  Best feasible weight: {w:.4f} kg")
    print(f"  Max constraint violation: {vio:.6e}")
else:
    x_best = result.x
    w, vio = evaluate_design(x_best, problem)
    print(f"  WARNING: No feasible solution found!")
    print(f"  Best weight: {w:.4f} kg, violation: {vio:.6e}")

# Write submission
submission = {
    "benchmark_id": "iscso_2015",
    "solution_vector": x_best.tolist(),
    "algorithm": "DifferentialEvolution",
    "num_evaluations": int(result.nfev),
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print(f"\n✓ submission.json written")
print(f"  Dimension: {len(x_best)}")
print(f"  Weight: {w:.4f} kg")
print(f"  Feasible: {vio <= 1e-6}")
# EVOLVE-BLOCK-END
