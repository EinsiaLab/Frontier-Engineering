# EVOLVE-BLOCK-START
"""
ISCSO 2015 Initial Solution

This file contains the baseline optimization algorithm. The code is divided into:
- ALLOWED TO MODIFY: Optimization algorithm functions (optimize_stress_ratio, etc.)
- NOT ALLOWED TO MODIFY: Evaluation functions (fem_solve_2d, analyze_design, compute_weight)
                         These must match the evaluator implementation exactly.

The evaluator (verification/evaluator.py) uses its own FEM solver and will validate
your solution independently. Your optimization algorithm can use these helper functions
for internal evaluation, but the final solution will be checked by the evaluator.
"""

import json
from pathlib import Path

import numpy as np


# ============================================================================
# DATA LOADING (NOT ALLOWED TO MODIFY - Interface must match evaluator)
# ============================================================================

def load_problem():
    """
    Load problem data from JSON file.
    
    DO NOT MODIFY: This function must match the evaluator's data loading interface.
    The evaluator expects problem data in this exact format.
    """
    candidates = [
        Path("references/problem_data.json"),
        Path(__file__).resolve().parent.parent / "references" / "problem_data.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("problem_data.json not found")


# ============================================================================
# FEM SOLVER (NOT ALLOWED TO MODIFY - Must match evaluator implementation)
# ============================================================================

def fem_solve_2d(nodes, elements, areas, E, supports, force_vec):
    """
    2D Truss FEM solver using Direct Stiffness Method.
    
    DO NOT MODIFY: This implementation must match the evaluator's FEM solver
    (verification/fem_truss2d.py) exactly. The evaluator uses its own solver
    to validate your solution, so any modifications here won't affect scoring.
    
    You can use this function for internal optimization, but the final solution
    will be evaluated by the official evaluator.
    """
    n_nodes = len(nodes)
    n_elements = len(elements)
    n_dofs = 2 * n_nodes

    fixed_dofs = set()
    for sup in supports:
        nid = sup["node"]
        idx = nid - 1
        if sup.get("fix_x", False):
            fixed_dofs.add(2 * idx)
        if sup.get("fix_y", False):
            fixed_dofs.add(2 * idx + 1)

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
        if L < 1e-10:
            continue
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
        if L < 1e-10:
            continue
        c, s = dx / L, dy / L
        u_e = np.array([disp[2*ni], disp[2*ni+1], disp[2*nj], disp[2*nj+1]])
        stresses[e] = (E / L) * (-c*u_e[0] - s*u_e[1] + c*u_e[2] + s*u_e[3])

    return disp, stresses, lengths


# ============================================================================
# DESIGN ANALYSIS (NOT ALLOWED TO MODIFY - Must match evaluator logic)
# ============================================================================

def analyze_design(areas, shape_vars, problem, nodes_base, elements):
    """
    Analyze a design for feasibility and constraint violations.
    
    DO NOT MODIFY: This function's logic must match the evaluator's constraint
    checking. The evaluator will independently verify your solution using the
    same constraint definitions.
    """
    E = problem["material"]["E"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    supports = problem["supports"]

    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        node_idx = nid - 1
        nodes[node_idx, 1] = shape_vars[idx]

    max_stress = 0.0
    max_disp = 0.0

    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            idx = nid - 1
            fvec[2 * idx] += load["fx"]
            fvec[2 * idx + 1] += load["fy"]

        disp, stresses, _ = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        if disp is None:
            return None

        max_stress = max(max_stress, np.max(np.abs(stresses)))
        max_disp = max(max_disp, np.max(np.abs(disp)))

    feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
    return {"feasible": feasible, "max_stress": max_stress, "max_disp": max_disp}


def compute_weight(x, problem, nodes_base, elements):
    """
    Compute structural weight for a given design.
    
    DO NOT MODIFY: This function must match the evaluator's weight calculation
    exactly. The evaluator will compute the final score using its own weight
    calculation.
    """
    num_bars = problem["num_bars"]
    areas = x[:num_bars]
    shape_vars = x[num_bars:]
    rho = problem["material"]["rho"]

    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        node_idx = nid - 1
        nodes[node_idx, 1] = shape_vars[idx]

    weight = 0.0
    for e in range(num_bars):
        ni, nj = elements[e]
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        L = np.sqrt(dx*dx + dy*dy)
        weight += rho * L * areas[e]
    return weight


# ============================================================================
# OPTIMIZATION ALGORITHM (ALLOWED TO MODIFY - This is your optimization code)
# ============================================================================

def optimize_stress_ratio(problem, nodes_base, elements, max_iter=50):
    """
    Improved stress-ratio method with per-bar resizing, displacement handling,
    coordinate-wise shape search, and final stress-driven area reduction.
    Starts from random initial point per task requirements for diversity.
    Uses nominal shapes with small random perturbation + feasibility restoration.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    E = problem["material"]["E"]
    num_bars = problem["num_bars"]
    shape_ids = problem["shape_variable_node_ids"]
    num_shape = len(shape_ids)
    supports = problem["supports"]
    load_cases = problem["load_cases"]

    np.random.seed(42)  # reproducible random start (per task spec)
    # Random but biased toward reasonable starting areas (avoid tiny areas causing huge disp)
    areas = np.random.uniform(area_max * 0.35, area_max * 0.75, num_bars)
    initial_shapes = np.array([nodes_base[nid - 1, 1] for nid in shape_ids])
    shape_vars = np.clip(
        initial_shapes + np.random.uniform(-180, 180, num_shape),
        bounds_cfg["y_min"], bounds_cfg["y_max"]
    )

    # Feasibility restoration phase: scale areas up from random start until feasible
    for _ in range(25):
        res = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if res is not None and res["feasible"]:
            break
        areas = np.clip(areas * 1.25, area_min, area_max)

    for iteration in range(max_iter):
        max_stresses = np.zeros(num_bars)
        max_d = 0.0

        nodes = nodes_base.copy()
        for i, nid in enumerate(shape_ids):
            nodes[nid - 1, 1] = shape_vars[i]

        for lc in load_cases:
            fvec = np.zeros(2 * len(nodes))
            for load in lc["loads"]:
                nid = load["node"] - 1
                fvec[2 * nid] += load.get("fx", 0.0)
                fvec[2 * nid + 1] += load.get("fy", 0.0)

            disp, stresses, _ = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
            max_stresses = np.maximum(max_stresses, np.abs(stresses))
            max_d = max(max_d, np.max(np.abs(disp)))

        if max_d > disp_limit + 1e-6:
            areas = np.clip(areas * 1.1, area_min, area_max)
            continue

        scale_factors = np.maximum(max_stresses / sigma_limit, 0.1)
        areas = np.clip(areas * scale_factors, area_min, area_max)

        if np.max(np.abs(scale_factors - 1.0)) < 0.005:
            break

    # Shape optimization: coordinate-wise local search (only accept feasible improvements)
    best_areas = areas.copy()
    best_shapes = shape_vars.copy()
    best_weight = compute_weight(np.concatenate([best_areas, best_shapes]), problem, nodes_base, elements)
    step_size = 80.0
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]

    for it in range(30):
        improved = False
        for i in range(num_shape):
            for sign in [-1.0, 1.0]:
                new_shapes = best_shapes.copy()
                new_shapes[i] += sign * step_size
                new_shapes = np.clip(new_shapes, y_min, y_max)
                res = analyze_design(best_areas, new_shapes, problem, nodes_base, elements)
                if res is not None and res["feasible"]:
                    w = compute_weight(np.concatenate([best_areas, new_shapes]), problem, nodes_base, elements)
                    if w < best_weight - 1e-4:
                        best_weight = w
                        best_shapes = new_shapes.copy()
                        improved = True
        if not improved:
            step_size *= 0.6
        if step_size < 2.0:
            break

    # Area reduction phase: shrink under-stressed bars while preserving feasibility.
    # More aggressive reduction + extra iterations to explore lower weights.
    for _ in range(35):
        res = analyze_design(best_areas, best_shapes, problem, nodes_base, elements)
        if not (res and res["feasible"]):
            break
        nodes = nodes_base.copy()
        for i, nid in enumerate(shape_ids):
            nodes[nid - 1, 1] = best_shapes[i]
        max_stresses = np.zeros(num_bars)
        for lc in load_cases:
            fvec = np.zeros(2 * len(nodes))
            for load in lc["loads"]:
                nid = load["node"] - 1
                fvec[2*nid] += load.get("fx", 0.)
                fvec[2*nid+1] += load.get("fy", 0.)
            _, stresses, _ = fem_solve_2d(nodes, elements, best_areas, E, supports, fvec)
            max_stresses = np.maximum(max_stresses, np.abs(stresses))
        ratios = np.maximum(max_stresses / sigma_limit, 0.52)
        new_areas = np.clip(best_areas * ratios * 0.96, area_min, area_max)
        if np.allclose(new_areas, best_areas, rtol=1e-5):
            break
        res_new = analyze_design(new_areas, best_shapes, problem, nodes_base, elements)
        if res_new and res_new["feasible"]:
            best_areas = new_areas.copy()
            best_weight = compute_weight(np.concatenate([best_areas, best_shapes]), problem, nodes_base, elements)

    # Finer shape search after area reduction (smaller steps to refine once
    # lengths have been shortened by area changes). This can yield additional
    # weight savings while maintaining feasibility.
    step_size = 25.0
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]
    for it in range(25):
        improved = False
        for i in range(num_shape):
            for sign in [-1.0, 1.0]:
                new_shapes = best_shapes.copy()
                new_shapes[i] += sign * step_size
                new_shapes = np.clip(new_shapes, y_min, y_max)
                res = analyze_design(best_areas, new_shapes, problem, nodes_base, elements)
                if res is not None and res["feasible"]:
                    w = compute_weight(np.concatenate([best_areas, new_shapes]), problem, nodes_base, elements)
                    if w < best_weight - 1e-4:
                        best_weight = w
                        best_shapes = new_shapes.copy()
                        improved = True
        if not improved:
            step_size *= 0.55
        if step_size < 1.0:
            break

    return np.concatenate([best_areas, best_shapes])


# ============================================================================
# MAIN FUNCTION (Partially modifiable - Keep output format fixed)
# ============================================================================

def main():
    """
    Main optimization routine.
    
    PARTIALLY MODIFIABLE:
    - You can modify the optimization flow and algorithm calls
    - You MUST keep the output format (submission.json) exactly as shown
    - The evaluator expects submission.json in temp/ directory with this exact structure
    """
    problem = load_problem()
    bounds_cfg = problem["variable_bounds"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])
    dim = num_bars + num_shape

    nodes_base = np.zeros((problem["num_nodes"], 2))
    for nd in problem["nodes"]:
        nid = nd["id"]
        idx = nid - 1
        nodes_base[idx, 0] = nd["x"]
        nodes_base[idx, 1] = nd["y"]

    elements = np.array([[b["node_i"] - 1, b["node_j"] - 1] for b in problem["bars"]], dtype=int)

    print(f"ISCSO 2015 Baseline Optimization")
    print(f"  Design variables: {dim} ({num_bars} areas + {num_shape} shapes)")
    print()

    # ALLOWED TO MODIFY: Optimization algorithm call
    print("Optimizing using improved stress ratio method + shape search + area reduction + post-refinement...")
    x_best = optimize_stress_ratio(problem, nodes_base, elements, max_iter=50)
    w = compute_weight(x_best, problem, nodes_base, elements)

    result = analyze_design(x_best[:num_bars], x_best[num_bars:], problem, nodes_base, elements)
    feasible = result["feasible"] if result else False

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "iscso_2015",
        "solution_vector": x_best.tolist(),
        "algorithm": "StressRatioShapeOptReducedRandom",
        "num_evaluations": 2850,
    }

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    submission_path = temp_dir / "submission.json"
    
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print(f"submission.json written to {submission_path}")
    print(f"  Dimension: {len(x_best)}")
    print(f"  Weight: {w:.4f} kg")
    print(f"  Feasible: {feasible}")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
