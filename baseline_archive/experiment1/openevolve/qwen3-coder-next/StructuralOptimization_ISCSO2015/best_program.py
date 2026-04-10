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

def optimize_stress_ratio(problem, nodes_base, elements, max_iter=15):
    """
    Stress ratio method optimization algorithm with shape optimization.
    
    ALLOWED TO MODIFY: This is the optimization algorithm. You can completely
    rewrite this function or replace it with your own optimization method.
    
    The function should return a solution vector x of length 54:
    - First 45 elements: area variables A_1 to A_45
    - Last 9 elements: shape variables y_2, y_4, y_6, y_8, y_10, y_12, y_14, y_16, y_18
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]

    # Try multiple restarts with different initial conditions to escape local minima
    best_solution = None
    best_weight = float('inf')
    
    for restart in range(7):  # Multiple restarts with different strategies
        # Different initialization strategies for each restart
        if restart == 0:
            areas = np.full(num_bars, area_max * 0.28)
        elif restart == 1:
            areas = np.full(num_bars, area_max * 0.33)
        elif restart == 2:
            areas = np.full(num_bars, area_max * 0.37)
        elif restart == 3:
            areas = np.full(num_bars, area_max * 0.42)
        elif restart == 4:
            areas = np.full(num_bars, area_max * 0.31)
        elif restart == 5:
            areas = np.full(num_bars, area_max * 0.36)
        else:
            areas = np.full(num_bars, area_max * 0.41)
        
        # Initialize shape variables with larger random perturbations for better exploration
        shape_vars = np.array([nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]])
        np.random.seed(100 + restart * 17)  # Changed seed strategy for better diversity
        shape_vars = np.clip(shape_vars + np.random.randn(num_shape) * 8.0, y_min, y_max)
        
        for iteration in range(max_iter):
            result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
            if result is None:
                areas = np.clip(areas * 1.18, area_min, area_max)
                continue

            if result["feasible"]:
                # Try to reduce areas while maintaining feasibility with more aggressive reduction
                improved = True
                while improved:
                    improved = False
                    # Try reducing areas one by one with more aggressive reduction
                    for i in range(num_bars):
                        areas_test = areas.copy()
                        areas_test[i] = max(area_min, areas[i] * 0.91)
                        result_test = analyze_design(areas_test, shape_vars, problem, nodes_base, elements)
                        if result_test is not None and result_test["feasible"]:
                            areas = areas_test
                            improved = True
                
                # Better shape optimization with larger search steps
                for shape_iter in range(12):  # Increased from 10 to 12 for better shape exploration
                    for j in range(num_shape):
                        # Try larger perturbation steps for better exploration
                        for delta in [-15, 15, -25, 25, -35, 35, -60, 60]:  # Larger perturbation steps
                            shape_test = shape_vars.copy()
                            shape_test[j] = np.clip(shape_vars[j] + delta, y_min, y_max)
                            result_test = analyze_design(areas, shape_test, problem, nodes_base, elements)
                            if result_test is not None and result_test["feasible"]:
                                # Check if weight decreases
                                test_x = np.concatenate([areas, shape_test])
                                current_x = np.concatenate([areas, shape_vars])
                                if compute_weight(test_x, problem, nodes_base, elements) < compute_weight(current_x, problem, nodes_base, elements):
                                    shape_vars = shape_test
                                    break
                
                # Additional shape optimization with targeted approach using smaller steps
                improved_shape = True
                while improved_shape:
                    improved_shape = False
                    for j in range(num_shape):
                        # Try smaller perturbation steps for fine-tuning
                        for delta in [-1, 1, -2, 2, -3, 3, -5, 5]:
                            shape_test = shape_vars.copy()
                            shape_test[j] = np.clip(shape_vars[j] + delta, y_min, y_max)
                            result_test = analyze_design(areas, shape_test, problem, nodes_base, elements)
                            if result_test is not None and result_test["feasible"]:
                                # Check if weight decreases
                                test_x = np.concatenate([areas, shape_test])
                                current_x = np.concatenate([areas, shape_vars])
                                if compute_weight(test_x, problem, nodes_base, elements) < compute_weight(current_x, problem, nodes_base, elements):
                                    shape_vars = shape_test
                                    improved_shape = True
                                    break
                
                # Check if this solution is better
                current_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
                if current_weight < best_weight:
                    best_weight = current_weight
                    best_solution = np.concatenate([areas, shape_vars])
                
                break

            max_stress = result["max_stress"]
            max_disp = result["max_disp"]
            
            # Scale factors for stress and displacement constraints with higher multipliers
            stress_scale = max_stress / sigma_limit if max_stress > 1e-6 else 1.0
            disp_scale = max_disp / disp_limit if max_disp > 1e-6 else 1.0
            
            # Use higher scale factors for more aggressive constraint satisfaction
            scale = max(stress_scale * 1.12, disp_scale * 1.12, 1.08)
            areas = np.clip(areas * scale, area_min, area_max)
        
        # Final check for this restart
        if best_solution is None:
            current_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if current_weight < best_weight:
                best_weight = current_weight
                best_solution = np.concatenate([areas, shape_vars])

    # If we found a feasible solution, use it; otherwise return last solution
    if best_solution is not None:
        return best_solution
    
    return np.concatenate([areas, shape_vars])


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
    print(f"  Area bounds: [{bounds_cfg['area_min']}, {bounds_cfg['area_max']}] mm^2")
    print(f"  Shape bounds: [{bounds_cfg['y_min']}, {bounds_cfg['y_max']}] mm")
    print()

    # ALLOWED TO MODIFY: Optimization algorithm call
    print("Optimizing using stress ratio method...")
    x_best = optimize_stress_ratio(problem, nodes_base, elements, max_iter=15)
    w = compute_weight(x_best, problem, nodes_base, elements)

    result = analyze_design(x_best[:num_bars], x_best[num_bars:], problem, nodes_base, elements)
    feasible = result["feasible"] if result else False

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "iscso_2015",
        "solution_vector": x_best.tolist(),
        "algorithm": "StressRatio",
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
