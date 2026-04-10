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

def get_stresses(areas, shape_vars, problem, nodes_base, elements):
    """Get per-element stresses and max displacement."""
    E = problem["material"]["E"]
    supports = problem["supports"]
    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        nodes[nid - 1, 1] = shape_vars[idx]
    max_stresses = np.zeros(len(areas))
    max_disp = 0.0
    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            idx = load["node"] - 1
            fvec[2*idx] += load["fx"]
            fvec[2*idx+1] += load["fy"]
        disp, stresses, _ = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        max_stresses = np.maximum(max_stresses, np.abs(stresses))
        max_disp = max(max_disp, np.max(np.abs(disp)))
    return max_stresses, max_disp

def single_run(areas, shape_vars, problem, nodes_base, elements, max_iter, cfg, sigma_lim, disp_lim, a_min, a_max, y_min, y_max, num_shape):
    """Single optimization run with enhanced FSD and shape optimization."""
    best_areas = areas.copy()
    best_shapes = shape_vars.copy()
    best_weight = float('inf')
    
    # Phase 0: Find feasible quickly
    for _ in range(25):
        stresses, max_disp = get_stresses(areas, shape_vars, problem, nodes_base, elements)
        if np.max(stresses) <= sigma_lim and max_disp <= disp_lim:
            best_areas = areas.copy()
            best_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            break
        sr = np.max(stresses) / sigma_lim if np.max(stresses) > 0 else 1.0
        dr = max_disp / disp_lim if max_disp > 0 else 1.0
        areas = np.clip(areas * max(sr, dr) * 1.12, a_min, a_max)
    
    areas = best_areas.copy()
    
    # Phase 1: FSD with adaptive exponent (start aggressive, become conservative)
    for it in range(max_iter):
        stresses, max_disp = get_stresses(areas, shape_vars, problem, nodes_base, elements)
        feasible = np.max(stresses) <= sigma_lim + 1e-6 and max_disp <= disp_lim + 1e-6
        if feasible:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if w < best_weight:
                best_weight, best_areas, best_shapes = w, areas.copy(), shape_vars.copy()
        
        # Adaptive exponent: more aggressive early, conservative later
        exp = 0.4 + 0.2 * (it / max_iter)
        ratios = np.maximum(stresses / sigma_lim, 0.05)
        new_areas = areas * (ratios ** exp)
        if max_disp > disp_lim:
            new_areas *= (max_disp / disp_lim) ** 0.5 * 1.05
        # Tighter move limits for better convergence
        move_lim = 0.5 + 0.3 * (it / max_iter)
        new_areas = np.clip(new_areas, areas * move_lim, areas / move_lim)
        areas = np.clip(new_areas, a_min, a_max)
    
    # Phase 2: Enhanced shape optimization with bidirectional search
    areas = best_areas.copy()
    shape_vars = best_shapes.copy()
    for _ in range(12):
        improved = False
        for i in range(num_shape):
            # Try both directions with finer resolution
            for delta in [400, 200, 100, 50, 25, 10, -10, -25, -50, -100, -200, -400]:
                ts = shape_vars.copy()
                ts[i] = np.clip(shape_vars[i] + delta, y_min, y_max)
                stresses, max_disp = get_stresses(areas, ts, problem, nodes_base, elements)
                if np.max(stresses) <= sigma_lim and max_disp <= disp_lim:
                    w = compute_weight(np.concatenate([areas, ts]), problem, nodes_base, elements)
                    if w < best_weight - 0.5:  # Require meaningful improvement
                        best_weight, best_shapes = w, ts.copy()
                        shape_vars = ts.copy()
                        improved = True
                        break
        if not improved: break
        # Re-optimize areas after shape improvement
        for _ in range(15):
            stresses, max_disp = get_stresses(areas, shape_vars, problem, nodes_base, elements)
            if np.max(stresses) > sigma_lim + 1e-6: break
            ratios = stresses / sigma_lim
            new_areas = np.array([max(a_min, a * (0.6 + 0.4 * ratios[i])) for i, a in enumerate(areas)])
            if np.allclose(new_areas, areas, rtol=1e-4): break
            areas = new_areas
    
    # Phase 3: Very aggressive refinement targeting understressed members
    areas = best_areas.copy()
    shape_vars = best_shapes.copy()
    for _ in range(50):
        stresses, max_disp = get_stresses(areas, shape_vars, problem, nodes_base, elements)
        if np.max(stresses) > sigma_lim + 1e-6 or max_disp > disp_lim + 1e-6: break
        w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
        if w < best_weight: best_weight, best_areas = w, areas.copy()
        ratios = stresses / sigma_lim
        # More aggressive reduction for understressed members
        new_areas = np.array([
            max(a_min, a * (0.5 + 0.5 * ratios[i])) if ratios[i] < 0.9 else a 
            for i, a in enumerate(areas)
        ])
        if np.allclose(new_areas, areas, rtol=1e-6): break
        areas = new_areas
    
    return best_areas, best_shapes, best_weight

def optimize_stress_ratio(problem, nodes_base, elements, max_iter=50):
    """Enhanced multi-start optimization with diverse initial configurations."""
    cfg = problem["variable_bounds"]
    a_min, a_max = cfg["area_min"], cfg["area_max"]
    y_min, y_max = cfg["y_min"], cfg["y_max"]
    sigma_lim = problem["constraints"]["stress_limit"]
    disp_lim = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])
    shape_ids = problem["shape_variable_node_ids"]
    
    np.random.seed(42)
    best_areas, best_shapes = None, None
    best_weight = float('inf')
    
    base_shapes = np.array([nodes_base[nid - 1, 1] for nid in shape_ids])
    
    # Multi-start with 8 diverse initial configurations
    for run in range(8):
        # Varied initial area strategies
        if run < 3:
            areas = np.random.uniform(a_min, a_min * 8, num_bars)
        elif run < 6:
            areas = np.random.uniform(a_min * 2, a_min * 15, num_bars)
        else:
            areas = np.full(num_bars, a_min * 5) + np.random.uniform(0, a_min * 5, num_bars)
        
        # Varied shape initialization
        if run % 3 == 0:
            shape_vars = base_shapes + np.random.uniform(-800, 800, num_shape)
        elif run % 3 == 1:
            shape_vars = np.full(num_shape, (y_min + y_max) / 2) + np.random.uniform(-400, 400, num_shape)
        else:
            shape_vars = base_shapes * (1 + np.random.uniform(-0.3, 0.3, num_shape))
        shape_vars = np.clip(shape_vars, y_min, y_max)
        
        a, s, w = single_run(areas, shape_vars, problem, nodes_base, elements, max_iter, cfg, sigma_lim, disp_lim, a_min, a_max, y_min, y_max, num_shape)
        if w < best_weight:
            best_weight, best_areas, best_shapes = w, a.copy(), s.copy()
    
    # Final polish from best solution with extended iterations
    a, s, w = single_run(best_areas, best_shapes, problem, nodes_base, elements, 30, cfg, sigma_lim, disp_lim, a_min, a_max, y_min, y_max, num_shape)
    if w < best_weight:
        best_areas, best_shapes = a.copy(), s.copy()
    
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
    print(f"  Area bounds: [{bounds_cfg['area_min']}, {bounds_cfg['area_max']}] mm^2")
    print(f"  Shape bounds: [{bounds_cfg['y_min']}, {bounds_cfg['y_max']}] mm")
    print()

    # ALLOWED TO MODIFY: Optimization algorithm call
    print("Optimizing using stress ratio method...")
    x_best = optimize_stress_ratio(problem, nodes_base, elements, max_iter=60)
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
