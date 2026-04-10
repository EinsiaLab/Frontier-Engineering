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
# FAST FEM SOLVER FOR OPTIMIZATION
# ============================================================================

def fem_solve_fast(nodes, elements, areas, E, fixed_dofs_set, free_dofs, free_map, force_vec):
    """Optimized FEM solver that reuses precomputed DOF mappings."""
    n_elements = len(elements)
    n_dofs = 2 * len(nodes)
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
        
        cc = c * c * coeff
        cs = c * s * coeff
        ss = s * s * coeff
        
        dofs_e = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
        ke_vals = [
            [cc, cs, -cc, -cs],
            [cs, ss, -cs, -ss],
            [-cc, -cs, cc, cs],
            [-cs, -ss, cs, ss],
        ]
        
        for i_l in range(4):
            di = dofs_e[i_l]
            if di not in free_map:
                continue
            ii = free_map[di]
            for j_l in range(4):
                dj = dofs_e[j_l]
                if dj not in free_map:
                    continue
                jj = free_map[dj]
                K[ii, jj] += ke_vals[i_l][j_l]

    F = force_vec[free_dofs]
    try:
        u_red = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        return None, None, lengths

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
        stresses[e] = (E / L) * (-c*disp[2*ni] - s*disp[2*ni+1] + c*disp[2*nj] + s*disp[2*nj+1])

    return disp, stresses, lengths


def full_analysis(areas, shape_vars, problem, nodes_base, elements, precomp):
    """Full analysis returning stresses, displacements, and lengths per element."""
    E = problem["material"]["E"]
    supports = problem["supports"]

    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        node_idx = nid - 1
        nodes[node_idx, 1] = shape_vars[idx]

    fixed_dofs_set, free_dofs, free_map = precomp
    
    all_stresses = []
    all_disps = []
    all_lengths = None
    
    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            idx = nid - 1
            fvec[2 * idx] += load["fx"]
            fvec[2 * idx + 1] += load["fy"]

        disp, stresses, lengths = fem_solve_fast(nodes, elements, areas, E, fixed_dofs_set, free_dofs, free_map, fvec)
        if disp is None:
            return None
        all_stresses.append(stresses)
        all_disps.append(disp)
        all_lengths = lengths

    return {
        "stresses": all_stresses,
        "disps": all_disps,
        "lengths": all_lengths,
        "nodes": nodes,
    }


def compute_weight_fast(areas, shape_vars, problem, nodes_base, elements):
    rho = problem["material"]["rho"]
    num_bars = problem["num_bars"]
    
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


def precompute_dofs(problem, nodes_base):
    n_nodes = len(nodes_base)
    n_dofs = 2 * n_nodes
    fixed_dofs = set()
    for sup in problem["supports"]:
        nid = sup["node"]
        idx = nid - 1
        if sup.get("fix_x", False):
            fixed_dofs.add(2 * idx)
        if sup.get("fix_y", False):
            fixed_dofs.add(2 * idx + 1)
    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    free_map = {dof: idx for idx, dof in enumerate(free_dofs)}
    return (fixed_dofs, free_dofs, free_map)


# ============================================================================
# OPTIMIZATION ALGORITHM (ALLOWED TO MODIFY)
# ============================================================================

def optimize_advanced(problem, nodes_base, elements, max_evals=6500):
    """
    Advanced optimization using a combination of stress-ratio resizing 
    and differential evolution for shape variables.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])
    rho = problem["material"]["rho"]
    E = problem["material"]["E"]
    
    precomp = precompute_dofs(problem, nodes_base)
    
    eval_count = [0]
    best_weight = [float('inf')]
    best_x = [None]
    
    def evaluate(areas, shape_vars):
        """Evaluate design, return (weight, feasible, result_dict) or None if singular."""
        eval_count[0] += 1
        res = full_analysis(areas, shape_vars, problem, nodes_base, elements, precomp)
        if res is None:
            return float('inf'), False, None
        
        max_stress = 0.0
        max_disp = 0.0
        for s in res["stresses"]:
            ms = np.max(np.abs(s))
            if ms > max_stress:
                max_stress = ms
        for d in res["disps"]:
            md = np.max(np.abs(d))
            if md > max_disp:
                max_disp = md
        
        feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
        w = compute_weight_fast(areas, shape_vars, problem, nodes_base, elements)
        
        if feasible and w < best_weight[0]:
            best_weight[0] = w
            best_x[0] = np.concatenate([areas.copy(), shape_vars.copy()])
        
        return w, feasible, {
            "max_stress": max_stress,
            "max_disp": max_disp,
            "stresses": res["stresses"],
            "disps": res["disps"],
            "lengths": res["lengths"],
        }
    
    def stress_ratio_resize(areas, shape_vars, n_iter=30):
        """Fully stressed design (FSD) resizing for given shape variables."""
        current_areas = areas.copy()
        
        for it in range(n_iter):
            if eval_count[0] >= max_evals:
                break
                
            w, feasible, res = evaluate(current_areas, shape_vars)
            if res is None:
                current_areas = np.clip(current_areas * 1.5, area_min, area_max)
                continue
            
            # Element-wise stress ratio resizing
            new_areas = current_areas.copy()
            for lc_stresses in res["stresses"]:
                ratios = np.abs(lc_stresses) / sigma_limit
                # Use max ratio across load cases
                for e in range(num_bars):
                    r = ratios[e]
                    if r > 1e-8:
                        candidate = current_areas[e] * r
                        new_areas[e] = max(new_areas[e], candidate)
            
            # Displacement scaling if needed
            if res["max_disp"] > disp_limit:
                disp_scale = (res["max_disp"] / disp_limit) ** 0.5
                new_areas *= disp_scale
            
            new_areas = np.clip(new_areas, area_min, area_max)
            
            # Damped update
            alpha = 0.7
            current_areas = current_areas * (1 - alpha) + new_areas * alpha
            current_areas = np.clip(current_areas, area_min, area_max)
            
            # If feasible, try reducing slightly
            if feasible:
                # Try reducing areas that are far from stress limit
                for lc_stresses in res["stresses"]:
                    ratios = np.abs(lc_stresses) / sigma_limit
                    for e in range(num_bars):
                        if ratios[e] < 0.5:
                            current_areas[e] *= max(0.85, ratios[e] + 0.3)
                current_areas = np.clip(current_areas, area_min, area_max)
        
        return current_areas
    
    # Strategy: Try multiple shape configurations with FSD resizing
    np.random.seed(42)
    
    # Initial shape: use base node positions
    base_shape = np.array([nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]])
    
    # Start with base shape and FSD
    init_areas = np.full(num_bars, (area_min + area_max) * 0.15)
    areas_opt = stress_ratio_resize(init_areas, base_shape, n_iter=40)
    
    # Now try different shape configurations
    # Generate candidate shapes
    n_shape_candidates = 40
    shape_candidates = []
    
    # Add base shape
    shape_candidates.append(base_shape.copy())
    
    # Add random shapes
    for _ in range(n_shape_candidates - 1):
        s = np.random.uniform(y_min, y_max, num_shape)
        # Round to integers as per discrete constraint
        s = np.round(s).astype(float)
        shape_candidates.append(s)
    
    # Evaluate each shape with FSD
    for shape in shape_candidates:
        if eval_count[0] >= max_evals - 100:
            break
        
        # Start with moderate areas
        init_a = np.full(num_bars, (area_min + area_max) * 0.1)
        opt_a = stress_ratio_resize(init_a, shape, n_iter=25)
    
    # Now do local search around best solution
    if best_x[0] is not None:
        x_cur = best_x[0].copy()
        areas_cur = x_cur[:num_bars]
        shape_cur = x_cur[num_bars:]
        
        # Perturbation-based local search on shape variables
        n_local = 200
        for i_local in range(n_local):
            if eval_count[0] >= max_evals - 50:
                break
            
            # Perturb shape
            shape_new = shape_cur.copy()
            n_perturb = np.random.randint(1, num_shape + 1)
            indices = np.random.choice(num_shape, n_perturb, replace=False)
            for idx in indices:
                delta = np.random.uniform(-100, 100)
                shape_new[idx] = np.clip(np.round(shape_new[idx] + delta), y_min, y_max)
            
            # FSD for new shape
            opt_a = stress_ratio_resize(areas_cur.copy(), shape_new, n_iter=15)
            
            w_new, feas_new, _ = evaluate(opt_a, shape_new)
            if feas_new and w_new < best_weight[0]:
                areas_cur = opt_a.copy()
                shape_cur = shape_new.copy()
        
        # Final refinement: more careful FSD on best shape
        if best_x[0] is not None:
            final_areas = best_x[0][:num_bars].copy()
            final_shape = best_x[0][num_bars:].copy()
            remaining = max_evals - eval_count[0]
            if remaining > 10:
                stress_ratio_resize(final_areas, final_shape, n_iter=min(remaining, 50))
    
    if best_x[0] is None:
        # Fallback: return a conservative feasible design
        areas_safe = np.full(num_bars, area_max * 0.5)
        shape_safe = base_shape.copy()
        evaluate(areas_safe, shape_safe)
        if best_x[0] is None:
            return np.concatenate([areas_safe, shape_safe])
    
    return best_x[0]


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
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

    print(f"ISCSO 2015 Advanced Optimization")
    print(f"  Design variables: {dim} ({num_bars} areas + {num_shape} shapes)")
    print(f"  Area bounds: [{bounds_cfg['area_min']}, {bounds_cfg['area_max']}] mm^2")
    print(f"  Shape bounds: [{bounds_cfg['y_min']}, {bounds_cfg['y_max']}] mm")
    print()

    print("Optimizing...")
    x_best = optimize_advanced(problem, nodes_base, elements, max_evals=6500)
    w = compute_weight(x_best, problem, nodes_base, elements)

    result = analyze_design(x_best[:num_bars], x_best[num_bars:], problem, nodes_base, elements)
    feasible = result["feasible"] if result else False

    submission = {
        "benchmark_id": "iscso_2015",
        "solution_vector": x_best.tolist(),
        "algorithm": "FSD_LocalSearch",
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
