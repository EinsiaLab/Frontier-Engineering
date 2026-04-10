# EVOLVE-BLOCK-START
"""
ISCSO 2023 Initial Solution

This file contains the baseline optimization algorithm. The code is divided into:
- ALLOWED TO MODIFY: Optimization algorithm functions (optimize_discrete_stress_ratio, etc.)
- NOT ALLOWED TO MODIFY: Evaluation functions (fem_solve_3d, analyze_design, compute_weight)
                         These must match the evaluator implementation exactly.

The evaluator (verification/evaluator.py) uses its own FEM solver and will validate
your solution independently. Your optimization algorithm can use these helper functions
for internal evaluation, but the final solution will be checked by the evaluator.
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


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


def get_section_database(problem):
    """
    Load section database from problem data.
    
    DO NOT MODIFY: This function must match the evaluator's section database loading.
    The evaluator uses the same section data to validate your solution.
    """
    if "section_database" in problem and "sections" in problem["section_database"]:
        return {s["id"]: s.get("area_mm2", s.get("area_cm2", 0.0) * 100) for s in problem["section_database"]["sections"]}
    candidates = [
        Path("references/section_database.json"),
        Path(__file__).resolve().parent.parent / "references" / "section_database.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {s["id"]: s.get("area_mm2", s.get("area_cm2", 0.0) * 100) for s in data["sections"]}
    raise FileNotFoundError("section_database not found")


# ============================================================================
# TOPOLOGY GENERATION (NOT ALLOWED TO MODIFY - Must match evaluator)
# ============================================================================

def generate_tower(problem):
    """
    Generate tower topology (nodes and elements).
    
    DO NOT MODIFY: This function must match the evaluator's topology generation
    (verification/fem_truss3d.py::generate_tower_topology) exactly. The evaluator
    will generate the same topology to validate your solution.
    """
    tp = problem["tower_parameters"]
    num_levels = tp["num_levels"]
    total_height = tp["total_height_mm"]
    bottom_hw = tp["bottom_half_width_mm"]
    top_hw = tp["top_half_width_mm"]
    cb_levels = tp["cross_bracing_levels"]

    n_nodes = num_levels * 4
    nodes = np.zeros((n_nodes, 3))
    for i in range(num_levels):
        t = i / (num_levels - 1) if num_levels > 1 else 0
        hw = bottom_hw + t * (top_hw - bottom_hw)
        z = t * total_height
        base = 4 * i
        nodes[base + 0] = [+hw, +hw, z]
        nodes[base + 1] = [-hw, +hw, z]
        nodes[base + 2] = [-hw, -hw, z]
        nodes[base + 3] = [+hw, -hw, z]

    elems = []
    for i in range(num_levels - 1):
        for j in range(4):
            elems.append((4*i + j, 4*(i+1) + j))
    for i in range(num_levels):
        b = 4 * i
        elems.extend([(b, b+1), (b+1, b+2), (b+2, b+3), (b+3, b)])
    for i in range(num_levels - 1):
        elems.append((4*i, 4*(i+1)+1))
        elems.append((4*i+1, 4*(i+1)+2))
        elems.append((4*i+2, 4*(i+1)+3))
        elems.append((4*i+3, 4*(i+1)))
    for i in cb_levels:
        if i < num_levels:
            b = 4 * i
            elems.extend([(b, b+2), (b+1, b+3)])

    return nodes, np.array(elems, dtype=int)


# ============================================================================
# FEM SOLVER (NOT ALLOWED TO MODIFY - Must match evaluator implementation)
# ============================================================================

def fem_solve_3d(nodes, elements, areas, E, supports, force_vec):
    """
    3D Truss FEM solver using Direct Stiffness Method with sparse matrices.
    
    DO NOT MODIFY: This implementation must match the evaluator's FEM solver
    (verification/fem_truss3d.py) exactly. The evaluator uses its own solver
    to validate your solution, so any modifications here won't affect scoring.
    
    You can use this function for internal optimization, but the final solution
    will be evaluated by the official evaluator.
    """
    n_nodes = len(nodes)
    n_elems = len(elements)
    n_dofs = 3 * n_nodes

    fixed = set()
    for sup in supports:
        nid = sup["node"]
        if sup.get("fix_x", False): fixed.add(3*nid)
        if sup.get("fix_y", False): fixed.add(3*nid + 1)
        if sup.get("fix_z", False): fixed.add(3*nid + 2)

    free = sorted(set(range(n_dofs)) - fixed)
    free_map = {d: i for i, d in enumerate(free)}
    n_free = len(free)

    rows, cols, vals = [], [], []
    lengths = np.zeros(n_elems)

    for e in range(n_elems):
        ni, nj = elements[e]
        d = nodes[nj] - nodes[ni]
        L = np.linalg.norm(d)
        lengths[e] = L
        if L < 1e-10:
            continue
        dc = d / L
        B = np.outer(dc, dc)
        coeff = E * areas[e] / L
        ke = coeff * np.block([[B, -B], [-B, B]])

        dofs_e = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2]
        for il in range(6):
            di = dofs_e[il]
            if di not in free_map: continue
            ii = free_map[di]
            for jl in range(6):
                dj = dofs_e[jl]
                if dj not in free_map: continue
                jj = free_map[dj]
                rows.append(ii)
                cols.append(jj)
                vals.append(ke[il, jl])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsc()
    F = force_vec[free]

    try:
        u_red = spsolve(K, F)
    except:
        return None, None, lengths

    disp = np.zeros(n_dofs)
    for idx, dof in enumerate(free):
        disp[dof] = u_red[idx]

    stresses = np.zeros(n_elems)
    for e in range(n_elems):
        ni, nj = elements[e]
        d = nodes[nj] - nodes[ni]
        L = lengths[e]
        if L < 1e-10:
            continue
        dc = d / L
        T = np.array([-dc[0], -dc[1], -dc[2], dc[0], dc[1], dc[2]])
        u_e = np.array([disp[3*ni], disp[3*ni+1], disp[3*ni+2],
                        disp[3*nj], disp[3*nj+1], disp[3*nj+2]])
        stresses[e] = (E / L) * T.dot(u_e)

    return disp, stresses, lengths


def apply_loads(lc, unsupported_nodes, num_unsupported):
    """
    Apply loads according to load case definition.
    
    DO NOT MODIFY: This function must match the evaluator's load application logic.
    The evaluator uses the same load definitions to validate your solution.
    """
    fvec = np.zeros(3 * (num_unsupported + 4))
    if len(lc.get("loads", [])) == 0:
        if lc["id"] == 0:
            load_per_node = 12000.0 / num_unsupported
            for nid in unsupported_nodes:
                fvec[3*nid] += load_per_node
        elif lc["id"] == 1:
            load_per_node = 12000.0 / num_unsupported
            for nid in unsupported_nodes:
                fvec[3*nid+1] += load_per_node
        elif lc["id"] == 2:
            load_per_node = 15000.0 / num_unsupported
            for nid in unsupported_nodes:
                fvec[3*nid+2] -= load_per_node
    else:
        for load in lc["loads"]:
            nid = load["node"]
            fvec[3*nid] += load["fx"]
            fvec[3*nid+1] += load["fy"]
            fvec[3*nid+2] += load["fz"]
    return fvec


# ============================================================================
# DESIGN ANALYSIS (NOT ALLOWED TO MODIFY - Must match evaluator logic)
# ============================================================================

def analyze_design(section_ids, problem, nodes, elements, section_db):
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

    areas = np.array([section_db[sid] for sid in section_ids])
    supported_nodes = {s["node"] for s in supports}
    unsupported_nodes = [i for i in range(len(nodes)) if i not in supported_nodes]
    num_unsupported = len(unsupported_nodes)

    max_stress = 0.0
    max_disp = 0.0

    for lc in problem["load_cases"]:
        fvec = apply_loads(lc, unsupported_nodes, num_unsupported)
        disp, stresses, _ = fem_solve_3d(nodes, elements, areas, E, supports, fvec)
        if disp is None:
            return None

        max_stress = max(max_stress, np.max(np.abs(stresses)))
        max_disp = max(max_disp, np.max(np.abs(disp)))

    feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
    return {"feasible": feasible, "max_stress": max_stress, "max_disp": max_disp}


def compute_weight(section_ids, problem, nodes, elements, section_db):
    """
    Compute structural weight for a given design.
    
    DO NOT MODIFY: This function must match the evaluator's weight calculation
    exactly. The evaluator will compute the final score using its own weight
    calculation.
    """
    rho = problem["material"]["rho"]
    areas = np.array([section_db[sid] for sid in section_ids])
    weight = 0.0
    for e in range(len(elements)):
        ni, nj = elements[e]
        L = np.linalg.norm(nodes[nj] - nodes[ni])
        weight += rho * L * areas[e]
    return weight


# ============================================================================
# OPTIMIZATION ALGORITHM (ALLOWED TO MODIFY - This is your optimization code)
# ============================================================================

import time

def get_initial_design(problem, nodes, elements, section_db, id_min, id_max):
    return np.random.randint(id_max - 5, id_max + 1, size=problem["dimension"])

def optimize_discrete_stress_ratio(problem, nodes, elements, section_db, max_eval=200000):
    start_time = time.time()
    bounds_cfg = problem["variable_bounds"]
    id_min, id_max = bounds_cfg["section_id_min"], bounds_cfg["section_id_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    E, dim = problem["material"]["E"], problem["dimension"]
    supports = problem["supports"]

    supported_nodes = {s["node"] for s in supports}
    unsupported_nodes = [i for i in range(len(nodes)) if i not in supported_nodes]
    fvecs = [apply_loads(lc, unsupported_nodes, len(unsupported_nodes)) for lc in problem["load_cases"]]

    sorted_secs = sorted([(sid, area) for sid, area in section_db.items()], key=lambda x: x[1])
    areas_list = np.array([a for s, a in sorted_secs])
    id_to_idx = {s: i for i, (s, a) in enumerate(sorted_secs)}

    section_ids = get_initial_design(problem, nodes, elements, section_db, id_min, id_max)
    best_feasible_ids, best_w, num_eval = None, float('inf'), 0

    current_ids = section_ids.copy()
    stagnation = 0
    visited = set()

    while num_eval < max_eval:
        if time.time() - start_time > 280:
            break
            
        areas = np.array([section_db[sid] for sid in current_ids])
        max_stresses = np.zeros(dim)
        max_disp = 0.0
        is_feasible = True

        for fvec in fvecs:
            disp, stresses, _ = fem_solve_3d(nodes, elements, areas, E, supports, fvec)
            if disp is None:
                is_feasible = False
                break
            max_stresses = np.maximum(max_stresses, np.abs(stresses))
            max_disp = max(max_disp, np.max(np.abs(disp)))

        num_eval += 1
        
        if not is_feasible:
            current_ids = np.clip(current_ids + 1, id_min, id_max)
            continue

        feasible = (np.max(max_stresses) <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
        
        if feasible:
            w = compute_weight(current_ids, problem, nodes, elements, section_db)
            if w < best_w:
                best_w = w
                best_feasible_ids = current_ids.copy()
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

        disp_ratio = max_disp / disp_limit
        new_ids = np.zeros(dim, dtype=int)
        
        overall_max_stress = np.max(max_stresses)
        if overall_max_stress < 1e-6:
            overall_max_stress = 1e-6
            
        for i in range(dim):
            stress_ratio = max_stresses[i] / sigma_limit
            req_area = areas[i] * stress_ratio
            
            if disp_ratio > 1.0:
                base_scale = disp_ratio ** 0.6
                stress_factor = (max_stresses[i] / overall_max_stress) ** 0.5
                disp_factor = base_scale * (1.0 + (disp_ratio - 1.0) * stress_factor)
                req_area = max(req_area, areas[i] * disp_factor)
            else:
                req_area = max(req_area, areas[i] * max(0.5, disp_ratio))
                
            idx = min(np.searchsorted(areas_list, req_area), len(areas_list) - 1)
            new_ids[i] = sorted_secs[idx][0]

        new_tuple = tuple(new_ids)
        
        if new_tuple in visited or np.array_equal(new_ids, current_ids) or stagnation > 15:
            if best_feasible_ids is not None and stagnation > 30:
                new_ids = best_feasible_ids.copy()
                stagnation = 0
                
            can_reduce = [i for i in range(dim) if id_to_idx[new_ids[i]] > 0]
            can_inc = [i for i in range(dim) if id_to_idx[new_ids[i]] < len(areas_list) - 1]
            
            if feasible and can_reduce:
                num_to_reduce = max(1, int(dim * random.uniform(0.02, 0.08)))
                least_stressed = sorted(can_reduce, key=lambda i: max_stresses[i])[:num_to_reduce*3]
                indices = random.sample(least_stressed, min(num_to_reduce, len(least_stressed)))
                for idx in indices:
                    new_ids[idx] = sorted_secs[id_to_idx[new_ids[idx]] - 1][0]
            elif not feasible and can_inc:
                num_to_inc = max(1, int(dim * random.uniform(0.02, 0.08)))
                most_stressed = sorted(can_inc, key=lambda i: max_stresses[i], reverse=True)[:num_to_inc*3]
                indices = random.sample(most_stressed, min(num_to_inc, len(most_stressed)))
                for idx in indices:
                    new_ids[idx] = sorted_secs[id_to_idx[new_ids[idx]] + 1][0]
            else:
                num_change = max(1, int(dim * 0.05))
                indices = random.sample(range(dim), num_change)
                for idx in indices:
                    curr_idx = id_to_idx[new_ids[idx]]
                    if curr_idx > 0 and random.random() > 0.5:
                        new_ids[idx] = sorted_secs[curr_idx - 1][0]
                    elif curr_idx < len(areas_list) - 1:
                        new_ids[idx] = sorted_secs[curr_idx + 1][0]
                        
            stagnation = 0

        visited.add(new_tuple)
        current_ids = new_ids

    return best_feasible_ids if best_feasible_ids is not None else current_ids, num_eval


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
    nodes, elements = generate_tower(problem)
    bounds_cfg = problem["variable_bounds"]
    dim = problem["dimension"]
    max_eval = problem.get("optimization", {}).get("max_evaluations", 200000)

    section_db = get_section_database(problem)

    print(f"ISCSO 2023 Baseline Optimization")
    print(f"  Design variables: {dim} (discrete section IDs {bounds_cfg['section_id_min']}-{bounds_cfg['section_id_max']})")
    print(f"  Max evaluations: {max_eval}")
    print()

    # ALLOWED TO MODIFY: Optimization algorithm call
    print("Optimizing using discrete stress ratio method...")
    x_best, num_eval = optimize_discrete_stress_ratio(problem, nodes, elements, section_db, max_eval)
    w = compute_weight(x_best, problem, nodes, elements, section_db)

    result = analyze_design(x_best, problem, nodes, elements, section_db)
    feasible = result["feasible"] if result else False

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "iscso_2023",
        "solution_vector": x_best.tolist(),
        "algorithm": "DiscreteStressRatio",
        "num_evaluations": num_eval,
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
    print(f"  Evaluations used: {num_eval}/{max_eval}")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
