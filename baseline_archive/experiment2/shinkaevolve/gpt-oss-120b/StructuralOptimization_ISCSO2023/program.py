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
    member_stresses = np.zeros(len(elements))  # Track per-member max stress

    for lc in problem["load_cases"]:
        fvec = apply_loads(lc, unsupported_nodes, num_unsupported)
        disp, stresses, _ = fem_solve_3d(nodes, elements, areas, E, supports, fvec)
        if disp is None:
            return None

        max_stress = max(max_stress, np.max(np.abs(stresses)))
        max_disp = max(max_disp, np.max(np.abs(disp)))
        member_stresses = np.maximum(member_stresses, np.abs(stresses))

    feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
    return {"feasible": feasible, "max_stress": max_stress, "max_disp": max_disp,
            "member_stresses": member_stresses}


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

def get_initial_design(problem, nodes, elements, section_db, id_min, id_max):
    """
    Generate initial design starting from minimum sections.

    ALLOWED TO MODIFY: You can change the initial design strategy. However,
    note that the problem requires starting from a random initial point.
    """
    dim = problem["dimension"]
    # Start with minimum sections for lighter initial design
    section_ids = np.full(dim, id_min, dtype=int)
    return section_ids


def optimize_discrete_stress_ratio(problem, nodes, elements, section_db, max_eval=200000):
    """
    Aggressive FSD optimization with extensive greedy reduction.

    Key strategy:
    1. Start from minimum sections only
    2. Conservative scaling to achieve feasibility quickly
    3. Aggressive multi-phase reduction using full evaluation budget

    ALLOWED TO MODIFY: This is the optimization algorithm.
    """
    import bisect

    bounds_cfg = problem["variable_bounds"]
    id_min = bounds_cfg["section_id_min"]
    id_max = bounds_cfg["section_id_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]

    dim = problem["dimension"]

    # Build sorted section list for area-based lookup
    section_areas = [(sid, section_db[sid]) for sid in range(id_min, id_max + 1)]
    section_areas.sort(key=lambda x: x[1])
    sorted_areas = [area for sid, area in section_areas]

    def get_section_id_for_area(target_area):
        """Find smallest section ID with area >= target_area using binary search"""
        idx = bisect.bisect_left(sorted_areas, target_area)
        if idx >= len(section_areas):
            return id_max
        return section_areas[idx][0]

    # Classify members by orientation and level
    member_type = []  # 0=vertical, 1=horizontal, 2=diagonal
    member_level = []
    member_length = []

    for e in range(dim):
        ni, nj = elements[e]
        d = nodes[nj] - nodes[ni]
        L = np.linalg.norm(d)
        member_length.append(L)

        level = min(ni, nj) // 4
        member_level.append(level)

        if L < 1e-10:
            member_type.append(1)
        else:
            z_ratio = abs(d[2]) / L
            if z_ratio > 0.9:
                member_type.append(0)  # Vertical
            elif z_ratio < 0.1:
                member_type.append(1)  # Horizontal
            else:
                member_type.append(2)  # Diagonal

    member_type = np.array(member_type)
    member_level = np.array(member_level)
    member_length = np.array(member_length)

    # Compute weight factors for prioritization
    rho = problem["material"]["rho"]
    weight_factor = rho * member_length

    total_eval = 0
    best_ids = None
    best_weight = float('inf')

    # Phase 1: FSD from minimum sections - conservative scaling
    section_ids = np.full(dim, id_min, dtype=int)

    prev_ids = None
    for iteration in range(100):
        if total_eval >= max_eval - dim:  # Reserve budget for reduction
            break

        result = analyze_design(section_ids, problem, nodes, elements, section_db)
        total_eval += 1

        if result is None:
            section_ids = np.clip(section_ids + 1, id_min, id_max)
            continue

        if result["feasible"]:
            w = compute_weight(section_ids, problem, nodes, elements, section_db)
            if w < best_weight:
                best_weight = w
                best_ids = section_ids.copy()
            break

        member_stresses = result["member_stresses"]
        max_stress = result["max_stress"]
        max_disp = result["max_disp"]

        new_ids = section_ids.copy()

        # Stress-based scaling - conservative, only scale overstressed members
        for i in range(dim):
            if member_stresses[i] > sigma_limit:
                stress_ratio = member_stresses[i] / sigma_limit
                current_area = section_db[section_ids[i]]
                target_area = current_area * stress_ratio * 1.02  # Very conservative
                new_ids[i] = get_section_id_for_area(target_area)

        # Displacement-based scaling - very targeted
        if max_disp > disp_limit:
            disp_ratio = max_disp / disp_limit

            # Only scale the most critical members (top 15%)
            stress_order = np.argsort(member_stresses)[::-1]
            n_critical = max(1, int(dim * 0.15))

            for idx in stress_order[:n_critical]:
                current_area = section_db[section_ids[idx]]
                if member_type[idx] == 0:  # Vertical
                    target_area = current_area * (disp_ratio ** 0.4) * 1.05
                else:
                    target_area = current_area * (disp_ratio ** 0.3) * 1.03
                new_ids[idx] = max(new_ids[idx], get_section_id_for_area(target_area))

        # Force progress if stuck
        if np.array_equal(new_ids, section_ids):
            max_idx = np.argmax(member_stresses)
            new_ids[max_idx] = min(new_ids[max_idx] + 1, id_max)

        if prev_ids is not None and np.array_equal(new_ids, prev_ids):
            max_idx = np.argmax(member_stresses)
            new_ids[max_idx] = min(new_ids[max_idx] + 1, id_max)

        prev_ids = section_ids.copy()
        section_ids = np.clip(new_ids, id_min, id_max)

    # Verify feasibility
    result = analyze_design(section_ids, problem, nodes, elements, section_db)
    total_eval += 1
    if result and result["feasible"]:
        w = compute_weight(section_ids, problem, nodes, elements, section_db)
        if w < best_weight:
            best_weight = w
            best_ids = section_ids.copy()

    if best_ids is None:
        return section_ids, total_eval

    section_ids = best_ids.copy()

    # Phase 2: Aggressive single-member greedy reduction
    # Use most of the evaluation budget for thorough reduction

    def get_reduction_priority(section_ids):
        """Get members sorted by reduction priority (lowest stress ratio first)"""
        result = analyze_design(section_ids, problem, nodes, elements, section_db)
        if result is None or not result["feasible"]:
            return None
        stress_ratios = result["member_stresses"] / sigma_limit
        # Weight by member weight contribution - prioritize heavy, low-stress members
        priority_score = stress_ratios - 0.1 * weight_factor / (weight_factor.max() + 1e-10)
        return np.argsort(priority_score)

    # Extended greedy reduction passes - continue while improving
    no_improvement_count = 0
    max_no_improvement = 3  # Stop after 3 consecutive passes with no improvement

    while total_eval < max_eval - dim and no_improvement_count < max_no_improvement:
        improved_this_pass = False

        priority = get_reduction_priority(section_ids)
        total_eval += 1

        if priority is None:
            break

        for idx in priority:
            if total_eval >= max_eval:
                break
            if section_ids[idx] > id_min:
                test_ids = section_ids.copy()
                test_ids[idx] -= 1
                result = analyze_design(test_ids, problem, nodes, elements, section_db)
                total_eval += 1
                if result and result["feasible"]:
                    w = compute_weight(test_ids, problem, nodes, elements, section_db)
                    if w < best_weight:
                        best_weight = w
                        best_ids = test_ids.copy()
                        section_ids = test_ids.copy()
                        improved_this_pass = True

        if improved_this_pass:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    # Phase 3: Pair reduction - try reducing pairs of low-stress members
    section_ids = best_ids.copy()
    priority = get_reduction_priority(section_ids)
    total_eval += 1

    if priority is not None:
        # Focus on members with very low stress
        result = analyze_design(section_ids, problem, nodes, elements, section_db)
        stress_ratios = result["member_stresses"] / sigma_limit
        low_stress_members = np.where(stress_ratios < 0.5)[0]
        low_stress_members = low_stress_members[np.argsort(stress_ratios[low_stress_members])]

        # Try reducing pairs
        for i in range(min(len(low_stress_members), 50)):
            if total_eval >= max_eval:
                break
            for j in range(i + 1, min(i + 10, len(low_stress_members))):
                if total_eval >= max_eval:
                    break
                idx1, idx2 = low_stress_members[i], low_stress_members[j]
                if section_ids[idx1] > id_min and section_ids[idx2] > id_min:
                    test_ids = section_ids.copy()
                    test_ids[idx1] -= 1
                    test_ids[idx2] -= 1
                    result = analyze_design(test_ids, problem, nodes, elements, section_db)
                    total_eval += 1
                    if result and result["feasible"]:
                        w = compute_weight(test_ids, problem, nodes, elements, section_db)
                        if w < best_weight:
                            best_weight = w
                            best_ids = test_ids.copy()
                            section_ids = test_ids.copy()

    # Phase 4: Level-based group reduction
    section_ids = best_ids.copy()
    for level in range(22, -1, -1):  # Start from top level
        if total_eval >= max_eval:
            break
        level_members = [i for i in range(dim) if member_level[i] == level]
        if not level_members:
            continue

        can_reduce = all(section_ids[i] > id_min for i in level_members)
        if can_reduce:
            test_ids = section_ids.copy()
            for i in level_members:
                test_ids[i] -= 1
            result = analyze_design(test_ids, problem, nodes, elements, section_db)
            total_eval += 1
            if result and result["feasible"]:
                w = compute_weight(test_ids, problem, nodes, elements, section_db)
                if w < best_weight:
                    best_weight = w
                    best_ids = test_ids.copy()
                    section_ids = test_ids.copy()

    # Phase 5: Member type group reduction
    section_ids = best_ids.copy()
    for mtype in [1, 2, 0]:  # Horizontal, Diagonal, Vertical
        if total_eval >= max_eval:
            break
        type_members = [i for i in range(dim) if member_type[i] == mtype]
        if not type_members:
            continue

        can_reduce = all(section_ids[i] > id_min for i in type_members)
        if can_reduce:
            test_ids = section_ids.copy()
            for i in type_members:
                test_ids[i] -= 1
            result = analyze_design(test_ids, problem, nodes, elements, section_db)
            total_eval += 1
            if result and result["feasible"]:
                w = compute_weight(test_ids, problem, nodes, elements, section_db)
                if w < best_weight:
                    best_weight = w
                    best_ids = test_ids.copy()
                    section_ids = test_ids.copy()

    # Phase 6: Final aggressive single-member reduction
    section_ids = best_ids.copy()
    no_improvement_count = 0

    while total_eval < max_eval - dim // 2 and no_improvement_count < 2:
        improved_this_pass = False

        priority = get_reduction_priority(section_ids)
        total_eval += 1

        if priority is None:
            break

        for idx in priority:
            if total_eval >= max_eval:
                break
            if section_ids[idx] > id_min:
                test_ids = section_ids.copy()
                test_ids[idx] -= 1
                result = analyze_design(test_ids, problem, nodes, elements, section_db)
                total_eval += 1
                if result and result["feasible"]:
                    w = compute_weight(test_ids, problem, nodes, elements, section_db)
                    if w < best_weight:
                        best_weight = w
                        best_ids = test_ids.copy()
                        section_ids = test_ids.copy()
                        improved_this_pass = True

        if improved_this_pass:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    return best_ids, total_eval


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