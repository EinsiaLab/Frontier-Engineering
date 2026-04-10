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

def _precompute_fem_data(problem, nodes_base, elements):
    """Precompute static FEM data that doesn't change with design variables."""
    n_nodes = len(nodes_base)
    n_dofs = 2 * n_nodes
    n_elements = len(elements)

    supports = problem["supports"]
    fixed_dofs = set()
    for sup in supports:
        nid = sup["node"]
        idx = nid - 1
        if sup.get("fix_x", False):
            fixed_dofs.add(2 * idx)
        if sup.get("fix_y", False):
            fixed_dofs.add(2 * idx + 1)

    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    free_map = np.full(n_dofs, -1, dtype=int)
    for i, dof in enumerate(free_dofs):
        free_map[dof] = i
    n_free = len(free_dofs)

    # Precompute element DOF indices and free DOF mappings
    elem_dofs = np.zeros((n_elements, 4), dtype=int)
    for e in range(n_elements):
        ni, nj = elements[e]
        elem_dofs[e] = [2*ni, 2*ni+1, 2*nj, 2*nj+1]

    # Precompute force vectors for all load cases
    force_vecs = []
    for lc in problem["load_cases"]:
        fvec = np.zeros(n_dofs)
        for load in lc["loads"]:
            nid = load["node"]
            idx = nid - 1
            fvec[2 * idx] += load["fx"]
            fvec[2 * idx + 1] += load["fy"]
        force_vecs.append(fvec[free_dofs])

    # Shape variable node indices
    shape_node_indices = np.array([nid - 1 for nid in problem["shape_variable_node_ids"]], dtype=int)

    # Precompute sparse assembly indices
    # For each element, compute which (row, col) pairs in the free stiffness matrix it contributes to
    row_indices = []
    col_indices = []
    elem_indices = []
    local_i_indices = []
    local_j_indices = []

    for e in range(n_elements):
        dofs_e = elem_dofs[e]
        for i_l in range(4):
            di = dofs_e[i_l]
            ii = free_map[di]
            if ii < 0:
                continue
            for j_l in range(4):
                dj = dofs_e[j_l]
                jj = free_map[dj]
                if jj < 0:
                    continue
                row_indices.append(ii)
                col_indices.append(jj)
                elem_indices.append(e)
                local_i_indices.append(i_l)
                local_j_indices.append(j_l)

    row_indices = np.array(row_indices, dtype=int)
    col_indices = np.array(col_indices, dtype=int)
    elem_indices = np.array(elem_indices, dtype=int)
    local_i_indices = np.array(local_i_indices, dtype=int)
    local_j_indices = np.array(local_j_indices, dtype=int)

    return {
        'n_nodes': n_nodes,
        'n_dofs': n_dofs,
        'n_elements': n_elements,
        'n_free': n_free,
        'free_dofs': np.array(free_dofs, dtype=int),
        'free_map': free_map,
        'elem_dofs': elem_dofs,
        'force_vecs': force_vecs,
        'shape_node_indices': shape_node_indices,
        'row_indices': row_indices,
        'col_indices': col_indices,
        'elem_indices': elem_indices,
        'local_i_indices': local_i_indices,
        'local_j_indices': local_j_indices,
        'E': problem["material"]["E"],
    }


def fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data):
    """
    Fast vectorized FEM solver for all load cases simultaneously.
    Uses sparse matrix assembly for speed.
    Returns (all_stresses, all_disps) or None on failure.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    E = fem_data['E']
    n_elements = fem_data['n_elements']
    n_free = fem_data['n_free']
    n_dofs = fem_data['n_dofs']
    free_dofs = fem_data['free_dofs']
    shape_node_indices = fem_data['shape_node_indices']

    # Update nodes with shape variables
    nodes = nodes_base.copy()
    nodes[shape_node_indices, 1] = shape_vars

    # Compute element geometry vectorized
    ni_arr = elements[:, 0]
    nj_arr = elements[:, 1]
    dx = nodes[nj_arr, 0] - nodes[ni_arr, 0]
    dy = nodes[nj_arr, 1] - nodes[ni_arr, 1]
    L = np.sqrt(dx*dx + dy*dy)

    # Avoid division by zero
    mask = L > 1e-10
    c = np.zeros(n_elements)
    s = np.zeros(n_elements)
    c[mask] = dx[mask] / L[mask]
    s[mask] = dy[mask] / L[mask]

    coeff = E * areas / L
    cc = c * c
    cs = c * s
    ss = s * s

    row_idx = fem_data['row_indices']
    col_idx = fem_data['col_indices']
    elem_idx = fem_data['elem_indices']

    _ke_type = fem_data.get('ke_type')
    if _ke_type is None:
        li_idx = fem_data['local_i_indices']
        lj_idx = fem_data['local_j_indices']
        type_map = np.array([[0,1,0,1],[1,2,1,2],[0,1,0,1],[1,2,1,2]])
        sign_map = np.array([[1,1,-1,-1],[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1]], dtype=np.float64)
        _ke_type_arr = type_map[li_idx, lj_idx]
        _ke_sign_arr = sign_map[li_idx, lj_idx]
        fem_data['ke_type'] = _ke_type_arr
        fem_data['ke_sign'] = _ke_sign_arr
    else:
        _ke_type_arr = fem_data['ke_type']
        _ke_sign_arr = fem_data['ke_sign']

    elem_cc = cc[elem_idx]
    elem_cs = cs[elem_idx]
    elem_ss = ss[elem_idx]
    elem_coeff = coeff[elem_idx]

    base_vals = np.where(_ke_type_arr == 0, elem_cc,
                np.where(_ke_type_arr == 1, elem_cs, elem_ss))
    values = _ke_sign_arr * elem_coeff * base_vals

    # Use sparse matrix for assembly and solving
    K_sparse = csc_matrix((values, (row_idx, col_idx)), shape=(n_free, n_free))

    # Solve for all load cases
    all_stresses = []
    all_disps = []

    force_vecs = fem_data['force_vecs']
    n_lc = len(force_vecs)

    try:
        lu = splu(K_sparse)
        for lc_idx in range(n_lc):
            u_red = lu.solve(force_vecs[lc_idx])
            disp = np.zeros(n_dofs)
            disp[free_dofs] = u_red
            u_ni_x = disp[2*ni_arr]
            u_ni_y = disp[2*ni_arr+1]
            u_nj_x = disp[2*nj_arr]
            u_nj_y = disp[2*nj_arr+1]
            stresses = (E / L) * (-c*u_ni_x - s*u_ni_y + c*u_nj_x + s*u_nj_y)
            all_stresses.append(stresses)
            all_disps.append(disp)
    except Exception:
        return None

    return all_stresses, all_disps


def fast_analyze(areas, shape_vars, problem, nodes_base, elements, fem_data):
    """Fast analysis using vectorized FEM."""
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]

    result = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
    if result is None:
        return None

    all_stresses, all_disps = result

    max_stress = 0.0
    max_disp = 0.0
    for stresses in all_stresses:
        max_stress = max(max_stress, np.max(np.abs(stresses)))
    for disp in all_disps:
        max_disp = max(max_disp, np.max(np.abs(disp)))

    feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
    return {"feasible": feasible, "max_stress": max_stress, "max_disp": max_disp}


def fast_compute_weight(areas, shape_vars, problem, nodes_base, elements):
    """Fast weight computation."""
    rho = problem["material"]["rho"]
    shape_node_indices = np.array([nid - 1 for nid in problem["shape_variable_node_ids"]], dtype=int)

    nodes = nodes_base.copy()
    nodes[shape_node_indices, 1] = shape_vars

    ni_arr = elements[:, 0]
    nj_arr = elements[:, 1]
    dx = nodes[nj_arr, 0] - nodes[ni_arr, 0]
    dy = nodes[nj_arr, 1] - nodes[ni_arr, 1]
    L = np.sqrt(dx*dx + dy*dy)

    return rho * np.sum(L * areas)


def get_element_stresses_and_disps(areas, shape_vars, problem, nodes_base, elements):
    """Get per-element stresses and full displacement vector for all load cases."""
    E = problem["material"]["E"]
    supports = problem["supports"]
    num_bars = problem["num_bars"]

    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        node_idx = nid - 1
        nodes[node_idx, 1] = shape_vars[idx]

    all_stresses = []
    all_disps = []

    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            idx = nid - 1
            fvec[2 * idx] += load["fx"]
            fvec[2 * idx + 1] += load["fy"]

        disp, stresses, lengths = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        if disp is None:
            return None, None
        all_stresses.append(stresses)
        all_disps.append(disp)

    return all_stresses, all_disps


def run_fsd(areas_init, shape_vars, problem, nodes_base, elements, max_iter=80, target_ratio=1.0, fem_data=None):
    """
    Run Fully Stressed Design iterations to optimize areas for given shape.
    Returns (areas, feasible, weight).
    Uses fast FEM if fem_data is provided.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]

    areas = areas_init.copy()
    prev_areas = None
    best_fsd_areas = None
    best_fsd_w = float('inf')

    for iteration in range(max_iter):
        if fem_data is not None:
            result = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
            if result is None:
                all_stresses, all_disps = None, None
            else:
                all_stresses, all_disps = result
        else:
            all_stresses, all_disps = get_element_stresses_and_disps(
                areas, shape_vars, problem, nodes_base, elements)

        if all_stresses is None:
            areas = np.clip(areas * 1.5, area_min, area_max)
            continue

        max_elem_stress = np.zeros(num_bars)
        for stresses in all_stresses:
            max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))

        max_disp = 0.0
        for disp in all_disps:
            max_disp = max(max_disp, np.max(np.abs(disp)))

        stress_feasible = np.max(max_elem_stress) <= sigma_limit * target_ratio + 1e-6
        disp_feasible = max_disp <= disp_limit * target_ratio + 1e-6

        # Track best feasible during FSD
        if stress_feasible and disp_feasible:
            w_cur = fast_compute_weight(areas, shape_vars, problem, nodes_base, elements)
            if w_cur < best_fsd_w:
                best_fsd_w = w_cur
                best_fsd_areas = areas.copy()

        stress_ratios = max_elem_stress / (sigma_limit * target_ratio)

        if not disp_feasible:
            disp_ratio = max_disp / (disp_limit * target_ratio)
            if disp_ratio > 2.0:
                disp_scale = disp_ratio ** 0.8
            elif disp_ratio > 1.5:
                disp_scale = disp_ratio ** 0.65
            elif disp_ratio > 1.1:
                disp_scale = disp_ratio ** 0.5
            else:
                disp_scale = disp_ratio ** 0.4
            stress_ratios = np.maximum(stress_ratios, disp_scale)

        # Adaptive damping - more aggressive early, gentler late
        if iteration < 2:
            damping = 0.95
        elif iteration < 5:
            damping = 0.8
        elif iteration < 10:
            damping = 0.6
        elif iteration < 20:
            damping = 0.45
        elif iteration < 40:
            damping = 0.3
        else:
            damping = 0.15

        new_areas = areas * (stress_ratios ** damping)

        # Aggressive reduction for very low stress elements
        low_stress = max_elem_stress < sigma_limit * 0.005
        new_areas[low_stress] = np.maximum(areas[low_stress] * 0.3, area_min)

        # Medium-low stress elements
        med_low = (max_elem_stress >= sigma_limit * 0.005) & (max_elem_stress < sigma_limit * 0.05)
        new_areas[med_low] = np.maximum(areas[med_low] * 0.6, area_min)

        areas = np.clip(new_areas, area_min, area_max)

        # Check convergence
        if prev_areas is not None:
            rel_change = np.max(np.abs(areas - prev_areas) / (prev_areas + 1e-10))
            if rel_change < 5e-5 and stress_feasible and disp_feasible:
                break
        prev_areas = areas.copy()

    # Use fast_analyze for feasibility check (much faster than analyze_design)
    if fem_data is not None:
        res = fast_analyze(areas, shape_vars, problem, nodes_base, elements, fem_data)
        if res and res["feasible"]:
            w = fast_compute_weight(areas, shape_vars, problem, nodes_base, elements)
            if best_fsd_areas is not None:
                res2 = fast_analyze(best_fsd_areas, shape_vars, problem, nodes_base, elements, fem_data)
                if res2 and res2["feasible"] and best_fsd_w < w:
                    return best_fsd_areas, True, best_fsd_w
            return areas, True, w
        else:
            if best_fsd_areas is not None:
                res2 = fast_analyze(best_fsd_areas, shape_vars, problem, nodes_base, elements, fem_data)
                if res2 and res2["feasible"]:
                    return best_fsd_areas, True, best_fsd_w
            return areas, False, float('inf')
    else:
        result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if result and result["feasible"]:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if best_fsd_areas is not None:
                res2 = analyze_design(best_fsd_areas, shape_vars, problem, nodes_base, elements)
                if res2 and res2["feasible"] and best_fsd_w < w:
                    return best_fsd_areas, True, best_fsd_w
            return areas, True, w
        else:
            if best_fsd_areas is not None:
                res2 = analyze_design(best_fsd_areas, shape_vars, problem, nodes_base, elements)
                if res2 and res2["feasible"]:
                    return best_fsd_areas, True, best_fsd_w
            return areas, False, float('inf')


def optimize_stress_ratio(problem, nodes_base, elements, max_iter=15):
    """
    Advanced optimization using FSD + Nelder-Mead/Powell shape optimization.
    Uses fast vectorized FEM for internal optimization.
    """
    from scipy.optimize import minimize as scipy_minimize, differential_evolution

    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])

    # Precompute FEM data for fast solving
    fem_data = _precompute_fem_data(problem, nodes_base, elements)

    best_weight = float('inf')
    best_x = None

    # Get default shape vars
    default_shape = np.array([nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]])
    default_norm = (default_shape - y_min) / (y_max - y_min)

    n_sv = num_shape  # 9 shape variables

    # Phase 1: Explore diverse shape configurations
    shape_candidates = []
    t = np.linspace(0, 1, n_sv + 2)[1:-1]  # parametric positions along span

    # Default shape
    shape_candidates.append(default_norm.copy())

    # Symmetric arch shapes at different heights - finer grid
    for height_frac in np.arange(0.02, 0.98, 0.015):
        arch = height_frac * 4 * t * (1 - t)
        shape_candidates.append(np.clip(arch, 0.01, 0.99))

    # Asymmetric shapes - loads on right side (nodes 15, 17, 19)
    for height_frac in np.arange(0.05, 0.95, 0.03):
        for skew in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            a_param = skew * 3
            b_param = (1 - skew) * 3
            arch = height_frac * (t ** a_param) * ((1 - t) ** b_param)
            arch = arch / (np.max(arch) + 1e-10) * height_frac
            shape_candidates.append(np.clip(arch, 0.01, 0.99))

    # Linear ramp shapes - more combinations
    for h_left in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for h_right in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ramp = h_left + (h_right - h_left) * t
            shape_candidates.append(np.clip(ramp, 0.01, 0.99))

    # Flat shapes
    for h in np.arange(0.02, 0.98, 0.02):
        shape_candidates.append(np.full(n_sv, h))

    # Triangular shapes (peak at different positions)
    for peak_pos in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for height in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            tri = np.where(t <= peak_pos, t / peak_pos, (1 - t) / (1 - peak_pos + 1e-10)) * height
            shape_candidates.append(np.clip(tri, 0.01, 0.99))

    # Catenary-like shapes
    for height in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cat = height * (np.cosh(3 * (t - 0.5)) - np.cosh(1.5)) / (1 - np.cosh(1.5))
        shape_candidates.append(np.clip(cat, 0.01, 0.99))

    # Parabolic with shifted vertex toward loads
    for vertex_pos in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for height in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
            par = height * (1 - ((t - vertex_pos) / max(vertex_pos, 1 - vertex_pos))**2)
            par = np.maximum(par, 0.0)
            shape_candidates.append(np.clip(par, 0.01, 0.99))

    # Sine-based shapes
    for n_half in [1, 2, 3]:
        for height in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]:
            sine = height * np.sin(n_half * np.pi * t)
            shape_candidates.append(np.clip(np.abs(sine), 0.01, 0.99))

    # Exponential growth shapes (increasing toward loaded end)
    for rate in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for height in [0.2, 0.4, 0.6, 0.8]:
            exp_shape = height * (np.exp(rate * t) - 1) / (np.exp(rate) - 1)
            shape_candidates.append(np.clip(exp_shape, 0.01, 0.99))

    # Quadratic shapes biased toward right (where loads are)
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for height in [0.2, 0.35, 0.5, 0.65, 0.8]:
            quad = height * (t / bias) ** 2
            quad = np.minimum(quad, height)
            shape_candidates.append(np.clip(quad, 0.01, 0.99))

    # Cubic spline-like shapes with 3 control points
    for h1 in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        for h2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for h3 in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
                # Interpolate through 3 height control points
                ctrl_t = [0.0, 0.5, 1.0]
                ctrl_h = [h1, h2, h3]
                interp = np.interp(t, ctrl_t, ctrl_h)
                shape_candidates.append(np.clip(interp, 0.01, 0.99))

    # 4-point control shapes for more complex profiles
    for h1 in [0.05, 0.2, 0.4]:
        for h2 in [0.2, 0.4, 0.6, 0.8]:
            for h3 in [0.2, 0.5, 0.8]:
                for h4 in [0.05, 0.2, 0.4]:
                    ctrl_t4 = [0.0, 0.33, 0.67, 1.0]
                    ctrl_h4 = [h1, h2, h3, h4]
                    interp4 = np.interp(t, ctrl_t4, ctrl_h4)
                    shape_candidates.append(np.clip(interp4, 0.01, 0.99))

    # Power-law shapes
    for power in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
        for height in [0.2, 0.4, 0.6, 0.8]:
            pw = height * t ** power
            shape_candidates.append(np.clip(pw, 0.01, 0.99))
            # Reversed
            pw_rev = height * (1 - t) ** power
            shape_candidates.append(np.clip(pw_rev, 0.01, 0.99))

    # Random perturbations around promising shapes (Latin Hypercube-like)
    np.random.seed(42)
    for _ in range(400):
        base = np.random.uniform(0.02, 0.98, n_sv)
        # Sort to create smooth shapes
        if np.random.random() < 0.3:
            base = np.sort(base)
        elif np.random.random() < 0.5:
            base = np.sort(base)[::-1]
        shape_candidates.append(np.clip(base, 0.01, 0.99))

    # Use fast FSD for screening - very quick first pass
    def shape_objective_quick(shape_normalized, areas_init=None):
        """Very quick objective for initial screening."""
        shape_vars = y_min + shape_normalized * (y_max - y_min)
        shape_vars = np.clip(shape_vars, y_min, y_max)

        if areas_init is None:
            areas_start = np.full(num_bars, area_max * 0.15)
        else:
            areas_start = areas_init.copy()

        areas, feasible, w = run_fsd(areas_start, shape_vars, problem, nodes_base, elements, max_iter=20, fem_data=fem_data)

        if not feasible:
            areas_start2 = np.full(num_bars, area_max * 0.3)
            areas, feasible, w = run_fsd(areas_start2, shape_vars, problem, nodes_base, elements, max_iter=30, fem_data=fem_data)

        if not feasible:
            return 1e8, areas
        return w, areas

    def shape_objective_thorough(shape_normalized, areas_init=None):
        """Thorough objective for top candidates."""
        nonlocal best_weight, best_x

        shape_vars = y_min + shape_normalized * (y_max - y_min)
        shape_vars = np.clip(shape_vars, y_min, y_max)

        if areas_init is None:
            areas_start = np.full(num_bars, area_max * 0.15)
        else:
            areas_start = areas_init.copy()

        areas, feasible, w = run_fsd(areas_start, shape_vars, problem, nodes_base, elements, max_iter=60, fem_data=fem_data)

        if not feasible:
            areas_start2 = np.full(num_bars, area_max * 0.3)
            areas, feasible, w = run_fsd(areas_start2, shape_vars, problem, nodes_base, elements, max_iter=70, fem_data=fem_data)

        if feasible and w < best_weight:
            best_weight = w
            best_x = np.concatenate([areas, shape_vars]).copy()

        if not feasible:
            return 1e8, areas
        return w, areas

    print(f"  Quick screening {len(shape_candidates)} shape candidates...")
    candidate_results = []
    for i, sv_norm in enumerate(shape_candidates):
        sv_norm = np.clip(sv_norm, 0.01, 0.99)
        w, areas_out = shape_objective_quick(sv_norm)
        candidate_results.append((w, i, sv_norm.copy(), areas_out.copy()))

    candidate_results.sort(key=lambda x: x[0])
    print(f"  Quick best weight: {candidate_results[0][0]:.2f} kg")

    # Thorough evaluation of top candidates
    top_thorough = min(40, len([c for c in candidate_results if c[0] < 1e7]))
    thorough_results = []
    for rank in range(top_thorough):
        w0, idx0, sv_norm0, areas0 = candidate_results[rank]
        if w0 >= 1e7:
            continue
        w_t, areas_t = shape_objective_thorough(sv_norm0, areas0)
        thorough_results.append((w_t, idx0, sv_norm0.copy(), areas_t.copy()))

    thorough_results.sort(key=lambda x: x[0])
    candidate_results = thorough_results
    print(f"  Thorough best weight: {candidate_results[0][0]:.2f} kg")

    # Phase 2: Refine top candidates with Nelder-Mead - invest heavily in top few
    top_k = min(6, len([c for c in candidate_results if c[0] < 1e7]))

    for rank in range(top_k):
        w0, idx0, sv_norm0, areas_from_screen = candidate_results[rank]
        if w0 >= 1e7:
            continue

        print(f"  Refining candidate {rank+1} (weight={w0:.2f})...")

        # First do a thorough FSD from the screened areas
        sv = y_min + sv_norm0 * (y_max - y_min)
        areas_opt, feas, w_refined = run_fsd(areas_from_screen, sv, problem, nodes_base, elements, max_iter=80, fem_data=fem_data)
        if not feas:
            areas_opt, feas, w_refined = run_fsd(np.full(num_bars, area_max * 0.2), sv, problem, nodes_base, elements, max_iter=80, fem_data=fem_data)

        current_areas = [areas_opt.copy()]

        def nm_objective(sv_norm_flat, _current_areas=current_areas):
            nonlocal best_weight, best_x
            sv_norm_c = np.clip(sv_norm_flat, 0.01, 0.99)
            shape_vars = y_min + sv_norm_c * (y_max - y_min)

            areas, feasible, w = run_fsd(_current_areas[0], shape_vars, problem, nodes_base, elements, max_iter=45, fem_data=fem_data)

            if not feasible:
                areas2 = np.clip(_current_areas[0] * 1.3, area_min, area_max)
                areas, feasible, w = run_fsd(areas2, shape_vars, problem, nodes_base, elements, max_iter=55, fem_data=fem_data)

            if feasible:
                _current_areas[0] = areas.copy()
                if w < best_weight:
                    best_weight = w
                    best_x = np.concatenate([areas, shape_vars]).copy()
                return w
            else:
                return 1e8

        try:
            if rank < 2:
                maxfev_nm = 2000
            elif rank < 4:
                maxfev_nm = 1000
            else:
                maxfev_nm = 500
            scipy_minimize(
                nm_objective,
                sv_norm0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': maxfev_nm,
                    'xatol': 0.00005,
                    'fatol': 0.002,
                    'adaptive': True,
                }
            )
        except Exception:
            pass

    # Phase 3: Powell method on the best solution found
    if best_x is not None:
        best_sv = best_x[num_bars:]
        best_sv_norm = (best_sv - y_min) / (y_max - y_min)
        best_areas_start = [best_x[:num_bars].copy()]

        def powell_objective(sv_norm_flat):
            nonlocal best_weight, best_x
            sv_norm_c = np.clip(sv_norm_flat, 0.01, 0.99)
            shape_vars = y_min + sv_norm_c * (y_max - y_min)

            areas, feasible, w = run_fsd(best_areas_start[0], shape_vars, problem, nodes_base, elements, max_iter=50, fem_data=fem_data)

            if feasible:
                best_areas_start[0] = areas.copy()
                if w < best_weight:
                    best_weight = w
                    best_x = np.concatenate([areas, shape_vars]).copy()
                return w
            else:
                return 1e8

        try:
            scipy_minimize(
                powell_objective,
                best_sv_norm,
                method='Powell',
                options={
                    'maxiter': 500,
                    'maxfev': 350,
                    'ftol': 0.02,
                }
            )
        except Exception:
            pass

    # Phase 3b: Coordinate descent on shape variables with fine steps and momentum
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()
        current_w = best_weight

        step_sizes = [(y_max - y_min) * s for s in [0.12, 0.08, 0.05, 0.03, 0.02, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005]]

        for cd_round in range(20):
            improved_any = False
            for si in range(num_shape):
                for step in step_sizes:
                    best_dir = 0
                    best_trial_areas = None
                    for direction in [step, -step]:
                        trial_sv = shape_vars.copy()
                        trial_sv[si] = np.clip(shape_vars[si] + direction, y_min, y_max)
                        trial_areas, feas, w = run_fsd(areas, trial_sv, problem, nodes_base, elements, max_iter=45, fem_data=fem_data)
                        if feas and w < current_w - 0.01:
                            current_w = w
                            best_dir = direction
                            best_trial_areas = trial_areas.copy()

                    if best_dir != 0:
                        shape_vars[si] = np.clip(shape_vars[si] + best_dir, y_min, y_max)
                        areas = best_trial_areas.copy()
                        best_weight = current_w
                        best_x = np.concatenate([areas, shape_vars]).copy()
                        improved_any = True
                        # Momentum: continue in same direction with acceleration
                        momentum_step = best_dir
                        for momentum_iter in range(10):
                            trial_sv = shape_vars.copy()
                            trial_sv[si] = np.clip(shape_vars[si] + momentum_step, y_min, y_max)
                            if trial_sv[si] == shape_vars[si]:
                                break
                            trial_areas2, feas2, w2 = run_fsd(areas, trial_sv, problem, nodes_base, elements, max_iter=45, fem_data=fem_data)
                            if feas2 and w2 < current_w - 0.01:
                                current_w = w2
                                shape_vars[si] = trial_sv[si]
                                areas = trial_areas2.copy()
                                best_weight = current_w
                                best_x = np.concatenate([areas, shape_vars]).copy()
                            else:
                                break
                        break  # move to next shape var

            if not improved_any:
                break

    # Phase 3c: Second Powell refinement after coordinate descent
    if best_x is not None:
        best_sv2 = best_x[num_bars:]
        best_sv_norm2 = (best_sv2 - y_min) / (y_max - y_min)
        best_areas_start2 = [best_x[:num_bars].copy()]

        def powell_objective2(sv_norm_flat):
            nonlocal best_weight, best_x
            sv_norm_c = np.clip(sv_norm_flat, 0.01, 0.99)
            shape_vars = y_min + sv_norm_c * (y_max - y_min)

            areas, feasible, w = run_fsd(best_areas_start2[0], shape_vars, problem, nodes_base, elements, max_iter=50, fem_data=fem_data)

            if feasible:
                best_areas_start2[0] = areas.copy()
                if w < best_weight:
                    best_weight = w
                    best_x = np.concatenate([areas, shape_vars]).copy()
                return w
            else:
                return 1e8

        try:
            scipy_minimize(
                powell_objective2,
                best_sv_norm2,
                method='Powell',
                options={
                    'maxiter': 800,
                    'maxfev': 800,
                    'ftol': 0.002,
                }
            )
        except Exception:
            pass

    # Phase 3c2: L-BFGS-B refinement
    if best_x is not None:
        best_sv3 = best_x[num_bars:]
        best_sv_norm3 = (best_sv3 - y_min) / (y_max - y_min)
        best_areas_start3 = [best_x[:num_bars].copy()]

        def lbfgsb_objective(sv_norm_flat):
            nonlocal best_weight, best_x
            sv_norm_c = np.clip(sv_norm_flat, 0.01, 0.99)
            shape_vars = y_min + sv_norm_c * (y_max - y_min)

            areas, feasible, w = run_fsd(best_areas_start3[0], shape_vars, problem, nodes_base, elements, max_iter=40, fem_data=fem_data)

            if feasible:
                best_areas_start3[0] = areas.copy()
                if w < best_weight:
                    best_weight = w
                    best_x = np.concatenate([areas, shape_vars]).copy()
                return w
            else:
                return 1e8

        try:
            scipy_minimize(
                lbfgsb_objective,
                best_sv_norm3,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * n_sv,
                options={
                    'maxiter': 100,
                    'maxfun': 400,
                    'ftol': 0.005,
                    'eps': 0.0008,
                }
            )
        except Exception:
            pass

    # Phase 3d: Pairwise shape variable optimization with multiple step sizes
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()
        current_w = best_weight

        for pair_step in [(y_max - y_min) * s for s in [0.01, 0.005, 0.002]]:
            for _ in range(2):
                improved_pair = False
                for si in range(num_shape):
                    for sj in range(si + 1, num_shape):
                        for di in [pair_step, -pair_step]:
                            for dj in [pair_step, -pair_step]:
                                trial_sv = shape_vars.copy()
                                trial_sv[si] = np.clip(shape_vars[si] + di, y_min, y_max)
                                trial_sv[sj] = np.clip(shape_vars[sj] + dj, y_min, y_max)
                                trial_areas, feas, w = run_fsd(areas, trial_sv, problem, nodes_base, elements, max_iter=40, fem_data=fem_data)
                                if feas and w < current_w - 0.02:
                                    current_w = w
                                    shape_vars = trial_sv.copy()
                                    areas = trial_areas.copy()
                                    best_weight = current_w
                                    best_x = np.concatenate([areas, shape_vars]).copy()
                                    improved_pair = True
                if not improved_pair:
                    break

    # Phase 4: Final FSD refinement with target ratios
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()

        targets = sorted(set(
            [round(x, 6) for x in np.arange(0.99999, 0.996, -0.0002).tolist()] +
            [round(x, 5) for x in np.arange(0.996, 0.98, -0.001).tolist()] +
            [round(x, 4) for x in np.arange(0.98, 0.95, -0.002).tolist()] +
            [round(x, 3) for x in np.arange(0.95, 0.90, -0.005).tolist()]
        ), reverse=True)

        for target in targets:
            areas_trial, feasible, w = run_fsd(areas, shape_vars, problem, nodes_base, elements,
                                                max_iter=80, target_ratio=target, fem_data=fem_data)
            if feasible and w < best_weight:
                best_weight = w
                best_x = np.concatenate([areas_trial, shape_vars]).copy()
                areas = areas_trial.copy()

    # Phase 5: Element-wise area trimming with binary search
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()

        # Get current stress ratios to guide trimming
        result_fem = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
        stress_util = None

        if result_fem is not None:
            all_stresses_list, all_disps_list = result_fem
            max_elem_stress = np.zeros(num_bars)
            for stresses in all_stresses_list:
                max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
            stress_util = max_elem_stress / sigma_limit

        for round_idx in range(30):
            improved = False
            nodes_tmp = nodes_base.copy()
            for idx_s, nid in enumerate(problem["shape_variable_node_ids"]):
                nodes_tmp[nid - 1, 1] = shape_vars[idx_s]

            ni_arr = elements[:, 0]
            nj_arr = elements[:, 1]
            dx_arr = nodes_tmp[nj_arr, 0] - nodes_tmp[ni_arr, 0]
            dy_arr = nodes_tmp[nj_arr, 1] - nodes_tmp[ni_arr, 1]
            L_arr = np.sqrt(dx_arr*dx_arr + dy_arr*dy_arr)
            elem_weights = areas * L_arr

            if stress_util is not None:
                potential_savings = elem_weights * (1 - stress_util)
                order = np.argsort(-potential_savings)
            else:
                order = np.argsort(-elem_weights)

            for e in order:
                if areas[e] <= area_min + 1e-6:
                    continue

                # Binary search for minimum feasible area
                a_low = area_min
                a_high = areas[e]

                # First check if minimum area is feasible
                trial_areas = areas.copy()
                trial_areas[e] = a_low
                res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                if res and res["feasible"]:
                    w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                    if w < best_weight:
                        best_weight = w
                        best_x = np.concatenate([trial_areas, shape_vars]).copy()
                        areas = trial_areas.copy()
                        improved = True
                    continue

                # Binary search
                for _ in range(16):
                    if a_high - a_low < 0.2:
                        break
                    a_mid = (a_low + a_high) / 2
                    trial_areas = areas.copy()
                    trial_areas[e] = a_mid
                    res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                    if res and res["feasible"]:
                        a_high = a_mid
                    else:
                        a_low = a_mid

                new_area = a_high * 1.0002
                if new_area < areas[e] - 0.05:
                    trial_areas = areas.copy()
                    trial_areas[e] = new_area
                    res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                    if res and res["feasible"]:
                        w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                        if w < best_weight:
                            best_weight = w
                            best_x = np.concatenate([trial_areas, shape_vars]).copy()
                            areas = trial_areas.copy()
                            improved = True

            # Update stress info
            result_fem = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
            if result_fem is not None:
                all_stresses_list, _ = result_fem
                max_elem_stress = np.zeros(num_bars)
                for stresses in all_stresses_list:
                    max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
                stress_util = max_elem_stress / sigma_limit

            if not improved:
                break

        # Try simultaneous reduction of low-utilization elements
        if stress_util is not None:
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for frac in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
                    trial_areas = areas.copy()
                    low_util = stress_util < threshold
                    if np.any(low_util):
                        trial_areas[low_util] = np.maximum(areas[low_util] * frac, area_min)
                        res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                        if res and res["feasible"]:
                            w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                            if w < best_weight:
                                best_weight = w
                                best_x = np.concatenate([trial_areas, shape_vars]).copy()
                                areas = trial_areas.copy()
                                result_fem = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
                                if result_fem is not None:
                                    all_stresses_list, _ = result_fem
                                    max_elem_stress = np.zeros(num_bars)
                                    for stresses in all_stresses_list:
                                        max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
                                    stress_util = max_elem_stress / sigma_limit

    # Phase 5a-extra: Try simultaneous area scaling
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()
        for scale in [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
            trial_areas = np.clip(areas * scale, area_min, area_max)
            res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
            if res and res["feasible"]:
                w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                if w < best_weight:
                    best_weight = w
                    best_x = np.concatenate([trial_areas, shape_vars]).copy()
                    areas = trial_areas.copy()

    # Phase 5b: Post-trimming coordinate descent to re-optimize shape
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()
        current_w = best_weight

        for cd_round_final in range(8):
            improved_final = False
            step_fracs_final = [(y_max - y_min) * s for s in [0.01, 0.005, 0.003, 0.001, 0.0005, 0.0002]]

            for si in range(num_shape):
                for step in step_fracs_final:
                    best_dir = 0
                    best_trial_areas = None
                    for direction in [step, -step]:
                        trial_sv = shape_vars.copy()
                        trial_sv[si] = np.clip(shape_vars[si] + direction, y_min, y_max)
                        trial_areas, feas, w = run_fsd(areas, trial_sv, problem, nodes_base, elements, max_iter=45, fem_data=fem_data)
                        if feas and w < current_w - 0.005:
                            current_w = w
                            best_dir = direction
                            best_trial_areas = trial_areas.copy()

                    if best_dir != 0:
                        shape_vars[si] = np.clip(shape_vars[si] + best_dir, y_min, y_max)
                        areas = best_trial_areas.copy()
                        best_weight = current_w
                        best_x = np.concatenate([areas, shape_vars]).copy()
                        improved_final = True
                        # Momentum
                        for _ in range(6):
                            trial_sv = shape_vars.copy()
                            trial_sv[si] = np.clip(shape_vars[si] + best_dir, y_min, y_max)
                            if trial_sv[si] == shape_vars[si]:
                                break
                            trial_areas2, feas2, w2 = run_fsd(areas, trial_sv, problem, nodes_base, elements, max_iter=45, fem_data=fem_data)
                            if feas2 and w2 < current_w - 0.005:
                                current_w = w2
                                shape_vars[si] = trial_sv[si]
                                areas = trial_areas2.copy()
                                best_weight = current_w
                                best_x = np.concatenate([areas, shape_vars]).copy()
                            else:
                                break
                        break

            if not improved_final:
                break

        # One more round of element-wise trimming after shape adjustment
        if best_x is not None:
            areas = best_x[:num_bars].copy()
            shape_vars = best_x[num_bars:].copy()
            result_fem = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
            if result_fem is not None:
                all_stresses_list, _ = result_fem
                max_elem_stress = np.zeros(num_bars)
                for stresses in all_stresses_list:
                    max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
                stress_util = max_elem_stress / sigma_limit

                nodes_tmp = nodes_base.copy()
                for idx_s, nid in enumerate(problem["shape_variable_node_ids"]):
                    nodes_tmp[nid - 1, 1] = shape_vars[idx_s]
                ni_arr = elements[:, 0]
                nj_arr = elements[:, 1]
                dx_arr = nodes_tmp[nj_arr, 0] - nodes_tmp[ni_arr, 0]
                dy_arr = nodes_tmp[nj_arr, 1] - nodes_tmp[ni_arr, 1]
                L_arr = np.sqrt(dx_arr*dx_arr + dy_arr*dy_arr)
                potential_savings = areas * L_arr * (1 - stress_util)
                order = np.argsort(-potential_savings)

                for e in order:
                    if areas[e] <= area_min + 1e-6:
                        continue
                    a_low = area_min
                    a_high = areas[e]
                    trial_areas = areas.copy()
                    trial_areas[e] = a_low
                    res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                    if res and res["feasible"]:
                        w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                        if w < best_weight:
                            best_weight = w
                            best_x = np.concatenate([trial_areas, shape_vars]).copy()
                            areas = trial_areas.copy()
                        continue
                    for _ in range(16):
                        if a_high - a_low < 0.2:
                            break
                        a_mid = (a_low + a_high) / 2
                        trial_areas = areas.copy()
                        trial_areas[e] = a_mid
                        res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                        if res and res["feasible"]:
                            a_high = a_mid
                        else:
                            a_low = a_mid
                    new_area = a_high * 1.0002
                    if new_area < areas[e] - 0.05:
                        trial_areas = areas.copy()
                        trial_areas[e] = new_area
                        res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                        if res and res["feasible"]:
                            w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                            if w < best_weight:
                                best_weight = w
                                best_x = np.concatenate([trial_areas, shape_vars]).copy()
                                areas = trial_areas.copy()

    # Phase 6: Aggressive group trimming - try reducing multiple elements simultaneously
    if best_x is not None:
        areas = best_x[:num_bars].copy()
        shape_vars = best_x[num_bars:].copy()

        # Get stress utilization
        result_fem = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
        if result_fem is not None:
            all_stresses_list, _ = result_fem
            max_elem_stress = np.zeros(num_bars)
            for stresses in all_stresses_list:
                max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
            stress_util = max_elem_stress / sigma_limit

            # Group elements by utilization and try reducing groups
            for n_groups in [3, 5, 8]:
                util_thresholds = np.linspace(0, 1, n_groups + 1)
                for gi in range(n_groups):
                    group_mask = (stress_util >= util_thresholds[gi]) & (stress_util < util_thresholds[gi + 1])
                    if not np.any(group_mask):
                        continue
                    for frac in [0.8, 0.85, 0.9, 0.95, 0.97, 0.99]:
                        trial_areas = areas.copy()
                        trial_areas[group_mask] = np.maximum(areas[group_mask] * frac, area_min)
                        res = fast_analyze(trial_areas, shape_vars, problem, nodes_base, elements, fem_data)
                        if res and res["feasible"]:
                            w = fast_compute_weight(trial_areas, shape_vars, problem, nodes_base, elements)
                            if w < best_weight:
                                best_weight = w
                                best_x = np.concatenate([trial_areas, shape_vars]).copy()
                                areas = trial_areas.copy()
                                # Update stress util
                                result_fem2 = fast_fem_solve(areas, shape_vars, nodes_base, elements, fem_data)
                                if result_fem2 is not None:
                                    all_stresses_list2, _ = result_fem2
                                    max_elem_stress = np.zeros(num_bars)
                                    for stresses in all_stresses_list2:
                                        max_elem_stress = np.maximum(max_elem_stress, np.abs(stresses))
                                    stress_util = max_elem_stress / sigma_limit

    # Final validation with official analyze_design
    if best_x is not None:
        areas_final = best_x[:num_bars]
        sv_final = best_x[num_bars:]
        result = analyze_design(areas_final, sv_final, problem, nodes_base, elements)
        if not (result and result["feasible"]):
            # If fast solver gave slightly different result, add small margin
            print("  Warning: Final validation failed, running safety FSD...")
            areas_safe, feas, w = run_fsd(areas_final * 1.02, sv_final, problem, nodes_base, elements, max_iter=80)
            if feas:
                best_x = np.concatenate([areas_safe, sv_final]).copy()
                best_weight = w

    if best_x is None:
        areas = np.full(num_bars, area_max * 0.3)
        shape_vars = np.array([nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]])
        best_x = np.concatenate([areas, shape_vars])

    return best_x


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
    print("Optimizing using FSD + Nelder-Mead/Powell shape optimization...")
    x_best = optimize_stress_ratio(problem, nodes_base, elements)
    w = compute_weight(x_best, problem, nodes_base, elements)

    result = analyze_design(x_best[:num_bars], x_best[num_bars:], problem, nodes_base, elements)
    feasible = result["feasible"] if result else False

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "iscso_2015",
        "solution_vector": x_best.tolist(),
        "algorithm": "FSD_NM_Powell_v2",
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