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

class FastFEMSolver:
    """Pre-computed FEM solver for repeated analyses with same topology - vectorized assembly."""

    def __init__(self, nodes, elements, supports):
        self.nodes = nodes
        self.elements = elements
        self.n_nodes = len(nodes)
        self.n_elems = len(elements)
        self.n_dofs = 3 * self.n_nodes

        # Compute fixed/free DOFs
        fixed = set()
        for sup in supports:
            nid = sup["node"]
            if sup.get("fix_x", False): fixed.add(3*nid)
            if sup.get("fix_y", False): fixed.add(3*nid + 1)
            if sup.get("fix_z", False): fixed.add(3*nid + 2)

        self.free = np.array(sorted(set(range(self.n_dofs)) - fixed))
        self.free_map = -np.ones(self.n_dofs, dtype=int)
        for i, d in enumerate(self.free):
            self.free_map[d] = i
        self.n_free = len(self.free)

        # Precompute element geometry
        ni_arr = elements[:, 0]
        nj_arr = elements[:, 1]
        d_vecs = nodes[nj_arr] - nodes[ni_arr]
        self.lengths = np.linalg.norm(d_vecs, axis=1)
        self.valid = self.lengths > 1e-10
        self.dc = np.zeros_like(d_vecs)
        self.dc[self.valid] = d_vecs[self.valid] / self.lengths[self.valid, np.newaxis]

        # Precompute DOF indices for each element
        self.dofs_e = np.zeros((self.n_elems, 6), dtype=int)
        self.dofs_e[:, 0] = 3 * ni_arr
        self.dofs_e[:, 1] = 3 * ni_arr + 1
        self.dofs_e[:, 2] = 3 * ni_arr + 2
        self.dofs_e[:, 3] = 3 * nj_arr
        self.dofs_e[:, 4] = 3 * nj_arr + 1
        self.dofs_e[:, 5] = 3 * nj_arr + 2

        self.free_dofs_e = self.free_map[self.dofs_e]  # (n_elems, 6)

        # Precompute normalized 6x6 element stiffness in flat form
        # ke_norm[e] = [[B, -B], [-B, B]] where B = outer(dc, dc)
        # Store as (n_elems, 6, 6)
        self.ke_flat = np.zeros((self.n_elems, 6, 6))
        for e in range(self.n_elems):
            if not self.valid[e]:
                continue
            dcv = self.dc[e]
            B = np.outer(dcv, dcv)
            self.ke_flat[e, :3, :3] = B
            self.ke_flat[e, :3, 3:] = -B
            self.ke_flat[e, 3:, :3] = -B
            self.ke_flat[e, 3:, 3:] = B

        # Vectorized sparse structure: precompute all (row, col, elem_idx, local_i, local_j)
        # for free-free DOF pairs
        all_rows = []
        all_cols = []
        all_elem_idx = []
        all_local_i = []
        all_local_j = []

        for e in range(self.n_elems):
            if not self.valid[e]:
                continue
            fidx = self.free_dofs_e[e]
            for il in range(6):
                ii = fidx[il]
                if ii < 0:
                    continue
                for jl in range(6):
                    jj = fidx[jl]
                    if jj < 0:
                        continue
                    all_rows.append(ii)
                    all_cols.append(jj)
                    all_elem_idx.append(e)
                    all_local_i.append(il)
                    all_local_j.append(jl)

        self.sp_rows = np.array(all_rows, dtype=np.int32)
        self.sp_cols = np.array(all_cols, dtype=np.int32)
        self.sp_elem_idx = np.array(all_elem_idx, dtype=np.int32)
        self.sp_local_i = np.array(all_local_i, dtype=np.int32)
        self.sp_local_j = np.array(all_local_j, dtype=np.int32)
        self.n_vals = len(all_rows)

        # Precompute the ke values for each entry (without the coeff)
        self.sp_ke_vals = self.ke_flat[self.sp_elem_idx, self.sp_local_i, self.sp_local_j]

        # Precompute stress transformation vectors
        self.T_vecs = np.zeros((self.n_elems, 6))
        self.T_vecs[:, 0] = -self.dc[:, 0]
        self.T_vecs[:, 1] = -self.dc[:, 1]
        self.T_vecs[:, 2] = -self.dc[:, 2]
        self.T_vecs[:, 3] = self.dc[:, 0]
        self.T_vecs[:, 4] = self.dc[:, 1]
        self.T_vecs[:, 5] = self.dc[:, 2]

        # E/L factor
        self.EoverL_factor = np.zeros(self.n_elems)
        self.EoverL_factor[self.valid] = 1.0 / self.lengths[self.valid]

        # Precompute E*A/L coefficients mapping: coeff[e] = E * areas[e] / lengths[e]
        # sp_coeff_factor[k] = 1.0 / lengths[elem_idx[k]]  (E and areas applied at solve time)
        self.sp_len_inv = 1.0 / np.maximum(self.lengths[self.sp_elem_idx], 1e-30)

        # Build a template CSC matrix for reuse
        temp_vals = np.ones(len(self.sp_rows))
        self._K_template = sparse.coo_matrix(
            (temp_vals, (self.sp_rows, self.sp_cols)),
            shape=(self.n_free, self.n_free)
        ).tocsc()
        # Store the mapping from COO to CSC for fast value updates
        self._csc_shape = self._K_template.shape
        self._csc_indices = self._K_template.indices.copy()
        self._csc_indptr = self._K_template.indptr.copy()

        # Build COO->CSC value mapping
        # We need to know how COO values map to CSC data array
        # Rebuild with actual indices to get the mapping
        self._use_cholmod = False
        try:
            from sksparse.cholmod import analyze as cholmod_analyze
            # Do a test factorization to cache symbolic
            test_K = self._K_template.copy()
            test_K.data[:] = 1.0
            # Make it positive definite for test
            diag_add = sparse.eye(self.n_free, format='csc') * 1e6
            test_K = test_K + diag_add
            self._chol_factor = cholmod_analyze(test_K)
            self._use_cholmod = True
            print("    Using CHOLMOD for faster solves")
        except (ImportError, Exception):
            self._use_cholmod = False
            self._symbolic_done = False

        # Pre-stack force vectors for batch solve
        self._force_matrix = None

    def solve(self, areas, E, force_vecs):
        """Solve FEM for multiple load cases using vectorized assembly + LU factorization."""
        from scipy.sparse.linalg import splu

        # Vectorized assembly: vals[k] = E * areas[elem_idx[k]] / lengths[elem_idx[k]] * ke_vals[k]
        coeffs = E * areas[self.sp_elem_idx] * self.sp_len_inv
        vals = coeffs * self.sp_ke_vals

        K = sparse.coo_matrix((vals, (self.sp_rows, self.sp_cols)),
                               shape=(self.n_free, self.n_free)).tocsc()

        n_lc = len(force_vecs)

        if self._use_cholmod:
            try:
                self._chol_factor.cholesky_inplace(K)
                all_disp = np.zeros((n_lc, self.n_dofs))
                all_stresses = np.zeros((n_lc, self.n_elems))
                EoverL = E * self.EoverL_factor
                for i, fvec in enumerate(force_vecs):
                    F = fvec[self.free]
                    u_red = self._chol_factor.solve_A(F)
                    all_disp[i, self.free] = u_red
                    u_elem = all_disp[i][self.dofs_e]
                    all_stresses[i] = EoverL * np.sum(self.T_vecs * u_elem, axis=1)
                return all_disp, all_stresses
            except:
                pass  # Fall through to LU

        try:
            lu = splu(K)
        except:
            return None, None

        all_disp = np.zeros((n_lc, self.n_dofs))
        all_stresses = np.zeros((n_lc, self.n_elems))
        EoverL = E * self.EoverL_factor

        for i, fvec in enumerate(force_vecs):
            F = fvec[self.free]
            try:
                u_red = lu.solve(F)
            except:
                return None, None
            all_disp[i, self.free] = u_red
            u_elem = all_disp[i][self.dofs_e]
            all_stresses[i] = EoverL * np.sum(self.T_vecs * u_elem, axis=1)

        return all_disp, all_stresses


def optimize_discrete_stress_ratio(problem, nodes, elements, section_db, max_eval=200000):
    """
    Advanced optimization: OC-based continuous FSD + smart discretization + extensive local search.
    """
    bounds_cfg = problem["variable_bounds"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    n_elems = len(elements)
    n_lc = len(problem["load_cases"])
    E = problem["material"]["E"]
    supports = problem["supports"]

    sorted_sections = sorted(section_db.items(), key=lambda x: x[1])
    n_sections = len(sorted_sections)
    sid_list = [s[0] for s in sorted_sections]
    area_list = np.array([s[1] for s in sorted_sections])
    sid_to_idx = {s[0]: i for i, s in enumerate(sorted_sections)}

    min_area = area_list[0]
    max_area = area_list[-1]

    elem_lengths = np.linalg.norm(nodes[elements[:, 1]] - nodes[elements[:, 0]], axis=1)
    rho = problem["material"]["rho"]
    elem_vol_factor = rho * elem_lengths  # weight = sum(elem_vol_factor * area)

    supported_nodes = {s["node"] for s in supports}
    unsupported_nodes = [i for i in range(len(nodes)) if i not in supported_nodes]
    num_unsupported = len(unsupported_nodes)

    solver = FastFEMSolver(nodes, elements, supports)

    force_vecs = []
    for lc in problem["load_cases"]:
        fvec = apply_loads(lc, unsupported_nodes, num_unsupported)
        force_vecs.append(fvec)

    def calc_weight_idx(idx_arr):
        return np.sum(elem_vol_factor * area_list[idx_arr])

    def calc_weight_areas(areas):
        return np.sum(elem_vol_factor * areas)

    def idx_to_sids(idxs):
        return np.array([sid_list[i] for i in idxs])

    def check_feas(ms, md):
        if ms is None:
            return False
        return (np.max(ms) <= sigma_limit + 1e-6) and (md <= disp_limit + 1e-6)

    def run_analysis(areas):
        """Run all load cases, return (max_elem_stress, max_disp, all_stresses)."""
        all_disp, all_stresses = solver.solve(areas, E, force_vecs)
        if all_disp is None:
            return None, None, None
        max_elem_stress = np.max(np.abs(all_stresses), axis=0)
        max_disp = np.max(np.abs(all_disp))
        return max_elem_stress, max_disp, all_stresses

    random.seed(42)
    np.random.seed(42)

    num_eval = 0
    best_weight = float('inf')
    best_idx = np.full(n_elems, n_sections - 1, dtype=int)
    best_feasible = idx_to_sids(best_idx)

    # ========================================================================
    # Identify symmetry groups for the tower structure
    # ========================================================================
    def identify_symmetry_groups(nodes, elements):
        """Group elements by structural symmetry (4-fold Z-axis rotation)."""
        n_e = len(elements)
        tp = problem["tower_parameters"]
        num_levels = tp["num_levels"]
        node_level = np.zeros(len(nodes), dtype=int)
        node_corner = np.zeros(len(nodes), dtype=int)
        for i in range(num_levels):
            for j in range(4):
                node_level[4*i + j] = i
                node_corner[4*i + j] = j
        elem_keys = {}
        groups = {}
        group_id = 0
        elem_to_group = np.zeros(n_e, dtype=int)
        for e in range(n_e):
            ni, nj = elements[e]
            li, ci = node_level[ni], node_corner[ni]
            lj, cj = node_level[nj], node_corner[nj]
            if li > lj:
                li, lj = lj, li
                ci, cj = cj, ci
            corner_diff = (cj - ci) % 4
            corner_diff_rev = (ci - cj) % 4
            canonical_diff = min(corner_diff, corner_diff_rev)
            key = (li, lj, canonical_diff)
            if key not in elem_keys:
                elem_keys[key] = group_id
                groups[group_id] = []
                group_id += 1
            gid = elem_keys[key]
            groups[gid].append(e)
            elem_to_group[e] = gid
        return elem_to_group, groups, group_id

    elem_to_group, sym_groups, n_groups = identify_symmetry_groups(nodes, elements)
    print(f"  Identified {n_groups} symmetry groups from {n_elems} elements")

    # ========================================================================
    # Phase 1: Continuous FSD with displacement scaling
    # ========================================================================
    print("  Phase 1: Continuous FSD with displacement scaling...")

    all_cont_solutions = []

    for start_frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        if num_eval >= max_eval - 160000:
            break

        areas = np.full(n_elems, max_area * start_frac)
        areas = np.clip(areas, min_area, max_area)

        best_fsd_areas = areas.copy()
        best_fsd_weight = float('inf')
        best_fsd_feasible = False

        for iteration in range(2000):
            if num_eval >= max_eval - 160000:
                break

            ms, md, all_stresses = run_analysis(areas)
            num_eval += 1

            if ms is None:
                areas = np.clip(areas * 2.0, min_area, max_area)
                continue

            stress_ratios = ms / sigma_limit
            disp_ratio = md / disp_limit
            max_sr = np.max(stress_ratios)

            is_feas = (max_sr <= 1.0 + 1e-6) and (disp_ratio <= 1.0 + 1e-6)
            w = calc_weight_areas(areas)

            if is_feas and w < best_fsd_weight:
                best_fsd_weight = w
                best_fsd_areas = areas.copy()
                best_fsd_feasible = True

            # Stress-ratio based resize
            resize = stress_ratios.copy()

            # If displacement is active, scale all areas up uniformly
            if disp_ratio > 0.95:
                # Scale up to satisfy displacement constraint
                disp_scale = max(1.0, (disp_ratio / 0.95) ** 0.5)
                resize = np.maximum(resize, disp_scale * np.ones(n_elems))
            elif disp_ratio > 0.8:
                # Mild scaling to prevent displacement violation
                disp_scale = (disp_ratio / 0.95) ** 0.3
                resize = np.maximum(resize, disp_scale * np.ones(n_elems))

            # Also use compliance sensitivity for displacement-aware resizing
            if disp_ratio > 0.7:
                comp_sens = np.zeros(n_elems)
                for lc_idx in range(n_lc):
                    comp_sens += all_stresses[lc_idx]**2 * elem_lengths / E
                comp_norm = comp_sens / (np.max(comp_sens) + 1e-30)
                # Elements with high compliance sensitivity get extra area
                if disp_ratio > 1.0:
                    extra = (disp_ratio - 1.0) * comp_norm * 2.0
                    resize = np.maximum(resize, 1.0 + extra)

            target_areas = areas * resize

            # Damping
            if iteration < 5:
                eta = 0.1
            elif iteration < 20:
                eta = 0.3
            elif iteration < 100:
                eta = 0.5
            else:
                eta = 0.7

            new_areas = areas * (1.0 - eta) + target_areas * eta
            new_areas = np.clip(new_areas, min_area, max_area)

            rel_change = np.max(np.abs(new_areas - areas) / (areas + 1e-10))
            areas = new_areas

            if rel_change < 1e-7 and iteration > 30:
                break

        w = calc_weight_areas(areas)
        all_cont_solutions.append((best_fsd_weight if best_fsd_feasible else float('inf'), best_fsd_areas.copy()))
        all_cont_solutions.append((w, areas.copy()))

    all_cont_solutions.sort(key=lambda x: x[0])
    print(f"    {len(all_cont_solutions)} continuous solutions, best={all_cont_solutions[0][0]:.0f}, evals={num_eval}")

    # ========================================================================
    # Phase 2: Smart discretization
    # ========================================================================
    print("  Phase 2: Discretization...")

    for sol_idx, (cont_w, cont_areas) in enumerate(all_cont_solutions[:10]):
        if num_eval >= max_eval - 155000:
            break

        ceil_idx = np.clip(np.searchsorted(area_list, cont_areas), 0, n_sections - 1)
        floor_idx = np.clip(np.searchsorted(area_list, cont_areas) - 1, 0, n_sections - 1)
        nearest_idx = np.array([np.argmin(np.abs(area_list - a)) for a in cont_areas])

        strategies = [
            ceil_idx.copy(),
            nearest_idx.copy(),
            np.clip(ceil_idx + 1, 0, n_sections - 1),
            np.clip(ceil_idx + 2, 0, n_sections - 1),
            np.clip(ceil_idx + 3, 0, n_sections - 1),
            np.clip(ceil_idx + 4, 0, n_sections - 1),
            floor_idx.copy(),
        ]

        # Probabilistic rounding
        for _ in range(3):
            prob_idx = np.zeros(n_elems, dtype=int)
            for e in range(n_elems):
                ci = ceil_idx[e]
                fi = floor_idx[e]
                if ci == fi:
                    prob_idx[e] = ci
                else:
                    a_cont = cont_areas[e]
                    p_ceil = (a_cont - area_list[fi]) / (area_list[ci] - area_list[fi] + 1e-30)
                    p_ceil = min(1.0, p_ceil + 0.3)
                    prob_idx[e] = ci if random.random() < p_ceil else fi
            strategies.append(prob_idx)

        for trial_idx in strategies:
            if num_eval >= max_eval - 155000:
                break
            trial_areas = area_list[trial_idx]
            ms, md, _ = run_analysis(trial_areas)
            num_eval += 1
            if ms is not None and check_feas(ms, md):
                w = calc_weight_idx(trial_idx)
                if w < best_weight:
                    best_weight = w
                    best_idx = trial_idx.copy()
                    best_feasible = idx_to_sids(best_idx)
                    print(f"    Sol {sol_idx}: weight={w:.1f}")

    if best_weight == float('inf'):
        best_idx = np.full(n_elems, n_sections - 1, dtype=int)
        best_weight = calc_weight_idx(best_idx)
        best_feasible = idx_to_sids(best_idx)
        print(f"    Fallback to max: weight={best_weight:.1f}")

    print(f"    Best discrete: weight={best_weight:.1f}, evals={num_eval}")

    # ========================================================================
    # Phase 3: Batch reduction
    # ========================================================================
    print("  Phase 3: Batch reduction...")
    cur_idx = best_idx.copy()

    for batch_pass in range(500):
        if num_eval >= max_eval - 145000:
            break

        cur_areas = area_list[cur_idx]
        ms, md, _ = run_analysis(cur_areas)
        num_eval += 1

        if ms is None or not check_feas(ms, md):
            cur_idx = best_idx.copy()
            continue

        w = calc_weight_idx(cur_idx)
        if w < best_weight:
            best_weight = w
            best_idx = cur_idx.copy()
            best_feasible = idx_to_sids(best_idx)

        stress_ratios = ms / sigma_limit
        disp_ratio = md / disp_limit
        min_margin = min(1.0 - np.max(stress_ratios), 1.0 - disp_ratio)

        if min_margin < 0.001:
            break

        trial_idx = cur_idx.copy()
        changed = False
        for e in range(n_elems):
            if trial_idx[e] == 0:
                continue
            sr = stress_ratios[e]
            if sr < 0.05:
                steps = min(trial_idx[e], 15)
            elif sr < 0.1:
                steps = min(trial_idx[e], 10)
            elif sr < 0.2:
                steps = min(trial_idx[e], 7)
            elif sr < 0.3:
                steps = min(trial_idx[e], 5)
            elif sr < 0.5:
                steps = min(trial_idx[e], 3)
            elif sr < 0.7:
                steps = min(trial_idx[e], 2)
            elif sr < 0.85:
                steps = min(trial_idx[e], 1)
            else:
                steps = 0
            if min_margin < 0.05:
                steps = min(steps, max(1, steps // 2))
            if steps > 0:
                trial_idx[e] -= steps
                changed = True

        if not changed:
            break

        ms2, md2, _ = run_analysis(area_list[trial_idx])
        num_eval += 1
        if ms2 is not None and check_feas(ms2, md2):
            w2 = calc_weight_idx(trial_idx)
            if w2 < best_weight:
                best_weight = w2
                best_idx = trial_idx.copy()
                best_feasible = idx_to_sids(best_idx)
                cur_idx = trial_idx.copy()
                print(f"    Batch {batch_pass}: weight={w2:.1f}")
                continue

        for frac in [0.5, 0.25, 0.75, 0.125, 0.0625]:
            if num_eval >= max_eval - 145000:
                break
            trial_idx_f = cur_idx.copy()
            for e in range(n_elems):
                diff = cur_idx[e] - trial_idx[e]
                if diff > 0:
                    trial_idx_f[e] = cur_idx[e] - max(1, int(diff * frac))
            if np.array_equal(trial_idx_f, cur_idx):
                continue
            ms2, md2, _ = run_analysis(area_list[trial_idx_f])
            num_eval += 1
            if ms2 is not None and check_feas(ms2, md2):
                w2 = calc_weight_idx(trial_idx_f)
                if w2 < best_weight:
                    best_weight = w2
                    best_idx = trial_idx_f.copy()
                    best_feasible = idx_to_sids(best_idx)
                    cur_idx = trial_idx_f.copy()
                    print(f"    Batch-frac {batch_pass}: weight={w2:.1f}")
                    break
        else:
            trial_idx3 = cur_idx.copy()
            for e in range(n_elems):
                if trial_idx3[e] > 0 and stress_ratios[e] < 0.5:
                    trial_idx3[e] -= 1
            if not np.array_equal(trial_idx3, cur_idx):
                ms2, md2, _ = run_analysis(area_list[trial_idx3])
                num_eval += 1
                if ms2 is not None and check_feas(ms2, md2):
                    w2 = calc_weight_idx(trial_idx3)
                    if w2 < best_weight:
                        best_weight = w2
                        best_idx = trial_idx3.copy()
                        best_feasible = idx_to_sids(best_idx)
                        cur_idx = trial_idx3.copy()
                        continue
            break

    print(f"    After batch: weight={best_weight:.1f}, evals={num_eval}")

    # ========================================================================
    # Phase 4: Element-by-element binary search
    # ========================================================================
    print("  Phase 4: Element-wise binary search...")

    for binary_round in range(15):
        if num_eval >= max_eval - 120000:
            break

        cur_idx = best_idx.copy()
        cur_areas = area_list[cur_idx].copy()
        round_improved = False

        savings = elem_vol_factor * (area_list[cur_idx] - area_list[0])
        savings[cur_idx == 0] = 0
        elem_order = np.argsort(-savings)

        for e in elem_order:
            if num_eval >= max_eval - 110000:
                break
            old_idx_e = cur_idx[e]
            if old_idx_e == 0:
                continue

            lo, hi = 0, old_idx_e - 1
            best_new_idx = old_idx_e
            old_area = cur_areas[e]

            while lo <= hi:
                if num_eval >= max_eval - 110000:
                    break
                mid = (lo + hi) // 2
                cur_areas[e] = area_list[mid]
                ms, md, _ = run_analysis(cur_areas)
                num_eval += 1
                if ms is not None and check_feas(ms, md):
                    best_new_idx = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            if best_new_idx < old_idx_e:
                cur_idx[e] = best_new_idx
                cur_areas[e] = area_list[best_new_idx]
                w = calc_weight_idx(cur_idx)
                if w < best_weight:
                    best_weight = w
                    best_idx = cur_idx.copy()
                    best_feasible = idx_to_sids(best_idx)
                    round_improved = True
                else:
                    cur_idx[e] = old_idx_e
                    cur_areas[e] = old_area
            else:
                cur_areas[e] = old_area

        if round_improved:
            print(f"    Binary round {binary_round}: weight={best_weight:.1f}, evals={num_eval}")
        else:
            break

    # ========================================================================
    # Phase 5: Single-step descent passes
    # ========================================================================
    print("  Phase 5: Single-step refinement...")
    cur_idx = best_idx.copy()
    cur_areas = area_list[cur_idx].copy()

    for pass_num in range(100):
        if num_eval >= max_eval - 100000:
            break

        improved = False
        step_savings = np.zeros(n_elems)
        mask = cur_idx > 0
        step_savings[mask] = elem_vol_factor[mask] * (area_list[cur_idx[mask]] - area_list[cur_idx[mask] - 1])
        elem_order = np.argsort(-step_savings)

        for e in elem_order:
            if num_eval >= max_eval - 100000:
                break
            if cur_idx[e] == 0:
                continue

            old_area = cur_areas[e]
            old_idx_e = cur_idx[e]
            cur_areas[e] = area_list[old_idx_e - 1]

            ms, md, _ = run_analysis(cur_areas)
            num_eval += 1

            if ms is not None and check_feas(ms, md):
                cur_idx[e] = old_idx_e - 1
                w = calc_weight_idx(cur_idx)
                if w < best_weight:
                    best_weight = w
                    best_idx = cur_idx.copy()
                    best_feasible = idx_to_sids(best_idx)
                    improved = True
                    continue
                else:
                    cur_idx[e] = old_idx_e
                    cur_areas[e] = old_area
            else:
                cur_areas[e] = old_area

        if not improved:
            break
        cur_idx = best_idx.copy()
        cur_areas = area_list[cur_idx].copy()
        print(f"    Refine pass {pass_num}: weight={best_weight:.1f}, evals={num_eval}")

    # ========================================================================
    # Phase 6: Simulated annealing + greedy local search (use remaining budget)
    # ========================================================================
    remaining = max_eval - num_eval
    if remaining > 1000:
        print(f"  Phase 6: SA + perturbation + greedy ({remaining} evals remaining)...")

        cur_idx_sa = best_idx.copy()
        cur_weight_sa = best_weight

        # SA parameters
        T_start = best_weight * 0.02
        T_end = best_weight * 0.0001

        n_restarts = 0
        sa_accepts = 0
        sa_total = max_eval - num_eval - 500

        while num_eval + 5 <= max_eval:
            n_restarts += 1

            progress = min(1.0, n_restarts / max(1, sa_total))
            T = T_start * (T_end / T_start) ** progress

            trial_idx = cur_idx_sa.copy()

            strategy = n_restarts % 14

            if strategy == 0:
                n_perturb = random.randint(1, 3)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    delta = random.randint(-3, 1)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 1:
                n_perturb = random.randint(3, 15)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    delta = random.randint(-4, 1)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 2:
                n_perturb = random.randint(n_elems // 4, n_elems // 2)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    delta = random.randint(-1, 0)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 3:
                weight_contrib = elem_vol_factor * area_list[trial_idx]
                top_n = random.randint(3, 20)
                top_elems = np.argsort(-weight_contrib)[:top_n]
                for e in top_elems:
                    delta = random.randint(-4, 0)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 4:
                e = random.randint(0, n_elems - 1)
                delta = random.randint(-6, -1)
                trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 5:
                e1 = random.randint(0, n_elems - 1)
                e2 = random.randint(0, n_elems - 1)
                if e1 != e2:
                    trial_idx[e1] = max(0, trial_idx[e1] - random.randint(1, 3))
                    trial_idx[e2] = min(n_sections - 1, trial_idx[e2] + random.randint(1, 2))
            elif strategy == 6:
                n_perturb = random.randint(2, 8)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    trial_idx[e] = max(0, trial_idx[e] - 2)
            elif strategy == 7:
                n_perturb = random.randint(n_elems // 2, n_elems)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    delta = random.randint(-2, 0)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 8:
                gid = random.randint(0, n_groups - 1)
                delta = random.randint(-3, 1)
                for e in sym_groups[gid]:
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 9:
                n_grps = random.randint(2, min(10, n_groups))
                grp_ids = random.sample(range(n_groups), n_grps)
                for gid in grp_ids:
                    delta = random.randint(-3, 1)
                    for e in sym_groups[gid]:
                        trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 10:
                g1 = random.randint(0, n_groups - 1)
                g2 = random.randint(0, n_groups - 1)
                if g1 != g2:
                    for e in sym_groups[g1]:
                        trial_idx[e] = max(0, trial_idx[e] - random.randint(1, 2))
                    for e in sym_groups[g2]:
                        trial_idx[e] = min(n_sections - 1, trial_idx[e] + random.randint(0, 1))
            elif strategy == 11:
                # Reduce from best, not current SA state
                trial_idx = best_idx.copy()
                n_perturb = random.randint(1, 5)
                perturb_elems = random.sample(range(n_elems), n_perturb)
                for e in perturb_elems:
                    delta = random.randint(-3, 1)
                    trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))
            elif strategy == 12:
                # Increase some, decrease others (rebalance)
                n_up = random.randint(1, 5)
                n_down = random.randint(3, 15)
                up_elems = random.sample(range(n_elems), n_up)
                down_elems = random.sample(range(n_elems), n_down)
                for e in up_elems:
                    trial_idx[e] = min(n_sections - 1, trial_idx[e] + random.randint(1, 3))
                for e in down_elems:
                    trial_idx[e] = max(0, trial_idx[e] - random.randint(1, 2))
            else:
                # Symmetry-aware from best
                trial_idx = best_idx.copy()
                n_grps = random.randint(1, min(5, n_groups))
                grp_ids = random.sample(range(n_groups), n_grps)
                for gid in grp_ids:
                    delta = random.randint(-2, 1)
                    for e in sym_groups[gid]:
                        trial_idx[e] = max(0, min(n_sections - 1, trial_idx[e] + delta))

            tw = calc_weight_idx(trial_idx)

            if tw < cur_weight_sa:
                pass
            elif tw < best_weight:
                pass
            elif T > 0 and tw < best_weight * 1.15:
                delta_w = tw - cur_weight_sa
                if delta_w > 0:
                    accept_prob = np.exp(-delta_w / T)
                    if random.random() > accept_prob:
                        continue
            else:
                continue

            trial_areas = area_list[trial_idx]
            ms, md, _ = run_analysis(trial_areas)
            num_eval += 1

            if ms is None or not check_feas(ms, md):
                continue

            sa_accepts += 1
            cur_idx_sa = trial_idx.copy()
            cur_weight_sa = tw

            if tw < best_weight:
                best_idx = trial_idx.copy()
                best_weight = tw
                best_feasible = idx_to_sids(best_idx)

                if n_restarts % 10000 == 0:
                    print(f"    Restart {n_restarts}: weight={best_weight:.1f}, evals={num_eval}")

                # Quick greedy descent
                cur_idx = best_idx.copy()
                cur_areas = area_list[cur_idx].copy()

                for sweep in range(10):
                    if num_eval + n_elems > max_eval:
                        break
                    sub_improved = False
                    step_savings = np.zeros(n_elems)
                    mask2 = cur_idx > 0
                    step_savings[mask2] = elem_vol_factor[mask2] * (area_list[cur_idx[mask2]] - area_list[cur_idx[mask2] - 1])
                    eo = np.argsort(-step_savings)

                    for e in eo:
                        if num_eval + 3 > max_eval:
                            break
                        if cur_idx[e] == 0:
                            continue
                        old_area_e = cur_areas[e]
                        old_idx_e = cur_idx[e]
                        cur_areas[e] = area_list[old_idx_e - 1]
                        ms2, md2, _ = run_analysis(cur_areas)
                        num_eval += 1
                        if ms2 is not None and check_feas(ms2, md2):
                            cur_idx[e] = old_idx_e - 1
                            cw = calc_weight_idx(cur_idx)
                            if cw < best_weight:
                                best_idx = cur_idx.copy()
                                best_weight = cw
                                best_feasible = idx_to_sids(best_idx)
                                sub_improved = True
                            else:
                                cur_idx[e] = old_idx_e
                                cur_areas[e] = old_area_e
                        else:
                            cur_areas[e] = old_area_e
                    if not sub_improved:
                        break

                cur_idx_sa = best_idx.copy()
                cur_weight_sa = best_weight

        print(f"    Phase 6 done: weight={best_weight:.1f}, evals={num_eval}, restarts={n_restarts}, accepts={sa_accepts}")

    print(f"  Final: weight={best_weight:.1f}, evals={num_eval}")
    return best_feasible, num_eval


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
    print("Optimizing using advanced FSD + binary search refinement...")
    x_best, num_eval = optimize_discrete_stress_ratio(problem, nodes, elements, section_db, max_eval)
    w = compute_weight(x_best, problem, nodes, elements, section_db)

    result = analyze_design(x_best, problem, nodes, elements, section_db)
    feasible = result["feasible"] if result else False

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "iscso_2023",
        "solution_vector": x_best.tolist(),
        "algorithm": "FastFEM_FSD_Sensitivity_BinarySearch",
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