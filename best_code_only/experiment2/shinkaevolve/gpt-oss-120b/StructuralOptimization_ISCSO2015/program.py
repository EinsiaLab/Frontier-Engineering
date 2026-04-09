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

def analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements):
    """
    Detailed analysis returning individual bar stresses for each load case.
    """
    E = problem["material"]["E"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    supports = problem["supports"]
    num_bars = len(areas)

    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        node_idx = nid - 1
        nodes[node_idx, 1] = shape_vars[idx]

    max_stress = 0.0
    max_disp = 0.0
    bar_stresses = np.zeros(num_bars)
    bar_forces = np.zeros(num_bars)
    lengths = np.zeros(num_bars)

    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            idx = nid - 1
            fvec[2 * idx] += load["fx"]
            fvec[2 * idx + 1] += load["fy"]

        disp, stresses, lens = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        if disp is None:
            return None

        max_stress = max(max_stress, np.max(np.abs(stresses)))
        max_disp = max(max_disp, np.max(np.abs(disp)))

        for b in range(num_bars):
            stress_ratio = abs(stresses[b]) / sigma_limit
            bar_stresses[b] = max(bar_stresses[b], stress_ratio)
            bar_forces[b] = max(bar_forces[b], abs(stresses[b] * areas[b]))
            lengths[b] = lens[b]

    feasible = (max_stress <= sigma_limit + 1e-6) and (max_disp <= disp_limit + 1e-6)
    return {
        "feasible": feasible,
        "max_stress": max_stress,
        "max_disp": max_disp,
        "bar_stress_ratios": bar_stresses,
        "bar_forces": bar_forces,
        "lengths": lengths
    }


def compute_direct_areas(areas, shape_vars, problem, nodes_base, elements, safety_factor=1.005):
    """
    Compute optimal areas directly from force constraints.
    A_new = max(A_min, |F| / sigma_allow * safety_factor)
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]

    result = analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements)
    if result is None:
        return np.clip(areas * 1.3, area_min, area_max)

    bar_forces = result["bar_forces"]
    new_areas = np.abs(bar_forces) / sigma_limit * safety_factor
    new_areas = np.clip(new_areas, area_min, area_max)

    return new_areas


def direct_sizing_optimize(areas, shape_vars, problem, nodes_base, elements, max_iter=25):
    """
    Direct stress-based sizing with rapid convergence.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]

    best_areas = areas.copy()
    best_weight = float('inf')

    # Phase 1: Achieve feasibility with force-based sizing
    for iteration in range(max_iter // 2):
        result = analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements)
        if result is None:
            areas = np.clip(areas * 1.25, area_min, area_max)
            continue

        if result["feasible"]:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if w < best_weight:
                best_weight = w
                best_areas = areas.copy()
            break

        bar_forces = result["bar_forces"]
        max_disp_ratio = result["max_disp"] / disp_limit
        safety = 1.01 + 0.02 * max(0, max_disp_ratio - 0.9)
        new_areas = np.abs(bar_forces) / sigma_limit * safety

        margin = 1.0 - max(result["bar_stress_ratios"]) if len(result["bar_stress_ratios"]) > 0 else 0
        move_limit = 0.4 if margin < 0 else min(0.6, 0.3 - margin * 0.5)

        for b in range(num_bars):
            change = new_areas[b] / areas[b]
            change = np.clip(change, 1.0 - move_limit, 1.0 + move_limit)
            new_areas[b] = areas[b] * change

        areas = np.clip(new_areas, area_min, area_max)

    # Phase 2: Refine to optimality
    for iteration in range(max_iter // 2, max_iter):
        result = analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements)
        if result is None:
            continue

        if result["feasible"]:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if w < best_weight:
                best_weight = w
                best_areas = areas.copy()

        bar_ratios = result["bar_stress_ratios"]
        max_stress_ratio = max(bar_ratios) if len(bar_ratios) > 0 else 0
        max_disp_ratio = result["max_disp"] / disp_limit

        bar_forces = result["bar_forces"]
        target_ratio = 0.995

        new_areas = areas.copy()
        for b in range(num_bars):
            if bar_forces[b] > 1e-6:
                optimal = bar_forces[b] / sigma_limit / target_ratio
                new_areas[b] = optimal
            else:
                new_areas[b] = area_min

        margin = min(1.0 - max_stress_ratio, 1.0 - max_disp_ratio)
        if margin > 0.05:
            move_limit = 0.25
        elif margin > 0.02:
            move_limit = 0.15
        else:
            move_limit = 0.08

        for b in range(num_bars):
            change = new_areas[b] / areas[b]
            change = np.clip(change, 1.0 - move_limit, 1.0 + move_limit)
            new_areas[b] = areas[b] * change

        areas = np.clip(new_areas, area_min, area_max)

        if result["max_stress"] <= sigma_limit * 0.99 and result["max_disp"] > disp_limit * 0.99:
            disp_ratio = result["max_disp"] / disp_limit
            areas = np.clip(areas * (disp_ratio ** 0.55), area_min, area_max)

    return best_areas if best_weight < float('inf') else areas


def sensitivity_shape_optimize(areas, shape_vars, problem, nodes_base, elements, max_iter=12):
    """
    Shape optimization using sensitivity-inspired updates.
    """
    bounds_cfg = problem["variable_bounds"]
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]
    num_shape = len(shape_vars)

    best_areas = areas.copy()
    best_shapes = shape_vars.copy()
    best_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)

    shape_step = (y_max - y_min) * 0.05

    for iteration in range(max_iter):
        improved = False

        result = analyze_design_detailed(best_areas, best_shapes, problem, nodes_base, elements)
        if result is None:
            continue

        for direction in [1, -1]:
            new_shapes = best_shapes.copy()
            for i in range(num_shape):
                x_norm = 1.0 - abs(i - (num_shape - 1) / 2) / ((num_shape - 1) / 2)
                bias = 0.5 + 0.5 * x_norm
                new_shapes[i] = np.clip(best_shapes[i] + direction * shape_step * bias, y_min, y_max)

            test_areas = direct_sizing_optimize(best_areas, new_shapes, problem, nodes_base, elements, max_iter=5)

            result = analyze_design(test_areas, new_shapes, problem, nodes_base, elements)
            if result and result["feasible"]:
                w = compute_weight(np.concatenate([test_areas, new_shapes]), problem, nodes_base, elements)
                if w < best_weight - 0.1:
                    best_weight = w
                    best_areas = test_areas.copy()
                    best_shapes = new_shapes.copy()
                    improved = True
                    break

        if not improved:
            for i in range(num_shape):
                for direction in [1, -1]:
                    new_shapes = best_shapes.copy()
                    new_shapes[i] = np.clip(best_shapes[i] + direction * shape_step * 0.5, y_min, y_max)

                    if new_shapes[i] == best_shapes[i]:
                        continue

                    test_areas = direct_sizing_optimize(best_areas, new_shapes, problem, nodes_base, elements, max_iter=4)

                    result = analyze_design(test_areas, new_shapes, problem, nodes_base, elements)
                    if result and result["feasible"]:
                        w = compute_weight(np.concatenate([test_areas, new_shapes]), problem, nodes_base, elements)
                        if w < best_weight - 0.05:
                            best_weight = w
                            best_areas = test_areas.copy()
                            best_shapes = new_shapes.copy()
                            improved = True
                            break

                if improved:
                    break

        if not improved:
            shape_step *= 0.6
            if shape_step < (y_max - y_min) * 0.002:
                break

    return best_areas, best_shapes


def aggressive_reduction(areas, shape_vars, problem, nodes_base, elements, max_iter=30):
    """
    Aggressive area reduction combining batch reduction and golden-section search.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    disp_limit = problem["constraints"]["displacement_limit"]
    sigma_limit = problem["constraints"]["stress_limit"]
    num_bars = problem["num_bars"]

    phi = (1 + np.sqrt(5)) / 2

    best_areas = areas.copy()
    best_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
    no_improve_count = 0

    result = analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements)
    if result is None or not result["feasible"]:
        return best_areas

    bar_ratios_init = result["bar_stress_ratios"]
    sorted_bars = np.argsort(bar_ratios_init)

    # Phase 1: Batch reduction of very understressed bars
    very_low_stress = [b for b in range(num_bars) if bar_ratios_init[b] < 0.25]
    if len(very_low_stress) > 0:
        test_areas = areas.copy()
        for b in very_low_stress:
            reduction = 0.55 + 0.35 * bar_ratios_init[b]
            test_areas[b] = max(area_min, areas[b] * reduction)
        test_result = analyze_design(test_areas, shape_vars, problem, nodes_base, elements)
        if test_result and test_result["feasible"]:
            areas = test_areas

    # Phase 1b: Golden-section search with 4 iterations per bar
    for b in sorted_bars:
        if bar_ratios_init[b] > 0.7:
            continue
        lo = area_min
        hi = areas[b]

        for _ in range(4):
            if hi - lo < area_min * 0.015:
                break
            mid1 = lo + (hi - lo) / (phi + 1)
            mid2 = hi - (hi - lo) / (phi + 1)

            test1 = areas.copy()
            test1[b] = mid1
            test2 = areas.copy()
            test2[b] = mid2

            r1 = analyze_design(test1, shape_vars, problem, nodes_base, elements)
            r2 = analyze_design(test2, shape_vars, problem, nodes_base, elements)

            if r1 and r1["feasible"]:
                if r2 and r2["feasible"]:
                    if mid1 < mid2:
                        hi = mid2
                    else:
                        lo = mid1
                else:
                    hi = mid2
                areas[b] = mid1
            elif r2 and r2["feasible"]:
                lo = mid1
                areas[b] = mid2
            else:
                lo = mid1

    result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
    if result and result["feasible"]:
        w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
        if w < best_weight:
            best_weight = w
            best_areas = areas.copy()

    # Phase 2: Coordinated group reductions
    for iteration in range(max_iter):
        result = analyze_design_detailed(areas, shape_vars, problem, nodes_base, elements)
        if result is None or not result["feasible"]:
            break

        w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
        if w < best_weight:
            best_weight = w
            best_areas = areas.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break

        bar_ratios = result["bar_stress_ratios"]
        max_ratio = max(bar_ratios)
        max_disp_ratio = result["max_disp"] / disp_limit
        margin = min(1.0 - max_ratio, 1.0 - max_disp_ratio)

        if margin > 0.10:
            new_areas = areas.copy()
            for b in range(num_bars):
                if bar_ratios[b] < 0.06:
                    new_areas[b] = max(area_min, areas[b] * 0.40)
                elif bar_ratios[b] < 0.10:
                    new_areas[b] = max(area_min, areas[b] * 0.52)
                elif bar_ratios[b] < 0.18:
                    new_areas[b] = max(area_min, areas[b] * 0.65)
                elif bar_ratios[b] < 0.28:
                    new_areas[b] = max(area_min, areas[b] * 0.76)
                elif bar_ratios[b] < 0.42:
                    new_areas[b] = max(area_min, areas[b] * 0.85)
                elif bar_ratios[b] < 0.58:
                    new_areas[b] = max(area_min, areas[b] * 0.92)
                elif bar_ratios[b] < 0.72:
                    new_areas[b] = max(area_min, areas[b] * 0.96)
                elif bar_ratios[b] < 0.86:
                    new_areas[b] = max(area_min, areas[b] * 0.99)
                elif bar_ratios[b] < 0.95:
                    new_areas[b] = max(area_min, areas[b] * 0.997)

            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                new_areas = np.clip(areas * 0.92, area_min, area_max)
                test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
                if test_result and test_result["feasible"]:
                    areas = new_areas

        elif margin > 0.06:
            new_areas = areas.copy()
            for b in range(num_bars):
                if bar_ratios[b] < 0.08:
                    new_areas[b] = max(area_min, areas[b] * 0.48)
                elif bar_ratios[b] < 0.15:
                    new_areas[b] = max(area_min, areas[b] * 0.60)
                elif bar_ratios[b] < 0.25:
                    new_areas[b] = max(area_min, areas[b] * 0.72)
                elif bar_ratios[b] < 0.40:
                    new_areas[b] = max(area_min, areas[b] * 0.82)
                elif bar_ratios[b] < 0.55:
                    new_areas[b] = max(area_min, areas[b] * 0.90)
                elif bar_ratios[b] < 0.70:
                    new_areas[b] = max(area_min, areas[b] * 0.95)
                elif bar_ratios[b] < 0.85:
                    new_areas[b] = max(area_min, areas[b] * 0.98)
                elif bar_ratios[b] < 0.95:
                    new_areas[b] = max(area_min, areas[b] * 0.996)

            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                new_areas = np.clip(areas * 0.93, area_min, area_max)
                test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
                if test_result and test_result["feasible"]:
                    areas = new_areas

        elif margin > 0.04:
            new_areas = areas.copy()
            for b in range(num_bars):
                if bar_ratios[b] < 0.12:
                    new_areas[b] = max(area_min, areas[b] * 0.62)
                elif bar_ratios[b] < 0.20:
                    new_areas[b] = max(area_min, areas[b] * 0.74)
                elif bar_ratios[b] < 0.32:
                    new_areas[b] = max(area_min, areas[b] * 0.84)
                elif bar_ratios[b] < 0.48:
                    new_areas[b] = max(area_min, areas[b] * 0.91)
                elif bar_ratios[b] < 0.62:
                    new_areas[b] = max(area_min, areas[b] * 0.95)
                elif bar_ratios[b] < 0.78:
                    new_areas[b] = max(area_min, areas[b] * 0.98)
                elif bar_ratios[b] < 0.92:
                    new_areas[b] = max(area_min, areas[b] * 0.995)

            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                break

        elif margin > 0.02:
            new_areas = areas.copy()
            for b in range(num_bars):
                if bar_ratios[b] < 0.20:
                    new_areas[b] = max(area_min, areas[b] * 0.78)
                elif bar_ratios[b] < 0.35:
                    new_areas[b] = max(area_min, areas[b] * 0.88)
                elif bar_ratios[b] < 0.50:
                    new_areas[b] = max(area_min, areas[b] * 0.94)
                elif bar_ratios[b] < 0.68:
                    new_areas[b] = max(area_min, areas[b] * 0.97)
                elif bar_ratios[b] < 0.85:
                    new_areas[b] = max(area_min, areas[b] * 0.992)

            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                break

        elif margin > 0.01:
            new_areas = np.clip(areas * 0.985, area_min, area_max)
            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                break

        elif margin > 0.003:
            new_areas = np.clip(areas * 0.994, area_min, area_max)
            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                break

        elif margin > 0.0005:
            new_areas = np.clip(areas * 0.9985, area_min, area_max)
            test_result = analyze_design(new_areas, shape_vars, problem, nodes_base, elements)
            if test_result and test_result["feasible"]:
                areas = new_areas
            else:
                break
        else:
            break

    return best_areas


def parabolic_arch_initialize(problem, nodes_base, curvature=0.08, height_factor=1.0):
    """
    Initialize shape variables as parabolic arch.
    """
    bounds_cfg = problem["variable_bounds"]
    y_max = bounds_cfg["y_max"]
    y_min = bounds_cfg["y_min"]
    num_shape = len(problem["shape_variable_node_ids"])

    shape_vars = np.zeros(num_shape)
    for i in range(num_shape):
        x_norm = (i - (num_shape - 1) / 2) / ((num_shape - 1) / 2)
        arch_factor = 1.0 - curvature * x_norm * x_norm
        shape_vars[i] = np.clip(y_max * height_factor * arch_factor, y_min, y_max)

    return shape_vars


def catenary_arch_initialize(problem, nodes_base, sag_ratio=0.12, height_factor=1.0):
    """
    Initialize shape variables as catenary arch.
    """
    bounds_cfg = problem["variable_bounds"]
    y_max = bounds_cfg["y_max"]
    y_min = bounds_cfg["y_min"]
    num_shape = len(problem["shape_variable_node_ids"])

    a_param = 1.0 / max(sag_ratio, 0.05)

    shape_vars = np.zeros(num_shape)
    for i in range(num_shape):
        x_norm = (i - (num_shape - 1) / 2) / ((num_shape - 1) / 2)
        y_catenary = a_param * (np.cosh(x_norm / a_param) - 1)
        y_max_catenary = a_param * (np.cosh(1.0 / a_param) - 1)
        y_norm = 1.0 - y_catenary / max(y_max_catenary, 0.01)
        shape_vars[i] = np.clip(y_max * height_factor * y_norm, y_min, y_max)

    return shape_vars


def circular_arch_initialize(problem, nodes_base, rise_ratio=0.15, height_factor=1.0):
    """
    Initialize shape variables as circular arch segment.
    """
    bounds_cfg = problem["variable_bounds"]
    y_max = bounds_cfg["y_max"]
    y_min = bounds_cfg["y_min"]
    num_shape = len(problem["shape_variable_node_ids"])

    span_half = 1.0
    rise = rise_ratio * 2 * span_half
    R = (span_half ** 2 + rise ** 2) / (2 * rise) if rise > 0.01 else 1000

    shape_vars = np.zeros(num_shape)
    for i in range(num_shape):
        x_norm = (i - (num_shape - 1) / 2) / ((num_shape - 1) / 2)
        x_pos = x_norm * span_half
        if R > 100:
            y_val = y_max * height_factor
        else:
            y_arch = np.sqrt(max(0, R ** 2 - x_pos ** 2)) - R + rise
            y_val = y_max * height_factor * (0.85 + 0.15 * y_arch / max(rise, 0.01))
        shape_vars[i] = np.clip(y_val, y_min, y_max)

    return shape_vars


def local_search_refinement(areas, shape_vars, problem, nodes_base, elements, max_iter=15):
    """
    Local search around current best solution.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    y_min = bounds_cfg["y_min"]
    y_max = bounds_cfg["y_max"]
    num_bars = problem["num_bars"]
    num_shape = len(shape_vars)

    best_areas = areas.copy()
    best_shapes = shape_vars.copy()
    best_weight = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)

    area_step = 0.03
    shape_step = (y_max - y_min) * 0.015

    for iteration in range(max_iter):
        improved = False

        result = analyze_design_detailed(best_areas, best_shapes, problem, nodes_base, elements)
        if result is None or not result["feasible"]:
            break

        bar_ratios = result["bar_stress_ratios"]
        sorted_bars = np.argsort(bar_ratios)

        for b in sorted_bars[:10]:
            if bar_ratios[b] < 0.80:
                test_areas = best_areas.copy()
                reduction = area_step * (1.0 - bar_ratios[b])
                test_areas[b] = max(area_min, best_areas[b] * (1.0 - reduction))

                test_result = analyze_design(test_areas, best_shapes, problem, nodes_base, elements)
                if test_result and test_result["feasible"]:
                    w = compute_weight(np.concatenate([test_areas, best_shapes]), problem, nodes_base, elements)
                    if w < best_weight - 0.01:
                        best_weight = w
                        best_areas = test_areas.copy()
                        improved = True
                        break

        if not improved:
            for i in range(num_shape):
                for direction in [-1, 1]:
                    test_shapes = best_shapes.copy()
                    test_shapes[i] = np.clip(best_shapes[i] + direction * shape_step, y_min, y_max)

                    if test_shapes[i] == best_shapes[i]:
                        continue

                    test_areas = compute_direct_areas(best_areas, test_shapes, problem, nodes_base, elements, safety_factor=1.01)

                    test_result = analyze_design(test_areas, test_shapes, problem, nodes_base, elements)
                    if test_result and test_result["feasible"]:
                        w = compute_weight(np.concatenate([test_areas, test_shapes]), problem, nodes_base, elements)
                        if w < best_weight - 0.05:
                            best_weight = w
                            best_areas = test_areas.copy()
                            best_shapes = test_shapes.copy()
                            improved = True
                            break

                if improved:
                    break

        if not improved:
            area_step *= 0.7
            shape_step *= 0.7
            if area_step < 0.005:
                break

    return best_areas, best_shapes


def optimize_stress_ratio(problem, nodes_base, elements, max_iter=40):
    """
    Enhanced direct sizing optimization with improved multi-start strategy.
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

    base_shapes = np.array([nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]])

    configs = [
        ("parabolic", 0.06, 1.0, 1.02),
        ("parabolic", 0.08, 1.0, 1.05),
        ("parabolic", 0.05, 1.0, 1.04),
        ("parabolic", 0.10, 1.0, 1.06),
        ("circular", 0.12, 1.0, 1.05),
        ("circular", 0.15, 1.0, 1.08),
        ("circular", 0.10, 1.0, 1.06),
        ("flat", None, 1.0, 1.08),
        ("flat", None, 1.0, 1.12),
        ("parabolic", 0.07, 0.98, 1.10),
        ("parabolic", 0.09, 1.0, 1.12),
        ("base", None, 1.15, 1.15),
    ]

    best_weight = float('inf')
    best_areas = None
    best_shapes = None

    for shape_type, curvature, height_factor, area_factor in configs:
        areas = np.full(num_bars, area_min * area_factor)

        if shape_type == "parabolic":
            shape_vars = parabolic_arch_initialize(problem, nodes_base, curvature, height_factor)
        elif shape_type == "circular":
            shape_vars = circular_arch_initialize(problem, nodes_base, curvature, height_factor)
        elif shape_type == "catenary":
            shape_vars = catenary_arch_initialize(problem, nodes_base, curvature, height_factor)
        elif shape_type == "flat":
            shape_vars = np.full(num_shape, y_max * height_factor)
        else:
            shape_vars = np.clip(base_shapes * height_factor, y_min, y_max)

        for _ in range(5):
            result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
            if result and result["feasible"]:
                break
            if result is None:
                areas = np.clip(areas * 1.20, area_min, area_max)
            else:
                stress_ratio = result["max_stress"] / sigma_limit if sigma_limit > 0 else 1.0
                disp_ratio = result["max_disp"] / disp_limit if disp_limit > 0 else 1.0
                scale = max(stress_ratio, disp_ratio, 1.0) * 1.01
                areas = np.clip(areas * scale, area_min, area_max)

        areas = direct_sizing_optimize(areas, shape_vars, problem, nodes_base, elements, max_iter=max_iter)
        areas, shape_vars = sensitivity_shape_optimize(areas, shape_vars, problem, nodes_base, elements, max_iter=6)
        areas = aggressive_reduction(areas, shape_vars, problem, nodes_base, elements, max_iter=15)
        areas, shape_vars = local_search_refinement(areas, shape_vars, problem, nodes_base, elements, max_iter=8)

        result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if result and result["feasible"]:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if w < best_weight:
                best_weight = w
                best_areas = areas.copy()
                best_shapes = shape_vars.copy()

    if best_areas is not None:
        areas = best_areas.copy()
        shape_vars = best_shapes.copy()

        areas, shape_vars = sensitivity_shape_optimize(areas, shape_vars, problem, nodes_base, elements, max_iter=4)
        areas = aggressive_reduction(areas, shape_vars, problem, nodes_base, elements, max_iter=20)
        areas, shape_vars = local_search_refinement(areas, shape_vars, problem, nodes_base, elements, max_iter=12)

        result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if result and result["feasible"]:
            w = compute_weight(np.concatenate([areas, shape_vars]), problem, nodes_base, elements)
            if w < best_weight:
                best_weight = w
                best_areas = areas.copy()
                best_shapes = shape_vars.copy()

    return np.concatenate([best_areas, best_shapes])


# ============================================================================
# MAIN FUNCTION (Partially modifiable - Keep output format fixed)
# ============================================================================

def main():
    """
    Main optimization routine.
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

    print("Optimizing using hybrid aggressive reduction...")
    x_best = optimize_stress_ratio(problem, nodes_base, elements, max_iter=50)
    w = compute_weight(x_best, problem, nodes_base, elements)

    result = analyze_design(x_best[:num_bars], x_best[num_bars:], problem, nodes_base, elements)
    feasible = result["feasible"] if result else False

    submission = {
        "benchmark_id": "iscso_2015",
        "solution_vector": x_best.tolist(),
        "algorithm": "HybridAggressiveReduction",
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