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

def _local_search(x0, problem, nodes_base, elements, max_evals=3000):
    """
    Simple stochastic local search that attempts to reduce areas (and
    optionally shape variables) while keeping the design feasible.
    It stops after ``max_evals`` feasibility checks.
    """
    rng = np.random.default_rng(0)
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])
    area_min = problem["variable_bounds"]["area_min"]
    area_max = problem["variable_bounds"]["area_max"]
    y_min = problem["variable_bounds"]["y_min"]
    y_max = problem["variable_bounds"]["y_max"]

    areas = x0[:num_bars].copy()
    shape = x0[num_bars:].copy()
    best_weight = compute_weight(x0, problem, nodes_base, elements)

    evals = 0
    while evals < max_evals:
        # choose whether to modify an area or a shape variable
        if rng.random() < 0.8:  # mostly tweak areas
            idx = rng.integers(num_bars)
            # try reducing the area by 10 % – a larger step helps escape shallow minima faster
            new_val = max(area_min, areas[idx] * 0.90)
            trial_areas = areas.copy()
            trial_areas[idx] = new_val
            trial_shape = shape
        else:
            idx = rng.integers(num_shape)
            # try a larger random downward move (‑5 mm … 0 mm) to explore more shape reduction
            delta = rng.integers(-5, 1)   # possible values: -5, -4, -3, -2, -1, 0
            new_val = np.clip(shape[idx] + delta, y_min, y_max)
            trial_shape = shape.copy()
            trial_shape[idx] = new_val
            trial_areas = areas

        result = analyze_design(trial_areas, trial_shape, problem, nodes_base, elements)
        evals += 1
        if result is None or not result["feasible"]:
            continue

        trial_x = np.concatenate([trial_areas, trial_shape])
        trial_weight = compute_weight(trial_x, problem, nodes_base, elements)
        if trial_weight < best_weight:
            areas, shape = trial_areas, trial_shape
            best_weight = trial_weight

    return np.concatenate([areas, shape])


# ----------------------------------------------------------------------
# Helper: greedy deterministic tightening of areas
# ----------------------------------------------------------------------
def _tighten_areas(x, problem, nodes_base, elements,
                  reduction=0.95, passes=3):
    """
    Greedy post‑processing: try to shrink each area by ``reduction``
    (default 5 %) while keeping feasibility and decreasing weight.
    Repeats up to ``passes`` times over all members.
    """
    num_bars = problem["num_bars"]
    area_min = problem["variable_bounds"]["area_min"]

    areas = x[:num_bars].copy()
    shape = x[num_bars:].copy()

    for _ in range(passes):
        improved = False
        for i in range(num_bars):
            new_area = max(area_min, areas[i] * reduction)
            # if no actual reduction, skip
            if new_area >= areas[i] - 1e-12:
                continue
            trial_areas = areas.copy()
            trial_areas[i] = new_area

            # feasibility check
            res = analyze_design(trial_areas, shape,
                                 problem, nodes_base, elements)
            if res is None or not res["feasible"]:
                continue

            # weight check
            trial_x = np.concatenate([trial_areas, shape])
            if compute_weight(trial_x, problem, nodes_base, elements) < \
               compute_weight(np.concatenate([areas, shape]),
                             problem, nodes_base, elements):
                areas = trial_areas
                improved = True

        if not improved:
            break

    return np.concatenate([areas, shape])


def optimize_stress_ratio(problem, nodes_base, elements, max_iter=15):
    """
    Stress ratio method optimization algorithm (enhanced).

    This version performs a *growth* phase to obtain a feasible design
    and then a *refinement* phase that repeatedly shrinks the areas
    while keeping the design feasible. The goal is to reduce the final
    weight, thereby improving the fitness score.
    """
    bounds_cfg = problem["variable_bounds"]
    area_min = bounds_cfg["area_min"]
    area_max = bounds_cfg["area_max"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    num_bars = problem["num_bars"]
    num_shape = len(problem["shape_variable_node_ids"])

    # ------------------------------------------------------------
    # Initial guess – random areas within bounds (adds diversity)
    # ------------------------------------------------------------
    rng = np.random.default_rng()
    areas = rng.uniform(area_min, area_max, size=num_bars)
    shape_vars = np.array(
        [nodes_base[nid - 1, 1] for nid in problem["shape_variable_node_ids"]]
    )

    # ------------------------------------------------------------
    # Phase 1 – Grow until feasible
    # ------------------------------------------------------------
    for iteration in range(max_iter):
        result = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if result is None:
            # In the very unlikely case of solver failure, enlarge a bit
            areas = np.clip(areas * 1.2, area_min, area_max)
            continue

        if result["feasible"]:
            # Feasible design found – move to shrinking phase
            break

        # Determine required scaling factor based on the worst constraint violation
        scale = 1.0
        max_stress = result["max_stress"]
        max_disp = result["max_disp"]
        if max_stress > 1e-6:
            scale = max(scale, max_stress / sigma_limit * 1.05)
        if max_disp > 1e-6:
            scale = max(scale, max_disp / disp_limit * 1.05)

        areas = np.clip(areas * scale, area_min, area_max)
    else:
        # If loop finishes without break, we may still be infeasible;
        # keep the last tried areas (they are the best we could get)
        pass

    # ------------------------------------------------------------
    # Phase 2 – Refine (shrink) while staying feasible
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Phase 2 – Refine (shrink) while staying feasible
    # ------------------------------------------------------------
    # Use the current slack (how far we are from the limits) to
    # shrink as much as possible in one step.
    while True:
        # evaluate the current design
        cur_res = analyze_design(areas, shape_vars, problem, nodes_base, elements)
        if cur_res is None or not cur_res["feasible"]:
            break

        # compute slack ratios (≤ 1).  The smaller one limits the shrink.
        stress_slack = problem["constraints"]["stress_limit"] / (cur_res["max_stress"] + 1e-12)
        disp_slack   = problem["constraints"]["displacement_limit"] / (cur_res["max_disp"] + 1e-12)
        slack = min(stress_slack, disp_slack)

        # if we are already within 0.1 % of the limits, stop shrinking
        if slack >= 0.999:
            break

        # apply a safety margin (0.5 %) and shrink uniformly
        shrink_factor = slack * 0.995
        candidate_areas = np.clip(areas * shrink_factor, area_min, area_max)

        # verify the candidate; if it becomes infeasible we stop
        cand_res = analyze_design(candidate_areas, shape_vars,
                                  problem, nodes_base, elements)
        if cand_res is None or not cand_res["feasible"]:
            break
        areas = candidate_areas

    # ------------------------------------------------------------
    # Phase 2.5 – Small uniform shrink of the shape variables
    # ------------------------------------------------------------
    shape_shrink_factor = 0.995
    while True:
        cand_shape = np.clip(shape_vars * shape_shrink_factor,
                             bounds_cfg["y_min"], bounds_cfg["y_max"])
        result = analyze_design(areas, cand_shape, problem, nodes_base, elements)
        if result is None or not result["feasible"]:
            break
        shape_vars = cand_shape

    # ------------------------------------------------------------
    # Return the final design vector (areas + shape variables)
    # ------------------------------------------------------------
    refined = _local_search(np.concatenate([areas, shape_vars]),
                           problem, nodes_base, elements,
                           max_evals=8000)          # more explorations
    refined = _tighten_areas(refined, problem, nodes_base, elements,
                            reduction=0.95, passes=7)  # deeper greedy tightening
    return refined


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
    # Run several independent optimisation restarts and keep the best feasible design.
    best_x = None
    best_w = float('inf')
    num_restarts = 12
    for run in range(num_restarts):
        print(f"Optimisation restart {run + 1}/{num_restarts} ...")
        x_candidate = optimize_stress_ratio(problem, nodes_base, elements, max_iter=15)
        w_candidate = compute_weight(x_candidate, problem, nodes_base, elements)

        # keep the lightest feasible design found so far
        result = analyze_design(x_candidate[:num_bars], x_candidate[num_bars:],
                                problem, nodes_base, elements)
        if result is not None and result["feasible"] and w_candidate < best_w:
            best_w = w_candidate
            best_x = x_candidate

    # fall‑back in the extremely unlikely case that none of the restarts were feasible
    if best_x is None:
        best_x = optimize_stress_ratio(problem, nodes_base, elements, max_iter=15)
        best_w = compute_weight(best_x, problem, nodes_base, elements)

    x_best = best_x
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
