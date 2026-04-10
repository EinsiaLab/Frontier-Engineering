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
    b = problem["variable_bounds"]
    amin, amax = b["area_min"], b["area_max"]
    ymin, ymax = b["y_min"], b["y_max"]
    slim = problem["constraints"]["stress_limit"]
    dlim = problem["constraints"]["displacement_limit"]
    nb = problem["num_bars"]
    y0 = np.array([nodes_base[n - 1, 1] for n in problem["shape_variable_node_ids"]], float)
    cache = {}

    def eva(a, y):
        k = (tuple(np.round(a, 5)), tuple(np.round(y, 2)))
        if k not in cache:
            r = analyze_design(a, y, problem, nodes_base, elements)
            cache[k] = (r, compute_weight(np.concatenate([a, y]), problem, nodes_base, elements))
        return cache[k]

    def make_feasible(a, y):
        r, _ = eva(a, y)
        for _ in range(max_iter):
            if r and r["feasible"]:
                return a
            g = 1.2 if not r else 1.02 * max(1.0, r["max_stress"] / slim, r["max_disp"] / dlim)
            a = np.clip(a * g, amin, amax)
            r, _ = eva(a, y)
        return a if r and r["feasible"] else None

    def reduce(a, y, fs=(0.8, 0.9, 0.95, 0.98), allow_equal=True):
        for f in fs:
            changed = True
            while changed:
                changed = False
                _, w0 = eva(a, y)
                for i in np.argsort(-a):
                    if a[i] <= amin:
                        continue
                    t = a.copy()
                    t[i] = max(amin, a[i] * f)
                    r, w = eva(t, y)
                    ok = w <= w0 if allow_equal else w < w0
                    if r and r["feasible"] and ok:
                        a, w0, changed = t, w, True
        return a

    def repair_reduce(a, y):
        for f in (0.7, 0.85, 0.93):
            for i in np.argsort(-a):
                if a[i] <= amin:
                    continue
                t = a.copy()
                t[i] = max(amin, a[i] * f)
                t = make_feasible(t, y)
                if t is None:
                    continue
                _, w0 = eva(a, y)
                _, w1 = eva(t, y)
                if w1 < w0:
                    a = t
        return a

    def tune_shape(a, y, ds=(-600.0, -300.0, -150.0, -75.0, 75.0, 150.0, 300.0, 600.0)):
        for d in ds:
            improved = True
            while improved:
                improved = False
                _, base = eva(a, y)
                for i in range(len(y)):
                    yt = y.copy()
                    yt[i] = np.clip(y[i] + d, ymin, ymax)
                    r, w = eva(a, yt)
                    if r and r["feasible"] and w < base:
                        y, base, improved = yt, w, True
        return y

    def polish(a, y):
        return reduce(a, y, (0.985, 0.992), False)

    def local_search(a, y):
        best = compute_weight(np.concatenate([a, y]), problem, nodes_base, elements)
        for s in (0.99, 0.995):
            for i in np.argsort(-a):
                if a[i] <= amin:
                    continue
                t = a.copy()
                t[i] = max(amin, a[i] * s)
                r, w = eva(t, y)
                if r and r["feasible"] and w < best:
                    a, best = t, w
        for d in (-60.0, -30.0, 30.0, 60.0):
            for i in range(len(y)):
                yt = y.copy()
                yt[i] = np.clip(y[i] + d, ymin, ymax)
                at = make_feasible(a.copy(), yt)
                if at is None:
                    continue
                at = polish(at, yt)
                _, w = eva(at, yt)
                if w < best:
                    a, y, best = at, yt, w
        return a, y

    ys = [y0, np.clip(y0 * 0.9, ymin, ymax), np.clip(y0 * 1.1, ymin, ymax)]
    ys += [np.clip(y0 + d, ymin, ymax) for d in (-1000.0, -500.0, 500.0, 1000.0)]
    best_x, best_w = None, 1e99
    for y in ys:
        for af in (0.06, 0.08, 0.12, 0.16, 0.22, 0.26, 0.3):
            a = make_feasible(np.full(nb, max(amin, amax * af)), y)
            if a is None:
                continue
            y2 = y.copy()
            for _ in range(2):
                a = reduce(a, y2)
                y2 = tune_shape(a, y2)
                a = repair_reduce(a, y2)
            a = polish(a, y2)
            y2 = tune_shape(a, y2, (-150.0, -75.0, -30.0, 30.0, 75.0, 150.0))
            a = repair_reduce(a, y2)
            a = polish(a, y2)
            a, y2 = local_search(a, y2)
            _, w = eva(a, y2)
            if w < best_w:
                best_w, best_x = w, np.concatenate([a, y2])
    return best_x if best_x is not None else np.concatenate([np.full(nb, amax), y0])


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
    print("Optimizing using repair-reduction + local search...")
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
