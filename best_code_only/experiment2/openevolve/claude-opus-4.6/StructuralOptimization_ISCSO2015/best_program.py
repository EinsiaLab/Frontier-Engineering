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

def get_elem_stresses(areas, sv, problem, nodes_base, elements):
    """Get per-element max stresses and max displacement across all load cases."""
    E = problem["material"]["E"]
    supports = problem["supports"]
    nodes = nodes_base.copy()
    for idx, nid in enumerate(problem["shape_variable_node_ids"]):
        nodes[nid - 1, 1] = sv[idx]
    max_s = np.zeros(problem["num_bars"])
    max_d = 0.0
    for lc in problem["load_cases"]:
        fvec = np.zeros(2 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            fvec[2*(nid-1)] += load["fx"]
            fvec[2*(nid-1)+1] += load["fy"]
        disp, stresses, _ = fem_solve_2d(nodes, elements, areas, E, supports, fvec)
        max_s = np.maximum(max_s, np.abs(stresses))
        max_d = max(max_d, np.max(np.abs(disp)))
    return max_s, max_d

def run_fsd(a0, sv, problem, nodes_base, elements, n_iter=80):
    """Run fully stressed design iterations."""
    bc = problem["variable_bounds"]
    amin, amax = bc["area_min"], bc["area_max"]
    sl = problem["constraints"]["stress_limit"]
    dl = problem["constraints"]["displacement_limit"]
    nb = problem["num_bars"]
    a = a0.copy()
    ba, bw = None, np.inf
    for it in range(n_iter):
        try:
            es, md = get_elem_stresses(a, sv, problem, nodes_base, elements)
        except Exception:
            a = np.clip(a * 1.5, amin, amax)
            continue
        sr = es / sl
        dr = md / dl
        alpha = 0.6 if it < 15 else (0.45 if it < 35 else (0.3 if it < 55 else 0.15))
        na = a.copy()
        for e in range(nb):
            r = max(sr[e], dr) if dr > 1.0 else sr[e]
            t = a[e] * r if r > 1e-6 else amin
            na[e] = a[e] * (1.0 - alpha) + t * alpha
            if sr[e] < 0.005:
                na[e] = amin
        a = np.clip(na, amin, amax)
        if it > 3:
            res = analyze_design(a, sv, problem, nodes_base, elements)
            if res and res["feasible"]:
                x = np.concatenate([a, sv])
                w = compute_weight(x, problem, nodes_base, elements)
                if w < bw:
                    bw = w
                    ba = a.copy()
    return ba, bw

def trim_design(a0, sv, problem, nodes_base, elements, n_iter=50):
    """Progressively trim feasible design."""
    bc = problem["variable_bounds"]
    amin = bc["area_min"]
    sl = problem["constraints"]["stress_limit"]
    dl = problem["constraints"]["displacement_limit"]
    nb = problem["num_bars"]
    a = a0.copy()
    ba = a.copy()
    bw = compute_weight(np.concatenate([a, sv]), problem, nodes_base, elements)
    for _ in range(n_iter):
        es, md = get_elem_stresses(a, sv, problem, nodes_base, elements)
        sr = es / sl
        dr = md / dl
        na = a.copy()
        changed = False
        for e in range(nb):
            if sr[e] < 0.98:
                if sr[e] < 0.01:
                    red = 0.5
                elif sr[e] < 0.5:
                    red = 0.85 + 0.12 * sr[e]
                else:
                    red = 0.92 + 0.08 * sr[e]
                if dr > 0.95:
                    red = max(red, 0.97)
                na[e] = max(a[e] * red, amin)
                changed = True
        if not changed:
            break
        res = analyze_design(na, sv, problem, nodes_base, elements)
        if res and res["feasible"]:
            a = na
            w = compute_weight(np.concatenate([a, sv]), problem, nodes_base, elements)
            if w < bw:
                bw = w
                ba = a.copy()
        else:
            mid = np.clip((a + na) / 2, amin, bc["area_max"])
            r2 = analyze_design(mid, sv, problem, nodes_base, elements)
            if r2 and r2["feasible"]:
                a = mid
                w = compute_weight(np.concatenate([a, sv]), problem, nodes_base, elements)
                if w < bw:
                    bw = w
                    ba = a.copy()
            else:
                break
    return ba, bw

def optimize_stress_ratio(problem, nodes_base, elements, max_iter=15):
    """FSD + shape exploration + trimming + binary search fine-tuning."""
    bc = problem["variable_bounds"]
    amin, amax = bc["area_min"], bc["area_max"]
    y_min, y_max = bc["y_min"], bc["y_max"]
    nb = problem["num_bars"]
    ns = len(problem["shape_variable_node_ids"])
    sv0 = np.array([nodes_base[nid-1, 1] for nid in problem["shape_variable_node_ids"]])
    bw, bx = np.inf, None
    shapes = [sv0.copy()]
    for s in [0.6, 0.8, 1.2, 1.5, 2.0]:
        shapes.append(np.clip(sv0 * s, y_min, y_max))
    for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        shapes.append(np.full(ns, y_min + (y_max - y_min) * f))
    xc = np.array([nodes_base[nid-1, 0] for nid in problem["shape_variable_node_ids"]])
    xm = (xc.min() + xc.max()) / 2.0
    xr = max((xc.max() - xc.min()) / 2.0, 1.0)
    for ph in [5000, 8000, 12000, 18000, 25000, 35000]:
        shapes.append(np.clip(ph * (1 - ((xc - xm)/xr)**2), y_min, y_max))

    # Asymmetric shapes (load is heavier on right side: nodes 15,17,19)
    for bias in [0.4, 0.6, 0.7, 0.8]:
        asym = np.clip(sv0 * (1.0 + bias * (xc - xm) / xr), y_min, y_max)
        shapes.append(asym)
    for bias in [-0.3, -0.5]:
        asym = np.clip(sv0 * (1.0 + bias * (xc - xm) / xr), y_min, y_max)
        shapes.append(asym)
    # Cubic profiles
    for ph in [10000, 20000, 30000]:
        cubic = ph * (1.0 - ((xc - xm) / xr) ** 4)
        shapes.append(np.clip(cubic, y_min, y_max))

    for sv in shapes:
        for sc in [3.0, 8.0, 15.0]:
            ba, w = run_fsd(np.full(nb, amin * sc), sv, problem, nodes_base, elements, 80)
            if ba is not None:
                ta, tw = trim_design(ba, sv, problem, nodes_base, elements, 50)
                if tw < bw:
                    bw, bx = tw, np.concatenate([ta, sv])

    if bx is not None:
        ao, so = bx[:nb].copy(), bx[nb:].copy()
        for rnd in range(3):
            imp = False
            ds = [500,200,100,50,20] if rnd==0 else ([100,50,20,10] if rnd==1 else [30,15,5])
            for si in range(ns):
                for d in ds:
                    for sgn in [1,-1]:
                        st = so.copy(); st[si] = np.clip(st[si]+sgn*d, y_min, y_max)
                        ba, w = run_fsd(ao, st, problem, nodes_base, elements, 60)
                        if ba is not None:
                            ta, tw = trim_design(ba, st, problem, nodes_base, elements, 40)
                            if tw < bw:
                                bw, bx = tw, np.concatenate([ta, st])
                                ao, so = ta.copy(), st.copy()
                                imp = True
            if not imp:
                break

    # Phase 3: Random shape perturbations around best
    if bx is not None:
        ao, so = bx[:nb].copy(), bx[nb:].copy()
        for trial in range(50):
            scale = 400 if trial < 15 else (150 if trial < 30 else 50)
            svt = np.clip(so + np.random.uniform(-scale, scale, ns), y_min, y_max)
            ba, w = run_fsd(ao, svt, problem, nodes_base, elements, 50)
            if ba is not None:
                ta, tw = trim_design(ba, svt, problem, nodes_base, elements, 35)
                if tw < bw:
                    bw, bx = tw, np.concatenate([ta, svt])
                    ao, so = ta.copy(), svt.copy()

    # Phase 4: Coarse element-wise area reduction
    if bx is not None:
        a, sv = bx[:nb].copy(), bx[nb:].copy()
        for _ in range(8):
            imp = False
            for e in range(nb):
                for fac in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
                    t = a.copy(); t[e] = max(a[e] * fac, amin)
                    r = analyze_design(t, sv, problem, nodes_base, elements)
                    if r and r["feasible"]:
                        w = compute_weight(np.concatenate([t, sv]), problem, nodes_base, elements)
                        if w < bw:
                            bw, bx = w, np.concatenate([t, sv])
                            a = t.copy(); imp = True; break
            if not imp:
                break

    # Phase 5: Binary search on each area
    if bx is not None:
        a, sv = bx[:nb].copy(), bx[nb:].copy()
        for e in range(nb):
            lo, hi = amin, a[e]
            if hi - lo < 1.0:
                continue
            for _ in range(12):
                mid = (lo + hi) / 2
                t = a.copy(); t[e] = mid
                r = analyze_design(t, sv, problem, nodes_base, elements)
                if r and r["feasible"]:
                    hi = mid
                else:
                    lo = mid
            a[e] = hi
        xc = np.concatenate([a, sv])
        r = analyze_design(a, sv, problem, nodes_base, elements)
        if r and r["feasible"]:
            w = compute_weight(xc, problem, nodes_base, elements)
            if w < bw:
                bw, bx = w, xc.copy()

    # Phase 6: Re-run FSD+trim on refined solution
    if bx is not None:
        a, sv = bx[:nb].copy(), bx[nb:].copy()
        ba, w = run_fsd(a, sv, problem, nodes_base, elements, 80)
        if ba is not None:
            ta, tw = trim_design(ba, sv, problem, nodes_base, elements, 50)
            if tw < bw:
                bw, bx = tw, np.concatenate([ta, sv])
        # Final binary search pass after FSD refinement
        a, sv = bx[:nb].copy(), bx[nb:].copy()
        for e in range(nb):
            lo, hi = amin, a[e]
            if hi - lo < 0.5:
                continue
            for _ in range(10):
                mid = (lo + hi) / 2
                t = a.copy(); t[e] = mid
                r = analyze_design(t, sv, problem, nodes_base, elements)
                if r and r["feasible"]:
                    hi = mid
                else:
                    lo = mid
            a[e] = hi
        xf = np.concatenate([a, sv])
        r = analyze_design(a, sv, problem, nodes_base, elements)
        if r and r["feasible"]:
            w = compute_weight(xf, problem, nodes_base, elements)
            if w < bw:
                bw, bx = w, xf.copy()

    if bx is None:
        bx = np.concatenate([np.full(nb, amax * 0.3), sv0])
    return bx


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
    np.random.seed(42)
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
