# EVOLVE-BLOCK-START
"""
ISCSO 2023 — 284-Member 3D Truss Sizing Optimization

Simple Random Search Baseline
A straightforward random sampling approach to find feasible solutions.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ===================== Inline 3D Tower Generator =====================

def generate_tower(problem):
    """Generate tower topology from problem data."""
    tp = problem["tower_parameters"]
    num_levels = tp["num_levels"]
    total_height = tp["total_height_mm"]
    bottom_hw = tp["bottom_half_width_mm"]
    top_hw = tp["top_half_width_mm"]
    cb_levels = tp["cross_bracing_levels"]

    n_nodes = num_levels * 4
    nodes = np.zeros((n_nodes, 3))
    for i in range(num_levels):
        t = i / (num_levels - 1)
        hw = bottom_hw + t * (top_hw - bottom_hw)
        z = t * total_height
        base = 4 * i
        nodes[base + 0] = [+hw, +hw, z]
        nodes[base + 1] = [-hw, +hw, z]
        nodes[base + 2] = [-hw, -hw, z]
        nodes[base + 3] = [+hw, -hw, z]

    elems = []
    # Verticals
    for i in range(num_levels - 1):
        for j in range(4):
            elems.append((4*i + j, 4*(i+1) + j))
    # Horizontal perimeter
    for i in range(num_levels):
        b = 4 * i
        elems.extend([(b, b+1), (b+1, b+2), (b+2, b+3), (b+3, b)])
    # Face diagonals
    for i in range(num_levels - 1):
        elems.append((4*i, 4*(i+1)+1))
        elems.append((4*i+1, 4*(i+1)+2))
        elems.append((4*i+2, 4*(i+1)+3))
        elems.append((4*i+3, 4*(i+1)))
    # Floor cross-bracing
    for i in cb_levels:
        if i < num_levels:
            b = 4 * i
            elems.extend([(b, b+2), (b+1, b+3)])

    return nodes, np.array(elems, dtype=int)


# ===================== Inline 3D FEM =====================

def fem_solve_3d(nodes, elements, areas, E, supports, force_vec):
    """Solve 3D truss FEM. Returns (displacements, stresses, lengths)."""
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
                rows.append(ii); cols.append(jj); vals.append(ke[il, jl])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsc()
    F = force_vec[free]
    u_red = spsolve(K, F)

    disp = np.zeros(n_dofs)
    for idx, dof in enumerate(free):
        disp[dof] = u_red[idx]

    stresses = np.zeros(n_elems)
    for e in range(n_elems):
        ni, nj = elements[e]
        d = nodes[nj] - nodes[ni]
        L = lengths[e]
        dc = d / L
        T = np.array([-dc[0], -dc[1], -dc[2], dc[0], dc[1], dc[2]])
        u_e = np.array([disp[3*ni], disp[3*ni+1], disp[3*ni+2],
                        disp[3*nj], disp[3*nj+1], disp[3*nj+2]])
        stresses[e] = (E / L) * T.dot(u_e)

    return disp, stresses, lengths


# ===================== Load Problem Data =====================

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


# ===================== Evaluation =====================

def evaluate_design(areas, problem, nodes, elements):
    E = problem["material"]["E"]
    rho = problem["material"]["rho"]
    sigma_limit = problem["constraints"]["stress_limit"]
    disp_limit = problem["constraints"]["displacement_limit"]
    supports = problem["supports"]

    max_vio = 0.0
    for lc in problem["load_cases"]:
        fvec = np.zeros(3 * len(nodes))
        for load in lc["loads"]:
            nid = load["node"]
            fvec[3*nid] += load["fx"]
            fvec[3*nid+1] += load["fy"]
            fvec[3*nid+2] += load["fz"]

        try:
            disp, stresses, lengths = fem_solve_3d(
                nodes, elements, areas, E, supports, fvec
            )
        except Exception:
            return 1e18, 1e18

        stress_vio = float(np.max(np.abs(stresses) - sigma_limit))
        disp_vio = float(np.max(np.abs(disp) - disp_limit))
        max_vio = max(max_vio, stress_vio, disp_vio)

    weight = 0.0
    for e in range(len(elements)):
        ni, nj = elements[e]
        L = np.linalg.norm(nodes[nj] - nodes[ni])
        weight += rho * L * areas[e]

    return weight, max(max_vio, 0.0)


# ===================== Main (top-level execution) =====================

problem = load_problem()
nodes, elements = generate_tower(problem)
bounds_cfg = problem["variable_bounds"]
dim = problem["dimension"]

print(f"ISCSO 2023 Simple Random Search Baseline")
print(f"  Nodes: {len(nodes)}, Elements: {len(elements)}")
print(f"  Design variables: {dim}")
print(f"  Area bounds: [{bounds_cfg['area_min']}, {bounds_cfg['area_max']}] mm^2")
print(f"  Stress limit: {problem['constraints']['stress_limit']} MPa")
print(f"  Displacement limit: {problem['constraints']['displacement_limit']} mm")
print(f"  Load cases: {len(problem['load_cases'])}")
print()

# Random search parameters
MAX_EVALS = 500  # Simple and fast (fewer than ISCSO2015 due to higher dimension)
np.random.seed(42)

best_feasible = None
best_weight = float("inf")
start_time = time.time()

print(f"Running Random Search ({MAX_EVALS} evaluations)...")
for i in range(MAX_EVALS):
    # Random sample in bounds
    x = np.random.uniform(bounds_cfg["area_min"], bounds_cfg["area_max"], dim)
    
    w, vio = evaluate_design(x, problem, nodes, elements)
    
    if vio <= 1e-6 and w < best_weight:
        best_weight = w
        best_feasible = x.copy()
        elapsed = time.time() - start_time
        print(f"  [{i+1:4d}] New best feasible: weight = {w:.4f} kg (time: {elapsed:.1f}s)")
    elif (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        status = "feasible" if vio <= 1e-6 else f"violation={vio:.2e}"
        print(f"  [{i+1:4d}] Current: weight = {w:.4f} kg ({status}), time: {elapsed:.1f}s")

elapsed_time = time.time() - start_time
print(f"\nRandom Search finished")
print(f"  Total evaluations: {MAX_EVALS}")
print(f"  Total time: {elapsed_time:.1f}s")

if best_feasible is not None:
    x_best = best_feasible
    w, vio = evaluate_design(x_best, problem, nodes, elements)
    print(f"  Best feasible weight: {w:.4f} kg")
    print(f"  Max constraint violation: {vio:.6e}")
else:
    # Use last evaluated point if no feasible found
    x_best = x
    w, vio = evaluate_design(x_best, problem, nodes, elements)
    print(f"  WARNING: No feasible solution found!")
    print(f"  Best weight: {w:.4f} kg, violation: {vio:.6e}")

submission = {
    "benchmark_id": "iscso_2023",
    "solution_vector": x_best.tolist(),
    "algorithm": "RandomSearch",
    "num_evaluations": MAX_EVALS,
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print(f"\n✓ submission.json written")
print(f"  Dimension: {len(x_best)}")
print(f"  Weight: {w:.4f} kg")
print(f"  Feasible: {vio <= 1e-6}")
# EVOLVE-BLOCK-END

