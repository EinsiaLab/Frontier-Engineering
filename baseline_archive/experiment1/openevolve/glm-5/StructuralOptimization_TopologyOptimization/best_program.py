# EVOLVE-BLOCK-START
"""
Topology Optimization — MBB Beam (SIMP Method)

- ALLOWED TO MODIFY: optimize_topology()
- NOT ALLOWED TO MODIFY: load_problem(), fem_solve_2d_quad(), compute_compliance(),
                         apply_density_filter(), output format
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================================
# DATA LOADING (NOT ALLOWED TO MODIFY - Interface must match evaluator)
# ============================================================================

def load_problem():
    """Load problem config. DO NOT MODIFY."""
    candidates = [
        Path("references/problem_config.json"),
        Path(__file__).resolve().parent.parent / "references" / "problem_config.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("problem_config.json not found")


# ============================================================================
# FEM SOLVER (NOT ALLOWED TO MODIFY - Must match evaluator implementation)
# ============================================================================

def _element_stiffness_matrix(nu):
    """8x8 element stiffness matrix for unit Q4 element, plane stress. DO NOT MODIFY."""
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    KE = (1.0 / (1.0 - nu**2)) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


def fem_solve_2d_quad(nelx, nely, density, config):
    """2D FEM solver with Q4 elements. Returns displacement vector. DO NOT MODIFY."""
    E0 = config["E0"]
    Emin = config["Emin"]
    nu = config["nu"]
    penal = config["penal"]

    KE = _element_stiffness_matrix(nu)

    n_dofs = 2 * (nelx + 1) * (nely + 1)

    # Assembly using COO format
    iK = np.zeros(64 * nelx * nely, dtype=int)
    jK = np.zeros(64 * nelx * nely, dtype=int)
    sK = np.zeros(64 * nelx * nely, dtype=float)

    for elx in range(nelx):
        for ely in range(nely):
            e_idx = elx * nely + ely
            n1 = elx * (nely + 1) + ely
            n2 = (elx + 1) * (nely + 1) + ely
            edof = np.array([
                2*n1, 2*n1+1,
                2*n2, 2*n2+1,
                2*n2+2, 2*n2+3,
                2*n1+2, 2*n1+3,
            ])
            Ee = Emin + density[ely, elx]**penal * (E0 - Emin)
            for i_local in range(8):
                for j_local in range(8):
                    idx = e_idx * 64 + i_local * 8 + j_local
                    iK[idx] = edof[i_local]
                    jK[idx] = edof[j_local]
                    sK[idx] = Ee * KE[i_local, j_local]

    K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs)).tocsc()

    # Force vector — load at top-left node (node 0), fy = -1
    F = np.zeros(n_dofs)
    F[1] = config["force"]["fy"]

    # MBB half-symmetry boundary conditions:
    # - Left edge: fix x-DOFs (symmetry)
    # - Bottom-right corner: fix y-DOF (roller support)
    fixed_dofs = []
    # Fix x-DOF on left edge
    for i in range(nely + 1):
        fixed_dofs.append(2 * i)
    # Fix y-DOF at bottom-right corner
    fixed_dofs.append(2 * (nelx * (nely + 1) + nely) + 1)

    fixed_dofs = np.array(fixed_dofs, dtype=int)
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[free_dofs, :][:, free_dofs]
    F_f = F[free_dofs]

    u = np.zeros(n_dofs)
    u[free_dofs] = spsolve(K_ff, F_f)

    return u


def compute_compliance(nelx, nely, density, u, config):
    """Compute total compliance and element sensitivities. DO NOT MODIFY."""
    E0 = config["E0"]
    Emin = config["Emin"]
    nu = config["nu"]
    penal = config["penal"]

    KE = _element_stiffness_matrix(nu)

    compliance = 0.0
    dc = np.zeros((nely, nelx))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = elx * (nely + 1) + ely
            n2 = (elx + 1) * (nely + 1) + ely
            edof = np.array([
                2*n1, 2*n1+1,
                2*n2, 2*n2+1,
                2*n2+2, 2*n2+3,
                2*n1+2, 2*n1+3,
            ])
            ue = u[edof]
            ce = float(ue @ KE @ ue)
            Ee = Emin + density[ely, elx]**penal * (E0 - Emin)
            compliance += Ee * ce
            dc[ely, elx] = -penal * density[ely, elx]**(penal - 1) * (E0 - Emin) * ce

    return compliance, dc


def apply_density_filter(nelx, nely, rmin, x, dc):
    """Apply density filter for mesh-independence. DO NOT MODIFY."""
    dc_new = np.zeros_like(dc)
    for i in range(nelx):
        for j in range(nely):
            sum_weight = 0.0
            for k in range(max(i - int(np.ceil(rmin)), 0),
                           min(i + int(np.ceil(rmin)) + 1, nelx)):
                for l in range(max(j - int(np.ceil(rmin)), 0),
                               min(j + int(np.ceil(rmin)) + 1, nely)):
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    if fac > 0:
                        dc_new[j, i] += fac * x[l, k] * dc[l, k]
                        sum_weight += fac * x[l, k]
            if sum_weight > 0:
                dc_new[j, i] /= sum_weight
    return dc_new


# ============================================================================
# OPTIMIZATION ALGORITHM (ALLOWED TO MODIFY - This is your optimization code)
# ============================================================================

def optimize_topology(nelx, nely, config, max_iter=200):
    """OC method with early Heaviside projection and continuation."""
    volfrac, rmin = config["volfrac"], config["rmin"]
    target_penal, rho_min = config["penal"], 1e-3

    x = np.full((nely, nelx), volfrac)
    best_compliance = float('inf')
    best_x = x.copy()
    
    for iteration in range(max_iter):
        # Aggressive continuation: reach target penal by iteration 12
        if iteration < 12:
            penal = 1.0 + (target_penal - 1.0) * (iteration / 12.0)
        else:
            penal = target_penal
        config['penal'] = penal
        
        # Adaptive move limit with aggressive early exploration
        if iteration < 15:
            move = 0.5
        elif iteration < 50:
            move = 0.3
        elif iteration < 100:
            move = 0.2
        else:
            move = 0.08
        
        u = fem_solve_2d_quad(nelx, nely, x, config)
        compliance, dc = compute_compliance(nelx, nely, x, u, config)
        dc = apply_density_filter(nelx, nely, rmin, x, dc)

        # Track best solution at target penalization
        if compliance < best_compliance and penal >= target_penal:
            best_compliance = compliance
            best_x = x.copy()

        # OC bisection with tighter tolerance
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-5:
            lmid = 0.5 * (l2 + l1)
            Be = np.sqrt(-dc / (lmid + 1e-30))
            x_new = np.maximum(rho_min,
                        np.maximum(x - move,
                            np.minimum(1.0, np.minimum(x + move, x * Be))))
            if np.mean(x_new) - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid

        # Standard OC damping for stability
        damping = 0.5
        x_new = (x_new + damping * x) / (1.0 + damping)
        
        # Earlier Heaviside projection with stronger parameters
        if iteration > 80:
            beta = 12.0  # Higher sharpness for cleaner 0/1 designs
            eta = volfrac  # Threshold
            x_proj = (np.tanh(beta * eta) + np.tanh(beta * (x_new - eta))) / \
                     (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))
            x_new = 0.55 * x_new + 0.45 * x_proj  # More aggressive blending
        
        # Ensure exact volume constraint
        vol_error = np.mean(x_new) - volfrac
        if abs(vol_error) > 0.0005:
            x_new = np.clip(x_new - vol_error, rho_min, 1.0)

        change = np.max(np.abs(x_new - x))
        x = x_new

        print(f"  Iter {iteration+1:3d}: compliance = {compliance:.4f}, vol = {np.mean(x):.4f}")

        if change < 0.0005 and penal >= target_penal and iteration > 120:
            print(f"  Converged at iteration {iteration+1}")
            break

    # Final evaluation to choose best solution
    config['penal'] = target_penal
    u = fem_solve_2d_quad(nelx, nely, best_x, config)
    best_comp, _ = compute_compliance(nelx, nely, best_x, u, config)
    u = fem_solve_2d_quad(nelx, nely, x, config)
    final_comp, _ = compute_compliance(nelx, nely, x, u, config)
    
    return best_x if best_comp < final_comp else x


# ============================================================================
# MAIN FUNCTION (Partially modifiable - Keep output format fixed)
# ============================================================================

def main():
    """Main routine. Keep output format (submission.json) fixed."""
    config = load_problem()
    nelx = config["nelx"]
    nely = config["nely"]

    print(f"Topology Optimization — MBB Beam")
    print(f"  Mesh: {nelx} x {nely} = {nelx * nely} elements")
    print(f"  Volume fraction: {config['volfrac']}")
    print(f"  Penalization: {config['penal']}")
    print(f"  Filter radius: {config['rmin']}")
    print()

    # ALLOWED TO MODIFY: Optimization algorithm call
    print("Optimizing using OC method with continuation...")
    density = optimize_topology(nelx, nely, config, max_iter=200)

    # Compute final compliance
    u = fem_solve_2d_quad(nelx, nely, density, config)
    compliance, _ = compute_compliance(nelx, nely, density, u, config)
    vol_frac = float(np.mean(density))

    print()
    print(f"  Final compliance: {compliance:.4f}")
    print(f"  Final volume fraction: {vol_frac:.4f}")

    # NOT ALLOWED TO MODIFY: Output format must match exactly
    submission = {
        "benchmark_id": "topology_optimization",
        "density_vector": density.flatten().tolist(),
        "nelx": nelx,
        "nely": nely,
    }

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    submission_path = temp_dir / "submission.json"

    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print(f"submission.json written to {submission_path}")
    print(f"  Elements: {nelx * nely}")
    print(f"  Compliance: {compliance:.4f}")
    print(f"  Volume fraction: {vol_frac:.4f}")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
