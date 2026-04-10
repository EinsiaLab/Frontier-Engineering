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

def optimize_topology(nelx, nely, config, max_iter=400):
    """OC method for topology optimization with vectorized FEM and continuation. ALLOWED TO MODIFY."""
    from scipy.sparse.linalg import splu

    volfrac = config["volfrac"]
    rmin = config["rmin"]
    penal_final = config["penal"]
    E0 = config["E0"]
    Emin = config["Emin"]
    nu = config["nu"]
    rho_min = 1e-3

    KE = _element_stiffness_matrix(nu)

    # Precompute element DOF indices (vectorized)
    elx_col = np.repeat(np.arange(nelx), nely)
    ely_col = np.tile(np.arange(nely), nelx)
    n1 = elx_col * (nely + 1) + ely_col
    n2 = (elx_col + 1) * (nely + 1) + ely_col
    n_elem = nelx * nely
    edof = np.zeros((n_elem, 8), dtype=int)
    edof[:, 0] = 2 * n1
    edof[:, 1] = 2 * n1 + 1
    edof[:, 2] = 2 * n2
    edof[:, 3] = 2 * n2 + 1
    edof[:, 4] = 2 * n2 + 2
    edof[:, 5] = 2 * n2 + 3
    edof[:, 6] = 2 * n1 + 2
    edof[:, 7] = 2 * n1 + 3

    # Precompute COO indices for assembly
    iK = np.repeat(edof, 8, axis=1).flatten()
    jK = np.tile(edof, (1, 8)).flatten()

    n_dofs = 2 * (nelx + 1) * (nely + 1)

    # Boundary conditions (MBB half-symmetry)
    fixed_dofs = np.zeros(nely + 2, dtype=int)
    for i in range(nely + 1):
        fixed_dofs[i] = 2 * i
    fixed_dofs[nely + 1] = 2 * (nelx * (nely + 1) + nely) + 1
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Force vector
    F = np.zeros(n_dofs)
    F[1] = config["force"]["fy"]
    F_f = F[free_dofs]

    # Precompute density filter weights as sparse matrix
    ceil_rmin = int(np.ceil(rmin))
    filter_rows = []
    filter_cols = []
    filter_vals = []
    for elx_i in range(nelx):
        for ely_i in range(nely):
            e1 = elx_i * nely + ely_i
            kmin = max(elx_i - ceil_rmin, 0)
            kmax = min(elx_i + ceil_rmin + 1, nelx)
            lmin = max(ely_i - ceil_rmin, 0)
            lmax = min(ely_i + ceil_rmin + 1, nely)
            for elx_j in range(kmin, kmax):
                dx = elx_i - elx_j
                for ely_j in range(lmin, lmax):
                    dy = ely_i - ely_j
                    fac = rmin - np.sqrt(dx * dx + dy * dy)
                    if fac > 0:
                        e2 = elx_j * nely + ely_j
                        filter_rows.append(e1)
                        filter_cols.append(e2)
                        filter_vals.append(fac)
    H = coo_matrix((filter_vals, (filter_rows, filter_cols)),
                    shape=(n_elem, n_elem)).tocsr()

    # Flat KE for vectorized assembly
    KE_flat = KE.flatten()

    # Initialize uniform density
    x = np.full((nely, nelx), volfrac)

    # Precompute sum of filter weights (constant)
    Hs = np.array(H.sum(axis=1)).flatten()
    Hs = np.maximum(Hs, 1e-30)

    # Fine-grained continuation scheme for better local minimum navigation
    penal_stages = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    iters_per_stage = [20, 15, 15, 15, 20, 15, 20, 15, 900]

    best_compliance = np.inf
    best_x = x.copy()

    # Sensitivity stabilization
    dc_old = None
    eta = 0.5  # OC damping parameter

    # Compliance history for stagnation detection
    compliance_history = []

    global_iter = 0
    for stage_idx, (penal, stage_iters) in enumerate(zip(penal_stages, iters_per_stage)):
        is_final = (penal == penal_final)
        converge_count = 0
        prev_compliance = np.inf

        # Reset sensitivity history when penalization changes
        if stage_idx > 0:
            dc_old = None

        for iteration in range(stage_iters):
            global_iter += 1

            # Adaptive move limit
            if is_final:
                if iteration < 30:
                    move = 0.2
                elif iteration < 80:
                    move = 0.15
                elif iteration < 150:
                    move = 0.1
                else:
                    move = 0.05
            else:
                move = 0.2

            # Vectorized FEM assembly
            x_flat = x.flatten(order='F')  # column-major: elx*nely + ely
            Ee = Emin + x_flat**penal * (E0 - Emin)
            sK = np.outer(Ee, KE_flat).flatten()
            K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs)).tocsc()
            K_ff = K[free_dofs, :][:, free_dofs]

            # Use LU factorization for faster solve
            u = np.zeros(n_dofs)
            try:
                lu = splu(K_ff)
                u[free_dofs] = lu.solve(F_f)
            except Exception:
                u[free_dofs] = spsolve(K_ff, F_f)

            # Vectorized compliance and sensitivity
            ue = u[edof]  # (n_elem, 8)
            ce = np.sum((ue @ KE) * ue, axis=1)
            compliance = float(np.sum(Ee * ce))
            dc_flat = -penal * x_flat**(penal - 1) * (E0 - Emin) * ce

            # Volume sensitivities
            dv_flat = np.ones_like(x_flat)

            # Apply sensitivity filter
            dc_filtered = H.dot(dc_flat) / Hs
            dv_filtered = H.dot(dv_flat) / Hs

            dc = dc_filtered.reshape((nelx, nely)).T  # back to [ely, elx]
            dv = dv_filtered.reshape((nelx, nely)).T

            # Sensitivity stabilization: average with previous iteration
            if dc_old is not None:
                dc = 0.5 * (dc + dc_old)
            dc_old = dc.copy()

            # OC update with bisection
            l1, l2 = 0.0, 1e9
            while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-8:
                lmid = 0.5 * (l2 + l1)
                Be = (-dc / (dv * lmid + 1e-30)) ** eta
                x_new = np.maximum(rho_min,
                            np.maximum(x - move,
                                np.minimum(1.0,
                                    np.minimum(x + move, x * Be))))
                if np.mean(x_new) - volfrac > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            change = np.max(np.abs(x_new - x))
            x = x_new

            # Track best at final penalization
            if is_final and compliance < best_compliance:
                best_compliance = compliance
                best_x = x.copy()

            if is_final:
                compliance_history.append(compliance)

            # Relative compliance change
            rel_change = abs(compliance - prev_compliance) / (abs(prev_compliance) + 1e-30)
            prev_compliance = compliance

            if global_iter % 20 == 0 or (is_final and change < 0.01):
                print(f"  Iter {global_iter:3d} (p={penal:.1f}): c = {compliance:.4f}, "
                      f"vol = {np.mean(x):.4f}, ch = {change:.4f}")

            # Convergence check
            if not is_final:
                if change < 0.01 and iteration > 5:
                    converge_count += 1
                    if converge_count >= 3:
                        print(f"  Converged at stage p={penal:.2f}, iter {iteration+1}")
                        break
                else:
                    converge_count = 0
            else:
                if change < 0.002:
                    converge_count += 1
                    if converge_count >= 12:
                        print(f"  Converged at stage p={penal:.1f}, iter {iteration+1}")
                        break
                else:
                    converge_count = 0
                # Stagnation detection
                if len(compliance_history) > 60:
                    recent_best = min(compliance_history[-30:])
                    older_best = min(compliance_history[-60:-30])
                    if abs(recent_best - older_best) / (abs(older_best) + 1e-30) < 1e-4:
                        print(f"  Stagnated at stage p={penal:.1f}, iter {global_iter}")
                        break

    print(f"  Best compliance: {best_compliance:.4f}")
    return best_x


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
    print("Optimizing using Optimality Criteria (OC) method...")
    density = optimize_topology(nelx, nely, config, max_iter=400)

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