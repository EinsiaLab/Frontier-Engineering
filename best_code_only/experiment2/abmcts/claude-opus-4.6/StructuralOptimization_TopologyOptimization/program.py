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
# FAST FEM SOLVER AND COMPLIANCE (vectorized versions for optimization loop)
# ============================================================================

def _fast_fem_solve(nelx, nely, density, config, KE, edof_mat, free_dofs, n_dofs, F):
    """Vectorized FEM solver for use in optimization loop."""
    E0 = config["E0"]
    Emin = config["Emin"]
    penal = config["penal"]

    dens_flat = density.T.flatten()
    Ee = Emin + dens_flat**penal * (E0 - Emin)

    n_elem = nelx * nely
    # Build COO arrays vectorized
    iK = np.repeat(edof_mat, 8, axis=1).flatten()
    jK = np.tile(edof_mat, (1, 8)).flatten()
    sK = (Ee[:, None, None] * KE[None, :, :]).flatten()

    K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs)).tocsc()

    K_ff = K[free_dofs, :][:, free_dofs]
    F_f = F[free_dofs]

    u = np.zeros(n_dofs)
    u[free_dofs] = spsolve(K_ff, F_f)
    return u


def _fast_compliance(nelx, nely, density, u, config, KE, edof_mat):
    """Vectorized compliance and sensitivity computation."""
    E0 = config["E0"]
    Emin = config["Emin"]
    penal = config["penal"]

    dens_flat = density.T.flatten()
    Ee = Emin + dens_flat**penal * (E0 - Emin)

    ue = u[edof_mat]
    ce = np.sum((ue @ KE) * ue, axis=1)

    compliance = float(np.sum(Ee * ce))

    dEe = -penal * dens_flat**(penal - 1) * (E0 - Emin)
    dc_flat = dEe * ce

    dc = dc_flat.reshape(nelx, nely).T

    return compliance, dc


# ============================================================================
# OPTIMIZATION ALGORITHM (ALLOWED TO MODIFY - This is your optimization code)
# ============================================================================

def optimize_topology(nelx, nely, config, max_iter=200):
    """OC method for topology optimization with continuation. ALLOWED TO MODIFY."""
    volfrac = config["volfrac"]
    rmin = config["rmin"]
    nu = config["nu"]
    penal_target = config["penal"]
    rho_min = 1e-3

    KE = _element_stiffness_matrix(nu)

    # Precompute edof_mat
    n_elem = nelx * nely
    n_dofs = 2 * (nelx + 1) * (nely + 1)
    
    elx_arr = np.repeat(np.arange(nelx), nely)
    ely_arr = np.tile(np.arange(nely), nelx)
    n1 = elx_arr * (nely + 1) + ely_arr
    n2 = (elx_arr + 1) * (nely + 1) + ely_arr
    edof_mat = np.column_stack([
        2*n1, 2*n1+1, 2*n2, 2*n2+1,
        2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
    ])

    # Precompute boundary conditions
    fixed_dofs = list(range(0, 2*(nely+1), 2))
    fixed_dofs.append(2 * (nelx * (nely + 1) + nely) + 1)
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Force vector
    F = np.zeros(n_dofs)
    F[1] = config["force"]["fy"]

    # Precompute filter as sparse matrix for fast application
    r_ceil = int(np.ceil(rmin))
    filt_rows = []
    filt_cols = []
    filt_vals = []
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j
            k_min = max(i - r_ceil, 0)
            k_max = min(i + r_ceil + 1, nelx)
            l_min = max(j - r_ceil, 0)
            l_max = min(j + r_ceil + 1, nely)
            for k in range(k_min, k_max):
                for l in range(l_min, l_max):
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    if fac > 0:
                        e2 = k * nely + l
                        filt_rows.append(e1)
                        filt_cols.append(e2)
                        filt_vals.append(fac)
    
    filt_rows = np.array(filt_rows, dtype=int)
    filt_cols = np.array(filt_cols, dtype=int)
    filt_vals = np.array(filt_vals, dtype=float)
    H = coo_matrix((filt_vals, (filt_rows, filt_cols)), shape=(n_elem, n_elem)).tocsr()

    def apply_filter_sparse(x, dc):
        # x is (nely, nelx), dc is (nely, nelx)
        # Element ordering: elx*nely+ely (column-major in x)
        x_flat = x.T.flatten()  # (n_elem,)
        dc_flat = dc.T.flatten()  # (n_elem,)
        
        numerator = H.dot(x_flat * dc_flat)
        denominator = H.dot(x_flat)
        
        dc_new_flat = np.where(denominator > 0, numerator / denominator, 0.0)
        dc_new = dc_new_flat.reshape(nelx, nely).T
        return dc_new

    # Initialize uniform density
    x = np.full((nely, nelx), volfrac)
    move = 0.2

    best_x = x.copy()
    best_compliance = np.inf

    # Use continuation: start with lower penal and increase
    penal_schedule = []
    if penal_target >= 3.0:
        # Start at penal=1.5, ramp to 3.0
        penal_schedule.append((1.0, 30))    # penal=1.0 for warmup
        penal_schedule.append((2.0, 40))    # penal=2.0
        penal_schedule.append((3.0, 130))   # penal=3.0 for final
    else:
        penal_schedule.append((penal_target, max_iter))

    total_iter = 0
    for penal_val, n_iters in penal_schedule:
        config_local = dict(config)
        config_local["penal"] = penal_val
        
        for iteration in range(n_iters):
            if total_iter >= max_iter:
                break
            total_iter += 1

            u = _fast_fem_solve(nelx, nely, x, config_local, KE, edof_mat, free_dofs, n_dofs, F)
            compliance, dc = _fast_compliance(nelx, nely, x, u, config_local, KE, edof_mat)

            # Track best with actual penal
            if penal_val == penal_target and compliance < best_compliance and np.mean(x) <= volfrac + 1e-6:
                best_compliance = compliance
                best_x = x.copy()

            dc = apply_filter_sparse(x, dc)

            # OC update
            l1, l2 = 0.0, 1e9
            while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                Be = np.sqrt(-dc / (lmid + 1e-30))
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

            if total_iter % 10 == 0 or change < 0.01:
                print(f"  Iter {total_iter:3d} (p={penal_val:.1f}): c = {compliance:.4f}, "
                      f"vol = {np.mean(x):.4f}, change = {change:.4f}")

            if change < 0.005 and iteration > 15 and penal_val == penal_target:
                print(f"  Converged at iteration {total_iter}")
                break

    # Final evaluation with target penal
    config_final = dict(config)
    config_final["penal"] = penal_target
    u = _fast_fem_solve(nelx, nely, x, config_final, KE, edof_mat, free_dofs, n_dofs, F)
    final_c, _ = _fast_compliance(nelx, nely, x, u, config_final, KE, edof_mat)
    if final_c < best_compliance and np.mean(x) <= volfrac + 1e-6:
        best_x = x.copy()
        best_compliance = final_c

    print(f"  Best compliance found: {best_compliance:.4f}")
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

    print("Optimizing using Optimality Criteria (OC) method with continuation...")
    density = optimize_topology(nelx, nely, config, max_iter=200)

    # Compute final compliance using the original (evaluator-matching) functions
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
