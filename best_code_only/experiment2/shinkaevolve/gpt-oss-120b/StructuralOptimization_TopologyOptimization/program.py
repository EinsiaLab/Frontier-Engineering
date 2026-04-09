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

def optimize_topology(nelx, nely, config, max_iter=150):
    """Three-phase OC optimizer with momentum and adaptive damping. ALLOWED TO MODIFY."""
    volfrac = config["volfrac"]
    rmin = config["rmin"]
    target_penal = config["penal"]
    rho_min = 1e-3

    # Initialize density field
    x = np.full((nely, nelx), volfrac)
    velocity = np.zeros_like(x)

    # Best solution tracking
    best_x = x.copy()
    best_compliance = float('inf')

    # Lagrange multiplier bounds
    l_lower, l_upper = 0.0, 1e9

    current_config = config.copy()
    total_iter = 0

    # ========== Phase 1: Exploration with momentum (rapid topology discovery) ==========
    penal_schedule = [1.0, 1.5, 2.0, 2.5, 3.0]
    penal_idx = 0
    move = 0.2
    momentum = 0.5

    for phase_iter in range(35):
        if total_iter >= max_iter:
            break
        total_iter += 1

        # Update penalization every 7 iterations (faster schedule)
        if penal_idx < len(penal_schedule) - 1 and phase_iter > 0 and phase_iter % 7 == 0:
            penal_idx += 1
        penal = penal_schedule[penal_idx]
        current_config["penal"] = penal

        # FEM solve
        u = fem_solve_2d_quad(nelx, nely, x, current_config)
        compliance, dc = compute_compliance(nelx, nely, x, u, current_config)
        dc = apply_density_filter(nelx, nely, rmin, x, dc)

        # Track best at target penal
        if penal >= target_penal and compliance < best_compliance:
            best_compliance = compliance
            best_x = x.copy()

        # OC update with bisection
        l1, l2 = l_lower, l_upper
        x_oc = np.zeros_like(x)

        for _ in range(30):
            if (l2 - l1) / (l1 + l2 + 1e-40) < 1e-6:
                break
            lmid = 0.5 * (l2 + l1)
            Be = np.sqrt(np.maximum(-dc, 1e-40) / (lmid + 1e-40))
            x_oc = np.maximum(rho_min,
                        np.maximum(x - move,
                            np.minimum(1.0, np.minimum(x + move, x * Be))))
            if np.mean(x_oc) - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid

        l_lower = max(0.0, l1 * 0.9)
        l_upper = min(1e12, l2 * 1.1)

        # Momentum update
        raw_update = x_oc - x
        velocity = momentum * velocity + (1 - momentum) * raw_update
        x_new = x + velocity
        x_new = np.clip(x_new, rho_min, 1.0)

        # Volume correction
        vol_error = np.mean(x_new) - volfrac
        if abs(vol_error) > 1e-10:
            x_new = np.clip(x_new * (volfrac / np.mean(x_new)), rho_min, 1.0)

        change = np.max(np.abs(x_new - x))
        x = x_new

        print(f"  Iter {total_iter:3d} [explore]: compliance = {compliance:.4f}, "
              f"vol = {np.mean(x):.4f}, change = {change:.4f}, penal = {penal:.1f}")

        # Early transition if stable at target penal
        if change < 0.025 and penal >= target_penal:
            break

    # ========== Phase 2: Refinement with damping ==========
    penal = target_penal
    current_config["penal"] = penal
    move = 0.12
    damping = 0.25
    dc_prev = None

    for phase_iter in range(75):
        if total_iter >= max_iter:
            break
        total_iter += 1

        # FEM solve
        u = fem_solve_2d_quad(nelx, nely, x, current_config)
        compliance, dc = compute_compliance(nelx, nely, x, u, current_config)
        dc = apply_density_filter(nelx, nely, rmin, x, dc)

        # Sensitivity damping for stability
        if dc_prev is not None:
            dc = (1 - damping) * dc + damping * dc_prev
        dc_prev = dc.copy()

        # Track best
        if compliance < best_compliance:
            best_compliance = compliance
            best_x = x.copy()

        # Adaptive move limit reduction
        if phase_iter > 25:
            move = max(0.04, move * 0.985)

        # OC update
        l1, l2 = l_lower, l_upper
        x_new = np.zeros_like(x)

        for _ in range(30):
            if (l2 - l1) / (l1 + l2 + 1e-40) < 1e-6:
                break
            lmid = 0.5 * (l2 + l1)
            Be = np.sqrt(np.maximum(-dc, 1e-40) / (lmid + 1e-40))
            x_new = np.maximum(rho_min,
                        np.maximum(x - move,
                            np.minimum(1.0, np.minimum(x + move, x * Be))))
            if np.mean(x_new) - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid

        l_lower = max(0.0, l1 * 0.9)
        l_upper = min(1e12, l2 * 1.1)

        # Volume correction
        vol_error = np.mean(x_new) - volfrac
        if abs(vol_error) > 1e-10:
            x_new = np.clip(x_new * (volfrac / np.mean(x_new)), rho_min, 1.0)

        change = np.max(np.abs(x_new - x))
        x = x_new

        print(f"  Iter {total_iter:3d} [refine ]: compliance = {compliance:.4f}, "
              f"vol = {np.mean(x):.4f}, change = {change:.4f}, penal = {penal:.1f}")

        # Convergence check
        if change < 0.002 and phase_iter > 35:
            break

    # ========== Phase 3: Polish ==========
    move = 0.02

    while total_iter < max_iter:
        total_iter += 1

        u = fem_solve_2d_quad(nelx, nely, x, current_config)
        compliance, dc = compute_compliance(nelx, nely, x, u, current_config)
        dc = apply_density_filter(nelx, nely, rmin, x, dc)

        if compliance < best_compliance:
            best_compliance = compliance
            best_x = x.copy()

        l1, l2 = 0.0, 1e9
        x_new = np.zeros_like(x)

        for _ in range(30):
            if (l2 - l1) / (l1 + l2 + 1e-40) < 1e-6:
                break
            lmid = 0.5 * (l2 + l1)
            Be = np.sqrt(np.maximum(-dc, 1e-40) / (lmid + 1e-40))
            x_new = np.maximum(rho_min,
                        np.maximum(x - move,
                            np.minimum(1.0, np.minimum(x + move, x * Be))))
            if np.mean(x_new) - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid

        vol_error = np.mean(x_new) - volfrac
        if abs(vol_error) > 1e-10:
            x_new = np.clip(x_new * (volfrac / np.mean(x_new)), rho_min, 1.0)

        change = np.max(np.abs(x_new - x))
        x = x_new

        print(f"  Iter {total_iter:3d} [polish ]: compliance = {compliance:.4f}, "
              f"vol = {np.mean(x):.4f}, change = {change:.4f}, penal = {penal:.1f}")

        if change < 0.001:
            break

    # Return best solution found
    if best_compliance < float('inf'):
        final_u = fem_solve_2d_quad(nelx, nely, best_x, config)
        final_compliance, _ = compute_compliance(nelx, nely, best_x, final_u, config)
        u_curr = fem_solve_2d_quad(nelx, nely, x, config)
        curr_compliance, _ = compute_compliance(nelx, nely, x, u_curr, config)

        if curr_compliance <= final_compliance:
            print(f"  Returning current solution with compliance = {curr_compliance:.4f}")
            return x
        else:
            print(f"  Returning best solution with compliance = {final_compliance:.4f}")
            return best_x

    return x


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
    print("Optimizing using Adaptive Continuation OC...")
    density = optimize_topology(nelx, nely, config, max_iter=150)

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