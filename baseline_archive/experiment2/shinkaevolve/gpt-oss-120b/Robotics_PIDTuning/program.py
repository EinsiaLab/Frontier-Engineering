# EVOLVE-BLOCK-START
"""CMA-ES optimizer for 2D quadrotor PID tuning.

DO NOT MODIFY: load_config(), simulate_quadrotor_2d(), compute_itae()
ALLOWED TO MODIFY: optimize_pid_gains()

Outputs submission.json with tuned PID gains.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# DO NOT MODIFY — Configuration loader
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Load pid_config.json from references/."""
    candidates = [
        Path(__file__).resolve().parent / "references" / "pid_config.json",
        Path(__file__).resolve().parent.parent / "references" / "pid_config.json",
    ]
    for p in candidates:
        if p.is_file():
            with p.open("r", encoding="utf-8-sig") as f:
                return json.load(f)
    raise FileNotFoundError("pid_config.json not found")


# ---------------------------------------------------------------------------
# DO NOT MODIFY — 2D Quadrotor Simulation
# ---------------------------------------------------------------------------

def simulate_quadrotor_2d(
    gains: dict[str, float],
    scenario: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Simulate a 2D quadrotor with cascaded PID control.

    States: [x, z, theta, x_dot, z_dot, theta_dot]
    Actuators: total thrust T, torque tau (with 1st-order motor lag).

    Returns dict with 'itae', 'feasible'.
    """
    quad = cfg["quadrotor"]
    cons = cfg["constraints"]
    sim = cfg["sim"]

    m = quad["mass"]
    I = quad["inertia"]
    g = quad["gravity"]
    drag = quad["drag"]
    angular_drag = quad.get("angular_drag", 0.0)
    tau_motor = quad["motor_time_constant"]
    dt = sim["dt"]
    switch_r = sim["waypoint_switch_radius"]

    max_pitch = cons["max_pitch_rad"]
    max_thrust = cons["max_thrust_factor"] * m * g
    min_z = cons["min_altitude"]

    duration = scenario["duration"]
    wind = np.array(scenario["wind"], dtype=float)
    waypoints = [np.array(wp, dtype=float) for wp in scenario["waypoints"]]
    start = np.array(scenario["start"], dtype=float)

    n_steps = int(duration / dt)

    # State
    x, z = float(start[0]), float(start[1])
    theta = 0.0
    x_dot, z_dot, theta_dot = 0.0, 0.0, 0.0

    # Motor commands (with lag)
    T_cmd, tau_cmd = m * g, 0.0
    T_act, tau_act = m * g, 0.0

    # PID integrators & filtered derivatives
    int_z, int_x, int_theta = 0.0, 0.0, 0.0
    df_z, df_x, df_theta = 0.0, 0.0, 0.0

    wp_idx = 0
    target = waypoints[wp_idx]

    # Initialize previous errors to initial values (avoid derivative kick)
    prev_ez = target[1] - z
    prev_ex = target[0] - x
    prev_etheta = 0.0

    itae = 0.0
    feasible = True

    for i in range(n_steps):
        t = i * dt

        # Waypoint switching
        if wp_idx < len(waypoints) - 1:
            dist_to_wp = math.sqrt((x - target[0]) ** 2 + (z - target[1]) ** 2)
            if dist_to_wp < switch_r:
                wp_idx += 1
                target = waypoints[wp_idx]

        # --- Altitude PID ---
        ez = target[1] - z
        int_z += ez * dt
        raw_dez = (ez - prev_ez) / dt
        alpha_z = dt * gains["N_z"] / (1.0 + dt * gains["N_z"])
        df_z = alpha_z * raw_dez + (1.0 - alpha_z) * df_z
        prev_ez = ez
        thrust_offset = gains["Kp_z"] * ez + gains["Ki_z"] * int_z + gains["Kd_z"] * df_z

        cos_theta = math.cos(theta)
        if abs(cos_theta) > 1e-6:
            T_cmd = (m * g + thrust_offset) / cos_theta
        else:
            T_cmd = max_thrust

        # --- Horizontal PID ---
        ex = target[0] - x
        int_x += ex * dt
        raw_dex = (ex - prev_ex) / dt
        alpha_x = dt * gains["N_x"] / (1.0 + dt * gains["N_x"])
        df_x = alpha_x * raw_dex + (1.0 - alpha_x) * df_x
        prev_ex = ex
        desired_pitch = -(gains["Kp_x"] * ex + gains["Ki_x"] * int_x + gains["Kd_x"] * df_x)
        desired_pitch = np.clip(desired_pitch, -max_pitch, max_pitch)

        # --- Pitch PID ---
        etheta = desired_pitch - theta
        int_theta += etheta * dt
        raw_detheta = (etheta - prev_etheta) / dt
        alpha_theta = dt * gains["N_theta"] / (1.0 + dt * gains["N_theta"])
        df_theta = alpha_theta * raw_detheta + (1.0 - alpha_theta) * df_theta
        prev_etheta = etheta
        tau_cmd = gains["Kp_theta"] * etheta + gains["Ki_theta"] * int_theta + gains["Kd_theta"] * df_theta

        # Clamp thrust
        T_cmd = float(np.clip(T_cmd, 0.0, max_thrust))

        # Motor lag (1st-order)
        alpha_m = dt / (tau_motor + dt)
        T_act = T_act + alpha_m * (T_cmd - T_act)
        tau_act = tau_act + alpha_m * (tau_cmd - tau_act)

        # Physics
        ax = -(T_act / m) * math.sin(theta) - drag * x_dot + wind[0]
        az = (T_act / m) * math.cos(theta) - g - drag * z_dot + wind[1]
        atheta = tau_act / I - angular_drag * theta_dot

        x_dot += ax * dt
        z_dot += az * dt
        theta_dot += atheta * dt
        x += x_dot * dt
        z += z_dot * dt
        theta += theta_dot * dt

        # Constraints
        if abs(theta) > max_pitch:
            feasible = False
            break
        if z < min_z:
            z = min_z
            z_dot = max(z_dot, 0.0)

        # ITAE accumulation
        pos_err = math.sqrt(ex ** 2 + ez ** 2)
        itae += t * pos_err * dt

    return {"itae": itae, "feasible": feasible}


# ---------------------------------------------------------------------------
# DO NOT MODIFY — Scoring
# ---------------------------------------------------------------------------

def compute_itae(gains: dict[str, float], cfg: dict[str, Any]) -> float:
    """Run all scenarios and return combined score (geometric mean of 1/ITAE).

    Returns 0.0 if any scenario is infeasible or ITAE is non-positive.
    """
    scenarios = cfg["scenarios"]
    inv_itaes: list[float] = []

    for sc in scenarios:
        result = simulate_quadrotor_2d(gains, sc, cfg)
        if not result["feasible"]:
            return 0.0
        itae = result["itae"]
        if itae <= 0.0:
            return 0.0
        inv_itaes.append(1.0 / itae)

    log_sum = sum(math.log(v) for v in inv_itaes)
    return float(math.exp(log_sum / len(inv_itaes)))


# ---------------------------------------------------------------------------
# ALLOWED TO MODIFY — Optimizer
# ---------------------------------------------------------------------------

def optimize_pid_gains() -> dict[str, float]:
    """Optimize PID gains using CMA-ES with local refinement.

    Returns the best gain dict found.
    """
    cfg = load_config()
    gain_ranges = cfg["gains"]
    rng = np.random.default_rng(42)

    keys_order = [
        ("altitude", "Kp", "Kp_z"), ("altitude", "Ki", "Ki_z"),
        ("altitude", "Kd", "Kd_z"), ("altitude", "N", "N_z"),
        ("horizontal", "Kp", "Kp_x"), ("horizontal", "Ki", "Ki_x"),
        ("horizontal", "Kd", "Kd_x"), ("horizontal", "N", "N_x"),
        ("pitch", "Kp", "Kp_theta"), ("pitch", "Ki", "Ki_theta"),
        ("pitch", "Kd", "Kd_theta"), ("pitch", "N", "N_theta"),
    ]

    # Extract bounds
    bounds = np.array([[gain_ranges[g][p][0], gain_ranges[g][p][1]]
                       for g, p, _ in keys_order])
    n_dims = len(keys_order)

    def vec_to_gains(vec: np.ndarray) -> dict[str, float]:
        return {key: float(vec[i]) for i, (_, _, key) in enumerate(keys_order)}

    # Evaluation cache to avoid redundant computations
    _eval_cache = {}
    _cache_hits = 0

    def objective(vec: np.ndarray) -> float:
        nonlocal _cache_hits
        key = tuple(round(v, 8) for v in vec)
        if key in _eval_cache:
            _cache_hits += 1
            return _eval_cache[key]
        result = compute_itae(vec_to_gains(vec), cfg)
        _eval_cache[key] = result
        return result

    # Known good solutions as seeds - updated with best found solutions
    # Best solution (score 0.163480): Kp_z=30, Kd_z~12.07, N_z~14.1, Kp_x~2.72, Kd_x~1.60, Kp_theta~6.58, Ki_theta~0.146
    seeds = [
        # Best known solution (score 0.163480) - from Nelder-Mead optimization
        [30.0, 0.0, 12.07, 14.12, 2.72, 0.0, 1.60, 100.0, 6.58, 0.146, 4.26, 100.0],
        # Variants around best solution
        [30.0, 0.0, 12.1, 14.0, 2.7, 0.0, 1.6, 100.0, 6.5, 0.15, 4.3, 100.0],
        [30.0, 0.0, 12.0, 14.5, 2.75, 0.0, 1.55, 100.0, 6.6, 0.14, 4.25, 100.0],
        [30.0, 0.0, 12.15, 13.5, 2.65, 0.0, 1.65, 100.0, 6.7, 0.16, 4.2, 100.0],
        [30.0, 0.0, 11.9, 15.0, 2.8, 0.0, 1.5, 100.0, 6.4, 0.17, 4.35, 100.0],
        # Exploration with different Kp_theta values
        [30.0, 0.0, 12.0, 14.0, 2.5, 0.0, 1.7, 100.0, 7.0, 0.13, 4.0, 100.0],
        [30.0, 0.0, 12.2, 13.0, 2.9, 0.0, 1.4, 100.0, 6.0, 0.18, 4.5, 100.0],
        [30.0, 0.0, 11.8, 16.0, 2.6, 0.0, 1.8, 100.0, 6.8, 0.12, 4.1, 100.0],
    ]

    # Evaluate seeds and find best
    best_vec = None
    best_score = 0.0
    for seed in seeds:
        vec = np.array(seed, dtype=float)
        for d in range(n_dims):
            vec[d] = np.clip(vec[d], bounds[d, 0], bounds[d, 1])
        score = objective(vec)
        if score > best_score:
            best_score = score
            best_vec = vec.copy()

    print(f"Best seed score: {best_score:.6f}")

    # Phase 0: Coarse grid search on key parameters
    print("Phase 0: Coarse grid search on key parameters...")

    # Grid search on Kp_z and Kd_z (most impactful altitude params) - focus near optimal
    # Kp_z optimal at 30.0 (upper bound), Kd_z optimal around 12.4
    kp_z_range = np.concatenate([
        np.linspace(28.0, 30.0, 11),  # Dense sampling at high end
    ])
    kd_z_range = np.linspace(11.0, 14.0, 16)  # Focus around optimal

    for kp_z in kp_z_range:
        for kd_z in kd_z_range:
            trial = best_vec.copy()
            trial[0] = kp_z
            trial[2] = kd_z
            score = objective(trial)
            if score > best_score:
                best_score = score
                best_vec = trial.copy()
                print(f"  Grid (Kp_z={kp_z:.1f}, Kd_z={kd_z:.1f}): {best_score:.6f}")

    # Grid search on Kp_x and Kd_x (horizontal params) - focus near optimal (Kp_x~2.7, Kd_x~1.6)
    kp_x_range = np.linspace(1.5, 4.5, 16)
    kd_x_range = np.linspace(0.8, 2.5, 21)

    for kp_x in kp_x_range:
        for kd_x in kd_x_range:
            trial = best_vec.copy()
            trial[4] = kp_x
            trial[6] = kd_x
            score = objective(trial)
            if score > best_score:
                best_score = score
                best_vec = trial.copy()
                print(f"  Grid (Kp_x={kp_x:.1f}, Kd_x={kd_x:.1f}): {best_score:.6f}")

    # CMA-ES implementation
    def cmaes_optimize(x0: np.ndarray, sigma0: float, max_evals: int) -> tuple[np.ndarray, float]:
        """Simple CMA-ES implementation with bounds handling."""
        n = len(x0)
        lam = 4 + int(3 * np.log(n))  # Population size
        mu = lam // 2  # Number of parents

        # Weights for recombination
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Initialize dynamic state
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        sigma = sigma0

        xmean = x0.copy()
        best_x = x0.copy()
        best_f = objective(x0)

        evals = 0
        stagnation = 0
        prev_best = best_f

        while evals < max_evals:
            # Generate population
            population = []
            fitness = []

            for _ in range(lam):
                # Sample from multivariate normal
                try:
                    z = rng.standard_normal(n)
                    x = xmean + sigma * np.dot(np.linalg.cholesky(C), z)
                except np.linalg.LinAlgError:
                    x = xmean + sigma * rng.standard_normal(n)

                # Clip to bounds
                x = np.clip(x, bounds[:, 0], bounds[:, 1])
                f = objective(x)
                evals += 1

                population.append(x)
                fitness.append(f)

                if f > best_f:
                    best_f = f
                    best_x = x.copy()

            # Sort by fitness (descending for maximization)
            order = np.argsort(-np.array(fitness))

            # Select parents
            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * population[order[i]]

            # Update evolution paths
            try:
                Cinv = np.linalg.inv(C)
                zmean = np.dot(np.linalg.cholesky(Cinv), (xmean - xold) / sigma)
            except np.linalg.LinAlgError:
                zmean = (xmean - xold) / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * evals / lam)) < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            # Update covariance matrix
            artmp = np.array([(population[order[i]] - xold) / sigma for i in range(mu)])
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * np.sum([weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu)], axis=0)

            # Update step size
            sigma = sigma * np.exp((np.linalg.norm(ps) / np.sqrt(n) - 1) * cs / damps)
            sigma = max(1e-10, min(sigma, 1.0))

            # Check stagnation
            if best_f - prev_best < 1e-8:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best_f

            # Restart if stagnated
            if stagnation > 20:
                sigma = sigma0 * 0.5
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                stagnation = 0

            if evals % 100 == 0 and best_f > 0:
                print(f"  CMA-ES evals {evals}: best = {best_f:.6f}")

        return best_x, best_f

    # Skip CMA-ES - local search is more effective for this problem
    # Focus computational budget on targeted local refinement
    print(f"Starting from best seed: {best_score:.6f}")

    # Priority: Altitude coupled search first (Kp_z, Kd_z) - most impactful
    print("Priority: Altitude coupled search (Kp_z, Kd_z) with aggressive scales...")
    d1, d2 = 0, 2  # Kp_z, Kd_z
    altitude_scales = [0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
    for scale in altitude_scales:
        improved = True
        while improved:
            improved = False
            for s1 in [-scale, -scale/2, 0.0, scale/2, scale]:
                for s2 in [-scale, -scale/2, 0.0, scale/2, scale]:
                    if s1 == 0 and s2 == 0:
                        continue
                    trial = best_vec.copy()
                    trial[d1] = np.clip(best_vec[d1] + s1 * (bounds[d1, 1] - bounds[d1, 0]),
                                        bounds[d1, 0], bounds[d1, 1])
                    trial[d2] = np.clip(best_vec[d2] + s2 * (bounds[d2, 1] - bounds[d2, 0]),
                                        bounds[d2, 0], bounds[d2, 1])
                    trial_score = objective(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_vec = trial.copy()
                        improved = True
                        print(f"  Altitude coupled (scale={scale:.3f}): {best_score:.6f}")

    # Intensive local search with multiple scales
    print("Starting local refinement...")
    scales = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    for scale in scales:
        improved = True
        while improved:
            improved = False
            for d in range(n_dims):
                for direction in [-1, 1]:
                    trial = best_vec.copy()
                    step = direction * scale * (bounds[d, 1] - bounds[d, 0])
                    trial[d] = np.clip(best_vec[d] + step, bounds[d, 0], bounds[d, 1])
                    trial_score = objective(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_vec = trial.copy()
                        improved = True
                        print(f"  Local ({scale:.4f}): {best_score:.6f}")

    # 2D coupled search for related parameters
    print("Starting coupled parameter search...")
    coupled_pairs = [
        (0, 2),   # Kp_z, Kd_z
        (0, 3),   # Kp_z, N_z
        (2, 3),   # Kd_z, N_z
        (4, 6),   # Kp_x, Kd_x
        (6, 7),   # Kd_x, N_x
        (8, 10),  # Kp_theta, Kd_theta
        (10, 11), # Kd_theta, N_theta
        (0, 4),   # Kp_z, Kp_x
        (2, 6),   # Kd_z, Kd_x
    ]

    for d1, d2 in coupled_pairs:
        improved = True
        while improved:
            improved = False
            for s1 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                for s2 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    if s1 == 0 and s2 == 0:
                        continue
                    trial = best_vec.copy()
                    trial[d1] = np.clip(best_vec[d1] + s1 * (bounds[d1, 1] - bounds[d1, 0]),
                                        bounds[d1, 0], bounds[d1, 1])
                    trial[d2] = np.clip(best_vec[d2] + s2 * (bounds[d2, 1] - bounds[d2, 0]),
                                        bounds[d2, 0], bounds[d2, 1])
                    trial_score = objective(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_vec = trial.copy()
                        improved = True
                        print(f"  Coupled ({d1},{d2}): {best_score:.6f}")

    # Ki_theta focused search - optimal is around 0.1-0.15 based on best solutions
    print("Ki_theta focused search...")
    ki_theta_idx = 9
    # Fine grid focusing on moderate values where optimum lies
    ki_theta_range = np.concatenate([
        [0.0],
        np.linspace(0.05, 0.25, 41)  # Dense sampling in optimal region
    ])
    for ki_val in ki_theta_range:
        trial = best_vec.copy()
        trial[ki_theta_idx] = ki_val
        trial_score = objective(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_vec = trial.copy()
            print(f"  Ki_theta={ki_val:.4f}: {best_score:.6f}")

    # N_z focused search
    print("N_z focused search...")
    n_z_idx = 3
    n_z_range = np.linspace(5.0, 20.0, 31)
    for n_val in n_z_range:
        trial = best_vec.copy()
        trial[n_z_idx] = n_val
        trial_score = objective(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_vec = trial.copy()
            print(f"  N_z={n_val:.4f}: {best_score:.6f}")

    # Kp_theta focused search
    print("Kp_theta focused search...")
    kp_theta_idx = 8
    kp_theta_range = np.linspace(2.0, 6.0, 41)
    for kp_val in kp_theta_range:
        trial = best_vec.copy()
        trial[kp_theta_idx] = kp_val
        trial_score = objective(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_vec = trial.copy()
            print(f"  Kp_theta={kp_val:.4f}: {best_score:.6f}")

    # Kd_theta focused search
    print("Kd_theta focused search...")
    kd_theta_idx = 10
    kd_theta_range = np.linspace(2.0, 6.0, 41)
    for kd_val in kd_theta_range:
        trial = best_vec.copy()
        trial[kd_theta_idx] = kd_val
        trial_score = objective(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_vec = trial.copy()
            print(f"  Kd_theta={kd_val:.4f}: {best_score:.6f}")

    # 3D coupled search for the most important parameter triplets
    print("3D coupled parameter search...")
    triplets = [
        (0, 2, 9),   # Kp_z, Kd_z, Ki_theta
        (0, 3, 9),   # Kp_z, N_z, Ki_theta
        (4, 6, 9),   # Kp_x, Kd_x, Ki_theta
        (8, 9, 10),  # Kp_theta, Ki_theta, Kd_theta
        (0, 4, 9),   # Kp_z, Kp_x, Ki_theta
        (2, 6, 9),   # Kd_z, Kd_x, Ki_theta
    ]

    for d1, d2, d3 in triplets:
        improved_3d = True
        while improved_3d:
            improved_3d = False
            for s1 in [-0.01, 0.0, 0.01]:
                for s2 in [-0.01, 0.0, 0.01]:
                    for s3 in [-0.01, 0.0, 0.01]:
                        if s1 == 0 and s2 == 0 and s3 == 0:
                            continue
                        trial = best_vec.copy()
                        trial[d1] = np.clip(best_vec[d1] + s1 * (bounds[d1, 1] - bounds[d1, 0]),
                                            bounds[d1, 0], bounds[d1, 1])
                        trial[d2] = np.clip(best_vec[d2] + s2 * (bounds[d2, 1] - bounds[d2, 0]),
                                            bounds[d2, 0], bounds[d2, 1])
                        trial[d3] = np.clip(best_vec[d3] + s3 * (bounds[d3, 1] - bounds[d3, 0]),
                                            bounds[d3, 0], bounds[d3, 1])
                        trial_score = objective(trial)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_vec = trial.copy()
                            improved_3d = True
                            print(f"  3D ({d1},{d2},{d3}): {best_score:.6f}")

    # 4D coupled search for the most important parameter combinations
    print("4D coupled parameter search...")
    quads = [
        (0, 2, 4, 9),   # Kp_z, Kd_z, Kp_x, Ki_theta
        (0, 2, 6, 9),   # Kp_z, Kd_z, Kd_x, Ki_theta
        (0, 4, 6, 9),   # Kp_z, Kp_x, Kd_x, Ki_theta
        (2, 4, 6, 9),   # Kd_z, Kp_x, Kd_x, Ki_theta
    ]

    for d1, d2, d3, d4 in quads:
        improved_4d = True
        iter_count = 0
        while improved_4d and iter_count < 10:
            improved_4d = False
            iter_count += 1
            for s1 in [-0.005, 0.0, 0.005]:
                for s2 in [-0.005, 0.0, 0.005]:
                    for s3 in [-0.005, 0.0, 0.005]:
                        for s4 in [-0.005, 0.0, 0.005]:
                            if s1 == 0 and s2 == 0 and s3 == 0 and s4 == 0:
                                continue
                            trial = best_vec.copy()
                            trial[d1] = np.clip(best_vec[d1] + s1 * (bounds[d1, 1] - bounds[d1, 0]),
                                                bounds[d1, 0], bounds[d1, 1])
                            trial[d2] = np.clip(best_vec[d2] + s2 * (bounds[d2, 1] - bounds[d2, 0]),
                                                bounds[d2, 0], bounds[d2, 1])
                            trial[d3] = np.clip(best_vec[d3] + s3 * (bounds[d3, 1] - bounds[d3, 0]),
                                                bounds[d3, 0], bounds[d3, 1])
                            trial[d4] = np.clip(best_vec[d4] + s4 * (bounds[d4, 1] - bounds[d4, 0]),
                                                bounds[d4, 0], bounds[d4, 1])
                            trial_score = objective(trial)
                            if trial_score > best_score:
                                best_score = trial_score
                                best_vec = trial.copy()
                                improved_4d = True
                                print(f"  4D ({d1},{d2},{d3},{d4}): {best_score:.6f}")

    # Fine-grained search with very small steps - more iterations for better convergence
    print("Fine-grained refinement...")
    for iteration in range(5):
        improved = True
        while improved:
            improved = False
            for d in range(n_dims):
                # Use smaller scales in later iterations
                scales = [0.0003, 0.0005, 0.001, 0.002] if iteration < 3 else [0.0001, 0.0002, 0.0003, 0.0005]
                for scale in scales:
                    for direction in [-1, 1]:
                        trial = best_vec.copy()
                        step = direction * scale * (bounds[d, 1] - bounds[d, 0])
                        trial[d] = np.clip(best_vec[d] + step, bounds[d, 0], bounds[d, 1])
                        trial_score = objective(trial)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_vec = trial.copy()
                            improved = True
                            print(f"  Fine (iter={iteration}): {best_score:.6f}")

    # Nelder-Mead simplex optimization for fine-tuning
    print("Nelder-Mead simplex optimization...")
    from scipy.optimize import minimize

    # Only optimize the key parameters that have shown sensitivity
    # Fixed parameters: Ki_z=0, Ki_x=0, N_x=100, N_theta=100
    key_indices = [0, 2, 3, 4, 6, 8, 9, 10]  # Kp_z, Kd_z, N_z, Kp_x, Kd_x, Kp_theta, Ki_theta, Kd_theta
    key_bounds = bounds[key_indices]
    x0_key = best_vec[key_indices]

    def key_objective(x_key):
        x_full = best_vec.copy()
        x_full[key_indices] = x_key
        x_full = np.clip(x_full, bounds[:, 0], bounds[:, 1])
        return -objective(x_full)

    result = minimize(
        key_objective,
        x0_key,
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10, 'disp': False}
    )

    trial = best_vec.copy()
    trial[key_indices] = np.clip(result.x, key_bounds[:, 0], key_bounds[:, 1])
    trial_score = objective(trial)
    if trial_score > best_score:
        best_score = trial_score
        best_vec = trial.copy()
        print(f"  Nelder-Mead: {best_score:.6f}")

    # Second Nelder-Mead pass with perturbed starting point
    x0_perturbed = best_vec[key_indices] + rng.uniform(-0.02, 0.02, len(key_indices)) * (key_bounds[:, 1] - key_bounds[:, 0])
    x0_perturbed = np.clip(x0_perturbed, key_bounds[:, 0], key_bounds[:, 1])

    result2 = minimize(
        key_objective,
        x0_perturbed,
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-10, 'disp': False}
    )

    trial2 = best_vec.copy()
    trial2[key_indices] = np.clip(result2.x, key_bounds[:, 0], key_bounds[:, 1])
    trial_score2 = objective(trial2)
    if trial_score2 > best_score:
        best_score = trial_score2
        best_vec = trial2.copy()
        print(f"  Nelder-Mead (restart): {best_score:.6f}")

    # Phase: Explore near-fixed parameters to verify optimality
    print("Exploring near-fixed parameters...")

    # Try small variations of Ki_z and Ki_x
    for ki_z in [0.0, 0.001, 0.005, 0.01]:
        for ki_x in [0.0, 0.001, 0.005, 0.01]:
            trial = best_vec.copy()
            trial[1] = ki_z
            trial[5] = ki_x
            score = objective(trial)
            if score > best_score:
                best_score = score
                best_vec = trial.copy()
                print(f"  Ki variation (Ki_z={ki_z}, Ki_x={ki_x}): {best_score:.6f}")

    # Try variations of N_x and N_theta
    for n_x in [80.0, 90.0, 100.0]:
        for n_theta in [80.0, 90.0, 100.0]:
            trial = best_vec.copy()
            trial[7] = n_x
            trial[11] = n_theta
            score = objective(trial)
            if score > best_score:
                best_score = score
                best_vec = trial.copy()
                print(f"  N variation (N_x={n_x}, N_theta={n_theta}): {best_score:.6f}")

    print(f"Final best score: {best_score:.6f}")
    print(f"Cache hits: {_cache_hits}, unique evaluations: {len(_eval_cache)}")
    return vec_to_gains(best_vec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gains = optimize_pid_gains()
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(gains, f, indent=2)
    print("Submission written to submission.json")
    print(json.dumps(gains, indent=2))
# EVOLVE-BLOCK-END