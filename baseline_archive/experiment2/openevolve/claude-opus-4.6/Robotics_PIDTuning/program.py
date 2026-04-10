# EVOLVE-BLOCK-START
"""Baseline PID tuning optimizer for 2D quadrotor hover stabilization.

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
    """Optimize PID gains using CMA-ES style evolution strategy.

    Returns the best gain dict found.
    """
    import time
    start_time = time.time()
    TIME_BUDGET = 270.0  # seconds, leave margin

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

    dim = len(keys_order)
    lo_arr = np.array([gain_ranges[g][p][0] for g, p, _ in keys_order])
    hi_arr = np.array([gain_ranges[g][p][1] for g, p, _ in keys_order])
    range_arr = hi_arr - lo_arr

    def vec_to_gains(v: np.ndarray) -> dict[str, float]:
        v_clip = np.clip(v, lo_arr, hi_arr)
        return {key: float(v_clip[i]) for i, (_, _, key) in enumerate(keys_order)}

    def gains_to_vec(g: dict[str, float]) -> np.ndarray:
        return np.array([g[key] for _, _, key in keys_order])

    def evaluate(v: np.ndarray) -> float:
        return compute_itae(vec_to_gains(v), cfg)

    # Best known solutions from all previous optimization runs
    baselines = [
        # BEST EVER from last run (score 0.160043)
        {"Kp_z": 21.364, "Ki_z": 0.0, "Kd_z": 8.350, "N_z": 35.310,
         "Kp_x": 2.680, "Ki_x": 0.0, "Kd_x": 1.602, "N_x": 89.668,
         "Kp_theta": 6.260, "Ki_theta": 0.555, "Kd_theta": 4.269, "N_theta": 99.507},
        # Previous best (score ~0.1585)
        {"Kp_z": 19.047, "Ki_z": 0.0, "Kd_z": 7.506, "N_z": 53.376,
         "Kp_x": 2.763, "Ki_x": 0.0, "Kd_x": 1.658, "N_x": 78.738,
         "Kp_theta": 5.496, "Ki_theta": 0.651, "Kd_theta": 4.382, "N_theta": 100.0},
        # Extrapolate: higher Kp_z, lower N_z, higher N_x
        {"Kp_z": 23.0, "Ki_z": 0.0, "Kd_z": 9.0, "N_z": 30.0,
         "Kp_x": 2.6, "Ki_x": 0.0, "Kd_x": 1.55, "N_x": 95.0,
         "Kp_theta": 6.5, "Ki_theta": 0.5, "Kd_theta": 4.2, "N_theta": 100.0},
        # Extrapolate further in same direction
        {"Kp_z": 24.5, "Ki_z": 0.0, "Kd_z": 9.5, "N_z": 25.0,
         "Kp_x": 2.5, "Ki_x": 0.0, "Kd_x": 1.5, "N_x": 98.0,
         "Kp_theta": 7.0, "Ki_theta": 0.4, "Kd_theta": 4.1, "N_theta": 100.0},
        # Explore Ki_theta variations around best
        {"Kp_z": 21.4, "Ki_z": 0.0, "Kd_z": 8.35, "N_z": 35.0,
         "Kp_x": 2.68, "Ki_x": 0.0, "Kd_x": 1.60, "N_x": 90.0,
         "Kp_theta": 6.26, "Ki_theta": 0.0, "Kd_theta": 4.27, "N_theta": 100.0},
        {"Kp_z": 21.4, "Ki_z": 0.0, "Kd_z": 8.35, "N_z": 35.0,
         "Kp_x": 2.68, "Ki_x": 0.0, "Kd_x": 1.60, "N_x": 90.0,
         "Kp_theta": 6.26, "Ki_theta": 1.0, "Kd_theta": 4.27, "N_theta": 100.0},
        # Explore N_z lower
        {"Kp_z": 22.0, "Ki_z": 0.0, "Kd_z": 8.5, "N_z": 20.0,
         "Kp_x": 2.65, "Ki_x": 0.0, "Kd_x": 1.58, "N_x": 92.0,
         "Kp_theta": 6.4, "Ki_theta": 0.55, "Kd_theta": 4.25, "N_theta": 100.0},
        # Explore Kp_theta higher
        {"Kp_z": 21.4, "Ki_z": 0.0, "Kd_z": 8.35, "N_z": 35.0,
         "Kp_x": 2.68, "Ki_x": 0.0, "Kd_x": 1.60, "N_x": 90.0,
         "Kp_theta": 8.0, "Ki_theta": 0.55, "Kd_theta": 4.0, "N_theta": 100.0},
    ]

    best_score = -1.0
    best_vec = None
    for bl in baselines:
        v = gains_to_vec(bl)
        s = evaluate(v)
        if s > best_score:
            best_score = s
            best_vec = v.copy()
    print(f"Best baseline score: {best_score:.6f}")

    # --- Phase 1: Focused random search near best known region ---
    n_random = 400
    for i in range(n_random):
        if time.time() - start_time > TIME_BUDGET * 0.15:
            break
        if rng.random() < 0.9:
            scale = rng.uniform(0.01, 0.15)
            v = best_vec + rng.normal(0, scale, dim) * range_arr
            v = np.clip(v, lo_arr, hi_arr)
        else:
            v = lo_arr + rng.random(dim) * range_arr
        s = evaluate(v)
        if s > best_score:
            best_score = s
            best_vec = v.copy()
            print(f"  random {i}: new best = {best_score:.6f}")

    # --- Phase 2: CMA-ES style optimization ---
    mu_es = best_vec.copy()
    sigma = 0.03  # very small step since we start very close to optimum
    pop_size = 16
    elite_size = 5
    C = np.eye(dim)
    
    generation = 0
    stagnation = 0
    
    while time.time() - start_time < TIME_BUDGET * 0.50:
        generation += 1
        # Generate population
        population = []
        scores = []
        for _ in range(pop_size):
            if time.time() - start_time > TIME_BUDGET * 0.85:
                break
            # Sample in normalized space then convert
            z = rng.multivariate_normal(np.zeros(dim), C)
            candidate = mu_es + sigma * range_arr * z
            candidate = np.clip(candidate, lo_arr, hi_arr)
            s = evaluate(candidate)
            population.append(candidate)
            scores.append(s)

        if not scores:
            break

        # Sort by score (descending)
        idx = np.argsort(scores)[::-1]
        
        # Update best
        if scores[idx[0]] > best_score:
            best_score = scores[idx[0]]
            best_vec = population[idx[0]].copy()
            stagnation = 0
            print(f"  gen {generation}: new best = {best_score:.6f}")
        else:
            stagnation += 1

        # Compute new mean from elite
        elite_vecs = np.array([population[idx[i]] for i in range(min(elite_size, len(scores)))])
        elite_scores = np.array([scores[idx[i]] for i in range(min(elite_size, len(scores)))])
        
        # Weighted recombination
        weights = np.log(elite_size + 0.5) - np.log(np.arange(1, len(elite_vecs) + 1))
        weights = weights / weights.sum()
        
        new_mu = np.sum(elite_vecs * weights[:, None], axis=0)
        
        # Update covariance (simplified rank-mu update)
        diffs = (elite_vecs - mu_es) / (sigma * range_arr + 1e-12)
        C_new = np.zeros((dim, dim))
        for j in range(len(elite_vecs)):
            C_new += weights[j] * np.outer(diffs[j], diffs[j])
        learning_rate = 0.3
        C = (1 - learning_rate) * C + learning_rate * C_new
        
        # Ensure C stays well-conditioned
        eigvals = np.linalg.eigvalsh(C)
        if eigvals.min() < 1e-6 or eigvals.max() / (eigvals.min() + 1e-12) > 1e4:
            C = np.eye(dim)
        
        mu_es = new_mu
        
        # Adapt sigma
        if stagnation > 3:
            sigma *= 0.8
            if sigma < 0.01:
                sigma = 0.15
                C = np.eye(dim)
                mu_es = best_vec.copy()
                stagnation = 0

    # --- Phase 3: Fine local search around best ---
    print(f"Starting fine refinement from score {best_score:.6f}")
    for scale in [0.03, 0.015, 0.008, 0.004, 0.002]:
        improved = True
        while improved and time.time() - start_time < TIME_BUDGET * 0.75:
            improved = False
            for _ in range(40):
                if time.time() - start_time > TIME_BUDGET:
                    break
                noise = rng.normal(0, scale, dim) * range_arr
                candidate = np.clip(best_vec + noise, lo_arr, hi_arr)
                s = evaluate(candidate)
                if s > best_score:
                    best_score = s
                    best_vec = candidate.copy()
                    improved = True
                    print(f"  fine({scale:.3f}): new best = {best_score:.6f}")

    # --- Phase 4: Coordinate-wise refinement with line search ---
    for pass_num in range(5):
        if time.time() - start_time > TIME_BUDGET:
            break
        any_improved = False
        for d in range(dim):
            if time.time() - start_time > TIME_BUDGET:
                break
            for delta_frac in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001]:
                candidate = best_vec.copy()
                candidate[d] = np.clip(candidate[d] + delta_frac * range_arr[d], lo_arr[d], hi_arr[d])
                s = evaluate(candidate)
                if s > best_score:
                    best_score = s
                    best_vec = candidate.copy()
                    any_improved = True
                    print(f"  coord {d} pass {pass_num}: new best = {best_score:.6f}")
                    # Continue in same direction (line search)
                    for _ in range(10):
                        candidate2 = best_vec.copy()
                        candidate2[d] = np.clip(candidate2[d] + delta_frac * range_arr[d], lo_arr[d], hi_arr[d])
                        s2 = evaluate(candidate2)
                        if s2 > best_score:
                            best_score = s2
                            best_vec = candidate2.copy()
                            print(f"    line {d}: {best_score:.6f}")
                        else:
                            break
                    break
        if not any_improved:
            break

    print(f"Final best score: {best_score:.6f}")
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
