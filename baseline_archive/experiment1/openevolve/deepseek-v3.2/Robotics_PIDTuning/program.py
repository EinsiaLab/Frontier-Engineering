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
    """Optimize PID gains using simulated annealing, multiple baselines, and local search.

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

    def random_gains() -> dict[str, float]:
        g: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            g[key] = float(rng.uniform(lo, hi))
        return g

    # Multiple baselines for diverse starting points, including patterns from successful runs
    baselines = [
        # Original baseline
        {
            "Kp_z": 8.0, "Ki_z": 0.5, "Kd_z": 4.0, "N_z": 20.0,
            "Kp_x": 0.1, "Ki_x": 0.01, "Kd_x": 0.1, "N_x": 10.0,
            "Kp_theta": 10.0, "Ki_theta": 0.5, "Kd_theta": 3.0, "N_theta": 20.0,
        },
        # Baseline with zero integral gains and high N (pattern from successful runs)
        {
            "Kp_z": 9.0, "Ki_z": 0.0, "Kd_z": 5.0, "N_z": 100.0,
            "Kp_x": 0.5, "Ki_x": 0.0, "Kd_x": 0.3, "N_x": 100.0,
            "Kp_theta": 9.0, "Ki_theta": 0.0, "Kd_theta": 3.5, "N_theta": 100.0,
        },
        # Variation with medium N values
        {
            "Kp_z": 10.0, "Ki_z": 0.0, "Kd_z": 5.0, "N_z": 50.0,
            "Kp_x": 0.4, "Ki_x": 0.0, "Kd_x": 0.25, "N_x": 50.0,
            "Kp_theta": 10.0, "Ki_theta": 0.0, "Kd_theta": 3.0, "N_theta": 50.0,
        },
        # Baseline with some integral action
        {
            "Kp_z": 12.0, "Ki_z": 0.3, "Kd_z": 6.0, "N_z": 15.0,
            "Kp_x": 0.2, "Ki_x": 0.005, "Kd_x": 0.2, "N_x": 5.0,
            "Kp_theta": 8.0, "Ki_theta": 1.0, "Kd_theta": 2.0, "N_theta": 25.0,
        },
        # Another zero integral baseline with different proportional gains
        {
            "Kp_z": 14.0, "Ki_z": 0.0, "Kd_z": 6.0, "N_z": 80.0,
            "Kp_x": 0.6, "Ki_x": 0.0, "Kd_x": 0.4, "N_x": 80.0,
            "Kp_theta": 12.0, "Ki_theta": 0.0, "Kd_theta": 4.0, "N_theta": 80.0,
        }
    ]

    best_gains = baselines[0]
    best_score = compute_itae(best_gains, cfg)
    print(f"Baseline 0 score: {best_score:.6f}")

    # Evaluate all baselines
    for idx, baseline in enumerate(baselines[1:], start=1):
        score = compute_itae(baseline, cfg)
        print(f"Baseline {idx} score: {score:.6f}")
        if score > best_score:
            best_score = score
            best_gains = baseline

    # Simulated annealing with adaptive exploration
    n_sa_iter = 150
    temperature = 1.0
    cooling_rate = 0.97
    current_gains = best_gains
    current_score = best_score

    for i in range(n_sa_iter):
        # Generate candidate by perturbing current gains
        candidate: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            scale = temperature * (hi - lo) * 0.1
            noise = rng.normal(0, scale)
            candidate[key] = float(np.clip(current_gains[key] + noise, lo, hi))

        score = compute_itae(candidate, cfg)

        # Acceptance probability
        if score > current_score:
            current_gains = candidate
            current_score = score
            if score > best_score:
                best_gains = candidate
                best_score = score
                print(f"SA iter {i}: new best = {best_score:.6f}")
        else:
            delta = current_score - score
            prob = math.exp(-delta / temperature)
            if rng.random() < prob:
                current_gains = candidate
                current_score = score

        temperature *= cooling_rate

    # Aggressive local search around best
    n_local_iter = 120
    for i in range(n_local_iter):
        perturbed: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            noise = rng.normal(0, 0.02 * (hi - lo))
            perturbed[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        score = compute_itae(perturbed, cfg)
        if score > best_score:
            best_score = score
            best_gains = perturbed
            print(f"Local iter {i}: new best = {best_score:.6f}")

    # Strategic pattern testing before coordinate descent
    print("Starting pattern testing...")
    pattern_tests = []
    # Test all Ki=0 with different N values (as seen in successful runs)
    for n_val in [100.0, 50.0, 75.0, 30.0, 20.0, 150.0]:
        pattern_tests.append({
            "Kp_z": best_gains["Kp_z"], "Ki_z": 0.0, "Kd_z": best_gains["Kd_z"], "N_z": n_val,
            "Kp_x": best_gains["Kp_x"], "Ki_x": 0.0, "Kd_x": best_gains["Kd_x"], "N_x": n_val,
            "Kp_theta": best_gains["Kp_theta"], "Ki_theta": 0.0, "Kd_theta": best_gains["Kd_theta"], "N_theta": n_val,
        })
    # Test small Ki values
    for ki_val in [0.0, 0.001, 0.005, 0.01]:
        pattern_tests.append({
            "Kp_z": best_gains["Kp_z"], "Ki_z": ki_val, "Kd_z": best_gains["Kd_z"], "N_z": best_gains["N_z"],
            "Kp_x": best_gains["Kp_x"], "Ki_x": ki_val, "Kd_x": best_gains["Kd_x"], "N_x": best_gains["N_x"],
            "Kp_theta": best_gains["Kp_theta"], "Ki_theta": ki_val, "Kd_theta": best_gains["Kd_theta"], "N_theta": best_gains["N_theta"],
        })
    
    for pattern in pattern_tests:
        score = compute_itae(pattern, cfg)
        if score > best_score:
            best_score = score
            best_gains = pattern
            print(f"Pattern test: new best = {best_score:.6f}")

    # Enhanced coordinate descent with strategic testing
    print("Starting coordinate descent...")
    for group, param, key in keys_order:
        lo, hi = gain_ranges[group][param]
        # Generate test values based on parameter type and observed patterns
        if param == 'N':
            # Derivative filter coefficients: high values tend to work well
            test_values = [lo, hi, (lo+hi)/2, hi*0.9, hi*0.75, hi*0.95, 100.0, 75.0, 50.0, 150.0]
        elif param == 'Ki':
            # Integral gains: zero and small values are often best
            test_values = [lo, hi, (lo+hi)/2, 0.0, 0.001, 0.01, 0.005, 0.0005]
        elif param == 'Kp':
            # Proportional gains: test around the current best and a few evenly spaced values
            test_values = np.linspace(lo, hi, 7)
            test_values = list(test_values) + [best_gains[key], best_gains[key]*0.8, best_gains[key]*1.2]
        else: # Kd
            test_values = np.linspace(lo, hi, 7)
            test_values = list(test_values) + [best_gains[key], best_gains[key]*0.8, best_gains[key]*1.2]
        
        # Remove duplicates and ensure they're within bounds
        test_values = [float(np.clip(v, lo, hi)) for v in test_values]
        test_values = list(set(test_values))
        
        for val in test_values:
            temp = best_gains.copy()
            temp[key] = val
            score = compute_itae(temp,  cfg)
            if score > best_score:
                best_score = score
                best_gains = temp
                print(f"Coordinate {key}={val:.3f}: new best = {best_score:.6f}")

    # Adaptive random walk with parameter-specific biases
    print("Adaptive random walk...")
    n_walk_iter = 60
    step_size = 0.05  # Relative to range
    for i in range(n_walk_iter):
        perturbed: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            # Parameter-specific step sizes
            if param == 'Ki':
                # Smaller steps for integral gains
                current_step = step_size * (hi - lo) * 0.3 * (1.0 - i/n_walk_iter)
            elif param == 'N':
                # Larger steps for N gains
                current_step = step_size * (hi - lo) * 1.5 * (1.0 - i/n_walk_iter)
            else:
                current_step = step_size * (hi - lo) * (1.0 - i/n_walk_iter)
            noise = rng.uniform(-current_step, current_step)
            perturbed[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        score = compute_itae(perturbed, cfg)
        if score > best_score:
            best_score = score
            best_gains = perturbed
            print(f"Walk {i}: new best = {best_score:.6f}")
            # Increase step size slightly to explore more
            step_size *= 1.1
        else:
            # Reduce step size to focus on refinement
            step_size *= 0.99
    
    # Final adaptive refinement with parameter-specific perturbation sizes
    print("Final adaptive refinement...")
    for i in range(50):
        perturbed: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            # Use smaller perturbations for integral gains and larger for proportional/derivative gains
            if param == 'Ki':
                noise = rng.normal(0, 0.002 * (hi - lo))
            elif param == 'N':
                noise = rng.normal(0, 0.01 * (hi - lo))  # N can vary more
            else:
                noise = rng.normal(0, 0.005 * (hi - lo))
            perturbed[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        score = compute_itae(perturbed, cfg)
        if score > best_score:
            best_score = score
            best_gains = perturbed
            print(f"Final refine {i}: new best = {best_score:.6f}")

    print(f"Final best score: {best_score:.6f}")
    return best_gains


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
