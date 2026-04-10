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
    """Optimize PID gains using targeted grid search + intensive refinement."""
    cfg = load_config()
    gain_ranges = cfg["gains"]
    keys_order = [
        ("altitude", "Kp", "Kp_z"), ("altitude", "Ki", "Ki_z"),
        ("altitude", "Kd", "Kd_z"), ("altitude", "N", "N_z"),
        ("horizontal", "Kp", "Kp_x"), ("horizontal", "Ki", "Ki_x"),
        ("horizontal", "Kd", "Kd_x"), ("horizontal", "N", "N_x"),
        ("pitch", "Kp", "Kp_theta"), ("pitch", "Ki", "Ki_theta"),
        ("pitch", "Kd", "Kd_theta"), ("pitch", "N", "N_theta"),
    ]
    # Best-known configuration from previous runs
    best_gains = {
        "Kp_z": 10.66, "Ki_z": 0.0, "Kd_z": 4.57, "N_z": 15.23,
        "Kp_x": 0.45, "Ki_x": 0.0, "Kd_x": 0.28, "N_x": 68.46,
        "Kp_theta": 10.78, "Ki_theta": 0.42, "Kd_theta": 3.46, "N_theta": 100.0,
    }
    best_score = compute_itae(best_gains, cfg)
    rng = np.random.default_rng(42)
    
    # Phase 1: Focused grid search on most sensitive parameters
    for ki_t in np.linspace(0.35, 0.55, 8):
        for n_x in np.linspace(60.0, 85.0, 8):
            for kp_t in np.linspace(10.0, 11.5, 6):
                cand = best_gains.copy()
                cand["Ki_theta"] = float(ki_t)
                cand["N_x"] = float(n_x)
                cand["Kp_theta"] = float(kp_t)
                s = compute_itae(cand, cfg)
                if s > best_score:
                    best_score, best_gains = s, cand.copy()
    
    # Phase 2: Multi-scale local refinement with adaptive tracking
    for scale in [0.08, 0.04, 0.02, 0.01, 0.005]:
        improved = True
        while improved:
            improved = False
            for _ in range(50):
                cand = {}
                for g, p, k in keys_order:
                    lo, hi = gain_ranges[g][p]
                    if k == "N_theta":
                        cand[k] = 100.0
                    elif k in ["Ki_z", "Ki_x"]:
                        cand[k] = 0.0
                    else:
                        noise = rng.normal(0, scale * (hi - lo))
                        cand[k] = float(np.clip(best_gains[k] + noise, lo, hi))
                s = compute_itae(cand, cfg)
                if s > best_score:
                    best_score, best_gains, improved = s, cand.copy(), True
    
    # Phase 3: Fine coordinate descent on free parameters
    for _ in range(4):
        for g, p, k in keys_order:
            if k in ["N_theta", "Ki_z", "Ki_x"]:
                continue
            lo, hi = gain_ranges[g][p]
            for d in [0.03, 0.015, 0.008, -0.008, -0.015, -0.03]:
                cand = best_gains.copy()
                cand[k] = float(np.clip(best_gains[k] + d * (hi - lo), lo, hi))
                s = compute_itae(cand, cfg)
                if s > best_score:
                    best_score, best_gains = s, cand.copy()
    
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
