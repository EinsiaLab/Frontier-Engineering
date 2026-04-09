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
    cfg = load_config()
    R = cfg["gains"]
    K = [
        ("altitude", "Kp", "Kp_z"), ("altitude", "Ki", "Ki_z"), ("altitude", "Kd", "Kd_z"), ("altitude", "N", "N_z"),
        ("horizontal", "Kp", "Kp_x"), ("horizontal", "Ki", "Ki_x"), ("horizontal", "Kd", "Kd_x"), ("horizontal", "N", "N_x"),
        ("pitch", "Kp", "Kp_theta"), ("pitch", "Ki", "Ki_theta"), ("pitch", "Kd", "Kd_theta"), ("pitch", "N", "N_theta"),
    ]
    bounds = {k: R[a][b] for a, b, k in K}
    tunable = ("Kp_z", "Kd_z", "N_z", "Kp_x", "Kd_x", "N_x", "Kp_theta", "Kd_theta", "N_theta")

    def shaped(g):
        h = {k: float(np.clip(g[k], *bounds[k])) for _, _, k in K}
        h["Ki_z"] = bounds["Ki_z"][0]
        h["Ki_x"] = bounds["Ki_x"][0]
        h["Ki_theta"] = bounds["Ki_theta"][0]
        h["Kd_z"] = float(np.clip(max(h["Kd_z"], 0.32 * h["Kp_z"]), *bounds["Kd_z"]))
        h["Kd_x"] = float(np.clip(max(h["Kd_x"], 0.72 * h["Kp_x"]), *bounds["Kd_x"]))
        h["Kd_theta"] = float(np.clip(max(h["Kd_theta"], 0.40 * h["Kp_theta"]), *bounds["Kd_theta"]))
        h["N_z"] = float(np.clip(max(h["N_z"], 25.0), *bounds["N_z"]))
        h["N_x"] = float(np.clip(max(h["N_x"], 40.0), *bounds["N_x"]))
        h["N_theta"] = float(np.clip(max(h["N_theta"], 40.0), *bounds["N_theta"]))
        return h

    def frac(d):
        return {k: float(bounds[k][0] + d.get(k, 0.5) * (bounds[k][1] - bounds[k][0])) for _, _, k in K}

    seeds = [
        {"Kp_z": 30.0, "Ki_z": 0.0, "Kd_z": 11.648495801572977, "N_z": 28.49570312320306, "Kp_x": 1.4571051743698404, "Ki_x": 0.0, "Kd_x": 1.064761196184092, "N_x": 100.0, "Kp_theta": 4.586422550008563, "Ki_theta": 0.0, "Kd_theta": 1.965108258453122, "N_theta": 100.0},
        {"Kp_z": 8.0, "Ki_z": 0.0, "Kd_z": 4.0, "N_z": 20.0, "Kp_x": 0.1, "Ki_x": 0.0, "Kd_x": 0.1, "N_x": 10.0, "Kp_theta": 10.0, "Ki_theta": 0.0, "Kd_theta": 3.0, "N_theta": 20.0},
        frac({"Kp_z": .85, "Kd_z": .78, "N_z": .28, "Kp_x": .29, "Kd_x": .22, "N_x": 1.0, "Kp_theta": .09, "Kd_theta": .18, "N_theta": 1.0}),
        frac({"Kp_z": .92, "Kd_z": .84, "N_z": .40, "Kp_x": .22, "Kd_x": .18, "N_x": .85, "Kp_theta": .06, "Kd_theta": .12, "N_theta": .85}),
        frac({"Kp_z": .75, "Kd_z": .65, "N_z": .55, "Kp_x": .18, "Kd_x": .16, "N_x": .65, "Kp_theta": .10, "Kd_theta": .16, "N_theta": .70}),
    ]
    seeds = [shaped(g) for g in seeds]
    best_gains, best_score = seeds[0], compute_itae(seeds[0], cfg)
    for g in seeds[1:]:
        s = compute_itae(g, cfg)
        if s > best_score:
            best_gains, best_score = g, s

    for axis in (("Kp_z", "Kd_z", "N_z"), ("Kp_x", "Kd_x", "N_x"), ("Kp_theta", "Kd_theta", "N_theta")):
        for a in (0.9, 0.96, 1.04, 1.1):
            for b in (0.9, 1.0, 1.1):
                g = dict(best_gains)
                g[axis[0]] *= a
                g[axis[1]] *= a
                g[axis[2]] *= b
                g = shaped(g)
                s = compute_itae(g, cfg)
                if s > best_score:
                    best_gains, best_score = g, s

    for px, dx, pt, dt in (
        (0.9, 0.9, 1.1, 1.1), (1.1, 1.1, 0.9, 0.9), (0.96, 0.96, 1.04, 1.04),
        (1.04, 1.04, 0.96, 0.96), (1.08, 1.02, 0.94, 0.98), (0.94, 0.98, 1.08, 1.02),
    ):
        g = dict(best_gains)
        g["Kp_x"] *= px; g["Kd_x"] *= dx; g["Kp_theta"] *= pt; g["Kd_theta"] *= dt
        g = shaped(g)
        s = compute_itae(g, cfg)
        if s > best_score:
            best_gains, best_score = g, s

    rng = np.random.default_rng(42)
    for sig, n in ((0.10, 12), (0.05, 12)):
        for _ in range(n):
            g = dict(best_gains)
            for k in tunable:
                g[k] *= float(np.exp(rng.normal(0.0, sig)))
            g = shaped(g)
            s = compute_itae(g, cfg)
            if s > best_score:
                best_gains, best_score = g, s

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
