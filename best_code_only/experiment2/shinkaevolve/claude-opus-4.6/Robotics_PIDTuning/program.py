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
    """Optimize PID gains using multi-phase search with warm starts.

    Returns the best gain dict found.
    """
    from scipy.optimize import minimize, differential_evolution
    import time

    cfg = load_config()
    gain_ranges = cfg["gains"]
    rng = np.random.default_rng(42)
    start_time = time.time()
    TIME_LIMIT = 260  # seconds, leave margin for evaluation

    keys_order = [
        ("altitude", "Kp", "Kp_z"), ("altitude", "Ki", "Ki_z"),
        ("altitude", "Kd", "Kd_z"), ("altitude", "N", "N_z"),
        ("horizontal", "Kp", "Kp_x"), ("horizontal", "Ki", "Ki_x"),
        ("horizontal", "Kd", "Kd_x"), ("horizontal", "N", "N_x"),
        ("pitch", "Kp", "Kp_theta"), ("pitch", "Ki", "Ki_theta"),
        ("pitch", "Kd", "Kd_theta"), ("pitch", "N", "N_theta"),
    ]

    bounds_lo = []
    bounds_hi = []
    for group, param, key in keys_order:
        lo, hi = gain_ranges[group][param]
        bounds_lo.append(lo)
        bounds_hi.append(hi)
    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)
    ranges = bounds_hi - bounds_lo
    scipy_bounds = list(zip(bounds_lo.tolist(), bounds_hi.tolist()))

    def vec_to_gains(vec: np.ndarray) -> dict[str, float]:
        g: dict[str, float] = {}
        for i, (group, param, key) in enumerate(keys_order):
            g[key] = float(np.clip(vec[i], bounds_lo[i], bounds_hi[i]))
        return g

    def gains_to_vec(gains: dict[str, float]) -> np.ndarray:
        return np.array([gains[key] for _, _, key in keys_order])

    best_score = 0.0
    best_gains = {}
    best_vec = np.zeros(len(keys_order))

    # Build a fast pure-Python simulation to avoid numpy overhead per call
    scenarios_data = []
    quad = cfg["quadrotor"]
    cons = cfg["constraints"]
    sim_cfg = cfg["sim"]
    _m = quad["mass"]
    _I = quad["inertia"]
    _g = quad["gravity"]
    _drag = quad["drag"]
    _angular_drag = quad.get("angular_drag", 0.0)
    _tau_motor = quad["motor_time_constant"]
    _dt = sim_cfg["dt"]
    _switch_r = sim_cfg["waypoint_switch_radius"]
    _max_pitch = cons["max_pitch_rad"]
    _max_thrust = cons["max_thrust_factor"] * _m * _g
    _min_z = cons["min_altitude"]
    _alpha_m = _dt / (_tau_motor + _dt)
    _mg = _m * _g

    for sc in cfg["scenarios"]:
        duration = sc["duration"]
        wind = sc["wind"]
        wps = sc["waypoints"]
        start = sc["start"]
        n_steps = int(duration / _dt)
        scenarios_data.append((n_steps, float(wind[0]), float(wind[1]),
                               [(float(wp[0]), float(wp[1])) for wp in wps],
                               float(start[0]), float(start[1])))

    def fast_compute_itae_vec(vec):
        """Fast scoring from a numpy vector - pure Python inner loop."""
        Kp_z = float(vec[0]); Ki_z = float(vec[1]); Kd_z = float(vec[2]); N_z = float(vec[3])
        Kp_x = float(vec[4]); Ki_x = float(vec[5]); Kd_x = float(vec[6]); N_x = float(vec[7])
        Kp_theta = float(vec[8]); Ki_theta = float(vec[9]); Kd_theta = float(vec[10]); N_theta = float(vec[11])

        alpha_z_c = _dt * N_z / (1.0 + _dt * N_z)
        alpha_x_c = _dt * N_x / (1.0 + _dt * N_x)
        alpha_theta_c = _dt * N_theta / (1.0 + _dt * N_theta)
        one_m_az = 1.0 - alpha_z_c
        one_m_ax = 1.0 - alpha_x_c
        one_m_at = 1.0 - alpha_theta_c

        log_sum = 0.0
        n_sc = len(scenarios_data)

        for sd in scenarios_data:
            n_steps, w0, w1, wps, sx, sz = sd
            n_wps = len(wps)

            x = sx; z = sz; theta = 0.0
            x_dot = 0.0; z_dot = 0.0; theta_dot = 0.0
            T_act = _mg; tau_act = 0.0
            int_z = 0.0; int_x = 0.0; int_theta = 0.0
            df_z = 0.0; df_x = 0.0; df_theta = 0.0
            wp_idx = 0
            tx, tz = wps[0]
            prev_ez = tz - z
            prev_ex = tx - x
            prev_etheta = 0.0
            itae = 0.0
            feasible = True

            for i in range(n_steps):
                t = i * _dt

                if wp_idx < n_wps - 1:
                    ddx = x - tx; ddz = z - tz
                    if ddx * ddx + ddz * ddz < _switch_r * _switch_r:
                        wp_idx += 1
                        tx, tz = wps[wp_idx]

                ez = tz - z
                int_z += ez * _dt
                raw_dez = (ez - prev_ez) / _dt
                df_z = alpha_z_c * raw_dez + one_m_az * df_z
                prev_ez = ez
                thrust_offset = Kp_z * ez + Ki_z * int_z + Kd_z * df_z

                cos_theta = math.cos(theta)
                if abs(cos_theta) > 1e-6:
                    T_cmd = (_mg + thrust_offset) / cos_theta
                else:
                    T_cmd = _max_thrust

                ex = tx - x
                int_x += ex * _dt
                raw_dex = (ex - prev_ex) / _dt
                df_x = alpha_x_c * raw_dex + one_m_ax * df_x
                prev_ex = ex
                desired_pitch = -(Kp_x * ex + Ki_x * int_x + Kd_x * df_x)
                if desired_pitch > _max_pitch:
                    desired_pitch = _max_pitch
                elif desired_pitch < -_max_pitch:
                    desired_pitch = -_max_pitch

                etheta = desired_pitch - theta
                int_theta += etheta * _dt
                raw_detheta = (etheta - prev_etheta) / _dt
                df_theta = alpha_theta_c * raw_detheta + one_m_at * df_theta
                prev_etheta = etheta
                tau_cmd = Kp_theta * etheta + Ki_theta * int_theta + Kd_theta * df_theta

                if T_cmd < 0.0:
                    T_cmd = 0.0
                elif T_cmd > _max_thrust:
                    T_cmd = _max_thrust

                T_act = T_act + _alpha_m * (T_cmd - T_act)
                tau_act = tau_act + _alpha_m * (tau_cmd - tau_act)

                ax = -(T_act / _m) * math.sin(theta) - _drag * x_dot + w0
                az = (T_act / _m) * cos_theta - _g - _drag * z_dot + w1
                atheta = tau_act / _I - _angular_drag * theta_dot

                x_dot += ax * _dt
                z_dot += az * _dt
                theta_dot += atheta * _dt
                x += x_dot * _dt
                z += z_dot * _dt
                theta += theta_dot * _dt

                if theta > _max_pitch or theta < -_max_pitch:
                    feasible = False
                    break
                if z < _min_z:
                    z = _min_z
                    if z_dot < 0.0:
                        z_dot = 0.0

                pos_err = math.sqrt(ex * ex + ez * ez)
                itae += t * pos_err * _dt

            if not feasible or itae <= 0.0:
                return 0.0
            log_sum += math.log(1.0 / itae)

        return math.exp(log_sum / n_sc)

    def objective(vec: np.ndarray) -> float:
        nonlocal best_score, best_gains, best_vec
        clipped = np.clip(vec, bounds_lo, bounds_hi)
        score = fast_compute_itae_vec(clipped)
        if score > best_score:
            best_score = score
            best_gains = vec_to_gains(clipped)
            best_vec = clipped.copy()
        return -score  # minimize negative score

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # ---- Best known gains from ALL previous runs ----
    # Score ~0.163900 (BEST EVER - from third prior program with quadratic line search)
    best_ever = {
        "Kp_z": 29.999073310674333, "Ki_z": 0.0, "Kd_z": 12.08328288962392, "N_z": 14.423790798007744,
        "Kp_x": 2.4605526089393392, "Ki_x": 0.0, "Kd_x": 1.4271376786406753, "N_x": 97.79197035185705,
        "Kp_theta": 8.053873096270586, "Ki_theta": 0.02496075362116908, "Kd_theta": 4.092063144092614, "N_theta": 100.0,
    }
    # Score ~0.163880 (second best - from first prior program)
    second_best = {
        "Kp_z": 29.75780409473378, "Ki_z": 0.0, "Kd_z": 11.998098546767507, "N_z": 14.274370196071565,
        "Kp_x": 2.4802578093012673, "Ki_x": 0.0, "Kd_x": 1.4396064576382217, "N_x": 97.33395299794758,
        "Kp_theta": 8.010862717131708, "Ki_theta": 0.1797858841399325, "Kd_theta": 4.097708998926449, "N_theta": 99.98065802243589,
    }
    # Score ~0.163848 (third best)
    third_best = {
        "Kp_z": 29.685597335632004, "Ki_z": 0.0, "Kd_z": 11.93351495995284, "N_z": 14.862461025155213,
        "Kp_x": 2.418852330853298, "Ki_x": 0.0, "Kd_x": 1.4024489748726787, "N_x": 97.00055753642516,
        "Kp_theta": 8.11705488545832, "Ki_theta": 0.18080008579965218, "Kd_theta": 4.084129444452825, "N_theta": 100.0,
    }
    # Score ~0.163194 (older region)
    old_region = {
        "Kp_z": 22.979913125057294, "Ki_z": 5.960860986549136e-05, "Kd_z": 9.257456168965223, "N_z": 14.082192247895085,
        "Kp_x": 2.43307770429677, "Ki_x": 0.0, "Kd_x": 1.4249073119469533, "N_x": 93.48133331659996,
        "Kp_theta": 7.865366086078917, "Ki_theta": 0.8846536445396934, "Kd_theta": 4.118827241666792, "N_theta": 99.19311883508941,
    }

    # CONVERGED REGION: Kp_z~30, Kd_z~12, N_z~14.4, Kp_x~2.46, Kd_x~1.43, N_x~97
    # Kp_theta~8.05, Ki_theta~0.02-0.18, Kd_theta~4.09, N_theta~100

    starting_points = [
        best_ever,
        second_best,
        third_best,
        old_region,
        # --- Fine grid around converged optimum ---
        # Kp_z: fine grid around 30.0
        {**best_ever, "Kp_z": 29.0}, {**best_ever, "Kp_z": 29.5},
        {**best_ever, "Kp_z": 30.5}, {**best_ever, "Kp_z": 31.0},
        {**best_ever, "Kp_z": 31.5}, {**best_ever, "Kp_z": 32.0},
        {**best_ever, "Kp_z": 33.0}, {**best_ever, "Kp_z": 34.0},
        # Kd_z: fine grid around 12.08
        {**best_ever, "Kd_z": 11.5}, {**best_ever, "Kd_z": 11.8},
        {**best_ever, "Kd_z": 12.3}, {**best_ever, "Kd_z": 12.5},
        {**best_ever, "Kd_z": 13.0}, {**best_ever, "Kd_z": 13.5},
        # N_z: fine grid around 14.4
        {**best_ever, "N_z": 12.0}, {**best_ever, "N_z": 13.0},
        {**best_ever, "N_z": 13.5}, {**best_ever, "N_z": 14.0},
        {**best_ever, "N_z": 15.0}, {**best_ever, "N_z": 16.0},
        {**best_ever, "N_z": 18.0},
        # Ki_theta: the big question - 0.025 vs 0.18 vs 0
        {**best_ever, "Ki_theta": 0.0}, {**best_ever, "Ki_theta": 0.01},
        {**best_ever, "Ki_theta": 0.05}, {**best_ever, "Ki_theta": 0.1},
        {**best_ever, "Ki_theta": 0.15}, {**best_ever, "Ki_theta": 0.2},
        {**best_ever, "Ki_theta": 0.3}, {**best_ever, "Ki_theta": 0.5},
        # Kp_theta: fine grid around 8.05
        {**best_ever, "Kp_theta": 7.5}, {**best_ever, "Kp_theta": 7.8},
        {**best_ever, "Kp_theta": 8.2}, {**best_ever, "Kp_theta": 8.5},
        {**best_ever, "Kp_theta": 9.0},
        # Kd_theta: fine grid around 4.09
        {**best_ever, "Kd_theta": 3.8}, {**best_ever, "Kd_theta": 4.0},
        {**best_ever, "Kd_theta": 4.2}, {**best_ever, "Kd_theta": 4.4},
        # Kp_x: around 2.46
        {**best_ever, "Kp_x": 2.2}, {**best_ever, "Kp_x": 2.3},
        {**best_ever, "Kp_x": 2.5}, {**best_ever, "Kp_x": 2.6},
        {**best_ever, "Kp_x": 2.8},
        # Kd_x: around 1.43
        {**best_ever, "Kd_x": 1.2}, {**best_ever, "Kd_x": 1.3},
        {**best_ever, "Kd_x": 1.5}, {**best_ever, "Kd_x": 1.6},
        # N_x: around 97.8
        {**best_ever, "N_x": 90.0}, {**best_ever, "N_x": 95.0},
        {**best_ever, "N_x": 100.0},
        # N_theta: at 100
        {**best_ever, "N_theta": 90.0}, {**best_ever, "N_theta": 80.0},
        # --- Combined variations ---
        {**best_ever, "Kp_z": 31.0, "Kd_z": 12.5},
        {**best_ever, "Kp_z": 31.0, "Kd_z": 12.5, "N_z": 14.0},
        {**best_ever, "Kp_z": 32.0, "Kd_z": 13.0, "N_z": 13.5},
        {**best_ever, "Kp_theta": 8.2, "Ki_theta": 0.0, "Kd_theta": 4.2},
        {**best_ever, "Kp_theta": 7.8, "Ki_theta": 0.1, "Kd_theta": 4.0},
        # Extrapolations from best_ever vs second_best
        {k: 2.0 * best_ever[k] - 1.0 * second_best[k] for k in best_ever},
        {k: 1.5 * best_ever[k] - 0.5 * second_best[k] for k in best_ever},
        {k: 1.3 * best_ever[k] - 0.3 * second_best[k] for k in best_ever},
        # Ki_z (small nonzero)
        {**best_ever, "Ki_z": 0.01}, {**best_ever, "Ki_z": 0.05},
        {**best_ever, "Ki_z": 0.1},
        # Ki_x (small nonzero)
        {**best_ever, "Ki_x": 0.01}, {**best_ever, "Ki_x": 0.03},
        # Higher gain exploration
        {"Kp_z": 35.0, "Ki_z": 0.0, "Kd_z": 14.0, "N_z": 13.0,
         "Kp_x": 2.5, "Ki_x": 0.0, "Kd_x": 1.5, "N_x": 97.0,
         "Kp_theta": 8.5, "Ki_theta": 0.0, "Kd_theta": 4.3, "N_theta": 100.0},
        {"Kp_z": 38.0, "Ki_z": 0.0, "Kd_z": 15.0, "N_z": 12.0,
         "Kp_x": 2.6, "Ki_x": 0.0, "Kd_x": 1.5, "N_x": 100.0,
         "Kp_theta": 9.0, "Ki_theta": 0.1, "Kd_theta": 4.5, "N_theta": 100.0},
    ]

    # Evaluate starting points using fast scorer
    for sp in starting_points:
        for i, (group, param, key) in enumerate(keys_order):
            lo, hi = gain_ranges[group][param]
            sp[key] = float(np.clip(sp[key], lo, hi))
        vec_sp = gains_to_vec(sp)
        score = fast_compute_itae_vec(vec_sp)
        if score > best_score:
            best_score = score
            best_gains = sp.copy()
            best_vec = vec_sp.copy()
    print(f"Best starting point score: {best_score:.6f}")

    dim = len(keys_order)

    # Phase 0: Quadratic line search along each dimension (proven effective)
    print(f"Phase 0: Quadratic line search per dimension")
    for j in range(dim):
        if time_left() < 230:
            break
        for width_frac in [0.1, 0.03, 0.01, 0.003, 0.001]:
            width = width_frac * ranges[j]
            center = best_vec[j]
            pts = np.linspace(max(bounds_lo[j], center - width),
                              min(bounds_hi[j], center + width), 11)
            scores_line = []
            for p in pts:
                trial = best_vec.copy()
                trial[j] = p
                s = fast_compute_itae_vec(trial)
                scores_line.append((s, p))
                if s > best_score:
                    best_score = s
                    best_vec = trial.copy()
                    best_gains = vec_to_gains(best_vec)
                    print(f"  Line search dim {j} width {width_frac}: new best = {best_score:.6f}")
            # Quadratic fit on top-5 points
            scores_line.sort(reverse=True)
            top_pts = scores_line[:5]
            if len(top_pts) >= 3:
                xs = np.array([p for _, p in top_pts])
                ys = np.array([s for s, _ in top_pts])
                try:
                    coeffs = np.polyfit(xs, ys, 2)
                    if coeffs[0] < 0:  # concave - has maximum
                        opt_x = -coeffs[1] / (2 * coeffs[0])
                        opt_x = np.clip(opt_x, bounds_lo[j], bounds_hi[j])
                        trial = best_vec.copy()
                        trial[j] = opt_x
                        s = fast_compute_itae_vec(trial)
                        if s > best_score:
                            best_score = s
                            best_vec = trial.copy()
                            best_gains = vec_to_gains(best_vec)
                            print(f"  Quadratic opt dim {j}: new best = {best_score:.6f}")
                except Exception:
                    pass

    # Phase 0b: Focused random sampling near converged optimum
    print(f"Phase 0b: Focused random sampling from {best_score:.6f}")
    n_biased = 400
    for i in range(n_biased):
        if time_left() < 218:
            break
        r = rng.random()
        if r < 0.50:
            noise = rng.normal(0, 0.015, dim) * ranges
            trial = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        elif r < 0.75:
            noise = rng.normal(0, 0.04, dim) * ranges
            trial = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        elif r < 0.90:
            noise = rng.normal(0, 0.10, dim) * ranges
            trial = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        else:
            trial = bounds_lo + rng.random(dim) * ranges
        score = fast_compute_itae_vec(trial)
        if score > best_score:
            best_score = score
            best_vec = trial.copy()
            best_gains = vec_to_gains(best_vec)
            print(f"  Random sample {i}: new best = {best_score:.6f}")

    # Phase 1: Differential Evolution with seeded population
    if time_left() > 185:
        print(f"Phase 1: Differential Evolution from score {best_score:.6f}")

        def de_objective(vec):
            nonlocal best_score, best_gains, best_vec
            clipped = np.clip(vec, bounds_lo, bounds_hi)
            score = fast_compute_itae_vec(clipped)
            if score > best_score:
                best_score = score
                best_gains = vec_to_gains(clipped)
                best_vec = clipped.copy()
            return -score

        try:
            pop_size_de = 50
            init_pop = np.zeros((pop_size_de, dim))
            init_pop[0] = best_vec.copy()
            init_pop[1] = gains_to_vec(second_best)
            init_pop[2] = gains_to_vec(third_best)
            init_pop[3] = gains_to_vec(old_region)
            # Tight perturbations around best
            for i in range(4, 18):
                noise = rng.normal(0, 0.015 + 0.003 * i, dim) * ranges
                init_pop[i] = np.clip(best_vec + noise, bounds_lo, bounds_hi)
            # Medium perturbations
            for i in range(18, 35):
                noise = rng.normal(0, 0.06, dim) * ranges
                init_pop[i] = np.clip(best_vec + noise, bounds_lo, bounds_hi)
            # Wide perturbations
            for i in range(35, 42):
                noise = rng.normal(0, 0.12, dim) * ranges
                init_pop[i] = np.clip(best_vec + noise, bounds_lo, bounds_hi)
            # Random for diversity
            for i in range(42, pop_size_de):
                for j in range(dim):
                    init_pop[i, j] = bounds_lo[j] + rng.random() * ranges[j]

            result = differential_evolution(
                de_objective,
                bounds=scipy_bounds,
                init=init_pop,
                maxiter=200,
                seed=42,
                tol=1e-14,
                mutation=(0.5, 1.5),
                recombination=0.9,
                strategy='best1bin',
                polish=False,
                updating='deferred',
                workers=1,
            )
            final_score = -result.fun
            if final_score > best_score:
                best_score = final_score
                best_gains = vec_to_gains(np.clip(result.x, bounds_lo, bounds_hi))
                best_vec = gains_to_vec(best_gains)
                print(f"  DE best1bin: improved to {best_score:.6f}")
        except Exception as e:
            print(f"  DE failed: {e}")

    # Phase 1a: Second DE run with currenttobest strategy (good for exploitation)
    if time_left() > 145:
        print(f"Phase 1a: DE currenttobest1bin from score {best_score:.6f}")
        try:
            pop_size_de2 = 35
            init_pop2 = np.zeros((pop_size_de2, dim))
            init_pop2[0] = best_vec.copy()
            for i in range(1, pop_size_de2 // 2):
                noise = rng.normal(0, 0.03, dim) * ranges
                init_pop2[i] = np.clip(best_vec + noise, bounds_lo, bounds_hi)
            for i in range(pop_size_de2 // 2, pop_size_de2):
                noise = rng.normal(0, 0.10, dim) * ranges
                init_pop2[i] = np.clip(best_vec + noise, bounds_lo, bounds_hi)

            result2 = differential_evolution(
                de_objective,
                bounds=scipy_bounds,
                init=init_pop2,
                maxiter=150,
                seed=789,
                tol=1e-14,
                mutation=(0.3, 1.0),
                recombination=0.8,
                strategy='currenttobest1bin',
                polish=False,
                updating='deferred',
                workers=1,
            )
            final_score2 = -result2.fun
            if final_score2 > best_score:
                best_score = final_score2
                best_gains = vec_to_gains(np.clip(result2.x, bounds_lo, bounds_hi))
                best_vec = gains_to_vec(best_gains)
                print(f"  DE currenttobest1bin: improved to {best_score:.6f}")
        except Exception as e:
            print(f"  DE2 failed: {e}")

    # Phase 1b: Powell from best and from diverse starts
    print(f"Phase 1b: Powell from score {best_score:.6f}")
    for pw_trial in range(18):
        if time_left() < 95:
            break
        if pw_trial == 0:
            x0 = best_vec.copy()
        elif pw_trial <= 4:
            noise_scale = 0.001 * pw_trial
            noise = rng.normal(0, noise_scale, dim) * ranges
            x0 = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        else:
            noise_scale = 0.005 + (pw_trial - 4) * 0.008
            noise = rng.normal(0, noise_scale, dim) * ranges
            x0 = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        try:
            result = minimize(
                objective,
                x0,
                method='Powell',
                bounds=scipy_bounds,
                options={'maxiter': 1500, 'maxfev': 2500, 'ftol': 1e-16}
            )
        except Exception:
            pass

    # Phase 1c: Nelder-Mead from best
    print(f"Phase 1c: Nelder-Mead from score {best_score:.6f}")
    for nm_trial in range(6):
        if time_left() < 75:
            break
        if nm_trial == 0:
            x0 = best_vec.copy()
        else:
            noise_scale = 0.002 + nm_trial * 0.005
            noise = rng.normal(0, noise_scale, dim) * ranges
            x0 = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'maxfev': 1200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
        except Exception:
            pass

    # Phase 2: CMA-ES
    print(f"Phase 2: CMA-ES from score {best_score:.6f}")

    def run_cma_es(x0, sigma0, max_time=60, pop_size=None):
        nonlocal best_score, best_gains, best_vec
        n = len(x0)
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(n))
        mu = pop_size // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        cma_start = time.time()
        gen = 0
        stagnation = 0
        local_best = -1e30

        while (time.time() - cma_start) < max_time and time_left() > 40:
            try:
                eigenvalues, B = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                D = np.sqrt(eigenvalues)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                eigenvalues = np.ones(n)
                B = np.eye(n)
                D = np.ones(n)

            arz = rng.standard_normal((pop_size, n))
            arx = np.zeros((pop_size, n))
            for k in range(pop_size):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], bounds_lo, bounds_hi)

            fitness = np.zeros(pop_size)
            for k in range(pop_size):
                score = fast_compute_itae_vec(arx[k])
                fitness[k] = score
                if score > best_score:
                    best_score = score
                    best_gains = vec_to_gains(arx[k])
                    best_vec = arx[k].copy()
                    print(f"  CMA gen {gen}: new best = {best_score:.6f}")

            idx = np.argsort(-fitness)
            arx = arx[idx]
            arz = arz[idx]
            fitness = fitness[idx]

            if fitness[0] > local_best:
                local_best = fitness[0]
                stagnation = 0
            else:
                stagnation += 1

            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[k]

            invsqrtC = B @ np.diag(1.0 / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (mean - old_mean) / sigma
            hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu_val * weights[k] * np.outer(artmp[k], artmp[k])
            C = (C + C.T) / 2

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0 * sigma0)

            gen += 1
            if stagnation > 20:
                break

    sigma0 = 0.02 * np.mean(ranges)
    run_cma_es(best_vec.copy(), sigma0, max_time=15, pop_size=16)

    if time_left() > 60:
        print(f"Phase 2b: CMA-ES wide from score {best_score:.6f}")
        run_cma_es(best_vec.copy(), sigma0 * 4.0, max_time=12, pop_size=18)

    if time_left() > 50:
        print(f"Phase 2c: CMA-ES focused from score {best_score:.6f}")
        run_cma_es(best_vec.copy(), sigma0 * 0.3, max_time=10, pop_size=14)

    # Phase 3: More Powell
    print(f"Phase 3: More Powell from score {best_score:.6f}")
    for pw_trial in range(8):
        if time_left() < 45:
            break
        if pw_trial == 0:
            x0 = best_vec.copy()
        else:
            noise = rng.normal(0, 0.003 + pw_trial * 0.005, dim) * ranges
            x0 = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        try:
            result = minimize(
                objective,
                x0,
                method='Powell',
                bounds=scipy_bounds,
                options={'maxiter': 800, 'maxfev': 1200, 'ftol': 1e-16}
            )
        except Exception:
            pass

    # Phase 3b: 2D pair search on correlated parameters
    print(f"Phase 3b: 2D pair search from score {best_score:.6f}")
    param_pairs = [
        (0, 2),   # Kp_z, Kd_z
        (8, 9),   # Kp_theta, Ki_theta
        (8, 10),  # Kp_theta, Kd_theta
        (4, 6),   # Kp_x, Kd_x
        (0, 3),   # Kp_z, N_z
        (9, 10),  # Ki_theta, Kd_theta
        (2, 3),   # Kd_z, N_z
        (4, 7),   # Kp_x, N_x
        (0, 8),   # Kp_z, Kp_theta
        (6, 7),   # Kd_x, N_x
        (10, 11), # Kd_theta, N_theta
    ]
    for j1, j2 in param_pairs:
        if time_left() < 30:
            break
        c1_val = best_vec[j1]
        c2_val = best_vec[j2]
        for w_scale in [0.03, 0.01, 0.005, 0.002]:
            w1 = w_scale * ranges[j1]
            w2 = w_scale * ranges[j2]
            n_grid = 7
            for g1 in range(n_grid):
                for g2 in range(n_grid):
                    v1 = c1_val + w1 * (2 * g1 / (n_grid - 1) - 1)
                    v2 = c2_val + w2 * (2 * g2 / (n_grid - 1) - 1)
                    v1 = np.clip(v1, bounds_lo[j1], bounds_hi[j1])
                    v2 = np.clip(v2, bounds_lo[j2], bounds_hi[j2])
                    trial = best_vec.copy()
                    trial[j1] = v1
                    trial[j2] = v2
                    score = fast_compute_itae_vec(trial)
                    if score > best_score:
                        best_score = score
                        best_gains = vec_to_gains(trial)
                        best_vec = trial.copy()
                        c1_val = best_vec[j1]
                        c2_val = best_vec[j2]
                        print(f"  2D pair ({j1},{j2}): new best = {best_score:.6f}")

    # Phase 3c: 3D triplet search on most correlated parameters
    print(f"Phase 3c: 3D triplet search from score {best_score:.6f}")
    param_triplets = [
        (0, 2, 3),   # Kp_z, Kd_z, N_z
        (8, 9, 10),  # Kp_theta, Ki_theta, Kd_theta
        (4, 6, 7),   # Kp_x, Kd_x, N_x
    ]
    for j1, j2, j3 in param_triplets:
        if time_left() < 22:
            break
        c1_val = best_vec[j1]
        c2_val = best_vec[j2]
        c3_val = best_vec[j3]
        for w_scale in [0.015, 0.006, 0.002]:
            w1 = w_scale * ranges[j1]
            w2 = w_scale * ranges[j2]
            w3 = w_scale * ranges[j3]
            n_grid = 5
            for g1 in range(n_grid):
                for g2 in range(n_grid):
                    for g3 in range(n_grid):
                        v1 = c1_val + w1 * (2 * g1 / (n_grid - 1) - 1)
                        v2 = c2_val + w2 * (2 * g2 / (n_grid - 1) - 1)
                        v3 = c3_val + w3 * (2 * g3 / (n_grid - 1) - 1)
                        v1 = np.clip(v1, bounds_lo[j1], bounds_hi[j1])
                        v2 = np.clip(v2, bounds_lo[j2], bounds_hi[j2])
                        v3 = np.clip(v3, bounds_lo[j3], bounds_hi[j3])
                        trial = best_vec.copy()
                        trial[j1] = v1; trial[j2] = v2; trial[j3] = v3
                        score = fast_compute_itae_vec(trial)
                        if score > best_score:
                            best_score = score
                            best_gains = vec_to_gains(trial)
                            best_vec = trial.copy()
                            c1_val = best_vec[j1]; c2_val = best_vec[j2]; c3_val = best_vec[j3]
                            print(f"  3D triplet ({j1},{j2},{j3}): new best = {best_score:.6f}")

    # Phase 4: Coordinate descent with adaptive step sizes
    print(f"Phase 4: Coordinate descent from score {best_score:.6f}")
    best_vec = gains_to_vec(best_gains)
    for scale in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005]:
        if time_left() < 12:
            break
        improved = True
        passes = 0
        while improved and passes < 6 and time_left() > 8:
            improved = False
            passes += 1
            for j in range(dim):
                for direction in [1, -1]:
                    trial = best_vec.copy()
                    step = direction * scale * ranges[j]
                    trial[j] += step
                    trial = np.clip(trial, bounds_lo, bounds_hi)
                    score = fast_compute_itae_vec(trial)
                    if score > best_score:
                        best_score = score
                        best_gains = vec_to_gains(trial)
                        best_vec = trial.copy()
                        improved = True
                        # Accelerate
                        for mult in [2, 3, 5, 8]:
                            trial2 = best_vec.copy()
                            trial2[j] += step * (mult - 1)
                            trial2 = np.clip(trial2, bounds_lo, bounds_hi)
                            score2 = fast_compute_itae_vec(trial2)
                            if score2 > best_score:
                                best_score = score2
                                best_gains = vec_to_gains(trial2)
                                best_vec = trial2.copy()
                            else:
                                break

    # Phase 4b: Pattern search with momentum
    print(f"Phase 4b: Pattern search from score {best_score:.6f}")
    best_vec = gains_to_vec(best_gains)
    momentum = np.zeros(dim)
    if time_left() > 8:
        for it in range(1500):
            if time_left() < 5:
                break
            n_dims = rng.integers(1, min(7, dim + 1))
            dims_to_perturb = rng.choice(dim, size=n_dims, replace=False)
            trial = best_vec.copy()
            scale_choice = rng.choice([0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005])
            for j in dims_to_perturb:
                perturbation = rng.normal(0, scale_choice) * ranges[j]
                trial[j] += perturbation
            # Also try adding momentum from previous improvements
            if it % 3 == 0 and np.any(momentum != 0):
                trial2 = best_vec.copy() + 0.5 * momentum
                trial2 = np.clip(trial2, bounds_lo, bounds_hi)
                s2 = fast_compute_itae_vec(trial2)
                if s2 > best_score:
                    best_score = s2
                    old_bv = best_vec.copy()
                    best_vec = trial2.copy()
                    best_gains = vec_to_gains(best_vec)
                    momentum = best_vec - old_bv
                    print(f"  Momentum search: new best = {best_score:.6f}")
                    continue
            trial = np.clip(trial, bounds_lo, bounds_hi)
            score = fast_compute_itae_vec(trial)
            if score > best_score:
                old_bv = best_vec.copy()
                best_score = score
                best_gains = vec_to_gains(trial)
                best_vec = trial.copy()
                momentum = best_vec - old_bv
                print(f"  Pattern search: new best = {best_score:.6f}")

    # Phase 5: Final Powell
    if time_left() > 5:
        print(f"Phase 5: Final Powell from score {best_score:.6f}")
        try:
            result = minimize(
                objective,
                best_vec.copy(),
                method='Powell',
                bounds=scipy_bounds,
                options={'maxiter': 500, 'maxfev': 500, 'ftol': 1e-16}
            )
        except Exception:
            pass

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