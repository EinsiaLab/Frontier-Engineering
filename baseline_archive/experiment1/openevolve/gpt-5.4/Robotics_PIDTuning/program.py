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
    """Exploit several strong anchors with normalized, cached local search."""
    cfg = load_config()
    rng = np.random.default_rng(42)

    keys_order = [
        ("altitude", "Kp", "Kp_z"), ("altitude", "Ki", "Ki_z"),
        ("altitude", "Kd", "Kd_z"), ("altitude", "N", "N_z"),
        ("horizontal", "Kp", "Kp_x"), ("horizontal", "Ki", "Ki_x"),
        ("horizontal", "Kd", "Kd_x"), ("horizontal", "N", "N_x"),
        ("pitch", "Kp", "Kp_theta"), ("pitch", "Ki", "Ki_theta"),
        ("pitch", "Kd", "Kd_theta"), ("pitch", "N", "N_theta"),
    ]

    meta = []
    bounds: dict[str, tuple[float, float]] = {}
    index: dict[str, int] = {}
    for i, (group, param, key) in enumerate(keys_order):
        lo, hi = map(float, cfg["gains"][group][param])
        meta.append((key, lo, hi, lo > 0.0 and hi > lo * 4.0))
        bounds[key] = (lo, hi)
        index[key] = i

    scenarios = cfg["scenarios"]
    order = sorted(
        range(len(scenarios)),
        key=lambda i: (
            float(np.linalg.norm(np.asarray(scenarios[i]["wind"], dtype=float))),
            len(scenarios[i]["waypoints"]),
            float(scenarios[i]["duration"]),
        ),
        reverse=True,
    )

    def encode(g: dict[str, float]) -> np.ndarray:
        vals = []
        for key, lo, hi, use_log in meta:
            x = float(np.clip(g[key], lo, hi))
            if hi <= lo:
                u = 0.0
            elif use_log:
                a, b = math.log(lo), math.log(hi)
                u = (math.log(max(x, lo)) - a) / (b - a)
            else:
                u = (x - lo) / (hi - lo)
            vals.append(u)
        return np.asarray(vals, dtype=float)

    def decode(v: np.ndarray) -> dict[str, float]:
        g: dict[str, float] = {}
        for u, (key, lo, hi, use_log) in zip(np.clip(v, 0.0, 1.0), meta):
            if hi <= lo:
                x = lo
            elif use_log:
                a, b = math.log(lo), math.log(hi)
                x = math.exp(a + float(u) * (b - a))
            else:
                x = lo + float(u) * (hi - lo)
            g[key] = float(x)
        return g

    cache: dict[tuple[float, ...], float] = {}

    def score_vec(v: np.ndarray) -> float:
        v = np.clip(v, 0.0, 1.0)
        key = tuple(np.round(v, 9))
        if key in cache:
            return cache[key]
        gains = decode(v)
        vals = []
        for i in order:
            result = simulate_quadrotor_2d(gains, scenarios[i], cfg)
            if (not result["feasible"]) or result["itae"] <= 0.0:
                cache[key] = 0.0
                return 0.0
            vals.append(1.0 / result["itae"])
        score = float(math.exp(sum(math.log(x) for x in vals) / len(vals)))
        cache[key] = score
        return score

    base = {
        "Kp_z": 8.0, "Ki_z": 0.5, "Kd_z": 4.0, "N_z": 20.0,
        "Kp_x": 0.1, "Ki_x": 0.01, "Kd_x": 0.1, "N_x": 10.0,
        "Kp_theta": 10.0, "Ki_theta": 0.5, "Kd_theta": 3.0, "N_theta": 20.0,
    }
    known = {
        "Kp_z": 10.947671729322126, "Ki_z": 0.0, "Kd_z": 4.566666974802083, "N_z": 13.925726596846756,
        "Kp_x": 0.24636485400356023, "Ki_x": 0.0, "Kd_x": 0.19708405882391564, "N_x": 10.841547181282369,
        "Kp_theta": 8.654892847374407, "Ki_theta": 0.0625, "Kd_theta": 3.4400432740651095, "N_theta": 19.85383092739354,
    }
    incumbent = {
        "Kp_z": 13.917453849596406, "Ki_z": 0.0, "Kd_z": 5.687838396854843, "N_z": 20.187682570469036,
        "Kp_x": 0.11481536214968828, "Ki_x": 0.0, "Kd_x": 0.15113616126691415, "N_x": 9.962669899707269,
        "Kp_theta": 9.96719068324189, "Ki_theta": 1.2734780362132097, "Kd_theta": 3.2948494937157067, "N_theta": 8.48807462675579,
    }
    prior = {
        "Kp_z": 11.330587725014334, "Ki_z": 0.0, "Kd_z": 4.8719613361449206, "N_z": 11.496412106856715,
        "Kp_x": 0.13261489812760063, "Ki_x": 0.0, "Kd_x": 0.14335712892832894, "N_x": 6.014222927938346,
        "Kp_theta": 9.659638023993317, "Ki_theta": 1.120597801833905, "Kd_theta": 3.29929780286154, "N_theta": 18.971870918362292,
    }
    current = {
        "Kp_z": 9.643220293896196, "Ki_z": 0.0, "Kd_z": 4.2185600316357315, "N_z": 14.271939389425281,
        "Kp_x": 0.11547772853851594, "Ki_x": 0.0, "Kd_x": 0.12795885362079945, "N_x": 5.582314490581659,
        "Kp_theta": 9.704977025261531, "Ki_theta": 6.530741881393164, "Kd_theta": 2.41601635259204, "N_theta": 10.250009648848138,
    }

    anchors = [
        base,
        known,
        prior,
        incumbent,
        current,
        {**known, "Ki_theta": bounds["Ki_theta"][0]},
        {**prior, "Ki_z": bounds["Ki_z"][0], "Ki_x": bounds["Ki_x"][0]},
        {**incumbent, "Ki_z": bounds["Ki_z"][0], "Ki_x": bounds["Ki_x"][0]},
        {**current, "Ki_z": bounds["Ki_z"][0], "Ki_x": bounds["Ki_x"][0]},
    ]

    print(f"Baseline score: {score_vec(encode(base)):.6f}")

    n = len(meta)
    anchor_vs = [encode(g) for g in anchors]
    seeds = list(anchor_vs)
    for av in anchor_vs:
        for s, m in ((0.01, 6), (0.03, 8), (0.06, 8)):
            for _ in range(m):
                seeds.append(np.clip(av + rng.normal(0.0, s, n), 0.0, 1.0))
    for i in range(len(anchor_vs)):
        for j in range(i + 1, len(anchor_vs)):
            seeds.append(np.clip(0.5 * anchor_vs[i] + 0.5 * anchor_vs[j], 0.0, 1.0))
    seeds += [rng.random(n) for _ in range(10)]

    seed_scores = np.asarray([score_vec(v) for v in seeds], dtype=float)
    best_idx = int(np.argmax(seed_scores))
    best_v = np.asarray(seeds[best_idx], dtype=float)
    best_score = float(seed_scores[best_idx])
    refs = [np.asarray(seeds[i], dtype=float) for i in np.argsort(seed_scores)[-8:]]
    print(f"Seed best score: {best_score:.6f}")

    def adopt(v: np.ndarray, label: str = "") -> bool:
        nonlocal best_v, best_score, refs
        v = np.clip(v, 0.0, 1.0)
        s = score_vec(v)
        if s > best_score:
            best_v = np.asarray(v, dtype=float)
            best_score = float(s)
            refs = (refs + [best_v.copy()])[-8:]
            if label:
                print(f"  {label}: new best = {best_score:.6f}")
            return True
        return False

    sweep_order = [
        "Kp_x", "Kd_x", "N_x",
        "Kp_theta", "Ki_theta", "Kd_theta", "N_theta",
        "Kp_z", "Kd_z", "N_z", "Ki_z", "Ki_x",
    ]
    coupled = [
        ("Kp_x", "Kd_x"), ("Kp_z", "Kd_z"), ("Kp_theta", "Kd_theta"),
        ("Kd_x", "N_x"), ("Kd_z", "N_z"), ("Kd_theta", "N_theta"),
    ]
    groups = [
        ("Kp_z", "Kd_z", "N_z"),
        ("Kp_x", "Kd_x", "N_x"),
        ("Kp_theta", "Ki_theta", "Kd_theta", "N_theta"),
    ]

    for step in (0.05, 0.025, 0.0125, 0.006, 0.003, 0.0015):
        for _ in range(5):
            improved = False

            for key in sweep_order:
                i = index[key]
                for d in (-step, -0.5 * step, 0.5 * step, step):
                    v = best_v.copy()
                    v[i] = np.clip(v[i] + d, 0.0, 1.0)
                    improved = adopt(v, key) or improved

            for a, b in coupled:
                ia, ib = index[a], index[b]
                for da, db in (
                    (step, step), (step, -step), (-step, step), (-step, -step),
                    (0.5 * step, -0.5 * step), (-0.5 * step, 0.5 * step),
                ):
                    v = best_v.copy()
                    v[ia] = np.clip(v[ia] + da, 0.0, 1.0)
                    v[ib] = np.clip(v[ib] + db, 0.0, 1.0)
                    improved = adopt(v) or improved

            g = decode(best_v)
            for group in groups:
                for scale in (1.0 - 1.5 * step, 1.0 - 0.75 * step, 1.0 + 0.75 * step, 1.0 + 1.5 * step):
                    h = g.copy()
                    for key in group:
                        lo, hi = bounds[key]
                        h[key] = float(np.clip(h[key] * scale, lo, hi))
                    improved = adopt(encode(h)) or improved

            for ref in refs + anchor_vs:
                d = ref - best_v
                if float(np.max(np.abs(d))) < 1e-12:
                    continue
                for t in (0.2, 0.4, -0.2):
                    improved = adopt(np.clip(best_v + t * d, 0.0, 1.0)) or improved

            for _ in range(16):
                v = np.clip(best_v + rng.normal(0.0, step * 0.6, n), 0.0, 1.0)
                if rng.random() < 0.8:
                    v[index["Ki_z"]] = 0.0
                    v[index["Ki_x"]] = 0.0
                improved = adopt(v) or improved

            if not improved:
                break

    pinned = decode(best_v)
    pinned["Ki_z"] = bounds["Ki_z"][0]
    pinned["Ki_x"] = bounds["Ki_x"][0]
    adopt(encode(pinned), "pin_I")

    for step in (0.00075, 0.00035):
        for key in sweep_order:
            i = index[key]
            for d in (-step, step):
                v = best_v.copy()
                v[i] = np.clip(v[i] + d, 0.0, 1.0)
                adopt(v)

    best_gains = decode(best_v)
    for a, b in (("Kp_x", "Kd_x"), ("Kp_z", "Kd_z"), ("Kp_theta", "Kd_theta")):
        for f in (0.94, 0.98, 1.02, 1.06):
            g = best_gains.copy()
            lo, hi = bounds[a]
            g[a] = float(np.clip(g[a] * f, lo, hi))
            lo, hi = bounds[b]
            g[b] = float(np.clip(g[b] / f, lo, hi))
            adopt(encode(g))

    best_gains = decode(best_v)
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
