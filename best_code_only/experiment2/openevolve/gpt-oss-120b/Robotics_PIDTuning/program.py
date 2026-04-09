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
import concurrent.futures


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

    Returns 0.0 if any scenario is infeasible or ITAE is non‑positive.
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
# Helper for parallel evaluation (must be top‑level so it is picklable)
# ---------------------------------------------------------------------------
def _evaluate_candidate(args: tuple[dict[str, float], dict[str, Any]]) -> float:
    """Unpack the arguments and return the ITAE‑based score."""
    gains, cfg = args
    return compute_itae(gains, cfg)


# ---------------------------------------------------------------------------
# ALLOWED TO MODIFY — Optimizer
# ---------------------------------------------------------------------------

def optimize_pid_gains() -> dict[str, float]:
    """Optimize PID gains using a simple random search.

    Returns the best gain dict found.
    """
    cfg = load_config()
    gain_ranges = cfg["gains"]
    # Use a fresh RNG without a fixed seed for broader exploration across runs
    rng = np.random.default_rng()
    # --------------------------------------------------------------------
    # Cache for evaluated gain sets – avoids re‑simulating identical vectors
    # --------------------------------------------------------------------
    _score_cache: dict[tuple[float, ...], float] = {}

    def evaluate(gains: dict[str, float]) -> float:
        """Return a cached score for *gains* (computes it once)."""
        # Build a deterministic hashable key from the ordered gain list
        key = tuple(round(gains[k], 6) for _, _, k in keys_order)
        if key not in _score_cache:
            _score_cache[key] = compute_itae(gains, cfg)
        return _score_cache[key]

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

    # Start with a hand-tuned baseline
    best_gains: dict[str, float] = {
        "Kp_z": 8.0, "Ki_z": 0.5, "Kd_z": 4.0, "N_z": 20.0,
        "Kp_x": 0.1, "Ki_x": 0.01, "Kd_x": 0.1, "N_x": 10.0,
        "Kp_theta": 10.0, "Ki_theta": 0.5, "Kd_theta": 3.0, "N_theta": 20.0,
    }
    # Evaluate the hand‑tuned baseline (cached)
    best_score = evaluate(best_gains)
    print(f"Baseline score: {best_score:.6f}")

    # Random search – evaluate candidates in parallel
    # More random candidates give a better chance of finding strong gains
    n_iter = 500                     # a little more random search budget
    batch_size = 20                  # how many candidates each parallel call evaluates
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(0, n_iter, batch_size):
            # generate a batch of candidates
            batch = [random_gains() for _ in range(min(batch_size, n_iter - i))]
            # map each (gains, cfg) pair to the worker pool
            scores = list(
                executor.map(_evaluate_candidate, [(g, cfg) for g in batch])
            )
            for candidate, score in zip(batch, scores):
                if score > best_score:
                    best_score = score
                    best_gains = candidate
                    print(f"  iter {i + batch.index(candidate)}: new best = {best_score:.6f}")

    # Local refinement around best – also run in parallel batches
    n_refine = 250                  # deeper local refinement
    batch_size = 10
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(n_refine):
            batch = []
            for _ in range(batch_size):
                perturbed: dict[str, float] = {}
                for group, param, key in keys_order:
                    lo, hi = gain_ranges[group][param]
                    noise = rng.normal(0, 0.05 * (hi - lo))
                    perturbed[key] = float(np.clip(best_gains[key] + noise, lo, hi))
                batch.append(perturbed)

            scores = list(
                executor.map(_evaluate_candidate, [(g, cfg) for g in batch])
            )
            for perturbed, score in zip(batch, scores):
                if score > best_score:
                    best_score = score
                    best_gains = perturbed
                    print(f"  refine {i}: new best = {best_score:.6f}")

    print(f"Final best score: {best_score:.6f}")

    # -----------------------------------------------------------------------
    # Lightweight pattern‑search refinement
    # -----------------------------------------------------------------------
    # Try small +/- steps for each gain individually. If any step improves the
    # score we keep it and continue the search. This is inexpensive because it
    # only evaluates O(num_gains) extra simulations per outer iteration.
    step_factor = 0.02   # 2 % of the allowed range
    improved = True
    while improved:
        improved = False
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            range_width = hi - lo
            step = step_factor * range_width

            for delta in (-step, step):
                trial = best_gains.copy()
                trial[key] = float(np.clip(trial[key] + delta, lo, hi))
                # Use cached evaluation for pattern‑search steps
                trial_score = evaluate(trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_gains = trial
                    improved = True
                    # Restart the loop immediately after an improvement
                    break
            if improved:
                break

    print(f"Post‑pattern‑search best score: {best_score:.6f}")
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
