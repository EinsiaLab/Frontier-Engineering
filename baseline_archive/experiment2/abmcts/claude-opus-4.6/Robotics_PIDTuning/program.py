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
import time
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
    """Optimize PID gains using a multi-phase optimization strategy.

    Returns the best gain dict found.
    """
    start_time = time.time()
    TIME_BUDGET = 270  # seconds, leave margin for evaluation

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

    dim = len(keys_order)
    bounds_lo = np.array([gain_ranges[g][p][0] for g, p, _ in keys_order])
    bounds_hi = np.array([gain_ranges[g][p][1] for g, p, _ in keys_order])
    bounds_range = bounds_hi - bounds_lo
    bounds_mid = 0.5 * (bounds_lo + bounds_hi)

    def vec_to_gains(v: np.ndarray) -> dict[str, float]:
        clipped = np.clip(v, bounds_lo, bounds_hi)
        return {key: float(clipped[i]) for i, (_, _, key) in enumerate(keys_order)}

    def gains_to_vec(g: dict[str, float]) -> np.ndarray:
        return np.array([g[key] for _, _, key in keys_order])

    def evaluate(v: np.ndarray) -> float:
        return compute_itae(vec_to_gains(v), cfg)

    def time_left():
        return TIME_BUDGET - (time.time() - start_time)

    # Starting points - wide variety
    starting_points = [
        {"Kp_z": 15.0, "Ki_z": 1.0, "Kd_z": 8.0, "N_z": 25.0,
         "Kp_x": 0.3, "Ki_x": 0.02, "Kd_x": 0.3, "N_x": 15.0,
         "Kp_theta": 15.0, "Ki_theta": 1.0, "Kd_theta": 5.0, "N_theta": 25.0},
        {"Kp_z": 20.0, "Ki_z": 2.0, "Kd_z": 10.0, "N_z": 30.0,
         "Kp_x": 0.5, "Ki_x": 0.05, "Kd_x": 0.5, "N_x": 20.0,
         "Kp_theta": 20.0, "Ki_theta": 2.0, "Kd_theta": 8.0, "N_theta": 30.0},
        {"Kp_z": 25.0, "Ki_z": 3.0, "Kd_z": 12.0, "N_z": 40.0,
         "Kp_x": 0.8, "Ki_x": 0.08, "Kd_x": 0.8, "N_x": 25.0,
         "Kp_theta": 25.0, "Ki_theta": 3.0, "Kd_theta": 10.0, "N_theta": 40.0},
        {"Kp_z": 30.0, "Ki_z": 1.5, "Kd_z": 15.0, "N_z": 50.0,
         "Kp_x": 1.0, "Ki_x": 0.1, "Kd_x": 1.0, "N_x": 30.0,
         "Kp_theta": 30.0, "Ki_theta": 1.5, "Kd_theta": 12.0, "N_theta": 50.0},
        {"Kp_z": 8.0, "Ki_z": 0.5, "Kd_z": 4.0, "N_z": 20.0,
         "Kp_x": 0.1, "Ki_x": 0.01, "Kd_x": 0.1, "N_x": 10.0,
         "Kp_theta": 10.0, "Ki_theta": 0.5, "Kd_theta": 3.0, "N_theta": 20.0},
        # Aggressive gains
        {"Kp_z": 40.0, "Ki_z": 4.0, "Kd_z": 18.0, "N_z": 60.0,
         "Kp_x": 1.5, "Ki_x": 0.15, "Kd_x": 1.5, "N_x": 40.0,
         "Kp_theta": 35.0, "Ki_theta": 4.0, "Kd_theta": 15.0, "N_theta": 50.0},
        # Conservative gains
        {"Kp_z": 12.0, "Ki_z": 0.8, "Kd_z": 6.0, "N_z": 15.0,
         "Kp_x": 0.2, "Ki_x": 0.015, "Kd_x": 0.2, "N_x": 12.0,
         "Kp_theta": 12.0, "Ki_theta": 0.8, "Kd_theta": 4.0, "N_theta": 15.0},
        # High derivative
        {"Kp_z": 18.0, "Ki_z": 0.5, "Kd_z": 20.0, "N_z": 35.0,
         "Kp_x": 0.4, "Ki_x": 0.01, "Kd_x": 1.0, "N_x": 20.0,
         "Kp_theta": 18.0, "Ki_theta": 0.5, "Kd_theta": 12.0, "N_theta": 35.0},
        # High integral
        {"Kp_z": 15.0, "Ki_z": 5.0, "Kd_z": 8.0, "N_z": 25.0,
         "Kp_x": 0.3, "Ki_x": 0.2, "Kd_x": 0.3, "N_x": 15.0,
         "Kp_theta": 15.0, "Ki_theta": 5.0, "Kd_theta": 5.0, "N_theta": 25.0},
        # Very high N (fast derivative filter)
        {"Kp_z": 20.0, "Ki_z": 2.0, "Kd_z": 10.0, "N_z": 80.0,
         "Kp_x": 0.5, "Ki_x": 0.05, "Kd_x": 0.5, "N_x": 80.0,
         "Kp_theta": 20.0, "Ki_theta": 2.0, "Kd_theta": 8.0, "N_theta": 80.0},
    ]

    best_score = -1.0
    best_vec = None

    for sp in starting_points:
        v = gains_to_vec(sp)
        v = np.clip(v, bounds_lo, bounds_hi)
        s = evaluate(v)
        if s > best_score:
            best_score = s
            best_vec = v.copy()

    print(f"Best starting score: {best_score:.6f}")

    rng = np.random.default_rng(42)

    # Phase 1: Latin Hypercube-like random search
    n_random = 500
    for i in range(n_random):
        if time_left() < 200:
            break
        v = bounds_lo + rng.random(dim) * bounds_range
        s = evaluate(v)
        if s > best_score:
            best_score = s
            best_vec = v.copy()
            print(f"  random {i}: new best = {best_score:.6f}")

    # Phase 2: Differential Evolution
    pop_size = 30
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate

    # Initialize population with best found + random
    population = []
    pop_scores = []

    population.append(best_vec.copy())
    pop_scores.append(best_score)

    for _ in range(pop_size - 1):
        v = bounds_lo + rng.random(dim) * bounds_range
        # Mix some with best
        if rng.random() < 0.3:
            v = best_vec + rng.normal(0, 0.1, dim) * bounds_range
            v = np.clip(v, bounds_lo, bounds_hi)
        s = evaluate(v)
        population.append(v)
        pop_scores.append(s)
        if s > best_score:
            best_score = s
            best_vec = v.copy()

    population = np.array(population)
    pop_scores = np.array(pop_scores)

    de_generations = 150
    for gen in range(de_generations):
        if time_left() < 80:
            break
        improved = False
        for i in range(pop_size):
            # DE/rand/1/bin with best bias
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = rng.choice(idxs, 3, replace=False)

            # Use best individual sometimes
            if rng.random() < 0.3:
                base = best_vec
            else:
                base = population[a]

            mutant = base + F * (population[b] - population[c])
            mutant = np.clip(mutant, bounds_lo, bounds_hi)

            # Binomial crossover
            cross_points = rng.random(dim) < CR
            if not np.any(cross_points):
                cross_points[rng.integers(dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            s = evaluate(trial)
            if s >= pop_scores[i]:
                population[i] = trial
                pop_scores[i] = s
                if s > best_score:
                    best_score = s
                    best_vec = trial.copy()
                    improved = True
                    print(f"  DE gen {gen}: new best = {best_score:.6f}")

        if gen % 10 == 0:
            print(f"  DE gen {gen}, best = {best_score:.6f}, time left = {time_left():.0f}s")

    # Phase 3: CMA-ES around best
    mean = best_vec.copy()
    sigma = 0.05 * bounds_range
    cma_pop_size = 24
    elite_size = 8

    for gen in range(200):
        if time_left() < 40:
            break

        pop = []
        scores = []
        for _ in range(cma_pop_size):
            noise = rng.normal(0, 1, dim) * sigma
            candidate = np.clip(mean + noise, bounds_lo, bounds_hi)
            s = evaluate(candidate)
            pop.append(candidate)
            scores.append(s)

        indices = np.argsort(scores)[::-1]
        if scores[indices[0]] > best_score:
            best_score = scores[indices[0]]
            best_vec = pop[indices[0]].copy()
            print(f"  CMA gen {gen}: new best = {best_score:.6f}")

        elite = np.array([pop[indices[j]] for j in range(elite_size)])
        weights = np.log(elite_size + 0.5) - np.log(np.arange(1, elite_size + 1))
        weights /= weights.sum()
        new_mean = np.sum(elite * weights[:, None], axis=0)

        diffs = elite - new_mean[None, :]
        new_sigma = np.sqrt(np.sum(weights[:, None] * diffs**2, axis=0))
        new_sigma = np.maximum(new_sigma, 0.0005 * bounds_range)
        new_sigma = np.minimum(new_sigma, 0.15 * bounds_range)

        mean = np.clip(new_mean, bounds_lo, bounds_hi)
        sigma = new_sigma

    # Phase 4: Fine local search - Nelder-Mead style coordinate descent
    for round_idx in range(10):
        if time_left() < 15:
            break
        improved_round = False
        for d in range(dim):
            for delta_frac in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                               -0.001, -0.002, -0.005, -0.01, -0.02, -0.05, -0.1]:
                if time_left() < 10:
                    break
                candidate = best_vec.copy()
                candidate[d] = np.clip(candidate[d] + delta_frac * bounds_range[d], bounds_lo[d], bounds_hi[d])
                s = evaluate(candidate)
                if s > best_score:
                    best_score = s
                    best_vec = candidate.copy()
                    improved_round = True
        if not improved_round:
            break

    # Phase 5: Very fine random perturbations
    for i in range(1000):
        if time_left() < 5:
            break
        scale = 0.005 * rng.random()
        noise = rng.normal(0, scale, dim) * bounds_range
        candidate = np.clip(best_vec + noise, bounds_lo, bounds_hi)
        s = evaluate(candidate)
        if s > best_score:
            best_score = s
            best_vec = candidate.copy()
            print(f"  fine {i}: new best = {best_score:.6f}")

    best_gains = vec_to_gains(best_vec)
    print(f"Final best score: {best_score:.6f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
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
