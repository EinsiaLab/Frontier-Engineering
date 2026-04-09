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
    """Optimize PID gains using evolutionary strategies.

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

    def evaluate_gains(g: dict[str, float]) -> float:
        """Evaluate gains and return score."""
        try:
            return compute_itae(g, cfg)
        except Exception:
            return 0.0

    # Start with multiple diverse initial configurations
    candidates: list[tuple[dict[str, float], float]] = []
    
    # Hand-tuned baseline
    baseline: dict[str, float] = {
        "Kp_z": 8.0, "Ki_z": 0.5, "Kd_z": 4.0, "N_z": 20.0,
        "Kp_x": 0.1, "Ki_x": 0.01, "Kd_x": 0.1, "N_x": 10.0,
        "Kp_theta": 10.0, "Ki_theta": 0.5, "Kd_theta": 3.0, "N_theta": 20.0,
    }
    candidates.append((baseline.copy(), evaluate_gains(baseline)))
    
    # Generate diverse random candidates
    for _ in range(20):
        g = random_gains()
        candidates.append((g, evaluate_gains(g)))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_gains, best_score = candidates[0]
    print(f"Initial best score: {best_score:.6f}")

    # Evolutionary optimization with improved strategies
    pop_size = 15
    generations = 40
    
    for gen in range(generations):
        # Create next generation using tournament selection and mutation
        new_pop: list[tuple[dict[str, float], float]] = []
        
        # Elitism - keep best individuals with increased preservation
        num_elites = min(5, len(candidates))
        for i in range(num_elites):
            new_pop.append((candidates[i][0].copy(), candidates[i][1]))
        
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection with stronger pressure (size 4)
            idxs = rng.choice(len(candidates), 4, replace=False)
            tournament = [(candidates[i][0], candidates[i][1]) for i in idxs]
            tournament.sort(key=lambda x: x[1], reverse=True)
            parent1 = tournament[0][0]
            parent2 = tournament[1][0]
            
            # Crossover with adaptive probability
            crossover_prob = 0.8 + 0.15 * (1.0 - gen / generations)
            child: dict[str, float] = {}
            for group, param, key in keys_order:
                if rng.random() < crossover_prob:
                    if rng.random() < 0.5:
                        child[key] = parent1[key]
                    else:
                        child[key] = parent2[key]
                else:
                    lo, hi = gain_ranges[group][param]
                    child[key] = float(rng.uniform(lo, hi))
            
            # Improved adaptive mutation rate with cooling schedule
            base_mutation_rate = 0.15 * (0.95 ** (gen / 10))
            mutation_rate = max(0.03, base_mutation_rate)
            
            # Mutate with dynamic noise scaling
            for group, param, key in keys_order:
                lo, hi = gain_ranges[group][param]
                if rng.random() < mutation_rate:
                    # Dynamic mutation strength based on search progress
                    progress = gen / generations
                    noise_scale = (1.0 - 0.5 * progress) * (hi - lo)
                    
                    mutation_type = rng.choice(['gaussian', 'uniform', 'large_step', 'local'], 
                                              p=[0.4, 0.2, 0.1, 0.3])
                    if mutation_type == 'gaussian':
                        noise = rng.normal(0, 0.03 * noise_scale)
                        child[key] = float(np.clip(child[key] + noise, lo, hi))
                    elif mutation_type == 'uniform':
                        child[key] = float(rng.uniform(lo, hi))
                    elif mutation_type == 'large_step':
                        child[key] = float(rng.uniform(lo, hi))
                    else:  # local refinement
                        noise = rng.normal(0, 0.01 * noise_scale)
                        child[key] = float(np.clip(child[key] + noise, lo, hi))
            
            new_pop.append((child, evaluate_gains(child)))
        
        # Add diversity injection periodically with improved strategy
        if gen > 0 and gen % 8 == 0:
            # Inject diverse candidates with different characteristics
            diversity_count = 3
            
            # Strategy 1: Random diverse candidates
            for _ in range(diversity_count):
                g = random_gains()
                new_pop.append((g, evaluate_gains(g)))
            
            # Strategy 2: Perturb current best to maintain diversity
            for _ in range(2):
                perturbed: dict[str, float] = {}
                for group, param, key in keys_order:
                    lo, hi = gain_ranges[group][param]
                    # Larger perturbation for diversity
                    noise = rng.normal(0, 0.15 * (hi - lo))
                    perturbed[key] = float(np.clip(candidates[0][0][key] + noise, lo, hi))
                new_pop.append((perturbed, evaluate_gains(perturbed)))
        
        candidates = new_pop
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates[0][1] > best_score:
            best_score = candidates[0][1]
            best_gains = candidates[0][0].copy()
            print(f"  gen {gen}: new best = {best_score:.6f}")
    
    # Enhanced local search with multiple strategies and adaptive noise scaling
    local_iter = 100
    # Store best gains from different search phases for diversity
    best_gains_history: list[tuple[dict[str, float], float]] = []
    
    for i in range(local_iter):
        candidate: dict[str, float] = {}
        
        # Adaptive search intensity with phase-based strategy
        progress = i / local_iter
        
        # Phase 1: Exploration (0-30%)
        if progress < 0.3:
            base_noise = 0.1 * (1.0 - 2.0 * progress)
            for group, param, key in keys_order:
                lo, hi = gain_ranges[group][param]
                noise = rng.normal(0, base_noise * (hi - lo))
                candidate[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        
        # Phase 2: Exploitation (30-70%)
        elif progress < 0.7:
            base_noise = 0.03 * (1.0 - (progress - 0.3) / 0.4)
            for group, param, key in keys_order:
                lo, hi = gain_ranges[group][param]
                noise = rng.normal(0, base_noise * (hi - lo))
                candidate[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        
        # Phase 3: Fine-tuning (70-100%)
        else:
            for group, param, key in keys_order:
                lo, hi = gain_ranges[group][param]
                # Very fine noise for precision
                noise = rng.normal(0, 0.005 * (hi - lo))
                candidate[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        
        score = evaluate_gains(candidate)
        if score > best_score:
            best_score = score
            best_gains = candidate.copy()
            print(f"  final: new best = {best_score:.6f}")
        
        # Store current best gains for history-based refinement
        if i % 20 == 0:
            best_gains_history.append((best_gains.copy(), best_score))
    
    # Additional targeted search around best gains from history
    # This helps escape local optima
    if len(best_gains_history) > 0:
        # Sort history by score
        best_gains_history.sort(key=lambda x: x[1], reverse=True)
        
        # Try small perturbations of top candidates from history
        for hist_idx in range(min(3, len(best_gains_history))):
            hist_gains, hist_score = best_gains_history[hist_idx]
            for _ in range(10):
                candidate: dict[str, float] = {}
                for group, param, key in keys_order:
                    lo, hi = gain_ranges[group][param]
                    # Small targeted perturbation
                    noise = rng.normal(0, 0.01 * (hi - lo))
                    candidate[key] = float(np.clip(hist_gains[key] + noise, lo, hi))
                
                score = evaluate_gains(candidate)
                if score > best_score:
                    best_score = score
                    best_gains = candidate.copy()
                    print(f"  history_refine: new best = {best_score:.6f}")
    
    # Aggressive local search around current best
    for _ in range(30):
        candidate: dict[str, float] = {}
        for group, param, key in keys_order:
            lo, hi = gain_ranges[group][param]
            noise = rng.normal(0, 0.02 * (hi - lo))
            candidate[key] = float(np.clip(best_gains[key] + noise, lo, hi))
        score = evaluate_gains(candidate)
        if score > best_score:
            best_score = score
            best_gains = candidate.copy()
            print(f"  final: new best = {best_score:.6f}")

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
