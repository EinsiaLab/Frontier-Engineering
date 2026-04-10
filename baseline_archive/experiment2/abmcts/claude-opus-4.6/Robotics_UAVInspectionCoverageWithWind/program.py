# EVOLVE-BLOCK-START
"""Baseline for UAVInspectionCoverageWithWind.

Strategy:
- Greedy TSP-like ordering of inspection points.
- Potential field navigation with strong obstacle/boundary avoidance.
- Multi-pass optimization with velocity management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _wind_velocity(scene: dict[str, Any], t: float) -> np.ndarray:
    wind = scene["wind"]
    base = np.array(wind["base"], dtype=float)
    amp = np.array(wind["amplitude"], dtype=float)
    freq = np.array(wind["frequency"], dtype=float)
    phase = np.array(wind["phase"], dtype=float)
    return base + amp * np.sin(freq * t + phase)


def _dynamic_obstacle_position(obstacle: dict[str, Any], t: float) -> np.ndarray:
    traj = obstacle.get("trajectory", [])
    if not isinstance(traj, list) or len(traj) == 0:
        return np.array([1e9, 1e9, 1e9], dtype=float)

    t_nodes = np.array([float(node["t"]) for node in traj], dtype=float)
    p_nodes = np.array([node["pos"] for node in traj], dtype=float)
    if t <= float(t_nodes[0]):
        return p_nodes[0]
    if t >= float(t_nodes[-1]):
        return p_nodes[-1]

    idx = int(np.searchsorted(t_nodes, t, side="right") - 1)
    idx = max(0, min(idx, len(t_nodes) - 2))
    t0, t1 = float(t_nodes[idx]), float(t_nodes[idx + 1])
    p0, p1 = p_nodes[idx], p_nodes[idx + 1]
    alpha = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
    return p0 + alpha * (p1 - p0)


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def _greedy_tsp_order(points: np.ndarray, start_pos: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor TSP ordering."""
    n = len(points)
    remaining = set(range(n))
    order = []
    current = start_pos.copy()
    for _ in range(n):
        best_idx = -1
        best_dist = float('inf')
        for idx in remaining:
            d = float(np.linalg.norm(points[idx] - current))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        order.append(best_idx)
        current = points[best_idx].copy()
        remaining.remove(best_idx)
    return order


def _is_in_no_fly(pos: np.ndarray, scene: dict[str, Any], margin: float = 0.0) -> bool:
    for zone in scene.get("no_fly_zones", []):
        if zone.get("type") == "box":
            pmin = np.array(zone["min"], dtype=float) - margin
            pmax = np.array(zone["max"], dtype=float) + margin
            if np.all(pos >= pmin) and np.all(pos <= pmax):
                return True
    return False


def _check_dyn_collision(pos: np.ndarray, scene: dict[str, Any], t: float, margin: float = 0.3) -> bool:
    for obs in scene.get("dynamic_obstacles", []):
        radius = float(obs.get("radius", 0.0))
        center = _dynamic_obstacle_position(obs, t)
        if float(np.linalg.norm(pos - center)) < radius + margin:
            return True
    return False


def _simulate_step(pos, vel, a_cmd, dt, wind_v, v_max):
    """Simulate one step and return new pos, vel."""
    vel_new = vel + a_cmd * dt
    vel_new = _clip_norm(vel_new, v_max)
    pos_new = pos + (vel_new + wind_v) * dt
    return pos_new, vel_new


def build_submission_for_scene(scene: dict[str, Any], dt: float, coverage_radius: float) -> dict[str, Any]:
    t_max = float(scene["T_max"])
    v_max = float(scene["uav"]["v_max"])
    a_max = float(scene["uav"]["a_max"])
    points = np.array(scene["inspection_points"], dtype=float)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    bounds = scene["bounds"]
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    state = np.array(scene["start"], dtype=float)
    pos = state[:3].copy()
    vel = state[3:].copy()

    # Compute greedy TSP order
    visit_order = _greedy_tsp_order(points, pos)
    current_target_idx_in_order = 0

    timestamps: list[float] = [0.0]
    controls: list[list[float]] = [[0.0, 0.0, 0.0]]

    t = 0.0
    
    # Safety margins
    safe_v = 0.92 * v_max
    safe_a = 0.90 * a_max
    bound_margin = 1.5
    bound_hard = 0.2

    while t + dt <= t_max + 1e-12:
        # Update coverage
        dists = np.linalg.norm(points - pos, axis=1)
        visited |= dists <= coverage_radius

        # Find next target using TSP order
        target = None
        while current_target_idx_in_order < n_points:
            candidate = visit_order[current_target_idx_in_order]
            if not visited[candidate]:
                target = points[candidate]
                break
            current_target_idx_in_order += 1
        
        if target is None:
            # All visited via order, check for any remaining
            unvisited = np.where(~visited)[0]
            if len(unvisited) > 0:
                nearest_idx = int(unvisited[np.argmin(dists[unvisited])])
                target = points[nearest_idx]
            else:
                # All covered - hover with minimal energy
                a_cmd = -vel * 0.5 / max(dt, 1e-6)
                a_cmd = _clip_norm(a_cmd, 0.3 * a_max)
                vel_new = vel + a_cmd * dt
                wind_v = _wind_velocity(scene, t)
                pos = pos + (vel_new + wind_v) * dt
                vel = vel_new
                t = round(t + dt, 10)
                timestamps.append(float(t))
                controls.append([float(a_cmd[0]), float(a_cmd[1]), float(a_cmd[2])])
                continue

        wind_v = _wind_velocity(scene, t)

        # Compute desired velocity toward target
        to_target = target - pos
        dist_target = float(np.linalg.norm(to_target))

        # Adaptive speed based on distance
        if dist_target > 5.0:
            speed_factor = 0.85
        elif dist_target > 2.0:
            speed_factor = 0.7
        elif dist_target > 1.0:
            speed_factor = 0.5
        else:
            speed_factor = 0.3

        if dist_target > 1e-6:
            desired_vel = (to_target / dist_target) * speed_factor * v_max
        else:
            desired_vel = np.zeros(3)

        # Repulsion forces
        repel = np.zeros(3, dtype=float)

        # Boundary avoidance with smooth gradient
        for dim, (lo, hi) in enumerate([(xmin, xmax), (ymin, ymax), (zmin, zmax)]):
            dist_lo = pos[dim] - lo
            dist_hi = hi - pos[dim]
            if dist_lo < bound_margin:
                strength = 8.0 * (bound_margin - dist_lo) / max(bound_margin, 1e-6)
                if dist_lo < bound_hard:
                    strength += 20.0
                repel[dim] += strength
            if dist_hi < bound_margin:
                strength = 8.0 * (bound_margin - dist_hi) / max(bound_margin, 1e-6)
                if dist_hi < bound_hard:
                    strength += 20.0
                repel[dim] -= strength

        # No-fly zone avoidance
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") != "box":
                continue
            pmin = np.array(zone["min"], dtype=float)
            pmax = np.array(zone["max"], dtype=float)
            center = 0.5 * (pmin + pmax)
            closest = np.clip(pos, pmin, pmax)
            delta = pos - closest
            dist = float(np.linalg.norm(delta))
            influence = 3.5

            if dist < influence:
                if dist < 1e-8:
                    delta = pos - center
                    n = float(np.linalg.norm(delta))
                    if n < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0])
                    else:
                        delta = delta / n
                else:
                    delta = delta / dist
                repel += 8.0 * (influence - dist) / max(influence, 1e-6) * delta
                if dist < 0.5:
                    repel += 15.0 * delta

        # Dynamic obstacle avoidance with prediction
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            if radius <= 0.0:
                continue
            for lookahead in range(8):
                future_t = t + lookahead * dt
                center = _dynamic_obstacle_position(obs, future_t)
                safe_dist = radius + 2.5
                delta = pos - center
                dist = float(np.linalg.norm(delta))

                if dist < safe_dist:
                    if dist < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0])
                    else:
                        delta = delta / dist
                    decay = max(0.0, 1.0 - lookahead * 0.1)
                    strength = 10.0 * (safe_dist - dist) / max(safe_dist, 1e-6) * decay
                    if dist < radius + 0.5:
                        strength += 20.0 * decay
                    repel += strength * delta

        # Wind compensation
        wind_comp = -0.6 * wind_v

        # PD control
        a_cmd = 2.0 * (desired_vel - vel) / max(dt * 10, 0.5) + wind_comp + repel
        a_cmd = _clip_norm(a_cmd, safe_a)

        # Simulate and check safety
        vel_new = vel + a_cmd * dt
        vel_new = _clip_norm(vel_new, safe_v)
        pos_new = pos + (vel_new + wind_v) * dt

        # Check bounds
        in_bounds = (xmin + bound_hard <= pos_new[0] <= xmax - bound_hard and
                     ymin + bound_hard <= pos_new[1] <= ymax - bound_hard and
                     zmin + bound_hard <= pos_new[2] <= zmax - bound_hard)

        in_no_fly = _is_in_no_fly(pos_new, scene)
        dyn_coll = _check_dyn_collision(pos_new, scene, t + dt, 0.3)

        if not in_bounds or in_no_fly or dyn_coll:
            # Emergency: strong braking + repulsion only
            a_cmd = _clip_norm(repel - 2.0 * vel, safe_a)
            vel_new = vel + a_cmd * dt
            vel_new = _clip_norm(vel_new, 0.4 * v_max)
            pos_new = pos + (vel_new + wind_v) * dt

            # If still bad, just brake hard
            if not (xmin + 0.05 <= pos_new[0] <= xmax - 0.05 and
                    ymin + 0.05 <= pos_new[1] <= ymax - 0.05 and
                    zmin + 0.05 <= pos_new[2] <= zmax - 0.05):
                a_cmd = _clip_norm(-vel / max(dt, 1e-6), safe_a)
                vel_new = vel + a_cmd * dt
                vel_new = _clip_norm(vel_new, 0.2 * v_max)
                pos_new = pos + (vel_new + wind_v) * dt

        vel = vel_new
        pos = pos_new
        t = round(t + dt, 10)

        timestamps.append(float(t))
        controls.append([float(a_cmd[0]), float(a_cmd[1]), float(a_cmd[2])])

    return {"id": scene["id"], "timestamps": timestamps, "controls": controls}


def main() -> None:
    task_root = Path(__file__).resolve().parents[1]
    scenarios_path = task_root / "references" / "scenarios.json"

    with scenarios_path.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    dt = float(cfg.get("dt", 0.1))
    coverage_radius = float(cfg.get("coverage_radius", 0.5))
    scenario_entries = [
        build_submission_for_scene(scene, dt=dt, coverage_radius=coverage_radius)
        for scene in cfg["scenarios"]
    ]

    submission = {"scenarios": scenario_entries}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print("Baseline submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END