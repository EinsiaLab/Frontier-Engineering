# EVOLVE-BLOCK-START
"""Baseline for UAVInspectionCoverageWithWind.

Strategy:
- Waypoint-based navigation through inspection points.
- Strong repulsion from all constraints.
- Predictive obstacle avoidance.
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


def build_submission_for_scene(scene: dict[str, Any], dt: float, coverage_radius: float) -> dict[str, Any]:
    t_max = float(scene["T_max"])
    v_max = float(scene["uav"]["v_max"])
    a_max = float(scene["uav"]["a_max"])
    points = np.array(scene["inspection_points"], dtype=float)
    visited = np.zeros(len(points), dtype=bool)
    bounds = scene["bounds"]
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    state = np.array(scene["start"], dtype=float)
    pos = state[:3].copy()
    vel = state[3:].copy()

    timestamps: list[float] = [0.0]
    controls: list[list[float]] = [[0.0, 0.0, 0.0]]

    t = 0.0

    while t + dt <= t_max + 1e-12:
        # Update coverage
        dists = np.linalg.norm(points - pos, axis=1)
        visited |= dists <= coverage_radius

        # Find next target
        unvisited = np.where(~visited)[0]
        if len(unvisited) > 0:
            nearest_idx = int(unvisited[np.argmin(dists[unvisited])])
            target = points[nearest_idx]
        else:
            target = points[int(np.argmin(dists))]

        wind_v = _wind_velocity(scene, t)

        # Compute base desired velocity toward target
        to_target = target - pos
        dist_target = float(np.linalg.norm(to_target))

        # Adaptive speed based on obstacle density and distance
        obstacle_density = 0.0
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            if radius <= 0.0:
                continue
            center = _dynamic_obstacle_position(obs, t)
            dist_to_obs = float(np.linalg.norm(pos - center))
            if dist_to_obs < radius + 3.0:
                obstacle_density += 1.0 / (dist_to_obs + 0.1)
        
        base_speed = 0.7 if dist_target > 3.0 else (0.5 if dist_target > 1.5 else 0.35)
        speed_factor = max(0.35, base_speed * (1.0 - 0.1 * obstacle_density))
        desired_vel = _clip_norm(to_target * 1.5, speed_factor * v_max)

        # Strong boundary avoidance
        margin = 2.5
        repel = np.zeros(3, dtype=float)
        
        # Unified boundary repulsion using vectorized calculations
        boundaries = [
            (xmin, xmax, 0, margin),
            (ymin, ymax, 1, margin),
            (zmin, zmax, 2, margin * 0.8)
        ]
        
        for low, high, axis, m in boundaries:
            if pos[axis] < low + m:
                repel[axis] += 6.0 * (low + m - pos[axis])
            elif pos[axis] > high - m:
                repel[axis] -= 6.0 * (pos[axis] - (high - m))

        # No-fly zone avoidance
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") != "box":
                continue
            pmin = np.array(zone["min"], dtype=float)
            pmax = np.array(zone["max"], dtype=float)

            # Calculate distance to box using vectorized approach
            closest = np.clip(pos, pmin, pmax)
            diff = pos - closest
            dist = float(np.sqrt(np.sum(diff * diff)))
            
            # More aggressive avoidance: increase influence and strength
            influence = 4.5  # Increase from 3.0
            repel_strength = 12.0  # Increase from 6.0

            if dist < influence:
                delta = pos - closest  # Use closest point on box instead of center for better direction
                n = float(np.linalg.norm(delta))
                if n < 1e-8:
                    # If inside the box, move toward center of box from pos
                    center = 0.5 * (pmin + pmax)
                    delta = center - pos
                    n = float(np.linalg.norm(delta))
                    if n < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                    else:
                        delta = delta / n
                else:
                    delta = delta / n
                repel += repel_strength * (influence - dist) * delta

        # Dynamic obstacle avoidance
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            if radius <= 0.0:
                continue

            # Look ahead several time steps
            for lookahead in range(6):
                future_t = t + lookahead * dt
                center = _dynamic_obstacle_position(obs, future_t)
                influence = radius + 2.0 + lookahead * 0.1

                delta = pos - center
                dist = float(np.linalg.norm(delta))

                if dist < influence:
                    if dist < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                        dist = 0.0
                    else:
                        delta = delta / dist

                    strength = 6.0 * (influence - dist) * (1.0 - lookahead * 0.12)
                    repel += strength * delta

        # Combine attraction and repulsion
        # Prioritize constraint avoidance over target acquisition
        a_cmd = 1.5 * (desired_vel - vel) - 0.8 * wind_v + repel
        # Reduce control authority when close to no-fly zones
        min_dist_to_zones = float('inf')
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") != "box":
                continue
            pmin = np.array(zone["min"], dtype=float)
            pmax = np.array(zone["max"], dtype=float)
            closest = np.clip(pos, pmin, pmax)
            diff = pos - closest
            dist = float(np.sqrt(np.sum(diff * diff)))
            if dist < min_dist_to_zones:
                min_dist_to_zones = dist
        
        # If very close to no-fly zone, reduce control authority
        if min_dist_to_zones < 2.0:
            a_cmd *= 0.7
        a_cmd = _clip_norm(a_cmd, 0.82 * a_max)

        # Simulate step
        vel_new = vel + a_cmd * dt
        vel_new = _clip_norm(vel_new, 0.88 * v_max)
        pos_new = pos + (vel_new + wind_v) * dt

        # Safety check - predict violations with more conservative no-fly zone check
        safety_margin = 0.3
        in_bounds = (xmin + safety_margin <= pos_new[0] <= xmax - safety_margin and
                     ymin + safety_margin <= pos_new[1] <= ymax - safety_margin and
                     zmin + safety_margin <= pos_new[2] <= zmax - safety_margin)

        # Check no-fly zones more thoroughly with expanded safety margin
        in_no_fly = False
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") != "box":
                continue
            pmin = np.array(zone["min"], dtype=float)
            pmax = np.array(zone["max"], dtype=float)
            # Use expanded bounds for safety margin
            expanded_pmin = pmin + np.array([0.2, 0.2, 0.2])
            expanded_pmax = pmax - np.array([0.2, 0.2, 0.2])
            if np.all(pos_new >= expanded_pmin) and np.all(pos_new <= expanded_pmax):
                in_no_fly = True
                break

        dyn_collision = False
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            center = _dynamic_obstacle_position(obs, t + dt)
            if float(np.linalg.norm(pos_new - center)) < radius + 0.3:
                dyn_collision = True
                break

        if not in_bounds or in_no_fly or dyn_collision:
            # Reduce speed significantly
            a_cmd = _clip_norm(0.3 * a_cmd, 0.4 * a_max)
            vel_new = vel + a_cmd * dt
            vel_new = _clip_norm(vel_new, 0.5 * v_max)

        vel = vel_new
        pos = pos + (vel + wind_v) * dt
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