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

        # Find next target with clustering awareness
        unvisited = np.where(~visited)[0]
        if len(unvisited) > 0:
            # Score candidates by distance minus cluster bonus
            sorted_idx = np.argsort(dists[unvisited])[:min(8, len(unvisited))]
            candidates = unvisited[sorted_idx]
            best_idx = candidates[0]
            best_score = float('inf')
            for idx in candidates:
                d = dists[idx]
                # Bonus for points near other unvisited points
                nearby = np.sum(np.linalg.norm(points[unvisited] - points[idx], axis=1) < coverage_radius * 3)
                score = d - nearby * 0.4
                if score < best_score:
                    best_score = score
                    best_idx = idx
            target = points[best_idx]
        else:
            target = points[int(np.argmin(dists))]

        wind_v = _wind_velocity(scene, t)

        # Compute base desired velocity toward target
        to_target = target - pos
        dist_target = float(np.linalg.norm(to_target))

        # Check if in safe zone (far from boundaries and dynamic obstacles)
        safe_dist = min(pos[0] - xmin, xmax - pos[0], pos[1] - ymin, ymax - pos[1], pos[2] - zmin, zmax - pos[2])
        
        # Check proximity to dynamic obstacles
        near_obstacle = False
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            center = _dynamic_obstacle_position(obs, t)
            if float(np.linalg.norm(pos - center)) < radius + 5.0:
                near_obstacle = True
                break
        
        is_safe = safe_dist > 4.0 and not near_obstacle
        
        # Adaptive speed - conservative near obstacles
        if is_safe:
            speed_factor = 0.75 if dist_target > 3.0 else (0.55 if dist_target > 1.5 else 0.4)
        else:
            speed_factor = 0.55 if dist_target > 3.0 else (0.4 if dist_target > 1.5 else 0.3)
        desired_vel = _clip_norm(to_target * 1.8, speed_factor * v_max)

        # Strong boundary avoidance
        margin = 2.5
        repel = np.zeros(3, dtype=float)

        # X boundary
        if pos[0] < xmin + margin:
            repel[0] += 6.0 * (xmin + margin - pos[0])
        elif pos[0] > xmax - margin:
            repel[0] -= 6.0 * (pos[0] - (xmax - margin))

        # Y boundary
        if pos[1] < ymin + margin:
            repel[1] += 6.0 * (ymin + margin - pos[1])
        elif pos[1] > ymax - margin:
            repel[1] -= 6.0 * (pos[1] - (ymax - margin))

        # Z boundary
        z_margin = margin * 0.8
        if pos[2] < zmin + z_margin:
            repel[2] += 6.0 * (zmin + z_margin - pos[2])
        elif pos[2] > zmax - z_margin:
            repel[2] -= 6.0 * (pos[2] - (zmax - z_margin))

        # No-fly zone avoidance - stronger and earlier
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") != "box":
                continue
            pmin = np.array(zone["min"], dtype=float)
            pmax = np.array(zone["max"], dtype=float)
            center = 0.5 * (pmin + pmax)

            closest = np.clip(pos, pmin, pmax)
            dist = float(np.linalg.norm(pos - closest))
            influence = 4.5  # Larger influence for earlier avoidance

            if dist < influence:
                delta = pos - center
                n = float(np.linalg.norm(delta))
                if n < 1e-8:
                    delta = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    delta = delta / n
                repel += 8.0 * (influence - dist) * delta

        # Dynamic obstacle avoidance - more lookahead and stronger repulsion
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            if radius <= 0.0:
                continue

            # Look ahead more time steps for better prediction
            for lookahead in range(10):
                future_t = t + lookahead * dt
                center = _dynamic_obstacle_position(obs, future_t)
                # Larger influence radius for earlier avoidance
                influence = radius + 3.0 + lookahead * 0.15

                delta = pos - center
                dist = float(np.linalg.norm(delta))

                if dist < influence:
                    if dist < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                        dist = 0.0
                    else:
                        delta = delta / dist

                    # Stronger repulsion, less decay with lookahead
                    strength = 8.0 * (influence - dist) * (1.0 - lookahead * 0.08)
                    repel += strength * delta

        # Combine attraction and repulsion with tuned gains
        a_cmd = 1.8 * (desired_vel - vel) - 0.9 * wind_v + repel
        a_cmd = _clip_norm(a_cmd, 0.85 * a_max)

        # Simulate step with conservative velocity limit
        vel_new = vel + a_cmd * dt
        vel_new = _clip_norm(vel_new, 0.9 * v_max)
        pos_new = pos + (vel_new + wind_v) * dt

        # Safety check - predict violations
        safety_margin = 0.3
        in_bounds = (xmin + safety_margin <= pos_new[0] <= xmax - safety_margin and
                     ymin + safety_margin <= pos_new[1] <= ymax - safety_margin and
                     zmin + safety_margin <= pos_new[2] <= zmax - safety_margin)

        in_no_fly = False
        nfz_margin = 0.5  # Safety margin around no-fly zones
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") == "box":
                pmin = np.array(zone["min"], dtype=float)
                pmax = np.array(zone["max"], dtype=float)
                # Check with margin - reject if too close to no-fly zone
                if (np.all(pos_new >= pmin - nfz_margin) and 
                    np.all(pos_new <= pmax + nfz_margin)):
                    in_no_fly = True
                    break

        dyn_collision = False
        for obs in scene.get("dynamic_obstacles", []):
            radius = float(obs.get("radius", 0.0))
            # Check multiple future positions for collision
            for check_ahead in range(3):
                center = _dynamic_obstacle_position(obs, t + dt * (1 + check_ahead))
                if float(np.linalg.norm(pos_new - center)) < radius + 0.5:
                    dyn_collision = True
                    break
            if dyn_collision:
                break

        if not in_bounds or in_no_fly or dyn_collision:
            # Emergency slowdown - more aggressive reduction
            a_cmd = _clip_norm(0.2 * a_cmd, 0.3 * a_max)
            vel_new = vel + a_cmd * dt
            vel_new = _clip_norm(vel_new, 0.4 * v_max)

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