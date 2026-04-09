# EVOLVE-BLOCK-START
"""UAV Inspection Coverage With Wind - Optimized.

Strategy:
- Greedy nearest-neighbor tour for inspection point ordering.
- Moderate repulsion from constraints with proper safety margins.
- Predictive obstacle avoidance.
- Careful wind handling without over-compensation.
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


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def _greedy_tsp_order(points: np.ndarray, start_pos: np.ndarray) -> list[int]:
    """Compute a greedy nearest-neighbor tour starting from start_pos, then improve with 2-opt."""
    n = len(points)
    if n == 0:
        return []
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

    # 2-opt improvement
    # Build position list: start_pos + tour points
    def tour_length(tour):
        total = float(np.linalg.norm(points[tour[0]] - start_pos))
        for i in range(len(tour) - 1):
            total += float(np.linalg.norm(points[tour[i+1]] - points[tour[i]]))
        return total

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment [i, j]
                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                # Only compute the affected edges
                # Before: edge(i-1, i) + edge(j, j+1)
                # After: edge(i-1, j) + edge(i, j+1)
                p_before_i = start_pos if i == 0 else points[order[i-1]]
                old_d1 = float(np.linalg.norm(points[order[i]] - p_before_i))
                new_d1 = float(np.linalg.norm(points[order[j]] - p_before_i))

                if j < n - 1:
                    old_d2 = float(np.linalg.norm(points[order[j+1]] - points[order[j]]))
                    new_d2 = float(np.linalg.norm(points[order[j+1]] - points[order[i]]))
                else:
                    old_d2 = 0.0
                    new_d2 = 0.0

                if new_d1 + new_d2 < old_d1 + old_d2 - 1e-10:
                    order = new_order
                    improved = True

    return order


def build_submission_for_scene(scene: dict[str, Any], dt: float, coverage_radius: float) -> dict[str, Any]:
    t_max = float(scene["T_max"])
    v_max = float(scene["uav"]["v_max"])
    a_max = float(scene["uav"]["a_max"])
    points = np.array(scene["inspection_points"], dtype=float)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    bounds = scene["bounds"]
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    no_fly_zones = scene.get("no_fly_zones", [])
    dynamic_obstacles = scene.get("dynamic_obstacles", [])

    state = np.array(scene["start"], dtype=float)
    pos = state[:3].copy()
    vel = state[3:].copy()

    # Pre-compute tour order
    tour = _greedy_tsp_order(points, pos)
    tour_idx = 0

    timestamps: list[float] = [0.0]
    controls: list[list[float]] = [[0.0, 0.0, 0.0]]

    t = 0.0

    # Precompute no-fly zone arrays
    nfz_data = []
    for zone in no_fly_zones:
        if zone.get("type") == "box":
            nfz_data.append((np.array(zone["min"], dtype=float), np.array(zone["max"], dtype=float)))

    # Precompute dynamic obstacle data
    dyn_obs_data = []
    for obs in dynamic_obstacles:
        radius = float(obs.get("radius", 0.0))
        if radius > 0.0:
            traj = obs.get("trajectory", [])
            if isinstance(traj, list) and len(traj) > 0:
                t_nodes = np.array([float(node["t"]) for node in traj], dtype=float)
                p_nodes = np.array([node["pos"] for node in traj], dtype=float)
                dyn_obs_data.append((radius, t_nodes, p_nodes))

    def _dyn_obs_pos(t_nodes, p_nodes, t_query):
        if t_query <= t_nodes[0]:
            return p_nodes[0]
        if t_query >= t_nodes[-1]:
            return p_nodes[-1]
        idx = int(np.searchsorted(t_nodes, t_query, side="right") - 1)
        idx = max(0, min(idx, len(t_nodes) - 2))
        t0, t1 = t_nodes[idx], t_nodes[idx + 1]
        p0, p1 = p_nodes[idx], p_nodes[idx + 1]
        alpha = 0.0 if t1 <= t0 else (t_query - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)

    # Use slightly less conservative limits to allow more agile movement
    v_safe = 0.92 * v_max
    a_safe = 0.85 * a_max

    while t + dt <= t_max + 1e-12:
        # Update coverage
        dists = np.linalg.norm(points - pos, axis=1)
        visited |= dists <= coverage_radius

        # Advance tour index past visited points
        while tour_idx < n_points and visited[tour[tour_idx]]:
            tour_idx += 1

        # Find next target from tour
        if tour_idx < n_points:
            target = points[tour[tour_idx]]
        else:
            # All tour points visited - check for any remaining unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited) > 0:
                target = points[unvisited[np.argmin(dists[unvisited])]]
            else:
                target = pos  # All covered, stay put

        wind_v = _wind_velocity(scene, t)

        # Compute desired velocity toward target
        to_target = target - pos
        dist_target = float(np.linalg.norm(to_target))

        # Adaptive speed - go fast when far, decelerate when close
        if dist_target > 5.0:
            speed_factor = 0.85
        elif dist_target > 3.0:
            speed_factor = 0.75
        elif dist_target > 1.5:
            speed_factor = 0.55
        elif dist_target > 0.5:
            speed_factor = 0.35
        else:
            speed_factor = 0.2

        desired_vel = _clip_norm(to_target * 2.0, speed_factor * v_max)

        # Boundary repulsion - use softer repulsion far away, hard repulsion close
        margin = 2.0
        hard_margin = 0.8
        repel = np.zeros(3, dtype=float)

        for dim, (lo, hi) in enumerate([(xmin, xmax), (ymin, ymax), (zmin, zmax)]):
            if pos[dim] < lo + margin:
                d = lo + margin - pos[dim]
                if pos[dim] < lo + hard_margin:
                    repel[dim] += 12.0 * (lo + hard_margin - pos[dim]) + 4.0 * d
                else:
                    repel[dim] += 4.0 * d
            elif pos[dim] > hi - margin:
                d = pos[dim] - (hi - margin)
                if pos[dim] > hi - hard_margin:
                    repel[dim] -= 12.0 * (pos[dim] - (hi - hard_margin)) + 4.0 * d
                else:
                    repel[dim] -= 4.0 * d

        # No-fly zone avoidance
        for pmin, pmax in nfz_data:
            closest = np.clip(pos, pmin, pmax)
            diff = pos - closest
            dist = float(np.linalg.norm(diff))
            influence = 2.5

            if dist < influence:
                if dist < 1e-8:
                    # Inside the zone - push toward nearest face
                    center = 0.5 * (pmin + pmax)
                    delta = pos - center
                    n_d = float(np.linalg.norm(delta))
                    if n_d < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                    else:
                        delta = delta / n_d
                else:
                    delta = diff / dist

                strength = 8.0 * (influence - dist) / max(dist, 0.1)
                strength = min(strength, 20.0)  # cap to avoid instability
                repel += strength * delta

        # Dynamic obstacle avoidance with prediction
        for radius, t_nodes, p_nodes in dyn_obs_data:
            for lookahead in range(8):
                future_t = t + lookahead * dt
                center = _dyn_obs_pos(t_nodes, p_nodes, future_t)
                # Predict our future position roughly
                if lookahead == 0:
                    future_pos = pos
                else:
                    future_pos = pos + (vel + wind_v) * (lookahead * dt)

                influence = radius + 2.5 + lookahead * 0.15

                delta = future_pos - center
                dist = float(np.linalg.norm(delta))

                if dist < influence:
                    if dist < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                        dist = 0.01
                    else:
                        delta = delta / dist

                    decay = max(0.0, 1.0 - lookahead * 0.1)
                    strength = 8.0 * (influence - dist) / max(dist - radius, 0.3) * decay
                    strength = min(strength, 15.0)
                    repel += strength * delta

        # Combine: proportional control + wind feedforward compensation + repulsion
        # Use higher gain for faster response, but compensate wind more precisely
        vel_error = desired_vel - vel
        a_cmd = 2.0 * vel_error - 0.6 * wind_v + repel
        a_cmd = _clip_norm(a_cmd, a_safe)

        # Simulate step
        vel_new = vel + a_cmd * dt
        vel_new = _clip_norm(vel_new, v_safe)
        pos_new = pos + (vel_new + wind_v) * dt

        # Safety check
        safety_margin = 0.3
        in_bounds = (xmin + safety_margin <= pos_new[0] <= xmax - safety_margin and
                     ymin + safety_margin <= pos_new[1] <= ymax - safety_margin and
                     zmin + safety_margin <= pos_new[2] <= zmax - safety_margin)

        in_no_fly = False
        for pmin, pmax in nfz_data:
            if np.all(pos_new >= pmin) and np.all(pos_new <= pmax):
                in_no_fly = True
                break

        dyn_collision = False
        for radius, t_nodes, p_nodes in dyn_obs_data:
            center = _dyn_obs_pos(t_nodes, p_nodes, t + dt)
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