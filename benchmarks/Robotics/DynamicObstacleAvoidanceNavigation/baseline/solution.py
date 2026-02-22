# EVOLVE-BLOCK-START
"""Baseline for DynamicObstacleAvoidanceNavigation.

Policy:
- Goal-guided differential-drive controller
- Speed clipping by heading error
- Simple time-domain avoidance by steering away from near obstacles
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def interp_dynamic_position(trajectory: list[list[float]], t: float) -> np.ndarray:
    pts = np.array(trajectory, dtype=float)
    if t <= pts[0, 0]:
        return pts[0, 1:3]
    if t >= pts[-1, 0]:
        return pts[-1, 1:3]
    idx = int(np.searchsorted(pts[:, 0], t, side="right") - 1)
    t0, x0, y0 = pts[idx]
    t1, x1, y1 = pts[idx + 1]
    ratio = (t - t0) / max(1e-9, (t1 - t0))
    return np.array([x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0)], dtype=float)


def nearest_obstacle_vector(pos: np.ndarray, t: float, scene: dict) -> tuple[np.ndarray, float]:
    best_vec = np.zeros(2, dtype=float)
    best_margin = 1e9
    robot_r = float(scene["robot"]["radius"])

    for obs in scene["static_obstacles"]:
        if obs["type"] == "circle":
            c = np.array(obs["center"], dtype=float)
            r = float(obs["radius"])
            vec = pos - c
            dist = np.linalg.norm(vec) - (robot_r + r)
            if dist < best_margin:
                best_margin = dist
                best_vec = vec
        elif obs["type"] == "rect":
            c = np.array(obs["center"], dtype=float)
            h = np.array(obs["half_extents"], dtype=float)
            closest = np.clip(pos, c - h, c + h)
            vec = pos - closest
            dist = np.linalg.norm(vec) - robot_r
            if dist < best_margin:
                best_margin = dist
                best_vec = vec

    for obs in scene["dynamic_obstacles"]:
        c = interp_dynamic_position(obs["trajectory"], t)
        r = float(obs["radius"])
        vec = pos - c
        dist = np.linalg.norm(vec) - (robot_r + r)
        if dist < best_margin:
            best_margin = dist
            best_vec = vec

    return best_vec, float(best_margin)


def build_controls_for_scene(scene: dict, dt: float, goal_tol: float) -> dict:
    vmax = float(scene["robot"]["v_max"])
    wmax = float(scene["robot"]["omega_max"])
    amax = float(scene["robot"]["a_max"])
    tmax = float(scene["T_max"])

    x, y, theta = map(float, scene["start"])
    goal = np.array(scene["goal"], dtype=float)

    timestamps: list[float] = [0.0]
    controls: list[list[float]] = [[0.0, 0.0]]

    v_prev = 0.0
    w_prev = 0.0
    t = 0.0

    while t + dt <= tmax + 1e-12:
        pos = np.array([x, y], dtype=float)
        to_goal = goal - pos
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal <= goal_tol:
            break

        goal_heading = float(np.arctan2(to_goal[1], to_goal[0]))
        heading_err = wrap_angle(goal_heading - theta)

        v_des = vmax * max(0.15, 1.0 - abs(heading_err) / np.pi)
        w_des = np.clip(2.2 * heading_err, -wmax, wmax)

        obs_vec, obs_margin = nearest_obstacle_vector(pos, t, scene)
        if obs_margin < 0.65:
            obs_angle = float(np.arctan2(obs_vec[1], obs_vec[0])) if np.linalg.norm(obs_vec) > 1e-6 else goal_heading + np.pi / 2
            avoid_err = wrap_angle(obs_angle - theta)
            w_des += np.clip(1.4 * avoid_err, -0.8 * wmax, 0.8 * wmax)
            v_des *= 0.35
        elif obs_margin < 1.0:
            v_des *= 0.65

        v_lb = max(-vmax, v_prev - amax * dt)
        v_ub = min(vmax, v_prev + amax * dt)
        w_lb = max(-wmax, w_prev - amax * dt)
        w_ub = min(wmax, w_prev + amax * dt)

        v = float(np.clip(v_des, v_lb, v_ub))
        w = float(np.clip(w_des, w_lb, w_ub))

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta = wrap_angle(theta + w * dt)
        t = round(t + dt, 10)

        timestamps.append(float(t))
        controls.append([v, w])
        v_prev, w_prev = v, w

    return {"id": scene["id"], "timestamps": timestamps, "controls": controls}


def main() -> None:
    task_root = Path(__file__).resolve().parents[1]
    cfg_path = task_root / "references" / "scenarios.json"

    with cfg_path.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    dt = float(cfg.get("dt", 0.05))
    goal_tol = float(cfg.get("goal_tolerance", 0.15))

    scenario_entries = [build_controls_for_scene(scene, dt=dt, goal_tol=goal_tol) for scene in cfg["scenarios"]]

    submission = {"scenarios": scenario_entries}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print("Baseline submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
