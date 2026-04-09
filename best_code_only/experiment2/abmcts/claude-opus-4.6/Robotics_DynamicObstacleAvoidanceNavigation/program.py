# EVOLVE-BLOCK-START
"""Optimized solution for DynamicObstacleAvoidanceNavigation.

Policy:
- Static path planning with 2D A* on an obstacle-inflated grid.
- Differential-drive waypoint tracking with strict speed/turn/acceleration clipping.
- Multi-step dynamic-obstacle safety with smarter avoidance (slow down or steer around).
- Tuned parameters for faster arrival times.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _interp_dynamic_position(trajectory: list[list[float]], t: float) -> np.ndarray:
    pts = trajectory
    if t <= pts[0][0]:
        return np.array([pts[0][1], pts[0][2]], dtype=float)
    if t >= pts[-1][0]:
        return np.array([pts[-1][1], pts[-1][2]], dtype=float)
    # Binary search for interval
    lo, hi = 0, len(pts) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0 = pts[lo]
    t1, x1, y1 = pts[lo + 1]
    ratio = (t - t0) / max(1e-9, (t1 - t0))
    return np.array([x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0)], dtype=float)


def _circle_rect_collision(cx: float, cy: float, circle_radius: float, rx: float, ry: float, rhx: float, rhy: float) -> bool:
    dx = abs(cx - rx) - rhx
    dy = abs(cy - ry) - rhy
    if dx < 0: dx = 0.0
    if dy < 0: dy = 0.0
    return (dx * dx + dy * dy) <= circle_radius * circle_radius


def _in_bounds(px: float, py: float, bounds: list[float], radius: float) -> bool:
    return (bounds[0] + radius <= px <= bounds[1] - radius) and (bounds[2] + radius <= py <= bounds[3] - radius)


def _static_collision_xy(px: float, py: float, scene: dict[str, Any], inflate: float) -> bool:
    radius = float(scene["robot"]["radius"]) + inflate
    bounds = scene["_bounds"]
    if not _in_bounds(px, py, bounds, radius):
        return True

    for obs in scene["static_obstacles"]:
        if obs["type"] == "circle":
            c = obs["_center"]
            r = float(obs["radius"])
            dx = px - c[0]
            dy = py - c[1]
            d = radius + r
            if dx * dx + dy * dy <= d * d:
                return True
        elif obs["type"] == "rect":
            c = obs["_center"]
            h = obs["_half"]
            if _circle_rect_collision(px, py, radius, c[0], c[1], h[0], h[1]):
                return True

    return False


def _static_collision(pos: np.ndarray, scene: dict[str, Any], inflate: float) -> bool:
    return _static_collision_xy(float(pos[0]), float(pos[1]), scene, inflate)


def _segment_in_collision(a: np.ndarray, b: np.ndarray, scene: dict[str, Any], inflate: float) -> bool:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    samples = max(2, int(math.ceil(length / 0.05)))
    for i in range(samples + 1):
        alpha = i / samples
        px = a[0] + alpha * dx
        py = a[1] + alpha * dy
        if _static_collision_xy(px, py, scene, inflate):
            return True
    return False


def _build_grid(scene: dict[str, Any], resolution: float, inflate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = float(scene["robot"]["radius"]) + inflate
    bounds = scene["_bounds"]
    xmin, xmax, ymin, ymax = bounds

    xs = np.arange(xmin + radius, xmax - radius + 1e-9, resolution, dtype=float)
    ys = np.arange(ymin + radius, ymax - radius + 1e-9, resolution, dtype=float)
    occ = np.zeros((len(xs), len(ys)), dtype=bool)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            occ[i, j] = _static_collision_xy(x, y, scene, inflate)

    return xs, ys, occ


def _nearest_index(value: float, grid_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(grid_values - value)))


def _astar_path(scene: dict[str, Any], resolution: float = 0.15, inflate: float = 0.02) -> list[np.ndarray]:
    xs, ys, occ = _build_grid(scene, resolution=resolution, inflate=inflate)
    start = np.array(scene["start"][:2], dtype=float)
    goal = np.array(scene["goal"], dtype=float)

    sx = _nearest_index(start[0], xs)
    sy = _nearest_index(start[1], ys)
    gx = _nearest_index(goal[0], xs)
    gy = _nearest_index(goal[1], ys)

    occ[sx, sy] = False
    occ[gx, gy] = False

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    gxv, gyv = xs[gx], ys[gy]

    open_heap: list[tuple[float, int, int]] = []
    heapq.heappush(open_heap, (math.hypot(xs[sx] - gxv, ys[sy] - gyv), sx, sy))
    g_cost = {(sx, sy): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    nx_len = len(xs)
    ny_len = len(ys)

    while open_heap:
        _, ix, iy = heapq.heappop(open_heap)
        node = (ix, iy)
        if node in closed:
            continue
        closed.add(node)

        if ix == gx and iy == gy:
            break

        gc = g_cost[node]
        for ddx, ddy in neighbors:
            nx, ny = ix + ddx, iy + ddy
            if not (0 <= nx < nx_len and 0 <= ny < ny_len):
                continue
            if occ[nx, ny]:
                continue

            step = math.hypot(xs[nx] - xs[ix], ys[ny] - ys[iy])
            cand = gc + step
            nnode = (nx, ny)
            if cand + 1e-12 < g_cost.get(nnode, float("inf")):
                g_cost[nnode] = cand
                parent[nnode] = node
                heapq.heappush(open_heap, (cand + math.hypot(xs[nx] - gxv, ys[ny] - gyv), nx, ny))

    if (gx, gy) not in g_cost:
        return [start, goal]

    path_idx: list[tuple[int, int]] = []
    cur = (gx, gy)
    while True:
        path_idx.append(cur)
        if cur == (sx, sy):
            break
        cur = parent[cur]
    path_idx.reverse()

    dense_path = [np.array([xs[i], ys[j]], dtype=float) for i, j in path_idx]
    simplified = [dense_path[0]]
    anchor = 0
    while anchor < len(dense_path) - 1:
        jump = len(dense_path) - 1
        while jump > anchor + 1 and _segment_in_collision(dense_path[anchor], dense_path[jump], scene, inflate=inflate):
            jump -= 1
        simplified.append(dense_path[jump])
        anchor = jump

    simplified[0] = start
    simplified[-1] = goal
    return simplified


def _track_control(
    scene: dict[str, Any],
    pos: np.ndarray,
    theta: float,
    target: np.ndarray,
    goal: np.ndarray,
    t_next: float,
    dt: float,
    v_prev: float,
    w_prev: float,
) -> tuple[float, float]:
    vmax = float(scene["robot"]["v_max"])
    wmax = float(scene["robot"]["omega_max"])
    amax = float(scene["robot"]["a_max"])

    heading = math.atan2(target[1] - pos[1], target[0] - pos[0])
    err = wrap_angle(heading - theta)

    # More aggressive speed profile
    abs_err = abs(err)
    if abs_err > 1.2:
        v_des = 0.05 * vmax
    elif abs_err > 0.6:
        v_des = 0.5 * vmax
    else:
        v_des = vmax

    dist_goal = math.sqrt((goal[0] - pos[0])**2 + (goal[1] - pos[1])**2)
    
    # Deceleration near goal - use kinematics-based braking
    # v^2 = 2 * a * d => v = sqrt(2*a*d)
    brake_v = math.sqrt(2.0 * amax * max(dist_goal - 0.1, 0.01))
    if dist_goal < 2.5:
        v_des = min(v_des, brake_v + 0.05)
    if dist_goal < 0.5:
        v_des = min(v_des, 0.08 + 0.35 * dist_goal)

    # Stronger steering gain for faster convergence
    w_des = float(np.clip(4.0 * err, -wmax, wmax))

    # Dynamic obstacle avoidance
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    robot_r = float(scene["robot"]["radius"])
    
    for obs in scene["dynamic_obstacles"]:
        dyn_r = float(obs["radius"])
        safe = robot_r + dyn_r + 0.06
        traj = obs["trajectory"]
        risk = False
        for h in range(1, 8):
            tt = t_next + (h - 1) * dt
            dyn = _interp_dynamic_position(traj, tt)
            pred_x = pos[0] + v_des * cos_t * dt * h
            pred_y = pos[1] + v_des * sin_t * dt * h
            ddx = pred_x - dyn[0]
            ddy = pred_y - dyn[1]
            if ddx * ddx + ddy * ddy < safe * safe:
                risk = True
                break
        if risk:
            # Try reduced speed first
            v_try = v_des * 0.3
            still_risk = False
            for h in range(1, 8):
                tt = t_next + (h - 1) * dt
                dyn = _interp_dynamic_position(traj, tt)
                pred_x = pos[0] + v_try * cos_t * dt * h
                pred_y = pos[1] + v_try * sin_t * dt * h
                ddx = pred_x - dyn[0]
                ddy = pred_y - dyn[1]
                if ddx * ddx + ddy * ddy < safe * safe:
                    still_risk = True
                    break
            if still_risk:
                v_des = 0.0
            else:
                v_des = v_try
            break

    v_lb = max(-vmax, v_prev - amax * dt)
    v_ub = min(vmax, v_prev + amax * dt)
    w_lb = max(-wmax, w_prev - amax * dt)
    w_ub = min(wmax, w_prev + amax * dt)

    v = max(v_lb, min(v_ub, v_des))
    w = max(w_lb, min(w_ub, w_des))

    pred_x = pos[0] + v * cos_t * dt
    pred_y = pos[1] + v * sin_t * dt
    if _static_collision_xy(pred_x, pred_y, scene, inflate=0.0):
        v = max(v_lb, min(v_ub, 0.0))
        pred_x = pos[0] + v * cos_t * dt
        pred_y = pos[1] + v * sin_t * dt
        if _static_collision_xy(pred_x, pred_y, scene, inflate=0.0):
            v = max(v_lb, min(v_ub, v_prev))

    return v, w


def _preprocess_scene(scene: dict[str, Any]) -> None:
    """Cache numeric conversions."""
    scene["_bounds"] = [float(b) for b in scene["bounds"]]
    for obs in scene["static_obstacles"]:
        if obs["type"] == "circle":
            obs["_center"] = [float(obs["center"][0]), float(obs["center"][1])]
        elif obs["type"] == "rect":
            obs["_center"] = [float(obs["center"][0]), float(obs["center"][1])]
            obs["_half"] = [float(obs["half_extents"][0]), float(obs["half_extents"][1])]


def build_controls_for_scene(scene: dict[str, Any], dt: float, goal_tol: float) -> dict[str, Any]:
    _preprocess_scene(scene)
    tmax = float(scene["T_max"])
    x, y, theta = map(float, scene["start"])
    goal = np.array(scene["goal"], dtype=float)
    amax = float(scene["robot"]["a_max"])
    vmax = float(scene["robot"]["v_max"])
    wmax = float(scene["robot"]["omega_max"])

    path = _astar_path(scene, resolution=0.15, inflate=0.02)
    wp_idx = 1 if len(path) > 1 else 0

    timestamps: list[float] = []
    controls: list[list[float]] = []
    v_prev = 0.0
    w_prev = 0.0
    t = 0.0

    while t <= tmax + 1e-12:
        pos = np.array([x, y], dtype=float)
        dist_to_goal = math.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)

        if dist_to_goal <= goal_tol:
            v_lb = max(-vmax, v_prev - amax * dt)
            v_ub = min(vmax, v_prev + amax * dt)
            w_lb = max(-wmax, w_prev - amax * dt)
            w_ub = min(wmax, w_prev + amax * dt)
            v = max(v_lb, min(v_ub, 0.0))
            w = max(w_lb, min(w_ub, 0.0))
        else:
            while wp_idx < len(path) - 1 and math.sqrt((path[wp_idx][0] - x)**2 + (path[wp_idx][1] - y)**2) < 0.25:
                wp_idx += 1
            target = path[wp_idx]

            v, w = _track_control(
                scene=scene, pos=pos, theta=theta,
                target=target, goal=goal,
                t_next=t + dt, dt=dt,
                v_prev=v_prev, w_prev=w_prev,
            )

        timestamps.append(float(t))
        controls.append([v, w])

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = wrap_angle(theta + w * dt)

        v_prev, w_prev = v, w
        dist_after = math.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)
        if dist_after <= goal_tol:
            break
        t += dt

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

    print("Optimized submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
