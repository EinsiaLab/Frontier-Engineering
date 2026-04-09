# EVOLVE-BLOCK-START
"""Improved solution for DynamicObstacleAvoidanceNavigation.

Policy:
- Static path planning with 2D A* on an obstacle-inflated grid.
- Aggressive differential-drive waypoint tracking with speed/turn/acceleration clipping.
- Dynamic-obstacle avoidance with steering and speed modulation.
"""

from __future__ import annotations

import bisect
import heapq
import math
import json
from pathlib import Path
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _interp_dynamic_position_fast(times: list, xs: list, ys: list, t: float) -> tuple[float, float]:
    """Fast interpolation using pre-split lists."""
    if t <= times[0]:
        return xs[0], ys[0]
    if t >= times[-1]:
        return xs[-1], ys[-1]
    idx = bisect.bisect_right(times, t) - 1
    t0 = times[idx]
    t1 = times[idx + 1]
    ratio = (t - t0) / max(1e-9, (t1 - t0))
    return xs[idx] + ratio * (xs[idx + 1] - xs[idx]), ys[idx] + ratio * (ys[idx + 1] - ys[idx])


def _circle_rect_collision_fast(px: float, py: float, circle_radius: float,
                                 cx: float, cy: float, hx: float, hy: float) -> bool:
    dx = abs(px - cx) - hx
    dy = abs(py - cy) - hy
    if dx < 0.0:
        dx = 0.0
    if dy < 0.0:
        dy = 0.0
    return (dx * dx + dy * dy) <= circle_radius * circle_radius


def _preprocess_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute obstacle data for fast collision checks."""
    bounds = [float(b) for b in scene["bounds"]]
    robot_radius = float(scene["robot"]["radius"])

    circle_obs = []
    rect_obs = []
    for obs in scene["static_obstacles"]:
        if obs["type"] == "circle":
            cx, cy = float(obs["center"][0]), float(obs["center"][1])
            r = float(obs["radius"])
            circle_obs.append((cx, cy, r))
        elif obs["type"] == "rect":
            cx, cy = float(obs["center"][0]), float(obs["center"][1])
            hx, hy = float(obs["half_extents"][0]), float(obs["half_extents"][1])
            rect_obs.append((cx, cy, hx, hy))

    # Pre-compute dynamic obstacle trajectories as separate lists for fast access
    dyn_obs = []
    for obs in scene["dynamic_obstacles"]:
        traj = obs["trajectory"]
        times = [float(p[0]) for p in traj]
        xs_list = [float(p[1]) for p in traj]
        ys_list = [float(p[2]) for p in traj]
        dyn_obs.append({
            "radius": float(obs["radius"]),
            "times": times,
            "xs": xs_list,
            "ys": ys_list,
        })

    scene["_bounds"] = bounds
    scene["_robot_radius"] = robot_radius
    scene["_circle_obs"] = circle_obs
    scene["_rect_obs"] = rect_obs
    scene["_dyn_obs"] = dyn_obs
    return scene


def _static_collision_fast(px: float, py: float, scene: dict[str, Any], inflate: float) -> bool:
    radius = scene["_robot_radius"] + inflate
    bounds = scene["_bounds"]
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

    if px < xmin + radius or px > xmax - radius or py < ymin + radius or py > ymax - radius:
        return True

    for cx, cy, r in scene["_circle_obs"]:
        dx = px - cx
        dy = py - cy
        dist_sq = dx * dx + dy * dy
        threshold = radius + r
        if dist_sq <= threshold * threshold:
            return True

    for cx, cy, hx, hy in scene["_rect_obs"]:
        if _circle_rect_collision_fast(px, py, radius, cx, cy, hx, hy):
            return True

    return False


def _segment_in_collision(a: np.ndarray, b: np.ndarray, scene: dict[str, Any], inflate: float) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    length = math.sqrt(dx * dx + dy * dy)
    step_size = 0.08
    samples = max(2, int(math.ceil(length / step_size)) + 1)
    for i in range(samples):
        alpha = i / (samples - 1)
        px = ax + alpha * dx
        py = ay + alpha * dy
        if _static_collision_fast(px, py, scene, inflate):
            return True
    return False


def _build_grid(scene: dict[str, Any], resolution: float, inflate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = scene["_robot_radius"] + inflate
    bounds = scene["_bounds"]
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

    xs = np.arange(xmin + radius, xmax - radius + 1e-9, resolution, dtype=float)
    ys = np.arange(ymin + radius, ymax - radius + 1e-9, resolution, dtype=float)
    occ = np.zeros((len(xs), len(ys)), dtype=bool)

    for i in range(len(xs)):
        for j in range(len(ys)):
            occ[i, j] = _static_collision_fast(xs[i], ys[j], scene, inflate)

    return xs, ys, occ


def _nearest_index(value: float, grid_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(grid_values - value)))


def _astar_path(scene: dict[str, Any], resolution: float = 0.15, inflate: float = 0.03) -> list[np.ndarray]:
    """Try A* with given params, fallback to smaller inflate if needed."""
    for inf in [inflate, inflate * 0.5, 0.01, 0.005, 0.001]:
        path = _astar_path_inner(scene, resolution, inf)
        if len(path) >= 2:
            return path
    start = np.array(scene["start"][:2], dtype=float)
    goal = np.array(scene["goal"], dtype=float)
    return [start, goal]


def _astar_path_inner(scene: dict[str, Any], resolution: float, inflate: float) -> list[np.ndarray]:
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
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    gx_val = xs[gx]
    gy_val = ys[gy]
    nx_len = len(xs)
    ny_len = len(ys)

    def heuristic(ix: int, iy: int) -> float:
        dx = xs[ix] - gx_val
        dy = ys[iy] - gy_val
        return math.sqrt(dx * dx + dy * dy)

    open_heap: list[tuple[float, int, int]] = []
    heapq.heappush(open_heap, (heuristic(sx, sy), sx, sy))
    g_cost = {(sx, sy): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    found = False
    while open_heap:
        _, ix, iy = heapq.heappop(open_heap)
        node = (ix, iy)
        if node in closed:
            continue
        closed.add(node)

        if ix == gx and iy == gy:
            found = True
            break

        cur_g = g_cost[node]
        cur_x = xs[ix]
        cur_y = ys[iy]
        for ddx, ddy in neighbors:
            nnx, nny = ix + ddx, iy + ddy
            if not (0 <= nnx < nx_len and 0 <= nny < ny_len):
                continue
            if occ[nnx, nny]:
                continue

            sx_d = xs[nnx] - cur_x
            sy_d = ys[nny] - cur_y
            step = math.sqrt(sx_d * sx_d + sy_d * sy_d)
            cand = cur_g + step
            nnode = (nnx, nny)
            if cand + 1e-12 < g_cost.get(nnode, float("inf")):
                g_cost[nnode] = cand
                parent[nnode] = node
                heapq.heappush(open_heap, (cand + heuristic(nnx, nny), nnx, nny))

    if not found:
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

    # Simplify path using line-of-sight checks
    simplified = [dense_path[0]]
    anchor = 0
    while anchor < len(dense_path) - 1:
        # Binary search for furthest visible point
        lo = anchor + 1
        hi = len(dense_path) - 1
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if not _segment_in_collision(dense_path[anchor], dense_path[mid], scene, inflate=inflate):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        simplified.append(dense_path[best])
        anchor = best

    simplified[0] = start
    simplified[-1] = goal

    # Further simplify iteratively
    changed = True
    while changed:
        changed = False
        new_simplified = [simplified[0]]
        i = 0
        while i < len(simplified) - 1:
            if i + 2 < len(simplified):
                if not _segment_in_collision(simplified[i], simplified[i + 2], scene, inflate=inflate):
                    new_simplified.append(simplified[i + 2])
                    i += 2
                    changed = True
                    continue
            new_simplified.append(simplified[i + 1])
            i += 1
        simplified = new_simplified

    return simplified


def _track_control(
    scene: dict[str, Any],
    px: float,
    py: float,
    theta: float,
    target_x: float,
    target_y: float,
    goal_x: float,
    goal_y: float,
    t_now: float,
    dt: float,
    v_prev: float,
    w_prev: float,
) -> tuple[float, float]:
    vmax = scene["_vmax"]
    wmax = scene["_wmax"]
    amax = scene["_amax"]

    heading = math.atan2(target_y - py, target_x - px)
    err = wrap_angle(heading - theta)
    abs_err = abs(err)

    # Speed profile based on heading error - be aggressive
    if abs_err > 1.5:
        v_des = 0.05 * vmax
    elif abs_err > 0.8:
        v_des = 0.3 * vmax
    elif abs_err > 0.4:
        v_des = 0.7 * vmax
    else:
        v_des = vmax

    dgx = goal_x - px
    dgy = goal_y - py
    dist_goal = math.sqrt(dgx * dgx + dgy * dgy)

    # Goal approach - need to decelerate smoothly
    # v^2 = 2*a*d => v = sqrt(2*a*d)
    v_brake = math.sqrt(2.0 * amax * max(dist_goal, 0.01))
    v_des = min(v_des, v_brake)

    if dist_goal < 0.3:
        v_des = min(v_des, 0.15 + 1.0 * dist_goal)

    # Angular control with proportional gain
    w_des = max(-wmax, min(wmax, 5.0 * err))

    cos_th = math.cos(theta)
    sin_th = math.sin(theta)

    # Dynamic obstacle avoidance
    dyn_obs = scene["_dyn_obs"]
    robot_radius = scene["_robot_radius"]

    # Collect all risky dynamic obstacles and find safest action
    any_risk = False
    best_v_des = v_des
    best_w_des = w_des

    for obs in dyn_obs:
        dyn_r = obs["radius"]
        safe = robot_radius + dyn_r + 0.08  # tighter safety margin
        safe_sq = safe * safe
        times = obs["times"]
        xs_list = obs["xs"]
        ys_list = obs["ys"]

        # Check current distance
        dyn_x_now, dyn_y_now = _interp_dynamic_position_fast(times, xs_list, ys_list, t_now)
        dx_now = px - dyn_x_now
        dy_now = py - dyn_y_now
        dist_now_sq = dx_now * dx_now + dy_now * dy_now

        # Only worry if obstacle is somewhat close
        worry_dist = safe + vmax * dt * 15
        if dist_now_sq > worry_dist * worry_dist:
            continue

        horizon = 8

        # Check at current desired speed over multiple horizons
        risk = False
        for h in range(1, horizon + 1):
            tt = t_now + h * dt
            dyn_x, dyn_y = _interp_dynamic_position_fast(times, xs_list, ys_list, tt)
            pred_x = px + best_v_des * cos_th * dt * h
            pred_y = py + best_v_des * sin_th * dt * h
            ddx = pred_x - dyn_x
            ddy = pred_y - dyn_y
            if ddx * ddx + ddy * ddy < safe_sq:
                risk = True
                break

        if not risk:
            continue

        any_risk = True

        # Try multiple speed reductions
        found_safe = False
        for frac in [0.5, 0.3, 0.1, 0.0, -0.1, -0.2]:
            v_try = frac * vmax if frac <= 0 else best_v_des * frac
            risk_try = False
            for h in range(1, horizon + 1):
                tt = t_now + h * dt
                dyn_x, dyn_y = _interp_dynamic_position_fast(times, xs_list, ys_list, tt)
                pred_x = px + v_try * cos_th * dt * h
                pred_y = py + v_try * sin_th * dt * h
                ddx = pred_x - dyn_x
                ddy = pred_y - dyn_y
                if ddx * ddx + ddy * ddy < safe_sq:
                    risk_try = True
                    break
            if not risk_try:
                best_v_des = v_try
                found_safe = True
                break

        if found_safe:
            continue

        # Need to steer away
        ang_to_obs = math.atan2(dyn_y_now - py, dyn_x_now - px)
        turn_away = wrap_angle(ang_to_obs - theta)

        if turn_away > 0:
            best_w_des = -wmax
        else:
            best_w_des = wmax

        # Try steering + various speeds
        found_steer = False
        for w_candidate in [best_w_des, -best_w_des]:
            for v_try_val in [0.3 * vmax, 0.1 * vmax, 0.0, -0.1 * vmax]:
                new_theta_c = theta + w_candidate * dt
                new_cos = math.cos(new_theta_c)
                new_sin = math.sin(new_theta_c)
                risk_steer = False
                for h in range(1, horizon + 1):
                    tt = t_now + h * dt
                    dyn_x, dyn_y = _interp_dynamic_position_fast(times, xs_list, ys_list, tt)
                    pred_x = px + v_try_val * new_cos * dt * h
                    pred_y = py + v_try_val * new_sin * dt * h
                    ddx = pred_x - dyn_x
                    ddy = pred_y - dyn_y
                    if ddx * ddx + ddy * ddy < safe_sq:
                        risk_steer = True
                        break
                if not risk_steer:
                    best_w_des = w_candidate
                    best_v_des = v_try_val
                    found_steer = True
                    break
            if found_steer:
                break

        if not found_steer:
            best_v_des = -0.15 * vmax

    v_des = best_v_des
    w_des = best_w_des

    v_lb = max(-vmax, v_prev - amax * dt)
    v_ub = min(vmax, v_prev + amax * dt)
    w_lb = max(-wmax, w_prev - amax * dt)
    w_ub = min(wmax, w_prev + amax * dt)

    v = max(v_lb, min(v_ub, v_des))
    w = max(w_lb, min(w_ub, w_des))

    # Static collision check with safety margin
    pred_x = px + v * cos_th * dt
    pred_y = py + v * sin_th * dt
    if _static_collision_fast(pred_x, pred_y, scene, inflate=0.005):
        # Try zero velocity
        v_zero = max(v_lb, min(v_ub, 0.0))
        pred_x2 = px + v_zero * cos_th * dt
        pred_y2 = py + v_zero * sin_th * dt
        if not _static_collision_fast(pred_x2, pred_y2, scene, inflate=0.005):
            v = v_zero
        else:
            # Try negative velocity
            v_neg = max(v_lb, min(v_ub, -0.1 * vmax))
            pred_x3 = px + v_neg * cos_th * dt
            pred_y3 = py + v_neg * sin_th * dt
            if not _static_collision_fast(pred_x3, pred_y3, scene, inflate=0.01):
                v = v_neg
            else:
                v = max(v_lb, min(v_ub, 0.0))

    # Final dynamic obstacle collision check for next step
    pred_x_final = px + v * cos_th * dt
    pred_y_final = py + v * sin_th * dt
    t_next = t_now + dt
    for obs in dyn_obs:
        dyn_r = obs["radius"]
        safe_d = robot_radius + dyn_r + 0.01
        safe_d_sq = safe_d * safe_d
        dyn_x, dyn_y = _interp_dynamic_position_fast(obs["times"], obs["xs"], obs["ys"], t_next)
        ddx = pred_x_final - dyn_x
        ddy = pred_y_final - dyn_y
        if ddx * ddx + ddy * ddy < safe_d_sq:
            # Emergency: try stopping
            v_stop = max(v_lb, min(v_ub, 0.0))
            px2 = px + v_stop * cos_th * dt
            py2 = py + v_stop * sin_th * dt
            ddx2 = px2 - dyn_x
            ddy2 = py2 - dyn_y
            if ddx2 * ddx2 + ddy2 * ddy2 >= safe_d_sq:
                v = v_stop
            else:
                v_back = max(v_lb, min(v_ub, -0.2 * vmax))
                px3 = px + v_back * cos_th * dt
                py3 = py + v_back * sin_th * dt
                ddx3 = px3 - dyn_x
                ddy3 = py3 - dyn_y
                if ddx3 * ddx3 + ddy3 * ddy3 >= safe_d_sq:
                    v = v_back
                else:
                    v = max(v_lb, min(v_ub, 0.0))
            break

    return v, w


def build_controls_for_scene(scene: dict[str, Any], dt: float, goal_tol: float) -> dict[str, Any]:
    # Preprocess scene for fast collision checks
    _preprocess_scene(scene)

    tmax = float(scene["T_max"])
    x, y, theta = map(float, scene["start"])
    goal_x = float(scene["goal"][0])
    goal_y = float(scene["goal"][1])
    amax = float(scene["robot"]["a_max"])
    vmax = float(scene["robot"]["v_max"])
    wmax = float(scene["robot"]["omega_max"])

    # Cache these in scene for fast access
    scene["_vmax"] = vmax
    scene["_wmax"] = wmax
    scene["_amax"] = amax

    path = _astar_path(scene, resolution=0.15, inflate=0.03)
    wp_idx = 1 if len(path) > 1 else 0

    # Convert path to list of (x, y) tuples for fast access
    path_xy = [(float(p[0]), float(p[1])) for p in path]

    timestamps: list[float] = []
    controls: list[list[float]] = []
    v_prev = 0.0
    w_prev = 0.0
    t = 0.0

    goal_tol_sq = goal_tol * goal_tol

    while t <= tmax + 1e-12:
        dgx = goal_x - x
        dgy = goal_y - y
        dist_sq = dgx * dgx + dgy * dgy

        if dist_sq <= goal_tol_sq:
            # At goal - decelerate to stop
            v_lb = max(-vmax, v_prev - amax * dt)
            v_ub = min(vmax, v_prev + amax * dt)
            w_lb = max(-wmax, w_prev - amax * dt)
            w_ub = min(wmax, w_prev + amax * dt)
            v = max(v_lb, min(v_ub, 0.0))
            w = max(w_lb, min(w_ub, 0.0))
        else:
            # Advance waypoint
            while wp_idx < len(path_xy) - 1:
                wpx, wpy = path_xy[wp_idx]
                ddx = wpx - x
                ddy = wpy - y
                if ddx * ddx + ddy * ddy < 0.25:  # ~0.5^2
                    wp_idx += 1
                else:
                    break

            target_x, target_y = path_xy[wp_idx]

            # Pure pursuit lookahead for smoother tracking
            if wp_idx < len(path_xy) - 1:
                wpx, wpy = path_xy[wp_idx]
                ddx = wpx - x
                ddy = wpy - y
                d_wp = math.sqrt(ddx * ddx + ddy * ddy)
                if d_wp < 0.7:
                    nxt_x, nxt_y = path_xy[wp_idx + 1]
                    alpha = max(0.0, min(1.0, 1.0 - d_wp / 0.7))
                    target_x = wpx + alpha * (nxt_x - wpx)
                    target_y = wpy + alpha * (nxt_y - wpy)

            v, w = _track_control(
                scene=scene,
                px=x,
                py=y,
                theta=theta,
                target_x=target_x,
                target_y=target_y,
                goal_x=goal_x,
                goal_y=goal_y,
                t_now=t,
                dt=dt,
                v_prev=v_prev,
                w_prev=w_prev,
            )

        timestamps.append(t)
        controls.append([v, w])

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = wrap_angle(theta + w * dt)

        v_prev, w_prev = v, w

        dgx2 = goal_x - x
        dgy2 = goal_y - y
        if dgx2 * dgx2 + dgy2 * dgy2 <= goal_tol_sq:
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

    print("Baseline submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END