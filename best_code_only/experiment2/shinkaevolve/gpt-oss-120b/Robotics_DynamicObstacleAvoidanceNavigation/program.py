# EVOLVE-BLOCK-START
"""Direct Navigation with Velocity Obstacles.

Architecture:
- DirectNavigator: Reactive goal-directed navigation with on-demand detours
- VelocityObstacle: Safe velocity computation for dynamic obstacles
- AggressiveController: High-performance speed and turning control

Key differences from A* approach:
- No grid construction or discretization
- Direct heading toward goal with reactive obstacle avoidance
- Tangent-based detour computation when blocked
- Velocity obstacle theory for dynamic avoidance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DirectNavigator:
    """Reactive navigation using direct goal heading and velocity obstacles."""

    def __init__(self, scene: dict[str, Any]):
        self.scene = scene
        self.robot_radius = float(scene["robot"]["radius"])
        self.v_max = float(scene["robot"]["v_max"])
        self.omega_max = float(scene["robot"]["omega_max"])
        self.a_max = float(scene["robot"]["a_max"])
        self.bounds = list(map(float, scene["bounds"]))
        self.goal = np.array(scene["goal"], dtype=float)
        self.t_max = float(scene["T_max"])

        # Robot state
        self.x = float(scene["start"][0])
        self.y = float(scene["start"][1])
        self.theta = float(scene["start"][2])
        self.v = 0.0
        self.w = 0.0

        # Preprocess static obstacles
        self.static_circles: list[tuple[np.ndarray, float]] = []
        self.static_rects: list[tuple[np.ndarray, np.ndarray]] = []
        for obs in scene["static_obstacles"]:
            if obs["type"] == "circle":
                self.static_circles.append((
                    np.array(obs["center"], dtype=float),
                    float(obs["radius"])
                ))
            elif obs["type"] == "rect":
                self.static_rects.append((
                    np.array(obs["center"], dtype=float),
                    np.array(obs["half_extents"], dtype=float)
                ))

        # Preprocess dynamic obstacles with trajectory arrays
        self.dynamic_obstacles: list[dict[str, Any]] = []
        for obs in scene["dynamic_obstacles"]:
            pts = np.array(obs["trajectory"], dtype=float)
            self.dynamic_obstacles.append({
                "radius": float(obs["radius"]),
                "times": pts[:, 0],
                "positions": pts[:, 1:3]
            })

        # Navigation state
        self.detour_target: np.ndarray | None = None
        self.detour_step = 0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def at_goal(self, tol: float) -> bool:
        return float(np.linalg.norm(self.goal - self.pos)) <= tol

    def static_collision(self, pos: np.ndarray, inflate: float = 0.0) -> bool:
        """Check for static collision at position."""
        r = self.robot_radius + inflate
        xmin, xmax, ymin, ymax = self.bounds

        # Bounds check
        if not (xmin + r <= pos[0] <= xmax - r and ymin + r <= pos[1] <= ymax - r):
            return True

        # Circle obstacles
        for center, radius in self.static_circles:
            if np.linalg.norm(pos - center) <= r + radius:
                return True

        # Rectangle obstacles
        for center, half in self.static_rects:
            diff = np.abs(pos - center) - half
            outside = np.maximum(diff, 0.0)
            if np.dot(outside, outside) <= r * r:
                return True

        return False

    def line_clear(self, a: np.ndarray, b: np.ndarray, step: float = 0.05) -> bool:
        """Check if line segment is collision-free."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = float(np.hypot(dx, dy))
        samples = max(2, int(np.ceil(length / step)))
        inv_samples = 1.0 / samples
        r = self.robot_radius + 0.008
        xmin, xmax, ymin, ymax = self.bounds
        for i in range(samples + 1):
            alpha = i * inv_samples
            px = a[0] + alpha * dx
            py = a[1] + alpha * dy
            # Inline bounds check
            if not (xmin + r <= px <= xmax - r and ymin + r <= py <= ymax - r):
                return False
            # Inline circle collision
            for center, radius in self.static_circles:
                if (px - center[0])**2 + (py - center[1])**2 <= (r + radius)**2:
                    return False
            # Inline rect collision
            for center, half in self.static_rects:
                diff0 = abs(px - center[0]) - half[0]
                diff1 = abs(py - center[1]) - half[1]
                outside0 = max(diff0, 0.0)
                outside1 = max(diff1, 0.0)
                if outside0 * outside0 + outside1 * outside1 <= r * r:
                    return False
        return True

    def interp_dynamic(self, obs_idx: int, t: float) -> np.ndarray:
        """Interpolate dynamic obstacle position at time t."""
        obs = self.dynamic_obstacles[obs_idx]
        times, positions = obs["times"], obs["positions"]

        if t <= times[0]:
            return positions[0].copy()
        if t >= times[-1]:
            return positions[-1].copy()

        idx = int(np.searchsorted(times, t, side="right") - 1)
        ratio = (t - times[idx]) / max(1e-9, times[idx + 1] - times[idx])
        return positions[idx] + ratio * (positions[idx + 1] - positions[idx])

    def find_detour_waypoint(self) -> np.ndarray | None:
        """Compute tangent-based detour when direct path blocked."""
        pos = self.pos
        goal = self.goal
        to_goal = goal - pos
        dist_goal = float(np.linalg.norm(to_goal))

        if dist_goal < 0.30:
            return None

        goal_dir = to_goal / dist_goal
        best_detour = None
        best_score = float('inf')

        # Check circle obstacles for blocking
        for center, radius in self.static_circles:
            to_obs = center - pos
            proj = np.dot(to_obs, goal_dir)

            # Check if obstacle is between robot and goal
            if 0 < proj < dist_goal + radius:
                perp_dist = float(np.linalg.norm(to_obs - proj * goal_dir))
                if perp_dist < radius + self.robot_radius + 0.08:  # Tighter check
                    # Compute tangent directions
                    dist_to_obs = float(np.linalg.norm(to_obs))
                    safe_dist = radius + self.robot_radius + 0.03  # Tighter margin
                    tangent_angle = np.arcsin(min(1.0, safe_dist / max(dist_to_obs, 1e-6)))
                    obs_angle = np.arctan2(to_obs[1], to_obs[0])

                    # Try both left and right tangents
                    for sign in [1, -1]:
                        angle = obs_angle + sign * (np.pi / 2 + tangent_angle)
                        detour_dir = np.array([np.cos(angle), np.sin(angle)])
                        detour_point = center + detour_dir * (radius + self.robot_radius + 0.08)

                        # Validate detour
                        if self.line_clear(pos, detour_point, step=0.05):
                            # Score by total path length through detour
                            to_detour = detour_point - pos
                            detour_dist = float(np.linalg.norm(to_detour))
                            to_goal_from_detour = goal - detour_point
                            remaining_dist = float(np.linalg.norm(to_goal_from_detour))
                            total_path = detour_dist + remaining_dist
                            score = total_path  # Prefer shorter total path

                            if score < best_score:
                                best_score = score
                                best_detour = detour_point.copy()

        # Check rectangle obstacles
        for center, half in self.static_rects:
            # Expand rectangle by robot radius
            inflated_half = half + self.robot_radius + 0.08

            # Check if path intersects rectangle
            rel_pos = pos - center
            rel_goal = goal - center

            # Simple AABB intersection test
            for axis in range(2):
                if (rel_pos[axis] > inflated_half[axis] and rel_goal[axis] > inflated_half[axis]) or \
                   (rel_pos[axis] < -inflated_half[axis] and rel_goal[axis] < -inflated_half[axis]):
                    continue

                # Path might intersect - compute corner-based detour
                corners = [
                    center + np.array([inflated_half[0], inflated_half[1]]),
                    center + np.array([inflated_half[0], -inflated_half[1]]),
                    center + np.array([-inflated_half[0], inflated_half[1]]),
                    center + np.array([-inflated_half[0], -inflated_half[1]]),
                ]

                for corner in corners:
                    if self.line_clear(pos, corner, step=0.04):
                        to_corner = corner - pos
                        alignment = np.dot(to_corner / max(np.linalg.norm(to_corner), 1e-6), goal_dir)
                        score = -alignment
                        if score < best_score:
                            best_score = score
                            best_detour = corner.copy()

        return best_detour

    def compute_control(self, t: float, dt: float) -> tuple[float, float]:
        """Compute optimal velocity commands."""
        pos = self.pos
        goal = self.goal

        # Determine navigation target
        if self.detour_target is not None:
            dist_to_detour = float(np.linalg.norm(self.detour_target - pos))
            if dist_to_detour < 0.12 or self.line_clear(pos, goal):
                self.detour_target = None
                target = goal
            else:
                target = self.detour_target
        else:
            if self.line_clear(pos, goal, step=0.04):
                target = goal
            else:
                self.detour_target = self.find_detour_waypoint()
                target = self.detour_target if self.detour_target is not None else goal

        # Compute heading error
        to_target = target - pos
        dist_target = float(np.linalg.norm(to_target))
        target_heading = float(np.arctan2(to_target[1], to_target[0]))
        heading_err = wrap_angle(target_heading - self.theta)
        abs_err = abs(heading_err)

        # Aggressive speed profile with tighter thresholds
        if abs_err < 0.45:
            v_des = self.v_max
        elif abs_err < 0.95:
            v_des = self.v_max * 0.97
        elif abs_err < 1.45:
            v_des = self.v_max * 0.75
        else:
            v_des = self.v_max * 0.40

        # Goal approach braking - less conservative
        dist_goal = float(np.linalg.norm(goal - pos))
        if dist_goal < 0.30:
            v_des = min(v_des, 0.02 + 0.97 * dist_goal)
        if dist_goal < 0.15:
            c, s = float(np.cos(self.theta)), float(np.sin(self.theta))
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            proj = dx * c + dy * s
            if proj > 0:
                v_des = min(v_des, max(0.0, (proj + 0.0003) / dt))

        # Velocity obstacle for dynamic obstacles
        v_des = self._velocity_obstacle_avoidance(v_des, t, dt)

        # High-gain angular control
        w_des = float(np.clip(4.8 * heading_err, -self.omega_max, self.omega_max))

        # Apply acceleration limits
        v_lb = max(-self.v_max, self.v - self.a_max * dt)
        v_ub = min(self.v_max, self.v + self.a_max * dt)
        w_lb = max(-self.omega_max, self.w - self.a_max * dt)
        w_ub = min(self.omega_max, self.w + self.a_max * dt)

        v = float(np.clip(v_des, v_lb, v_ub))
        w = float(np.clip(w_des, w_lb, w_ub))

        # Static collision safety check
        pred = np.array([
            self.x + v * np.cos(self.theta) * dt,
            self.y + v * np.sin(self.theta) * dt
        ])
        if self.static_collision(pred, 0.0):
            v = float(np.clip(0.0, v_lb, v_ub))
            pred = np.array([
                self.x + v * np.cos(self.theta) * dt,
                self.y + v * np.sin(self.theta) * dt
            ])
            if self.static_collision(pred, 0.0):
                v = float(np.clip(self.v * 0.5, v_lb, v_ub))

        return v, w

    def _velocity_obstacle_avoidance(self, v_des: float, t: float, dt: float) -> float:
        """Compute safe velocity using velocity obstacle theory."""
        safe_v = v_des
        r = self.robot_radius

        for i, obs in enumerate(self.dynamic_obstacles):
            obs_r = obs["radius"]
            combined_r = r + obs_r + 0.008  # Tighter margin

            # Get obstacle velocity
            pos_now = self.interp_dynamic(i, t)
            pos_next = self.interp_dynamic(i, t + dt)
            obs_vel = (pos_next - pos_now) / dt
            obs_speed = float(np.linalg.norm(obs_vel))

            # Multi-horizon collision prediction (reduced horizon for speed)
            for h in range(1, 4):
                tt = t + (h - 1) * dt
                obs_pos = self.interp_dynamic(i, tt)

                # Predict robot position at current desired velocity
                pred_x = self.x + v_des * np.cos(self.theta) * dt * h
                pred_y = self.y + v_des * np.sin(self.theta) * dt * h

                dx = pred_x - obs_pos[0]
                dy = pred_y - obs_pos[1]
                dist = float(np.hypot(dx, dy))
                margin = combined_r * (1.0 + 0.002 * h)  # Tighter growth

                if dist < margin:
                    # Check relative motion - is obstacle approaching?
                    to_robot_x = self.x - obs_pos[0]
                    to_robot_y = self.y - obs_pos[1]
                    to_robot_n = np.hypot(to_robot_x, to_robot_y)

                    if to_robot_n > 1e-6 and obs_speed > 0.008:
                        obs_dir_x = obs_vel[0] / obs_speed
                        obs_dir_y = obs_vel[1] / obs_speed
                        robot_dir_x = to_robot_x / to_robot_n
                        robot_dir_y = to_robot_y / to_robot_n

                        # If obstacle moving away from robot, less aggressive slowdown
                        if obs_dir_x * robot_dir_x + obs_dir_y * robot_dir_y < 0.02:
                            scale = max(0.45, dist / margin)
                            safe_v = min(safe_v, v_des * scale)
                            continue

                    # Standard velocity obstacle scaling
                    scale = max(0.15, dist / margin)
                    safe_v = min(safe_v, v_des * scale)
                    break

        return safe_v

    def step(self, v: float, w: float, dt: float):
        """Update robot state."""
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = wrap_angle(self.theta + w * dt)
        self.v = v
        self.w = w


def build_controls_for_scene(scene: dict[str, Any], dt: float, goal_tol: float) -> dict[str, Any]:
    """Build control sequence for a single scene using direct navigation."""
    nav = DirectNavigator(scene)

    timestamps: list[float] = []
    controls: list[list[float]] = []
    t = 0.0

    while t <= nav.t_max + 1e-12:
        if nav.at_goal(goal_tol):
            # Braking control at goal
            v_lb = max(-nav.v_max, nav.v - nav.a_max * dt)
            v_ub = min(nav.v_max, nav.v + nav.a_max * dt)
            w_lb = max(-nav.omega_max, nav.w - nav.a_max * dt)
            w_ub = min(nav.omega_max, nav.w + nav.a_max * dt)
            v = float(np.clip(0.0, v_lb, v_ub))
            w = float(np.clip(0.0, w_lb, w_ub))
        else:
            v, w = nav.compute_control(t, dt)

        timestamps.append(float(t))
        controls.append([v, w])

        nav.step(v, w, dt)

        if nav.at_goal(goal_tol):
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
    scenario_entries = [build_controls_for_scene(scene, dt=dt, goal_tol=goal_tol)
                        for scene in cfg["scenarios"]]

    submission = {"scenarios": scenario_entries}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print("Baseline submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END