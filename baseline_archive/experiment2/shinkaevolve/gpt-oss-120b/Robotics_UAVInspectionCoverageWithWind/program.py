# EVOLVE-BLOCK-START
"""UAV Inspection Coverage with Wind - Modular Architecture.

Strategy:
- Pre-compute visiting order using nearest neighbor heuristic.
- Class-based modular architecture for wind, obstacles, environment.
- Improved controller with predictive avoidance.
- Better coverage through optimized waypoint sequencing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List
import numpy as np


class WindModel:
    """Encapsulates wind dynamics computation."""
    
    def __init__(self, wind_data: dict[str, Any]):
        self.base = np.array(wind_data["base"], dtype=float)
        self.amplitude = np.array(wind_data["amplitude"], dtype=float)
        self.frequency = np.array(wind_data["frequency"], dtype=float)
        self.phase = np.array(wind_data["phase"], dtype=float)
    
    def velocity(self, t: float) -> np.ndarray:
        return self.base + self.amplitude * np.sin(self.frequency * t + self.phase)


class DynamicObstacle:
    """Represents a dynamic obstacle with piecewise-linear trajectory."""
    
    def __init__(self, obstacle_data: dict[str, Any]):
        self.radius = float(obstacle_data.get("radius", 0.0))
        traj = obstacle_data.get("trajectory", [])
        
        if traj and isinstance(traj, list) and len(traj) > 0:
            self.t_nodes = np.array([float(n["t"]) for n in traj], dtype=float)
            self.p_nodes = np.array([n["pos"] for n in traj], dtype=float)
        else:
            self.t_nodes = np.array([0.0, 1e9], dtype=float)
            self.p_nodes = np.array([[1e9, 1e9, 1e9], [1e9, 1e9, 1e9]], dtype=float)
    
    def position(self, t: float) -> np.ndarray:
        if t <= self.t_nodes[0]:
            return self.p_nodes[0].copy()
        if t >= self.t_nodes[-1]:
            return self.p_nodes[-1].copy()
        
        idx = int(np.searchsorted(self.t_nodes, t, side="right") - 1)
        idx = max(0, min(idx, len(self.t_nodes) - 2))
        
        t0, t1 = self.t_nodes[idx], self.t_nodes[idx + 1]
        p0, p1 = self.p_nodes[idx], self.p_nodes[idx + 1]
        
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)


class Environment:
    """Encapsulates all environmental constraints and queries."""
    
    def __init__(self, scene: dict[str, Any]):
        bounds = scene["bounds"]
        self.xmin, self.xmax = bounds[0], bounds[1]
        self.ymin, self.ymax = bounds[2], bounds[3]
        self.zmin, self.zmax = bounds[4], bounds[5]
        
        self.wind = WindModel(scene["wind"])
        
        # Parse no-fly zones
        self.no_fly_zones: List[dict] = []
        for zone in scene.get("no_fly_zones", []):
            if zone.get("type") == "box":
                pmin = np.array(zone["min"], dtype=float)
                pmax = np.array(zone["max"], dtype=float)
                self.no_fly_zones.append({
                    "min": pmin,
                    "max": pmax,
                    "center": 0.5 * (pmin + pmax)
                })
        
        # Parse dynamic obstacles
        self.dynamic_obstacles: List[DynamicObstacle] = [
            DynamicObstacle(obs) for obs in scene.get("dynamic_obstacles", [])
            if float(obs.get("radius", 0.0)) > 0
        ]
    
    def is_in_bounds(self, pos: np.ndarray, margin: float = 0.0) -> bool:
        return (self.xmin + margin <= pos[0] <= self.xmax - margin and
                self.ymin + margin <= pos[1] <= self.ymax - margin and
                self.zmin + margin <= pos[2] <= self.zmax - margin)
    
    def is_in_no_fly_zone(self, pos: np.ndarray) -> bool:
        for zone in self.no_fly_zones:
            if np.all(pos >= zone["min"]) and np.all(pos <= zone["max"]):
                return True
        return False
    
    def check_dynamic_collision(self, pos: np.ndarray, t: float, margin: float = 0.0) -> bool:
        for obs in self.dynamic_obstacles:
            center = obs.position(t)
            if float(np.linalg.norm(pos - center)) < obs.radius + margin:
                return True
        return False


class UAVController:
    """Handles UAV state dynamics and control computation."""
    
    def __init__(self, start: np.ndarray, v_max: float, a_max: float):
        self.pos = start[:3].copy()
        self.vel = start[3:].copy()
        self.v_max = v_max
        self.a_max = a_max
    
    @staticmethod
    def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n <= max_norm or n < 1e-12:
            return vec
        return vec * (max_norm / n)
    
    def compute_repulsion(self, env: Environment, t: float, dt: float) -> np.ndarray:
        repel = np.zeros(3, dtype=float)
        margin = 2.5
        
        # Boundary repulsion
        if self.pos[0] < env.xmin + margin:
            repel[0] += 6.0 * (env.xmin + margin - self.pos[0])
        elif self.pos[0] > env.xmax - margin:
            repel[0] -= 6.0 * (self.pos[0] - (env.xmax - margin))
        
        if self.pos[1] < env.ymin + margin:
            repel[1] += 6.0 * (env.ymin + margin - self.pos[1])
        elif self.pos[1] > env.ymax - margin:
            repel[1] -= 6.0 * (self.pos[1] - (env.ymax - margin))
        
        z_margin = margin * 0.8
        if self.pos[2] < env.zmin + z_margin:
            repel[2] += 6.0 * (env.zmin + z_margin - self.pos[2])
        elif self.pos[2] > env.zmax - z_margin:
            repel[2] -= 6.0 * (self.pos[2] - (env.zmax - z_margin))
        
        # No-fly zone repulsion
        for zone in env.no_fly_zones:
            closest = np.clip(self.pos, zone["min"], zone["max"])
            dist = float(np.linalg.norm(self.pos - closest))
            influence = 3.0
            
            if dist < influence:
                delta = self.pos - zone["center"]
                n = float(np.linalg.norm(delta))
                if n < 1e-8:
                    delta = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    delta = delta / n
                repel += 6.0 * (influence - dist) * delta
        
        # Dynamic obstacle repulsion with multi-step lookahead
        for obs in env.dynamic_obstacles:
            for lookahead in range(6):
                future_t = t + lookahead * dt
                center = obs.position(future_t)
                influence = obs.radius + 2.0 + lookahead * 0.1
                
                delta = self.pos - center
                dist = float(np.linalg.norm(delta))
                
                if dist < influence:
                    if dist < 1e-8:
                        delta = np.array([1.0, 0.0, 0.0], dtype=float)
                        dist = 0.0
                    else:
                        delta = delta / dist
                    strength = 6.0 * (influence - dist) * (1.0 - lookahead * 0.12)
                    repel += strength * delta
        
        return repel
    
    def compute_control(self, target: np.ndarray, env: Environment, t: float, dt: float) -> np.ndarray:
        wind_v = env.wind.velocity(t)
        
        to_target = target - self.pos
        dist_target = float(np.linalg.norm(to_target))
        
        # Adaptive speed based on distance
        speed_factor = 0.7 if dist_target > 3.0 else (0.5 if dist_target > 1.5 else 0.35)
        desired_vel = self.clip_norm(to_target * 1.5, speed_factor * self.v_max)
        
        repel = self.compute_repulsion(env, t, dt)
        
        a_cmd = 1.5 * (desired_vel - self.vel) - 0.8 * wind_v + repel
        a_cmd = self.clip_norm(a_cmd, 0.82 * self.a_max)
        
        return a_cmd
    
    def step(self, a_cmd: np.ndarray, wind_v: np.ndarray, dt: float, 
             env: Environment, t: float) -> np.ndarray:
        vel_new = self.vel + a_cmd * dt
        vel_new = self.clip_norm(vel_new, 0.88 * self.v_max)
        pos_new = self.pos + (vel_new + wind_v) * dt
        
        # Safety check with margin
        safety_margin = 0.3
        safe = (env.is_in_bounds(pos_new, safety_margin) and 
                not env.is_in_no_fly_zone(pos_new) and
                not env.check_dynamic_collision(pos_new, t + dt, 0.3))
        
        if not safe:
            a_cmd = self.clip_norm(0.3 * a_cmd, 0.4 * self.a_max)
            vel_new = self.vel + a_cmd * dt
            vel_new = self.clip_norm(vel_new, 0.5 * self.v_max)
        
        self.vel = vel_new
        self.pos = self.pos + (self.vel + wind_v) * dt
        
        return a_cmd


def compute_visiting_order(points: np.ndarray, start_pos: np.ndarray) -> List[int]:
    """Compute visiting order using nearest neighbor heuristic."""
    n = len(points)
    if n == 0:
        return []
    
    order = []
    remaining = set(range(n))
    current = start_pos.copy()
    
    while remaining:
        best_idx = min(remaining, key=lambda i: float(np.linalg.norm(points[i] - current)))
        order.append(best_idx)
        current = points[best_idx].copy()
        remaining.remove(best_idx)
    
    return order


def build_submission_for_scene(scene: dict[str, Any], dt: float, coverage_radius: float) -> dict[str, Any]:
    t_max = float(scene["T_max"])
    v_max = float(scene["uav"]["v_max"])
    a_max = float(scene["uav"]["a_max"])
    points = np.array(scene["inspection_points"], dtype=float)
    visited = np.zeros(len(points), dtype=bool)
    
    env = Environment(scene)
    start = np.array(scene["start"], dtype=float)
    controller = UAVController(start, v_max, a_max)
    
    # Pre-compute visiting order for better coverage
    visiting_order = compute_visiting_order(points, start[:3])
    current_target_idx = 0
    
    timestamps: List[float] = [0.0]
    controls: List[List[float]] = [[0.0, 0.0, 0.0]]
    
    t = 0.0
    
    while t + dt <= t_max + 1e-12:
        # Update coverage
        dists = np.linalg.norm(points - controller.pos, axis=1)
        visited |= dists <= coverage_radius
        
        # Select next target using pre-computed order
        while current_target_idx < len(visiting_order) and visited[visiting_order[current_target_idx]]:
            current_target_idx += 1
        
        if current_target_idx < len(visiting_order):
            target = points[visiting_order[current_target_idx]]
        else:
            # All pre-planned targets visited, find nearest unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited) > 0:
                target = points[unvisited[np.argmin(dists[unvisited])]]
            else:
                target = controller.pos
        
        wind_v = env.wind.velocity(t)
        a_cmd = controller.compute_control(target, env, t, dt)
        a_cmd = controller.step(a_cmd, wind_v, dt, env, t)
        
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
    
    print("Modular submission written to submission.json")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END