# EVOLVE-BLOCK-START
"""Crossover solver for RobotArmCycleTimeOptimization.

Architecture:
1. RobotModel: Encapsulates PyBullet robot and obstacle interactions
2. TrajectoryChecker: Validates constraints with early termination
3. TimeOptimizer: Binary search for minimum feasible time
4. PathOptimizer: Coordinates multi-phase via-point optimization
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import CubicSpline


# Constants
Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)
MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 30
OBS_CENTER = [0.45, -0.35, 0.65]
OBS_HALF = [0.08, 0.20, 0.08]


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""
    waypoints: list[list[float]]
    timestamps: list[float]
    total_time: float


class RobotModel:
    """Encapsulates PyBullet robot and obstacle model with collision caching."""

    def __init__(self):
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        obs_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=OBS_HALF)
        self.obs_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=obs_shape,
            basePosition=OBS_CENTER
        )

        self.joint_idxs = self._get_joint_indices()
        self.joint_limits = self._get_joint_limits()
        self._collision_cache = {}  # Cache for collision results

    def _get_joint_indices(self) -> list[int]:
        indices = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                indices.append(j)
        return indices[:7]

    def _get_joint_limits(self) -> np.ndarray:
        return np.array([
            [p.getJointInfo(self.robot_id, j)[8], p.getJointInfo(self.robot_id, j)[9]]
            for j in self.joint_idxs
        ], dtype=float)

    def set_configuration(self, q: np.ndarray) -> None:
        for i, joint_idx in enumerate(self.joint_idxs):
            p.resetJointState(self.robot_id, joint_idx, float(q[i]))

    def check_collision(self, q: np.ndarray) -> bool:
        """Returns True if collision exists. Uses caching for speed."""
        # Quantize for cache key (4 decimal places)
        key = tuple(np.round(q, 4))
        if key in self._collision_cache:
            return self._collision_cache[key]

        self.set_configuration(q)
        result = len(p.getClosestPoints(self.robot_id, self.obs_id, distance=0.0)) > 0
        self._collision_cache[key] = result
        return result

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clip joint values to limits."""
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def cleanup(self) -> None:
        p.disconnect(self.physics_client)


class TrajectoryChecker:
    """Validates trajectory constraints with early termination and vectorized checks."""

    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.joint_limits = robot.joint_limits
        self.joint_min = self.joint_limits[:, 0]
        self.joint_max = self.joint_limits[:, 1]

    def is_feasible(self, total_time: float, waypoints: np.ndarray) -> bool:
        """Full feasibility check with early termination."""
        n_pts = len(waypoints)
        timestamps = np.linspace(0.0, total_time, n_pts)

        cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
        cs_vel = cs.derivative(1)
        cs_acc = cs.derivative(2)

        for seg in range(n_pts - 1):
            t0, t1 = float(timestamps[seg]), float(timestamps[seg + 1])
            t_samples = np.linspace(t0, t1, SAMPLES_PER_SEG, endpoint=False)

            q_batch = cs(t_samples)
            v_batch = cs_vel(t_samples)
            a_batch = cs_acc(t_samples)

            # Vectorized kinematic constraint check (much faster)
            if np.any(q_batch < self.joint_min - 1e-4):
                return False
            if np.any(q_batch > self.joint_max + 1e-4):
                return False
            if np.any(np.abs(v_batch) > MAX_VEL + 1e-4):
                return False
            if np.any(np.abs(a_batch) > MAX_ACC + 1e-4):
                return False

            # Collision check with early termination
            for q in q_batch:
                if self.robot.check_collision(q):
                    return False

        return True


class TimeOptimizer:
    """Binary search for minimum feasible trajectory time."""

    def __init__(self, checker: TrajectoryChecker):
        self.checker = checker

    def find_min_time(self, waypoints: np.ndarray, max_iterations: int = 17) -> float:
        """Find minimum feasible time for given waypoints."""
        # Compute tighter lower bound from velocity and acceleration limits
        delta = np.abs(Q_GOAL - Q_START)
        for wp in waypoints[1:-1]:  # Include via-points
            delta = np.maximum(delta, np.abs(wp - Q_START))
            delta = np.maximum(delta, np.abs(Q_GOAL - wp))

        # Lower bound from velocity limits (tighter bound)
        vel_bound = np.max(delta / (MAX_VEL * 0.96))
        # Lower bound from acceleration (need time to accelerate/decelerate)
        acc_bound = np.sqrt(np.max(delta / MAX_ACC))
        lower_bound = max(vel_bound, acc_bound)
        lo = max(0.3, lower_bound)
        hi = max(lo * 2.0, 3.5)

        # Find upper bound with limited iterations
        for _ in range(6):
            if self.checker.is_feasible(hi, waypoints):
                break
            hi *= 1.5
        else:
            if hi > 50.0:
                return float('inf')

        # Binary search with sufficient precision
        for _ in range(max_iterations):
            mid = 0.5 * (lo + hi)
            if self.checker.is_feasible(mid, waypoints):
                hi = mid
            else:
                lo = mid

        return hi


class PathOptimizer:
    """Coordinates via-point optimization with obstacle-aware strategies."""

    # Primary obstacle-avoidance bias for obstacle at [0.45, -0.35, 0.65]
    # Joint 1 (base rotation) and Joint 3 (elbow) most affect position in obstacle region
    OBSTACLE_AVOID_BIAS = np.array([0.22, 0.06, 0.18, 0.02, 0.0, 0.0, 0.0])
    # Alternative bias for exploring different paths (opposite direction)
    OBSTACLE_AVOID_BIAS_ALT = np.array([-0.20, 0.10, -0.15, 0.03, 0.0, 0.0, 0.0])

    def __init__(self, robot: RobotModel, time_optimizer: TimeOptimizer):
        self.robot = robot
        self.time_optimizer = time_optimizer
        self.joint_limits = robot.joint_limits

    def _clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        return self.robot.clip_to_limits(q)

    def _interpolate_via(self, alpha: float, add_bias: bool = False, use_alt_bias: bool = False) -> np.ndarray:
        """Linear interpolation between start and goal with optional obstacle avoidance bias."""
        base = Q_START + alpha * (Q_GOAL - Q_START)
        if add_bias:
            # Bias is strongest at middle of path (alpha=0.5), weaker at ends
            bias_strength = 0.5 - abs(alpha - 0.5)
            bias = self.OBSTACLE_AVOID_BIAS_ALT if use_alt_bias else self.OBSTACLE_AVOID_BIAS
            base = base + bias * bias_strength
        return base

    def _evaluate_path(self, via_points: list[np.ndarray]) -> float:
        """Evaluate total time for a path."""
        all_waypoints = np.vstack([Q_START] + via_points + [Q_GOAL])
        return self.time_optimizer.find_min_time(all_waypoints)

    def _create_obstacle_aware_via(self, alpha: float, use_alt_bias: bool = False) -> np.ndarray:
        """Create a via-point biased away from obstacle region."""
        base = Q_START + alpha * (Q_GOAL - Q_START)
        bias = (self.OBSTACLE_AVOID_BIAS_ALT if use_alt_bias else self.OBSTACLE_AVOID_BIAS) * np.random.uniform(0.5, 1.5)
        perturbed = base + bias + np.random.uniform(-0.15, 0.15, 7)
        return self._clip_to_limits(perturbed)

    def _generate_initial_candidates(self, n_via: int) -> list[list[np.ndarray]]:
        """Generate diverse initial candidates with obstacle-aware strategies."""
        candidates = []
        # Denser coverage in collision zone for better via-point placement
        alphas = np.concatenate([
            np.linspace(0.08, 0.28, 6),   # Before obstacle region
            np.linspace(0.30, 0.70, 16),  # Very dense coverage in collision zone
            np.linspace(0.72, 0.92, 6)    # After obstacle region
        ])

        if n_via == 1:
            for a1 in alphas:
                via = self._interpolate_via(a1, add_bias=True)
                candidates.append([self._clip_to_limits(via)])
                # Also try alternative bias
                via_alt = self._interpolate_via(a1, add_bias=True, use_alt_bias=True)
                candidates.append([self._clip_to_limits(via_alt)])
        elif n_via == 2:
            for a1 in alphas[:10]:
                for a2 in alphas[8:]:
                    if a1 < a2:
                        v1 = self._interpolate_via(a1, add_bias=True)
                        v2 = self._interpolate_via(a2, add_bias=True)
                        candidates.append([self._clip_to_limits(v1), self._clip_to_limits(v2)])
        elif n_via == 3:
            for a1 in alphas[:6]:
                for a2 in alphas[5:12]:
                    for a3 in alphas[10:]:
                        if a1 < a2 < a3:
                            v1 = self._interpolate_via(a1, add_bias=True)
                            v2 = self._interpolate_via(a2, add_bias=True)
                            v3 = self._interpolate_via(a3, add_bias=True)
                            candidates.append([
                                self._clip_to_limits(v1),
                                self._clip_to_limits(v2),
                                self._clip_to_limits(v3)
                            ])

        return candidates

    def optimize(self, n_via: int = 2, seed: int = 42, early_stop_time: float = 1.18) -> OptimizationResult:
        """Main optimization routine with multi-phase search and early stopping."""
        np.random.seed(seed)

        best_time = float('inf')
        best_vias = None

        # Phase 1: Grid-based initialization with obstacle bias
        grid_candidates = self._generate_initial_candidates(n_via)
        for vias in grid_candidates:
            t = self._evaluate_path(vias)
            if t < best_time:
                best_time = t
                best_vias = [v.copy() for v in vias]

        # If no valid candidates found, create obstacle-aware random ones
        if best_vias is None:
            best_vias = [self._create_obstacle_aware_via((i + 1) / (n_via + 1)) for i in range(n_via)]
            best_time = self._evaluate_path(best_vias)

        # Phase 2: Focused random exploration with both bias directions
        for _ in range(50):
            use_alt = np.random.random() < 0.35
            vias = [self._create_obstacle_aware_via((i + 1) / (n_via + 1), use_alt_bias=use_alt) for i in range(n_via)]
            t = self._evaluate_path(vias)
            if t < best_time:
                best_time = t
                best_vias = [v.copy() for v in vias]

        # Phase 3: Local refinement with decreasing scales
        scales = [0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]
        iterations = [50, 45, 40, 35, 30, 25, 20]

        for scale, iters in zip(scales, iterations):
            for _ in range(iters):
                vias = [
                    self._clip_to_limits(v + np.random.uniform(-scale, scale, 7))
                    for v in best_vias
                ]
                t = self._evaluate_path(vias)
                if t < best_time:
                    best_time = t
                    best_vias = [v.copy() for v in vias]
            # Early exit only if very good solution found
            if best_time < early_stop_time:
                break

        # Build result
        all_waypoints = [Q_START.tolist()] + [v.tolist() for v in best_vias] + [Q_GOAL.tolist()]
        timestamps = np.linspace(0.0, best_time, len(all_waypoints)).tolist()

        return OptimizationResult(
            waypoints=all_waypoints,
            timestamps=timestamps,
            total_time=best_time
        )


def solve() -> tuple[list[list[float]], list[float]]:
    """Main entry point for trajectory optimization."""
    robot = RobotModel()

    try:
        checker = TrajectoryChecker(robot)
        time_optimizer = TimeOptimizer(checker)
        path_optimizer = PathOptimizer(robot, time_optimizer)

        best_result = None
        best_time = float('inf')

        # Try 2 via-points first (typically optimal), then 3 and 1
        for n_via in [2, 3, 1]:
            # Try with different seeds for robustness
            for seed_offset in range(3):
                result = path_optimizer.optimize(
                    n_via=n_via,
                    seed=42 + n_via * 10 + seed_offset,
                    early_stop_time=1.18
                )
                if result.total_time < best_time:
                    best_time = result.total_time
                    best_result = result
                # Early exit only if very good solution found
                if best_time < 1.18:
                    break
            if best_time < 1.18:
                break

        return best_result.waypoints, best_result.timestamps

    finally:
        robot.cleanup()


def main() -> None:
    waypoints, timestamps = solve()
    submission = {"waypoints": waypoints, "timestamps": timestamps}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    total_time = timestamps[-1]
    print("Crossover submission written to submission.json")
    print(f"Total time: {total_time:.4f} s")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END