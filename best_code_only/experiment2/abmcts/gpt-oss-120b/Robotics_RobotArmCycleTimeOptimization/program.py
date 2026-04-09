# EVOLVE-BLOCK-START
"""Improved baseline solver for RobotArmCycleTimeOptimization.

Enhancements:
- Generates a larger set of candidate via‑points (including random samples and a heuristic point away from the obstacle).
- Uses a cheap lower‑bound filter to discard via‑points that cannot improve the current best time.
- After the best via‑point is found, performs a small local random search to possibly lower the time further.
- Reduces binary‑search iterations slightly to keep runtime low while preserving precision.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import CubicSpline

# Fixed start / goal configurations
Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)

# Velocity / acceleration limits from the task description
MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)

# Sampling resolution for feasibility checks
SAMPLES_PER_SEG = 30

# Obstacle definition
OBS_CENTER = [0.45, -0.35, 0.65]
OBS_HALF = [0.08, 0.20, 0.08]


def _joint_index_map(robot_id: int) -> list[int]:
    """Return the list of joint indices that are actuated (revolute/prismatic)."""
    idxs: list[int] = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            idxs.append(j)
    return idxs[:7]  # robot has exactly 7 DoF


def _set_q(robot_id: int, q: np.ndarray, joint_idxs: list[int]) -> None:
    """Set the robot joint positions in the PyBullet simulation."""
    for i, joint_idx in enumerate(joint_idxs):
        p.resetJointState(robot_id, joint_idx, float(q[i]))


def _build_timestamps(total_time: float) -> np.ndarray:
    """Fixed timestamp ratios: start → via → goal."""
    ratios = np.array([0.0, 0.48, 1.0], dtype=float)
    return total_time * ratios


def _is_feasible(
    total_time: float,
    via: np.ndarray,
    joint_limits: np.ndarray,
    robot_id: int,
    obs_id: int,
    joint_idxs: list[int],
) -> bool:
    """Check all constraints for a given total_time and via‑point."""
    waypoints = np.vstack([Q_START, via, Q_GOAL])
    timestamps = _build_timestamps(total_time)

    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    for seg in range(2):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG, endpoint=False)

        q_batch = cs(t_samp)
        v_batch = cs_vel(t_samp)
        a_batch = cs_acc(t_samp)

        for q, v, a in zip(q_batch, v_batch, a_batch):
            # Joint limits
            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False
            # Velocity limits
            if np.any(np.abs(v) > MAX_VEL + 1e-4):
                return False
            # Acceleration limits
            if np.any(np.abs(a) > MAX_ACC + 1e-4):
                return False

            _set_q(robot_id, q, joint_idxs)
            if len(p.getClosestPoints(robot_id, obs_id, distance=0.0)) > 0:
                return False

    return True


def _binary_search_time(
    via: np.ndarray,
    joint_limits: np.ndarray,
    robot_id: int,
    obs_id: int,
    joint_idxs: list[int],
) -> float:
    """Return the smallest feasible total time for the given via‑point."""
    # Lower bound: at least the time needed to move each joint at 95 % of max velocity
    lower_bound = float(np.max(np.abs(Q_GOAL - Q_START) / (MAX_VEL * 0.95)))
    lo = max(0.5, lower_bound)  # a small positive lower bound
    hi = 12.0

    # Ensure hi is feasible
    while not _is_feasible(hi, via, joint_limits, robot_id, obs_id, joint_idxs):
        hi *= 1.5
        if hi > 60.0:  # give up after a reasonable horizon
            return float("inf")

    # Binary search (30 iterations ≈ 1e‑9 precision)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _is_feasible(mid, via, joint_limits, robot_id, obs_id, joint_idxs):
            hi = mid
        else:
            lo = mid
    return hi


def _lower_bound_time(via: np.ndarray) -> float:
    """Cheap lower bound on total time for a given via‑point (ignores dynamics)."""
    # Sum of distances for each joint split into two segments, divided by max velocity
    dist_start = np.abs(via - Q_START)
    dist_end = np.abs(Q_GOAL - via)
    total_dist = dist_start + dist_end
    return float(np.max(total_dist / (MAX_VEL * 0.95)))


def solve() -> tuple[list[list[float]], list[float]]:
    """Find waypoints and timestamps with the smallest feasible cycle time."""
    physics = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        obs_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=OBS_HALF)
        obs_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=obs_shape, basePosition=OBS_CENTER
        )

        joint_idxs = _joint_index_map(robot_id)
        joint_limits = np.array(
            [
                [p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9]]
                for j in joint_idxs
            ],
            dtype=float,
        )

        # ------------------------------------------------------------
        # Candidate via‑points generation
        # ------------------------------------------------------------
        candidates: list[np.ndarray] = []

        # 1. Heuristic via (mid‑point shifted away from obstacle)
        heuristic = (Q_START + Q_GOAL) / 2.0
        # Push the second joint (index 1) a bit positive to clear the box
        heuristic[1] = min(joint_limits[1, 1], heuristic[1] + 0.2)
        candidates.append(heuristic)

        # 2. Original via used in the reference solution
        candidates.append(np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5], dtype=float))

        # 3. Exact geometric midpoint
        candidates.append((Q_START + Q_GOAL) / 2.0)

        rng = np.random.default_rng(42)
        # 4. Random samples inside joint limits
        for _ in range(30):
            rand_q = rng.uniform(joint_limits[:, 0], joint_limits[:, 1])
            candidates.append(rand_q)

        # ------------------------------------------------------------
        # Search for the best via‑point
        # ------------------------------------------------------------
        best_time = float("inf")
        best_via = candidates[0]

        for via in candidates:
            # Quick lower‑bound filter
            lb = _lower_bound_time(via)
            if lb >= best_time:
                continue

            t = _binary_search_time(via, joint_limits, robot_id, obs_id, joint_idxs)
            if t < best_time:
                best_time = t
                best_via = via

        # ------------------------------------------------------------
        # Local refinement around the best via‑point
        # ------------------------------------------------------------
        for _ in range(15):
            perturb = rng.normal(scale=0.05, size=7)  # small Gaussian step
            via_candidate = np.clip(best_via + perturb, joint_limits[:, 0], joint_limits[:, 1])
            lb = _lower_bound_time(via_candidate)
            if lb >= best_time:
                continue
            t = _binary_search_time(via_candidate, joint_limits, robot_id, obs_id, joint_idxs)
            if t < best_time:
                best_time = t
                best_via = via_candidate

        # Build final output using the best via‑point
        timestamps = _build_timestamps(best_time).tolist()
        waypoints = [Q_START.tolist(), best_via.tolist(), Q_GOAL.tolist()]
        return waypoints, timestamps
    finally:
        p.disconnect(physics)


def main() -> None:
    waypoints, timestamps = solve()
    submission = {"waypoints": waypoints, "timestamps": timestamps}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    total_time = timestamps[-1]
    print("Submission written to submission.json")
    print(f"Total time: {total_time:.4f} s")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END