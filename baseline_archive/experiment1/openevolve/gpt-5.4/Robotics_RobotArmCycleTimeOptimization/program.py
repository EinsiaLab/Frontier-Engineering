# EVOLVE-BLOCK-START
"""Baseline solver for RobotArmCycleTimeOptimization.

Approach:
1. Use a fixed 3-waypoint path (start -> via -> goal).
2. Time-scale the path and binary-search the smallest feasible total time.
3. Feasibility is checked with the same constraints as the evaluator
   (joint limits, velocity, acceleration, and collision in PyBullet).
"""

from __future__ import annotations

import json

import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import CubicSpline

Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)
Q_MID = 0.5 * (Q_START + Q_GOAL)
Q_DIR = Q_GOAL - Q_START
Q_BEND = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)
Q_POST = np.array([0.35, 0.0, 0.25, 0.0, 0.2, 0.0, 0.2], dtype=float)
Q_VIAS = np.array(
    [Q_MID + a * Q_BEND + b * Q_DIR for a in (0.0, 0.12, 0.24, -0.12) for b in (-0.12, 0.0, 0.12)]
    + [Q_MID + a * Q_BEND + c * Q_POST for a in (0.0, 0.12, 0.24) for c in (-0.35, 0.35)],
    dtype=float,
)

MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 30

OBS_CENTER = [0.45, -0.35, 0.65]
OBS_HALF = [0.08, 0.20, 0.08]


def _joint_index_map(robot_id: int) -> list[int]:
    idxs: list[int] = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            idxs.append(j)
    return idxs[:7]


def _set_q(robot_id: int, q: np.ndarray, joint_idxs: list[int]) -> None:
    for i, joint_idx in enumerate(joint_idxs):
        p.resetJointState(robot_id, joint_idx, float(q[i]))


def _build_timestamps(total_time: float, split: float) -> np.ndarray:
    return total_time * np.array([0.0, split, 1.0], dtype=float)


def _is_feasible(total_time: float, split: float, via: np.ndarray, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int]) -> bool:
    t = _build_timestamps(total_time, split)
    cs = CubicSpline(t, np.vstack([Q_START, via, Q_GOAL]), bc_type="clamped")
    ts = np.r_[np.linspace(t[0], t[1], SAMPLES_PER_SEG, endpoint=False), np.linspace(t[1], t[2], SAMPLES_PER_SEG, endpoint=False)]
    q = cs(ts)
    if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
        return False
    d1 = cs.derivative(1)(ts)
    if np.any(np.abs(d1) > MAX_VEL + 1e-4):
        return False
    if np.any(np.abs(cs.derivative(2)(ts)) > MAX_ACC + 1e-4):
        return False
    for x in q:
        _set_q(robot_id, x, joint_idxs)
        if p.getClosestPoints(robot_id, obs_id, distance=0.0):
            return False
    return True


def solve() -> tuple[list[list[float]], list[float]]:
    physics = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        obs_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=OBS_HALF)
        obs_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=obs_shape, basePosition=OBS_CENTER)

        joint_idxs = _joint_index_map(robot_id)
        joint_limits = np.array(
            [[p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9]] for j in joint_idxs],
            dtype=float,
        )

        for split in (0.35, 0.45, 0.55, 0.65):
            via = Q_START + split * Q_DIR
            if _is_feasible(1.55, split, via, joint_limits, robot_id, obs_id, joint_idxs):
                lo, hi = 1.0, 1.55
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    if _is_feasible(mid, split, via, joint_limits, robot_id, obs_id, joint_idxs):
                        hi = mid
                    else:
                        lo = mid
                return [Q_START.tolist(), via.tolist(), Q_GOAL.tolist()], _build_timestamps(hi, split).tolist()

        best_t, best_split, best_via = 60.0, 0.5, Q_VIAS[0]
        for via in np.vstack([Q_VIAS, Q_START + np.outer((0.3, 0.4, 0.5, 0.6, 0.7), Q_DIR)]):
            d0, d1 = np.abs(via - Q_START), np.abs(Q_GOAL - via)
            for split in (0.3, 0.4, 0.5, 0.6, 0.7):
                lo = max(float(np.max(d0 / (MAX_VEL * split))), float(np.max(d1 / (MAX_VEL * (1.0 - split)))), 1.0)
                hi = min(best_t, 2.2)
                while hi < best_t and hi <= 60.0 and not _is_feasible(hi, split, via, joint_limits, robot_id, obs_id, joint_idxs):
                    hi *= 1.25
                if hi > best_t or hi > 60.0:
                    continue
                for _ in range(32):
                    mid = 0.5 * (lo + hi)
                    if _is_feasible(mid, split, via, joint_limits, robot_id, obs_id, joint_idxs):
                        hi = mid
                    else:
                        lo = mid
                if hi < best_t:
                    best_t, best_split, best_via = hi, split, via.copy()

        return [Q_START.tolist(), best_via.tolist(), Q_GOAL.tolist()], _build_timestamps(best_t, best_split).tolist()
    finally:
        p.disconnect(physics)


def main() -> None:
    waypoints, timestamps = solve()
    submission = {"waypoints": waypoints, "timestamps": timestamps}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    total_time = timestamps[-1]
    print("Baseline submission written to submission.json")
    print(f"Total time: {total_time:.4f} s")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
