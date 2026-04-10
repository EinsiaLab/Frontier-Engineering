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

# Generate candidate via points with more diversity
def generate_via_candidates():
    base = (Q_START + Q_GOAL) / 2.0
    candidates = []
    # Perturb multiple joints to explore different paths
    for j0 in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for j2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for j4 in [0.0, 0.3, 0.5]:
                via = base.copy()
                via[0] = j0
                via[2] = j2
                via[4] = j4
                candidates.append(via)
    # Also add points closer to start or goal
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        via = Q_START + alpha * (Q_GOAL - Q_START)
        candidates.append(via)
    return candidates

MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 40  # Increased sampling for better constraint checking without being too slow

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


def _build_timestamps(total_time: float) -> np.ndarray:
    ratios = np.array([0.0, 0.5, 1.0], dtype=float)
    return total_time * ratios


def _is_feasible(total_time: float, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int], via_point: np.ndarray) -> bool:
    waypoints = np.vstack([Q_START, via_point, Q_GOAL])
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

        for k in range(len(t_samp)):
            q = q_batch[k]
            v = v_batch[k]
            a = a_batch[k]

            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False
            if np.any(np.abs(v) > MAX_VEL + 1e-4):
                return False
            if np.any(np.abs(a) > MAX_ACC + 1e-4):
                return False

            _set_q(robot_id, q, joint_idxs)
            if len(p.getClosestPoints(robot_id, obs_id, distance=0.0)) > 0:
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

        candidates = generate_via_candidates()
        best_time = float('inf')
        best_via = None
        
        for via in candidates:
            # Compute a tighter lower bound using both velocity and acceleration constraints
            delta_q = np.abs(Q_GOAL - Q_START)
            vel_bound = np.max(delta_q / MAX_VEL)
            # Estimate acceleration bound: assume maximum acceleration needed to reach velocity
            acc_bound = np.sqrt(np.max(delta_q / MAX_ACC))
            lower_bound = max(vel_bound, acc_bound)
            lo = max(0.5, lower_bound)  # Start with a tighter bound
            hi = lo * 3.0  # Increase initial range
            
            # Ensure hi is feasible
            # Try to find a feasible upper bound quickly
            for expand_iter in range(10):
                if _is_feasible(hi, joint_limits, robot_id, obs_id, joint_idxs, via):
                    break
                hi *= 1.5
                if hi > 60.0:
                    break
            if hi > 60.0:
                continue  # Candidate not feasible within reasonable time
            
            # Binary search with more iterations for precision
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _is_feasible(mid, joint_limits, robot_id, obs_id, joint_idxs, via):
                    hi = mid
                else:
                    lo = mid
            
            if hi < best_time:
                best_time = hi
                best_via = via
        
        # Fallback if no candidate works
        if best_via is None:
            best_via = np.array([0.8, 0.1, 0.6, -1.8, 0.2, 0.9, 0.5], dtype=float)
            delta_q = np.abs(Q_GOAL - Q_START)
            vel_bound = np.max(delta_q / MAX_VEL)
            acc_bound = np.sqrt(np.max(delta_q / MAX_ACC))
            lower_bound = max(vel_bound, acc_bound)
            lo = max(1.0, lower_bound)
            hi = lo * 2.0
            for expand_iter in range(10):
                if _is_feasible(hi, joint_limits, robot_id, obs_id, joint_idxs, best_via):
                    break
                hi *= 1.5
                if hi > 60.0:
                    break
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _is_feasible(mid, joint_limits, robot_id, obs_id, joint_idxs, best_via):
                    hi = mid
                else:
                    lo = mid
            best_time = hi
        
        waypoints = [Q_START.tolist(), best_via.tolist(), Q_GOAL.tolist()]
        timestamps = _build_timestamps(best_time).tolist()
        return waypoints, timestamps
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
