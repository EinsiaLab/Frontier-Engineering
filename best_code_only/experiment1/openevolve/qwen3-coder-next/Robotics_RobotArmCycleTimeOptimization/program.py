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
Q_VIA = np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)

# Optimize via point with improved gradient-free search (optimized for speed)
def _optimize_via_point(joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int], base_time: float = 8.0) -> np.ndarray:
    """Improved local search to find better via point."""
    best_via = Q_VIA.copy()
    best_score = float('inf')
    
    # Coarse to fine search with early termination and more intelligent stepping
    for step_size in [0.1, 0.05, 0.02]:
        improved = True
        search_count = 0
        max_searches = 80  # Increased search budget for better results
        
        while improved and search_count < max_searches:
            improved = False
            # Randomize axis order for better exploration
            axes = np.random.permutation(7)
            
            for axis in axes:
                if search_count >= max_searches:
                    break
                    
                # Try multiple delta values per axis for better exploration
                for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                    test_via = best_via.copy()
                    test_via[axis] += delta
                    search_count += 1
                    
                    # Check bounds with tighter safety margin for optimization
                    if np.any(test_via < joint_limits[:, 0] + 0.02) or np.any(test_via > joint_limits[:, 1] - 0.02):
                        continue
                    
                    waypoints = np.vstack([Q_START, test_via, Q_GOAL])
                    timestamps = np.array([0.0, 0.48 * base_time, base_time], dtype=float)
                    
                    try:
                        # Quick feasibility check with fewer samples first
                        quick_samp = np.linspace(0.0, base_time, 20)
                        cs_quick = CubicSpline(timestamps, waypoints, bc_type="clamped")
                        v_quick = cs_quick.derivative(1)(quick_samp)
                        
                        # Early termination if velocity exceeds limits significantly
                        if np.any(np.abs(v_quick) > MAX_VEL * 1.2):
                            continue
                        
                        cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
                        cs_vel = cs.derivative(1)
                        cs_acc = cs.derivative(2)
                        
                        # Use more samples for better feasibility estimation
                        t_samp = np.linspace(0.0, base_time, 50)
                        q_batch = cs(t_samp)
                        v_batch = cs_vel(t_samp)
                        a_batch = cs_acc(t_samp)
                        
                        feasible = True
                        # Calculate usage metrics for scoring (lower usage = better)
                        max_v_usage = 0.0
                        max_a_usage = 0.0
                        
                        for k in range(len(t_samp)):
                            q = q_batch[k]
                            v = v_batch[k]
                            a = a_batch[k]
                            
                            # Check joint limits with safety margin
                            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                                feasible = False
                                break
                            if np.any(np.abs(v) > MAX_VEL + 1e-4) or np.any(np.abs(a) > MAX_ACC + 1e-4):
                                feasible = False
                                break
                            
                            # Calculate usage metrics
                            v_usage = np.max(np.abs(v) / MAX_VEL)
                            a_usage = np.max(np.abs(a) / MAX_ACC)
                            max_v_usage = max(max_v_usage, v_usage)
                            max_a_usage = max(max_a_usage, a_usage)
                            
                            # Collision check with early termination
                            _set_q(robot_id, q, joint_idxs)
                            if len(p.getClosestPoints(robot_id, obs_id, distance=0.01)) > 0:
                                feasible = False
                                break
                        
                        if feasible:
                            # Score based on max usage (lower is better)
                            score = max(max_v_usage, max_a_usage)
                            if score < best_score:
                                best_via = test_via.copy()
                                best_score = score
                                improved = True
                                break  # Break axis loop to restart search from new best
                    except:
                        pass
    
    return best_via

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


def _build_timestamps(total_time: float) -> np.ndarray:
    ratios = np.array([0.0, 0.48, 1.0], dtype=float)
    return total_time * ratios


# Global variable to store optimized via point
Q_VIA_OPT = None

def _is_feasible(total_time: float, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int], via_point: np.ndarray = None, use_optimized_via: bool = True) -> bool:
    if via_point is None:
        via_point = Q_VIA_OPT if use_optimized_via else Q_VIA
    waypoints = np.vstack([Q_START, via_point, Q_GOAL])
    timestamps = _build_timestamps(total_time)

    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    for seg in range(2):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        # Include endpoint to avoid gaps in sampling
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG)
        
        q_batch = cs(t_samp)
        v_batch = cs_vel(t_samp)
        a_batch = cs_acc(t_samp)

        for k in range(len(t_samp)):
            q = q_batch[k]
            v = v_batch[k]
            a = a_batch[k]

            # Check all constraints in order of computational cost (cheapest first)
            if np.any(np.abs(v) > MAX_VEL + 1e-4):
                return False
            if np.any(np.abs(a) > MAX_ACC + 1e-4):
                return False
            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False

            # Collision check with early termination
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

        # Optimize via point first
        global Q_VIA_OPT
        Q_VIA_OPT = _optimize_via_point(joint_limits, robot_id, obs_id, joint_idxs)
        
        # Better initial bounds based on joint limits and obstacle awareness
        # Calculate minimum time for straight line motion
        straight_line_time = float(np.max(np.abs(Q_GOAL - Q_START) / MAX_VEL))
        
        # Use a conservative obstacle detour factor based on obstacle size and distance
        # The obstacle is at [0.45, -0.35, 0.65] with half-extents [0.08, 0.20, 0.08]
        # Calculate distance from midpoint between start and goal to obstacle center
        mid_q = (Q_START + Q_GOAL) / 2.0
        _set_q(robot_id, mid_q, joint_idxs)
        
        # Get approximate end-effector position using forward kinematics simulation
        # For KUKA LBR iiwa, the end-effector is typically link 7 (but we need to verify)
        # Since we don't know the exact link index, we'll use a simpler approach
        obstacle_distance = 0.5  # Assume minimum distance to obstacle
        obstacle_buffer = max(0.2, min(0.6, 1.0 - obstacle_distance))  # Normalize and cap
        obstacle_detour_factor = 1.0 + obstacle_buffer
        
        lo = max(1.0, straight_line_time * obstacle_detour_factor)
        hi = lo * 1.5

        # First, expand hi until we find a feasible time
        while not _is_feasible(hi, joint_limits, robot_id, obs_id, joint_idxs, via_point=Q_VIA_OPT):
            hi *= 1.5
            if hi > 60.0:
                break

        # Binary search with more iterations for better precision
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            if _is_feasible(mid, joint_limits, robot_id, obs_id, joint_idxs, via_point=Q_VIA_OPT):
                hi = mid
            else:
                lo = mid

        best_t = hi
        waypoints = [Q_START.tolist(), Q_VIA_OPT.tolist(), Q_GOAL.tolist()]
        timestamps = _build_timestamps(best_t).tolist()
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
