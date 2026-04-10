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


def _build_timestamps(total_time: float, ratios: np.ndarray) -> np.ndarray:
    return total_time * ratios


def _is_feasible(total_time: float, waypoints: np.ndarray, ratios: np.ndarray, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int]) -> bool:
    timestamps = _build_timestamps(total_time, ratios)
    n_seg = len(timestamps) - 1
    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    for seg in range(n_seg):
        t0, t1 = float(timestamps[seg]), float(timestamps[seg + 1])
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG, endpoint=False)
        q_batch, v_batch, a_batch = cs(t_samp), cs_vel(t_samp), cs_acc(t_samp)

        for k in range(len(t_samp)):
            q, v, a = q_batch[k], v_batch[k], a_batch[k]
            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False
            if np.any(np.abs(v) > MAX_VEL + 1e-4) or np.any(np.abs(a) > MAX_ACC + 1e-4):
                return False
            _set_q(robot_id, q, joint_idxs)
            if len(p.getClosestPoints(robot_id, obs_id, distance=0.0)) > 0:
                return False
    return True


def _generate_via_candidates(joint_limits: np.ndarray) -> list[np.ndarray]:
    """Generate diverse via point candidates for collision avoidance."""
    direct = 0.5 * (Q_START + Q_GOAL)
    delta = Q_GOAL - Q_START
    candidates = [direct]
    # Linear interpolation candidates
    for scale in np.linspace(0.1, 0.9, 9):
        candidates.append(Q_START + scale * delta)
    # Perturbation vectors designed to avoid obstacle region
    perturbs = [
        np.array([0.35, -0.25, 0.25, -0.2, 0.15, -0.15, 0.25]),
        np.array([-0.25, 0.35, -0.15, 0.25, -0.2, 0.15, -0.15]),
        np.array([0.25, 0.15, 0.35, -0.25, 0.2, -0.1, 0.15]),
        np.array([0.45, -0.35, 0.15, -0.15, 0.25, 0.05, 0.35]),
        np.array([0.3, 0.3, 0.3, -0.3, 0.2, 0.2, 0.2]),
        np.array([-0.2, -0.2, 0.4, -0.1, 0.3, 0.1, 0.3]),
        np.array([0.4, 0.1, 0.2, -0.35, 0.25, 0.0, 0.2]),
    ]
    for pvec in perturbs:
        for scale in [0.3, 0.5, 0.7, 1.0]:
            candidates.append(np.clip(direct + scale * pvec, joint_limits[:, 0], joint_limits[:, 1]))
            candidates.append(np.clip(direct - scale * pvec, joint_limits[:, 0], joint_limits[:, 1]))
    return candidates


def _is_feasible_direct(total_time: float, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int]) -> bool:
    """Check feasibility for direct path (no via point)."""
    waypoints = np.vstack([Q_START, Q_GOAL])
    timestamps = np.array([0.0, total_time])
    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel, cs_acc = cs.derivative(1), cs.derivative(2)
    t_samp = np.linspace(0, total_time, 60, endpoint=False)
    for q, v, a in zip(cs(t_samp), cs_vel(t_samp), cs_acc(t_samp)):
        if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
            return False
        if np.any(np.abs(v) > MAX_VEL + 1e-4) or np.any(np.abs(a) > MAX_ACC + 1e-4):
            return False
        _set_q(robot_id, q, joint_idxs)
        if len(p.getClosestPoints(robot_id, obs_id, distance=0.0)) > 0:
            return False
    return True


def _is_feasible_2via(total_time: float, via1: np.ndarray, via2: np.ndarray, ratios: np.ndarray, joint_limits: np.ndarray, robot_id: int, obs_id: int, joint_idxs: list[int]) -> bool:
    """Check feasibility for path with 2 via points."""
    waypoints = np.vstack([Q_START, via1, via2, Q_GOAL])
    timestamps = total_time * ratios
    n_seg = len(timestamps) - 1
    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel, cs_acc = cs.derivative(1), cs.derivative(2)
    for seg in range(n_seg):
        t0, t1 = float(timestamps[seg]), float(timestamps[seg + 1])
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG, endpoint=False)
        for q, v, a in zip(cs(t_samp), cs_vel(t_samp), cs_acc(t_samp)):
            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False
            if np.any(np.abs(v) > MAX_VEL + 1e-4) or np.any(np.abs(a) > MAX_ACC + 1e-4):
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
        joint_limits = np.array([[p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9]] for j in joint_idxs], dtype=float)
        
        lower_bound = float(np.max(np.abs(Q_GOAL - Q_START) / (MAX_VEL * 0.95)))
        
        # Try direct path first
        best_t = 60.0
        best_waypoints = None
        best_timestamps = None
        
        lo, hi = max(0.5, lower_bound), 15.0
        for _ in range(6):
            if _is_feasible_direct(hi, joint_limits, robot_id, obs_id, joint_idxs):
                break
            hi *= 1.5
        else:
            hi = 60.0
        if hi < 60.0:
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                if _is_feasible_direct(mid, joint_limits, robot_id, obs_id, joint_idxs):
                    hi = mid
                else:
                    lo = mid
            if hi < best_t:
                best_t = hi
                best_waypoints = [Q_START.tolist(), Q_GOAL.tolist()]
                best_timestamps = [0.0, hi]
        
        # Single via point search
        candidates = _generate_via_candidates(joint_limits)
        timing_options = [np.array([0.0, r, 1.0]) for r in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]]
        best_via, best_ratio = 0.5 * (Q_START + Q_GOAL), 0.5
        
        for via in candidates:
            for ratios in timing_options:
                lo, hi = max(0.5, lower_bound), min(15.0, best_t * 1.1)
                wps = np.vstack([Q_START, via, Q_GOAL])
                if not _is_feasible(hi, wps, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                    continue
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if _is_feasible(mid, wps, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                        hi = mid
                    else:
                        lo = mid
                if hi < best_t:
                    best_t, best_via, best_ratio = hi, via.copy(), ratios[1]
                    best_waypoints = [Q_START.tolist(), via.tolist(), Q_GOAL.tolist()]
                    best_timestamps = (hi * ratios).tolist()
        
        # Two via points search
        via_shortlist = candidates[:min(12, len(candidates))]
        for i, via1 in enumerate(via_shortlist):
            for via2 in via_shortlist[i+1:i+6]:
                for r1, r2 in [(0.2, 0.4), (0.25, 0.5), (0.3, 0.55), (0.3, 0.6), (0.35, 0.65), (0.4, 0.7), (0.45, 0.75)]:
                    ratios = np.array([0.0, r1, r2, 1.0])
                    lo, hi = max(0.5, lower_bound), min(15.0, best_t * 1.05)
                    if not _is_feasible_2via(hi, via1, via2, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                        continue
                    for _ in range(20):
                        mid = 0.5 * (lo + hi)
                        if _is_feasible_2via(mid, via1, via2, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                            hi = mid
                        else:
                            lo = mid
                    if hi < best_t:
                        best_t = hi
                        best_waypoints = [Q_START.tolist(), via1.tolist(), via2.tolist(), Q_GOAL.tolist()]
                        best_timestamps = (hi * ratios).tolist()
        
        # Local refinement around best solution with adaptive perturbation
        if best_waypoints is not None and len(best_waypoints) == 3:
            np.random.seed(42)
            for iteration in range(35):
                scale = 0.08 * (0.85 ** (iteration // 10))
                perturb = np.random.uniform(-scale, scale, 7)
                via_refined = np.clip(np.array(best_via) + perturb, joint_limits[:, 0], joint_limits[:, 1])
                for dr in np.linspace(-0.08, 0.08, 5):
                    new_r = np.clip(best_ratio + dr, 0.2, 0.8)
                    ratios = np.array([0.0, new_r, 1.0])
                    lo, hi = max(0.5, lower_bound), best_t * 1.02
                    wps = np.vstack([Q_START, via_refined, Q_GOAL])
                    if not _is_feasible(hi, wps, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                        continue
                    for _ in range(28):
                        mid = 0.5 * (lo + hi)
                        if _is_feasible(mid, wps, ratios, joint_limits, robot_id, obs_id, joint_idxs):
                            hi = mid
                        else:
                            lo = mid
                    if hi < best_t:
                        best_t = hi
                        best_waypoints = [Q_START.tolist(), via_refined.tolist(), Q_GOAL.tolist()]
                        best_timestamps = (hi * ratios).tolist()

        if best_waypoints is None:
            best_waypoints = [Q_START.tolist(), Q_GOAL.tolist()]
            best_timestamps = [0.0, 10.0]
        return best_waypoints, best_timestamps
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
