# EVOLVE-BLOCK-START
"""Optimized solver for RobotArmCycleTimeOptimization.

Approach:
1. Use multiple via-points optimized to avoid the obstacle.
2. Binary-search for the smallest feasible total time.
3. Optimize via-point placement and time allocation using scipy.optimize.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)

MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 30

OBS_CENTER = [0.45, -0.35, 0.65]
OBS_HALF = [0.08, 0.20, 0.08]


def _joint_index_map(robot_id):
    idxs = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            idxs.append(j)
    return idxs[:7]


def _set_q(robot_id, q, joint_idxs):
    for i, joint_idx in enumerate(joint_idxs):
        p.resetJointState(robot_id, joint_idx, float(q[i]))


def _check_collision(robot_id, obs_id, q, joint_idxs):
    _set_q(robot_id, q, joint_idxs)
    contacts = p.getClosestPoints(robot_id, obs_id, distance=0.0)
    return len(contacts) > 0


def _is_feasible_fast(waypoints_arr, timestamps, joint_limits, robot_id, obs_id, joint_idxs, samples_per_seg=SAMPLES_PER_SEG):
    """Faster feasibility check - vectorized where possible."""
    n_seg = len(timestamps) - 1
    cs = CubicSpline(timestamps, waypoints_arr, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    # Collect all sample times
    all_times = []
    for seg in range(n_seg):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        t_samp = np.linspace(t0, t1, samples_per_seg, endpoint=False)
        all_times.append(t_samp)
    all_times = np.concatenate(all_times)

    q_all = cs(all_times)
    v_all = cs_vel(all_times)
    a_all = cs_acc(all_times)

    # Vectorized checks
    if np.any(q_all < joint_limits[:, 0] - 1e-4) or np.any(q_all > joint_limits[:, 1] + 1e-4):
        return False
    if np.any(np.abs(v_all) > MAX_VEL + 1e-4):
        return False
    if np.any(np.abs(a_all) > MAX_ACC + 1e-4):
        return False

    # Collision checks
    for k in range(len(all_times)):
        if _check_collision(robot_id, obs_id, q_all[k], joint_idxs):
            return False

    q_final = waypoints_arr[-1]
    if _check_collision(robot_id, obs_id, q_final, joint_idxs):
        return False

    return True


def _try_direct_path(total_time, joint_limits, robot_id, obs_id, joint_idxs):
    waypoints = np.vstack([Q_START, Q_GOAL])
    timestamps = np.array([0.0, total_time])
    return _is_feasible_fast(waypoints, timestamps, joint_limits, robot_id, obs_id, joint_idxs)


def _try_via_path(via_points, total_time, time_ratios, joint_limits, robot_id, obs_id, joint_idxs):
    all_waypoints = [Q_START]
    for vp in via_points:
        all_waypoints.append(vp)
    all_waypoints.append(Q_GOAL)
    waypoints = np.vstack(all_waypoints)
    timestamps = total_time * np.array(time_ratios)
    return _is_feasible_fast(waypoints, timestamps, joint_limits, robot_id, obs_id, joint_idxs)


def _try_via_path_quick(via_points, total_time, time_ratios, joint_limits, robot_id, obs_id, joint_idxs, samples=15):
    """Quick feasibility check with fewer samples."""
    all_waypoints = [Q_START]
    for vp in via_points:
        all_waypoints.append(vp)
    all_waypoints.append(Q_GOAL)
    waypoints = np.vstack(all_waypoints)
    timestamps = total_time * np.array(time_ratios)
    return _is_feasible_fast(waypoints, timestamps, joint_limits, robot_id, obs_id, joint_idxs, samples_per_seg=samples)


def solve():
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

        lower_bound = float(np.max(np.abs(Q_GOAL - Q_START) / MAX_VEL))

        best_waypoints = None
        best_timestamps = None
        best_time = float('inf')

        # --- Attempt 1: Direct path ---
        lo_d = max(0.5, lower_bound)
        hi_d = 10.0
        while not _try_direct_path(hi_d, joint_limits, robot_id, obs_id, joint_idxs):
            hi_d *= 1.5
            if hi_d > 60.0:
                break

        if hi_d <= 60.0:
            for _ in range(40):
                mid = 0.5 * (lo_d + hi_d)
                if _try_direct_path(mid, joint_limits, robot_id, obs_id, joint_idxs):
                    hi_d = mid
                else:
                    lo_d = mid
            best_time = hi_d
            best_waypoints = [Q_START.tolist(), Q_GOAL.tolist()]
            best_timestamps = [0.0, best_time]

        # --- Attempt 2: One via-point with many candidates ---
        via_candidates = []
        # Generate a grid of via-points
        for alpha in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            base = (1 - alpha) * Q_START + alpha * Q_GOAL
            via_candidates.append(base.copy())
            for d3 in [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]:
                v = base.copy()
                v[3] += d3
                via_candidates.append(v)
            for d1 in [-0.4, -0.2, 0.2, 0.4]:
                v = base.copy()
                v[1] += d1
                via_candidates.append(v)
            for d0 in [-0.3, -0.15, 0.15, 0.3]:
                v = base.copy()
                v[0] += d0
                via_candidates.append(v)
            for d2 in [-0.3, -0.15, 0.15, 0.3]:
                v = base.copy()
                v[2] += d2
                via_candidates.append(v)
            # Combined perturbations
            for d1 in [-0.3, 0.3]:
                for d3 in [-0.5, 0.5]:
                    v = base.copy()
                    v[1] += d1
                    v[3] += d3
                    via_candidates.append(v)

        # Specific hand-crafted candidates
        via_candidates.extend([
            np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5]),
            np.array([0.6, 0.1, 0.4, -1.15, 0.25, 0.9, 0.5]),
            np.array([0.5, 0.1, 0.3, -1.2, 0.2, 0.95, 0.4]),
            np.array([0.7, 0.0, 0.5, -1.0, 0.3, 0.85, 0.6]),
            np.array([0.6, 0.1, 0.4, -2.0, 0.2, 0.9, 0.5]),
            np.array([0.4, 0.2, 0.2, -1.5, 0.15, 0.95, 0.3]),
            np.array([0.8, -0.1, 0.6, -0.9, 0.35, 0.85, 0.7]),
            np.array([0.6, 0.3, 0.4, -1.3, 0.25, 0.9, 0.5]),
            np.array([0.6, -0.1, 0.4, -1.3, 0.25, 0.9, 0.5]),
        ])

        time_ratio_candidates = [
            [0.0, 0.3, 1.0],
            [0.0, 0.35, 1.0],
            [0.0, 0.4, 1.0],
            [0.0, 0.45, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.55, 1.0],
            [0.0, 0.6, 1.0],
            [0.0, 0.65, 1.0],
            [0.0, 0.7, 1.0],
        ]

        for via in via_candidates:
            via_clipped = np.clip(via, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
            for tr in time_ratio_candidates:
                # Quick pre-check with fewer samples at a moderate time
                test_time = min(best_time * 0.95, 8.0)
                if test_time <= lower_bound:
                    test_time = lower_bound * 1.1
                
                # First check if feasible at a reasonable upper bound
                hi_v = min(best_time * 0.999, 10.0)
                if hi_v <= lower_bound:
                    continue
                    
                if not _try_via_path_quick([via_clipped], hi_v, tr, joint_limits, robot_id, obs_id, joint_idxs, samples=12):
                    # Try larger time
                    hi_v2 = hi_v * 1.5
                    if hi_v2 > 20.0 or hi_v2 >= best_time:
                        continue
                    if not _try_via_path_quick([via_clipped], hi_v2, tr, joint_limits, robot_id, obs_id, joint_idxs, samples=12):
                        continue
                    hi_v = hi_v2

                lo_v = max(0.5, lower_bound)
                
                # Full check at hi_v
                if not _try_via_path([via_clipped], hi_v, tr, joint_limits, robot_id, obs_id, joint_idxs):
                    hi_v_test = hi_v
                    while not _try_via_path([via_clipped], hi_v_test, tr, joint_limits, robot_id, obs_id, joint_idxs):
                        hi_v_test *= 1.3
                        if hi_v_test > 20.0 or hi_v_test >= best_time:
                            break
                    if hi_v_test > 20.0 or hi_v_test >= best_time:
                        continue
                    hi_v = hi_v_test

                for _ in range(30):
                    mid = 0.5 * (lo_v + hi_v)
                    if _try_via_path([via_clipped], mid, tr, joint_limits, robot_id, obs_id, joint_idxs):
                        hi_v = mid
                    else:
                        lo_v = mid

                if hi_v < best_time:
                    best_time = hi_v
                    wp_list = [Q_START.tolist(), via_clipped.tolist(), Q_GOAL.tolist()]
                    ts_list = (best_time * np.array(tr)).tolist()
                    best_waypoints = wp_list
                    best_timestamps = ts_list

        # --- Attempt 3: Two via-points ---
        if best_time > lower_bound * 1.1:
            two_via_candidates = []
            for a1 in [0.25, 0.3, 0.35, 0.4]:
                for a2 in [0.6, 0.65, 0.7, 0.75]:
                    v1 = (1 - a1) * Q_START + a1 * Q_GOAL
                    v2 = (1 - a2) * Q_START + a2 * Q_GOAL
                    for d3_1 in [-0.5, -0.3, 0.0, 0.3, 0.5]:
                        for d3_2 in [-0.5, -0.3, 0.0, 0.3, 0.5]:
                            vv1 = v1.copy()
                            vv1[3] += d3_1
                            vv2 = v2.copy()
                            vv2[3] += d3_2
                            vv1 = np.clip(vv1, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                            vv2 = np.clip(vv2, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                            two_via_candidates.append((vv1, vv2))

            two_via_tr = [
                [0.0, 0.3, 0.65, 1.0],
                [0.0, 0.33, 0.67, 1.0],
                [0.0, 0.35, 0.7, 1.0],
                [0.0, 0.4, 0.7, 1.0],
                [0.0, 0.3, 0.7, 1.0],
                [0.0, 0.4, 0.65, 1.0],
            ]

            for (v1, v2) in two_via_candidates:
                for tr in two_via_tr:
                    hi_v = min(best_time * 0.999, 10.0)
                    if hi_v <= lower_bound:
                        continue
                    
                    if not _try_via_path_quick([v1, v2], hi_v, tr, joint_limits, robot_id, obs_id, joint_idxs, samples=10):
                        continue

                    lo_v = max(0.5, lower_bound)
                    if not _try_via_path([v1, v2], hi_v, tr, joint_limits, robot_id, obs_id, joint_idxs):
                        hi_v_test = hi_v
                        while not _try_via_path([v1, v2], hi_v_test, tr, joint_limits, robot_id, obs_id, joint_idxs):
                            hi_v_test *= 1.3
                            if hi_v_test > 20.0 or hi_v_test >= best_time:
                                break
                        if hi_v_test > 20.0 or hi_v_test >= best_time:
                            continue
                        hi_v = hi_v_test

                    for _ in range(25):
                        mid = 0.5 * (lo_v + hi_v)
                        if _try_via_path([v1, v2], mid, tr, joint_limits, robot_id, obs_id, joint_idxs):
                            hi_v = mid
                        else:
                            lo_v = mid

                    if hi_v < best_time:
                        best_time = hi_v
                        wp_list = [Q_START.tolist(), v1.tolist(), v2.tolist(), Q_GOAL.tolist()]
                        ts_list = (best_time * np.array(tr)).tolist()
                        best_waypoints = wp_list
                        best_timestamps = ts_list

        # --- Attempt 4: Local optimization (Nelder-Mead) ---
        if best_waypoints is not None and len(best_waypoints) == 3:
            best_via = np.array(best_waypoints[1])
            best_tr_mid = best_timestamps[1] / best_time

            def objective(x):
                via = x[:7]
                tr_mid = x[7]
                T = x[8]
                if T <= lower_bound or tr_mid <= 0.05 or tr_mid >= 0.95:
                    return 100.0
                via_clipped = np.clip(via, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                tr = [0.0, tr_mid, 1.0]
                if _try_via_path([via_clipped], T, tr, joint_limits, robot_id, obs_id, joint_idxs):
                    return T
                return 100.0

            x0 = np.concatenate([best_via, [best_tr_mid, best_time]])
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 0.0005, 'fatol': 0.0005, 'adaptive': True})
                if result.fun < best_time:
                    via_opt = np.clip(result.x[:7], joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                    tr_opt = result.x[7]
                    T_opt = result.x[8]
                    if _try_via_path([via_opt], T_opt, [0.0, tr_opt, 1.0], joint_limits, robot_id, obs_id, joint_idxs):
                        best_time = T_opt
                        best_waypoints = [Q_START.tolist(), via_opt.tolist(), Q_GOAL.tolist()]
                        best_timestamps = [0.0, best_time * tr_opt, best_time]
            except Exception:
                pass

        elif best_waypoints is not None and len(best_waypoints) == 4:
            best_v1 = np.array(best_waypoints[1])
            best_v2 = np.array(best_waypoints[2])
            best_tr1 = best_timestamps[1] / best_time
            best_tr2 = best_timestamps[2] / best_time

            def objective2(x):
                v1 = x[:7]
                v2 = x[7:14]
                tr1 = x[14]
                tr2 = x[15]
                T = x[16]
                if T <= lower_bound or tr1 <= 0.05 or tr2 <= tr1 + 0.05 or tr2 >= 0.95:
                    return 100.0
                v1c = np.clip(v1, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                v2c = np.clip(v2, joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                tr = [0.0, tr1, tr2, 1.0]
                if _try_via_path([v1c, v2c], T, tr, joint_limits, robot_id, obs_id, joint_idxs):
                    return T
                return 100.0

            x0 = np.concatenate([best_v1, best_v2, [best_tr1, best_tr2, best_time]])
            try:
                result = minimize(objective2, x0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 0.0005, 'fatol': 0.0005, 'adaptive': True})
                if result.fun < best_time:
                    v1_opt = np.clip(result.x[:7], joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                    v2_opt = np.clip(result.x[7:14], joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                    tr1_opt = result.x[14]
                    tr2_opt = result.x[15]
                    T_opt = result.x[16]
                    if _try_via_path([v1_opt, v2_opt], T_opt, [0.0, tr1_opt, tr2_opt, 1.0], joint_limits, robot_id, obs_id, joint_idxs):
                        best_time = T_opt
                        best_waypoints = [Q_START.tolist(), v1_opt.tolist(), v2_opt.tolist(), Q_GOAL.tolist()]
                        best_timestamps = [0.0, best_time * tr1_opt, best_time * tr2_opt, best_time]
            except Exception:
                pass

        if best_waypoints is None:
            best_time = 12.0
            via = np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5])
            best_waypoints = [Q_START.tolist(), via.tolist(), Q_GOAL.tolist()]
            best_timestamps = [0.0, best_time * 0.48, best_time]

        return best_waypoints, best_timestamps
    finally:
        p.disconnect(physics)


def main():
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