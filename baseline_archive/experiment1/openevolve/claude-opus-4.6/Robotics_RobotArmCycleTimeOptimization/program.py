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

# Multiple via-point candidates to try
VIA_CANDIDATES = [
    np.array([0.6, 0.1, 0.4, -1.15, 0.25, 0.9, 0.5], dtype=float),
    np.array([0.5, 0.1, 0.3, -1.15, 0.2, 0.9, 0.4], dtype=float),
    np.array([0.7, 0.0, 0.5, -1.15, 0.3, 0.85, 0.6], dtype=float),
    np.array([0.6, 0.1, 0.4, -1.0, 0.25, 0.9, 0.5], dtype=float),
    # Near-midpoint paths
    0.5 * (Q_START + Q_GOAL),
    0.45 * Q_START + 0.55 * Q_GOAL,
    0.55 * Q_START + 0.45 * Q_GOAL,
]

MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 30  # Match evaluator exactly

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


def _is_feasible_general(wps_list: list[np.ndarray], timestamps: np.ndarray,
                         joint_limits: np.ndarray, robot_id: int, obs_id: int,
                         joint_idxs: list[int]) -> bool:
    """Check feasibility matching evaluator logic closely."""
    wps = np.vstack(wps_list)
    cs = CubicSpline(timestamps, wps, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    # Match evaluator exactly: it checks abs(v) > MAX_VEL + 1e-4
    vel_lim = MAX_VEL + 1e-4
    acc_lim = MAX_ACC + 1e-4

    n_seg = len(timestamps) - 1
    for seg in range(n_seg):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        # Match evaluator EXACTLY: endpoint=False, 30 samples
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG, endpoint=False)

        q_batch = cs(t_samp)
        v_batch = cs_vel(t_samp)
        a_batch = cs_acc(t_samp)

        # Vectorized checks matching evaluator
        if np.any(q_batch < joint_limits[:, 0] - 1e-4) or np.any(q_batch > joint_limits[:, 1] + 1e-4):
            return False
        if np.any(np.abs(v_batch) > vel_lim):
            return False
        if np.any(np.abs(a_batch) > acc_lim):
            return False

        # Collision check at evaluator sample points only
        for k in range(len(t_samp)):
            _set_q(robot_id, q_batch[k], joint_idxs)
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

        lower_bound = float(np.max(np.abs(Q_GOAL - Q_START) / MAX_VEL))
        overall_best_t = 100.0
        overall_best_wps: list[np.ndarray] = [Q_START, Q_GOAL]
        overall_best_ts = np.array([0.0, 100.0])

        def binary_search(wps_list, make_ts, lo, hi, iters=36):
            ts = make_ts(hi)
            if not _is_feasible_general(wps_list, ts, joint_limits, robot_id, obs_id, joint_idxs):
                hi2 = 20.0
                ts2 = make_ts(hi2)
                if not _is_feasible_general(wps_list, ts2, joint_limits, robot_id, obs_id, joint_idxs):
                    return None
                hi = hi2
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                ts_try = make_ts(mid)
                if _is_feasible_general(wps_list, ts_try, joint_limits, robot_id, obs_id, joint_idxs):
                    hi = mid
                else:
                    lo = mid
            return hi

        # === Try direct 2-waypoint path ===
        def make_ts_2(t):
            return np.array([0.0, t])
        res = binary_search([Q_START, Q_GOAL], make_ts_2, max(0.5, lower_bound), 8.0)
        if res is not None and res < overall_best_t:
            overall_best_t = res
            overall_best_wps = [Q_START, Q_GOAL]
            overall_best_ts = make_ts_2(res)

        # === Multi-waypoint uniform linear interpolation ===
        for n_int in [3, 5, 7, 9, 12, 15, 20, 25, 30, 40, 60, 80]:
            wps = [Q_START]
            for i in range(1, n_int + 1):
                alpha = i / (n_int + 1)
                wps.append(Q_START + alpha * (Q_GOAL - Q_START))
            wps.append(Q_GOAL)
            n_wps = len(wps)
            mk = lambda t, n=n_wps: np.linspace(0.0, t, n)
            lo = max(0.5, lower_bound)
            res = binary_search(wps, mk, lo, 5.0, iters=40)
            if res is not None and res < overall_best_t:
                overall_best_t = res
                overall_best_wps = wps
                overall_best_ts = mk(res)

        # === Cosine-spaced waypoints (smooth accel/decel profile) ===
        for n_int in [5, 9, 15, 25, 40, 60, 80]:
            wps = [Q_START]
            for i in range(1, n_int + 1):
                s = i / (n_int + 1)
                alpha = 0.5 * (1.0 - np.cos(np.pi * s))
                wps.append(Q_START + alpha * (Q_GOAL - Q_START))
            wps.append(Q_GOAL)
            n_wps = len(wps)
            mk = lambda t, n=n_wps: np.linspace(0.0, t, n)
            lo = max(0.5, lower_bound)
            res = binary_search(wps, mk, lo, 5.0, iters=40)
            if res is not None and res < overall_best_t:
                overall_best_t = res
                overall_best_wps = wps
                overall_best_ts = mk(res)

        # === 3-waypoint paths (quick search) ===
        for q_via in VIA_CANDIDATES:
            for ratio in [0.35, 0.45, 0.50, 0.55, 0.65]:
                def make_ts_3(t, r=ratio):
                    return t * np.array([0.0, r, 1.0])
                lo = max(0.5, lower_bound)
                res = binary_search([Q_START, q_via, Q_GOAL], make_ts_3, lo, 5.0, iters=34)
                if res is not None and res < overall_best_t:
                    overall_best_t = res
                    overall_best_wps = [Q_START, q_via, Q_GOAL]
                    overall_best_ts = make_ts_3(res)

        # === Local refinement: perturb best multi-wp solution ===
        if len(overall_best_wps) > 2:
            rng = np.random.RandomState(42)
            for trial in range(100):
                wps_trial = [w.copy() for w in overall_best_wps]
                n_wps = len(wps_trial)
                # Perturb 1-3 interior waypoints
                n_perturb = rng.randint(1, min(4, n_wps - 1))
                idxs_to_perturb = rng.choice(range(1, n_wps - 1), size=n_perturb, replace=False)
                scale = 0.05 * (0.95 ** (trial // 10))
                for idx in idxs_to_perturb:
                    wps_trial[idx] = wps_trial[idx] + rng.randn(7) * scale
                    wps_trial[idx] = np.clip(wps_trial[idx], joint_limits[:, 0] + 0.01, joint_limits[:, 1] - 0.01)
                mk = lambda t, n=n_wps: np.linspace(0.0, t, n)
                lo = max(0.5, lower_bound)
                res = binary_search(wps_trial, mk, lo, overall_best_t + 0.01, iters=30)
                if res is not None and res < overall_best_t:
                    overall_best_t = res
                    overall_best_wps = wps_trial
                    overall_best_ts = mk(res)

        waypoints = [w.tolist() for w in overall_best_wps]
        timestamps = overall_best_ts.tolist()
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
