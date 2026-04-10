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

        def get_min_time_generic(waypoints_inner: np.ndarray, timestamps_inner: list[float], best_t: float) -> float:
            waypoints = np.vstack([Q_START, waypoints_inner, Q_GOAL])
            timestamps = np.array([0.0] + timestamps_inner + [1.0], dtype=float)

            cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
            t_samp = np.concatenate([
                np.linspace(timestamps[i], timestamps[i+1], SAMPLES_PER_SEG, endpoint=False)
                for i in range(len(timestamps) - 1)
            ])
            
            q_batch = cs(t_samp)
            if np.any(q_batch < joint_limits[:, 0] - 1e-4) or np.any(q_batch > joint_limits[:, 1] + 1e-4):
                return float('inf')

            v_batch = cs.derivative(1)(t_samp)
            a_batch = cs.derivative(2)(t_samp)
            
            T_v = np.max(np.abs(v_batch) / MAX_VEL)
            T_a = np.sqrt(np.max(np.abs(a_batch) / MAX_ACC))
            t_cand = float(max(T_v, T_a))

            if t_cand >= best_t:
                return float('inf')

            for k in range(len(t_samp)):
                _set_q(robot_id, q_batch[k], joint_idxs)
                if p.getClosestPoints(robot_id, obs_id, distance=0.0):
                    return float('inf')
            
            return t_cand

        def eval_1(ind: np.ndarray, best_t: float) -> float:
            ratio = 0.05 + 0.90 * ind[7]
            return get_min_time_generic(ind[:7], [ratio], best_t)

        def eval_2(ind: np.ndarray, best_t: float) -> float:
            r1 = 0.05 + 0.85 * ind[14]
            r2 = r1 + 0.05 + (0.90 - r1) * ind[15]
            return get_min_time_generic(np.vstack([ind[:7], ind[7:14]]), [r1, r2], best_t)

        def run_de(dim: int, pop_size: int, max_gen: int, init_pop: np.ndarray, eval_func, bounds: np.ndarray) -> tuple[float, np.ndarray]:
            pop = init_pop.copy()
            pop_time = np.array([eval_func(ind, float('inf')) for ind in pop])
            
            best_idx = int(np.argmin(pop_time))
            best_t = pop_time[best_idx]
            best_ind = pop[best_idx].copy()
            
            for gen in range(max_gen):
                for i in range(pop_size):
                    cands = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = np.random.choice(cands, 3, replace=False)
                    
                    F = np.random.uniform(0.5, 0.9)
                    CR = np.random.uniform(0.7, 0.9)
                    
                    if np.random.rand() < 0.5:
                        mut = pop[best_idx] + F * (pop[a] - pop[b])
                    else:
                        mut = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                        
                    cross_points = np.random.rand(dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(dim)] = True
                        
                    trial = np.where(cross_points, mut, pop[i])
                    trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                    
                    t_cand = eval_func(trial, pop_time[i])
                    if t_cand < pop_time[i]:
                        pop[i] = trial
                        pop_time[i] = t_cand
                        if t_cand < best_t:
                            best_t = t_cand
                            best_ind = trial.copy()
                            best_idx = i
                            
            return best_t, best_ind

        np.random.seed(42)
        
        # Phase 1: Optimize 1 via point
        bounds_1 = np.zeros((8, 2))
        bounds_1[:7, 0] = joint_limits[:, 0]
        bounds_1[:7, 1] = joint_limits[:, 1]
        bounds_1[7, :] = [0.0, 1.0]
        
        pop_size_1 = 60
        max_gen_1 = 500
        pop_1 = np.zeros((pop_size_1, 8))
        pop_1[0, :7] = Q_VIA
        pop_1[0, 7] = (0.48 - 0.05) / 0.90
        
        for i in range(1, pop_size_1):
            alpha = np.random.uniform(0.1, 0.9)
            pop_1[i, :7] = np.clip(Q_START + alpha * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_1[:7, 0], bounds_1[:7, 1])
            pop_1[i, 7] = np.random.uniform(0.0, 1.0)
            
        best_t_1, best_ind_1 = run_de(8, pop_size_1, max_gen_1, pop_1, eval_1, bounds_1)

        # Phase 2: Optimize 2 via points
        bounds_2 = np.zeros((16, 2))
        bounds_2[:7, 0] = joint_limits[:, 0]
        bounds_2[:7, 1] = joint_limits[:, 1]
        bounds_2[7:14, 0] = joint_limits[:, 0]
        bounds_2[7:14, 1] = joint_limits[:, 1]
        bounds_2[14:16, :] = [0.0, 1.0]
        
        pop_size_2 = 120
        max_gen_2 = 1000
        pop_2 = np.zeros((pop_size_2, 16))
        
        # Convert best 1-via to 2-via
        ratio_1 = 0.05 + 0.90 * best_ind_1[7]
        cs_init = CubicSpline([0.0, ratio_1, 1.0], np.vstack([Q_START, best_ind_1[:7], Q_GOAL]), bc_type="clamped")
        t1 = np.clip(ratio_1 * 0.5, 0.05, 0.90)
        t2 = np.clip(ratio_1 + (1.0 - ratio_1) * 0.5, t1 + 0.05, 0.95)
        
        pop_2[0, :7] = np.clip(cs_init(t1), bounds_2[:7, 0], bounds_2[:7, 1])
        pop_2[0, 7:14] = np.clip(cs_init(t2), bounds_2[7:14, 0], bounds_2[7:14, 1])
        pop_2[0, 14] = (t1 - 0.05) / 0.85
        pop_2[0, 15] = (t2 - t1 - 0.05) / (0.90 - t1) if t1 < 0.90 else 0.0
        
        for i in range(1, pop_size_2):
            if np.random.rand() < 0.2:
                alpha1 = np.random.uniform(0.1, 0.5)
                alpha2 = np.random.uniform(0.5, 0.9)
                pop_2[i, :7] = np.clip(Q_START + alpha1 * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_2[:7, 0], bounds_2[:7, 1])
                pop_2[i, 7:14] = np.clip(Q_START + alpha2 * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_2[7:14, 0], bounds_2[7:14, 1])
                pop_2[i, 14] = np.random.uniform(0, 1)
                pop_2[i, 15] = np.random.uniform(0, 1)
            else:
                pop_2[i, :7] = np.clip(pop_2[0, :7] + np.random.normal(0, 0.2, 7), bounds_2[:7, 0], bounds_2[:7, 1])
                pop_2[i, 7:14] = np.clip(pop_2[0, 7:14] + np.random.normal(0, 0.2, 7), bounds_2[7:14, 0], bounds_2[7:14, 1])
                pop_2[i, 14] = np.clip(pop_2[0, 14] + np.random.normal(0, 0.1), 0.0, 1.0)
                pop_2[i, 15] = np.clip(pop_2[0, 15] + np.random.normal(0, 0.1), 0.0, 1.0)
                
        best_t_2, best_ind_2 = run_de(16, pop_size_2, max_gen_2, pop_2, eval_2, bounds_2)

        # Polish Phase 2
        for k in range(500):
            scale = 0.01 if k % 2 == 0 else 0.002
            trial = np.clip(best_ind_2 + np.random.normal(0, scale, 16), bounds_2[:, 0], bounds_2[:, 1])
            t_cand = eval_2(trial, best_t_2)
            if t_cand < best_t_2:
                best_t_2 = t_cand
                best_ind_2 = trial.copy()

        # Phase 3: Optimize 3 via points
        bounds_3 = np.zeros((24, 2))
        for i in range(3):
            bounds_3[i*7:(i+1)*7, 0] = joint_limits[:, 0]
            bounds_3[i*7:(i+1)*7, 1] = joint_limits[:, 1]
        bounds_3[21:24, :] = [0.0, 1.0]
        
        pop_size_3 = 120
        max_gen_3 = 1000
        pop_3 = np.zeros((pop_size_3, 24))
        
        r1_2 = 0.05 + 0.85 * best_ind_2[14]
        r2_2 = r1_2 + 0.05 + (0.90 - r1_2) * best_ind_2[15]
        cs_init_2 = CubicSpline([0.0, r1_2, r2_2, 1.0], np.vstack([Q_START, best_ind_2[:7], best_ind_2[7:14], Q_GOAL]), bc_type="clamped")
        
        t1 = np.clip(r1_2 * 0.6, 0.05, 0.80)
        t2 = np.clip(r1_2 + (r2_2 - r1_2) * 0.4, t1 + 0.05, 0.85)
        t3 = np.clip(r2_2 + (1.0 - r2_2) * 0.4, t2 + 0.05, 0.90)
        
        pop_3[0, :7] = np.clip(cs_init_2(t1), bounds_3[:7, 0], bounds_3[:7, 1])
        pop_3[0, 7:14] = np.clip(cs_init_2(t2), bounds_3[7:14, 0], bounds_3[7:14, 1])
        pop_3[0, 14:21] = np.clip(cs_init_2(t3), bounds_3[14:21, 0], bounds_3[14:21, 1])
        pop_3[0, 21] = (t1 - 0.05) / 0.80
        pop_3[0, 22] = (t2 - t1 - 0.05) / (0.85 - t1) if t1 < 0.85 else 0.0
        pop_3[0, 23] = (t3 - t2 - 0.05) / (0.90 - t2) if t2 < 0.90 else 0.0
        
        for i in range(1, pop_size_3):
            if np.random.rand() < 0.2:
                alpha1 = np.random.uniform(0.1, 0.4)
                alpha2 = np.random.uniform(0.4, 0.7)
                alpha3 = np.random.uniform(0.7, 0.9)
                pop_3[i, :7] = np.clip(Q_START + alpha1 * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_3[:7, 0], bounds_3[:7, 1])
                pop_3[i, 7:14] = np.clip(Q_START + alpha2 * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_3[7:14, 0], bounds_3[7:14, 1])
                pop_3[i, 14:21] = np.clip(Q_START + alpha3 * (Q_GOAL - Q_START) + np.random.normal(0, 0.6, 7), bounds_3[14:21, 0], bounds_3[14:21, 1])
                pop_3[i, 21] = np.random.uniform(0, 1)
                pop_3[i, 22] = np.random.uniform(0, 1)
                pop_3[i, 23] = np.random.uniform(0, 1)
            else:
                pop_3[i, :7] = np.clip(pop_3[0, :7] + np.random.normal(0, 0.2, 7), bounds_3[:7, 0], bounds_3[:7, 1])
                pop_3[i, 7:14] = np.clip(pop_3[0, 7:14] + np.random.normal(0, 0.2, 7), bounds_3[7:14, 0], bounds_3[7:14, 1])
                pop_3[i, 14:21] = np.clip(pop_3[0, 14:21] + np.random.normal(0, 0.2, 7), bounds_3[14:21, 0], bounds_3[14:21, 1])
                pop_3[i, 21] = np.clip(pop_3[0, 21] + np.random.normal(0, 0.1), 0.0, 1.0)
                pop_3[i, 22] = np.clip(pop_3[0, 22] + np.random.normal(0, 0.1), 0.0, 1.0)
                pop_3[i, 23] = np.clip(pop_3[0, 23] + np.random.normal(0, 0.1), 0.0, 1.0)

        def eval_3(ind: np.ndarray, best_t: float) -> float:
            r1 = 0.05 + 0.80 * ind[21]
            r2 = r1 + 0.05 + (0.85 - r1) * ind[22]
            r3 = r2 + 0.05 + (0.90 - r2) * ind[23]
            return get_min_time_generic(np.vstack([ind[:7], ind[7:14], ind[14:21]]), [r1, r2, r3], best_t)

        best_t_3, best_ind_3 = run_de(24, pop_size_3, max_gen_3, pop_3, eval_3, bounds_3)

        # Polish Phase 3
        for k in range(1200):
            scale = 0.01 if k % 2 == 0 else 0.002
            trial = np.clip(best_ind_3 + np.random.normal(0, scale, 24), bounds_3[:, 0], bounds_3[:, 1])
            t_cand = eval_3(trial, best_t_3)
            if t_cand < best_t_3:
                best_t_3 = t_cand
                best_ind_3 = trial.copy()

        if best_t_3 < best_t_2 and best_t_3 != float('inf'):
            r1 = 0.05 + 0.80 * best_ind_3[21]
            r2 = r1 + 0.05 + (0.85 - r1) * best_ind_3[22]
            r3 = r2 + 0.05 + (0.90 - r2) * best_ind_3[23]
            waypoints = [Q_START.tolist(), best_ind_3[:7].tolist(), best_ind_3[7:14].tolist(), best_ind_3[14:21].tolist(), Q_GOAL.tolist()]
            timestamps = [0.0, float((best_t_3 + 1e-5) * r1), float((best_t_3 + 1e-5) * r2), float((best_t_3 + 1e-5) * r3), float(best_t_3 + 1e-5)]
            return waypoints, timestamps
        elif best_t_2 != float('inf'):
            r1 = 0.05 + 0.85 * best_ind_2[14]
            r2 = r1 + 0.05 + (0.90 - r1) * best_ind_2[15]
            waypoints = [Q_START.tolist(), best_ind_2[:7].tolist(), best_ind_2[7:14].tolist(), Q_GOAL.tolist()]
            timestamps = [0.0, float((best_t_2 + 1e-5) * r1), float((best_t_2 + 1e-5) * r2), float(best_t_2 + 1e-5)]
            return waypoints, timestamps
        else:
            # Fallback to Phase 1
            ratio = 0.05 + 0.90 * best_ind_1[7]
            waypoints = [Q_START.tolist(), best_ind_1[:7].tolist(), Q_GOAL.tolist()]
            timestamps = [0.0, float((best_t_1 + 1e-5) * ratio), float(best_t_1 + 1e-5)]
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
