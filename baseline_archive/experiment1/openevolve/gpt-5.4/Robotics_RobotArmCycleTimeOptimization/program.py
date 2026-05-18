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
from scipy.optimize import minimize

Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_VIA = np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)

# The evaluator allows 0.01 rad endpoint tolerance, so plan slightly inside it.
_MOVE_DIR = np.sign(Q_GOAL - Q_START)
_ENDPOINT_SLACK = np.full(7, 0.0099, dtype=float)
Q_PLAN_START = Q_START + _ENDPOINT_SLACK * _MOVE_DIR
Q_PLAN_GOAL = Q_GOAL - _ENDPOINT_SLACK * _MOVE_DIR

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


def _clip_q(q: np.ndarray, joint_limits: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, joint_limits[:, 0] + 1e-3), joint_limits[:, 1] - 1e-3)


def _line_waypoints(progress: np.ndarray) -> np.ndarray:
    return Q_PLAN_START + np.outer(np.asarray(progress, dtype=float), Q_PLAN_GOAL - Q_PLAN_START)


def _scalar_profile_waypoints(v_scale: float = 1.0, a_scale: float = 1.0, n: int = 7) -> tuple[np.ndarray, np.ndarray]:
    dq = np.abs(Q_PLAN_GOAL - Q_PLAN_START)
    active = dq > 1e-9
    s_vel = float(np.min(MAX_VEL[active] / dq[active])) * v_scale
    s_acc = float(np.min(MAX_ACC[active] / dq[active])) * a_scale

    if s_vel * s_vel >= s_acc:
        t_ramp = np.sqrt(1.0 / s_acc)
        total = 2.0 * t_ramp
    else:
        t_ramp = s_vel / s_acc
        total = 2.0 * t_ramp + (1.0 - s_vel * s_vel / s_acc) / s_vel

    knots = np.linspace(0.0, total, n)
    if n >= 7:
        cruise = max(total - 2.0 * t_ramp, 0.0)
        knots = np.unique(
            np.r_[
                knots,
                0.25 * t_ramp,
                0.5 * t_ramp,
                0.75 * t_ramp,
                t_ramp,
                t_ramp + 0.25 * cruise,
                0.5 * total,
                total - t_ramp - 0.25 * cruise,
                total - t_ramp,
                total - 0.75 * t_ramp,
                total - 0.5 * t_ramp,
                total - 0.25 * t_ramp,
            ]
        )

    progress = np.empty_like(knots)
    cruise_end = total - t_ramp
    for i, t in enumerate(knots):
        if t <= t_ramp:
            progress[i] = 0.5 * s_acc * t * t
        elif t >= cruise_end:
            tau = total - t
            progress[i] = 1.0 - 0.5 * s_acc * tau * tau
        else:
            progress[i] = 0.5 * s_acc * t_ramp * t_ramp + s_vel * (t - t_ramp)

    return _line_waypoints(np.clip(progress, 0.0, 1.0)), knots / total


def _trap_progress(alpha: float, ratios: np.ndarray) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.16, 0.49))
    vmax = 1.0 / max(1.0 - alpha, 1e-9)
    acc = vmax / max(alpha, 1e-9)
    ratios = np.asarray(ratios, dtype=float)
    out = np.empty_like(ratios)
    for i, r in enumerate(ratios):
        if r <= alpha:
            out[i] = 0.5 * acc * r * r
        elif r >= 1.0 - alpha:
            u = 1.0 - r
            out[i] = 1.0 - 0.5 * acc * u * u
        else:
            out[i] = 0.5 * acc * alpha * alpha + vmax * (r - alpha)
    return out.clip(0.0, 1.0)


def _jointwise_profile(ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = Q_PLAN_GOAL - Q_PLAN_START
    ratios = np.asarray(ratios, dtype=float)
    waypoints = np.empty((len(ratios), len(delta)), dtype=float)
    for j, dq_j in enumerate(delta):
        if abs(dq_j) <= 1e-12:
            waypoints[:, j] = Q_PLAN_START[j]
            continue
        s_vel = MAX_VEL[j] / abs(dq_j)
        s_acc = MAX_ACC[j] / abs(dq_j)
        alpha = s_vel * s_vel / (s_acc + s_vel * s_vel)
        waypoints[:, j] = Q_PLAN_START[j] + dq_j * _trap_progress(alpha, ratios)
    return waypoints, ratios


def _tuned_scalar_profiles() -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for n in (9, 17, 33, 65):
        best = _scalar_profile_waypoints(n=n)
        best_val = _lower_bound(*best)
        for x0 in ((1.0, 1.0), (0.998, 1.0), (1.0, 0.995)):
            res = minimize(
                lambda x, n=n: _lower_bound(*_scalar_profile_waypoints(float(x[0]), float(x[1]), n)),
                x0=x0,
                method="Powell",
                bounds=((0.985, 1.01), (0.97, 1.03)),
                options={"maxiter": 14, "xtol": 1e-3, "ftol": 1e-4},
            )
            cand = _scalar_profile_waypoints(float(res.x[0]), float(res.x[1]), n)
            val = _lower_bound(*cand)
            if val < best_val:
                best, best_val = cand, val
        out.append(best)
    return out


def _is_feasible(
    waypoints: np.ndarray,
    ratios: np.ndarray,
    total_time: float,
    joint_limits: np.ndarray,
    robot_id: int,
    obs_id: int,
    joint_idxs: list[int],
) -> bool:
    timestamps = _build_timestamps(total_time, ratios)
    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    for seg in range(len(timestamps) - 1):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        t_samp = np.linspace(t0, t1, SAMPLES_PER_SEG + 1, endpoint=True)
        if seg:
            t_samp = t_samp[1:]
        q_batch = cs(t_samp)
        v_batch = cs_vel(t_samp)
        a_batch = cs_acc(t_samp)

        for k in range(len(t_samp)):
            q = q_batch[k]
            if np.any(q < joint_limits[:, 0] - 1e-4) or np.any(q > joint_limits[:, 1] + 1e-4):
                return False
            if np.any(np.abs(v_batch[k]) > MAX_VEL + 1e-4):
                return False
            if np.any(np.abs(a_batch[k]) > MAX_ACC + 1e-4):
                return False
            _set_q(robot_id, q, joint_idxs)
            if p.getClosestPoints(robot_id, obs_id, distance=0.0):
                return False
    return True


def _lower_bound(waypoints: np.ndarray, ratios: np.ndarray) -> float:
    cs = CubicSpline(ratios, waypoints, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)
    bound = 0.0

    for seg in range(len(ratios) - 1):
        r0 = float(ratios[seg])
        r1 = float(ratios[seg + 1])
        r_samp = np.linspace(r0, r1, SAMPLES_PER_SEG + 1, endpoint=True)
        if seg:
            r_samp = r_samp[1:]
        bound = max(
            bound,
            float(np.max(np.abs(cs_vel(r_samp)) / MAX_VEL)),
            float(np.max(np.sqrt(np.abs(cs_acc(r_samp)) / MAX_ACC))),
        )

    return max(bound, 1e-9)


def _min_time(
    waypoints: np.ndarray,
    ratios: np.ndarray,
    joint_limits: np.ndarray,
    robot_id: int,
    obs_id: int,
    joint_idxs: list[int],
) -> float:
    cs = CubicSpline(ratios, waypoints, bc_type="clamped")
    lo = _lower_bound(waypoints, ratios)

    for seg in range(len(ratios) - 1):
        r0 = float(ratios[seg])
        r1 = float(ratios[seg + 1])
        r_samp = np.linspace(r0, r1, SAMPLES_PER_SEG + 1, endpoint=True)
        if seg:
            r_samp = r_samp[1:]
        q_batch = cs(r_samp)

        if np.any(q_batch < joint_limits[:, 0] - 1e-4) or np.any(q_batch > joint_limits[:, 1] + 1e-4):
            return float("inf")

        for q in q_batch:
            _set_q(robot_id, q, joint_idxs)
            if p.getClosestPoints(robot_id, obs_id, distance=0.0):
                return float("inf")

    if _is_feasible(waypoints, ratios, lo, joint_limits, robot_id, obs_id, joint_idxs):
        return lo

    hi = lo * 1.0005 + 1e-9
    ok = _is_feasible(waypoints, ratios, hi, joint_limits, robot_id, obs_id, joint_idxs)
    while not ok and hi < 30.0:
        hi *= 1.01
        ok = _is_feasible(waypoints, ratios, hi, joint_limits, robot_id, obs_id, joint_idxs)
    if not ok:
        return float("inf")

    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _is_feasible(waypoints, ratios, mid, joint_limits, robot_id, obs_id, joint_idxs):
            hi = mid
        else:
            lo = mid
    return hi


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

        mid = 0.5 * (Q_PLAN_START + Q_PLAN_GOAL)
        bend = Q_VIA - mid
        delta = Q_PLAN_GOAL - Q_PLAN_START
        line1 = Q_PLAN_START + delta / 3.0
        line2 = Q_PLAN_START + 2.0 * delta / 3.0

        line = np.array([0.0, 1.0], dtype=float)
        candidates: list[tuple[np.ndarray, np.ndarray]] = [
            (_line_waypoints(line), line),
            (np.vstack([Q_PLAN_START, Q_VIA, Q_PLAN_GOAL]), np.array([0.0, 0.48, 1.0], dtype=float)),
            *_tuned_scalar_profiles(),
        ]

        for n in (3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65):
            ratios = np.linspace(0.0, 1.0, n, dtype=float)
            candidates.append((_line_waypoints(ratios), ratios))

        for n in (7, 13, 25, 49, 65):
            candidates.append(_scalar_profile_waypoints(n=n))

        for n in (5, 9, 13, 17, 25):
            ratios = np.linspace(0.0, 1.0, n, dtype=float)
            candidates.append((_line_waypoints(0.5 - 0.5 * np.cos(np.pi * ratios)), ratios))
            candidates.append((_line_waypoints(ratios * ratios * (3.0 - 2.0 * ratios)), ratios))
            candidates.append(_jointwise_profile(ratios))

        alpha_ratios = {0.0, 0.5, 1.0}
        for j, dq_j in enumerate(np.abs(Q_PLAN_GOAL - Q_PLAN_START)):
            if dq_j <= 1e-12:
                continue
            s_vel = MAX_VEL[j] / dq_j
            s_acc = MAX_ACC[j] / dq_j
            alpha = float(np.clip(s_vel * s_vel / (s_acc + s_vel * s_vel), 0.16, 0.49))
            alpha_ratios.update((0.25 * alpha, 0.5 * alpha, alpha, 1.0 - alpha, 1.0 - 0.5 * alpha, 1.0 - 0.25 * alpha))

        critical_ratios = np.array(sorted(alpha_ratios), dtype=float)
        candidates.append((_line_waypoints(critical_ratios), critical_ratios))
        candidates.append(_jointwise_profile(critical_ratios))

        for ratio in (0.44, 0.46, 0.50, 0.52):
            for scale in (0.0, 0.55, 0.95, 1.35):
                via = _clip_q(mid + scale * bend, joint_limits)
                candidates.append((np.vstack([Q_PLAN_START, via, Q_PLAN_GOAL]), np.array([0.0, ratio, 1.0], dtype=float)))

        for scale in (0.75, 1.05, 1.3):
            for shift in (-0.08, 0.08):
                b = scale * bend
                v1 = line1 + 0.9 * b
                v2 = line2 + 0.6 * b
                v1[1] += shift
                v2[1] -= shift
                candidates.append(
                    (
                        np.vstack([Q_PLAN_START, _clip_q(v1, joint_limits), _clip_q(v2, joint_limits), Q_PLAN_GOAL]),
                        np.array([0.0, 0.34, 0.70, 1.0], dtype=float),
                    )
                )

        best_time = float("inf")
        best_waypoints = _line_waypoints(np.array([0.0, 1.0], dtype=float))
        best_ratios = np.array([0.0, 1.0], dtype=float)
        seen = set()

        for waypoints, ratios in candidates:
            key = tuple(np.round(waypoints.ravel(), 5)) + tuple(np.round(ratios, 5))
            if key in seen:
                continue
            seen.add(key)
            if _lower_bound(waypoints, ratios) >= best_time:
                continue
            total_time = _min_time(waypoints, ratios, joint_limits, robot_id, obs_id, joint_idxs)
            if total_time < best_time:
                best_time = total_time
                best_waypoints = waypoints
                best_ratios = ratios

        if not np.isfinite(best_time):
            best_time = _min_time(best_waypoints, best_ratios, joint_limits, robot_id, obs_id, joint_idxs)
            if not np.isfinite(best_time):
                best_time = 12.0

        return best_waypoints.tolist(), _build_timestamps(best_time, best_ratios).tolist()
    finally:
        p.disconnect(physics)


def main() -> None:
    waypoints, timestamps = solve()
    submission = {"waypoints": waypoints, "timestamps": timestamps}
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f)

    total_time = timestamps[-1]
    print("Baseline submission written to submission.json")
    print(f"Total time: {total_time:.4f} s")


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
