# EVOLVE-BLOCK-START
"""Optimized solver for RobotArmCycleTimeOptimization.

Key insights:
1. Use time-parameterized waypoints from synchronized per-joint trapezoidal profiles
2. The cubic spline with clamped BCs needs enough waypoints to avoid oscillation
3. Joint optimization of waypoint positions for small configurations
4. Iterative refinement: find good config, then optimize positions and ratios
5. Try 2-waypoint (start+goal only) as simplest possible trajectory
6. Focus computation on most promising configurations
"""

from __future__ import annotations

import json
import time as time_mod

import numpy as np
import pybullet as p
import pybullet_data
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution

Q_START = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0], dtype=float)
Q_GOAL = np.array([1.2, -0.3, 0.8, -0.8, 0.5, 0.8, 1.0], dtype=float)

MAX_VEL = np.array([1.48, 1.48, 1.74, 1.74, 2.27, 2.27, 2.27], dtype=float)
MAX_ACC = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
SAMPLES_PER_SEG = 30

OBS_CENTER = np.array([0.45, -0.35, 0.65])
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


def _build_timestamps(total_time: float, n_wps: int, time_ratios: np.ndarray | None = None) -> np.ndarray:
    if time_ratios is not None:
        all_ratios = np.concatenate([[0.0], np.sort(time_ratios), [1.0]])
    else:
        all_ratios = np.linspace(0.0, 1.0, n_wps)
    return total_time * all_ratios


def _is_feasible(total_time: float, wps: np.ndarray, timestamps: np.ndarray,
                 joint_limits: np.ndarray, robot_id: int, obs_id: int,
                 joint_idxs: list[int], samples: int = SAMPLES_PER_SEG,
                 check_collision: bool = True) -> bool:
    n_wps = len(wps)
    cs = CubicSpline(timestamps, wps, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    all_t = []
    for seg in range(n_wps - 1):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        all_t.append(np.linspace(t0, t1, samples, endpoint=False))
    all_t = np.concatenate(all_t)

    q_all = cs(all_t)
    v_all = cs_vel(all_t)
    a_all = cs_acc(all_t)

    if np.any(q_all < joint_limits[:, 0] - 1e-4) or np.any(q_all > joint_limits[:, 1] + 1e-4):
        return False
    if np.any(np.abs(v_all) > MAX_VEL[None, :] + 1e-4):
        return False
    if np.any(np.abs(a_all) > MAX_ACC[None, :] + 1e-4):
        return False

    if check_collision:
        for k in range(len(all_t)):
            _set_q(robot_id, q_all[k], joint_idxs)
            if len(p.getClosestPoints(robot_id, obs_id, distance=0.0)) > 0:
                return False

    return True


def _max_violation(total_time: float, wps: np.ndarray, timestamps: np.ndarray,
                   joint_limits: np.ndarray, samples: int = 20) -> float:
    """Return max constraint violation (positive = infeasible)."""
    n_wps = len(wps)
    cs = CubicSpline(timestamps, wps, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    all_t = []
    for seg in range(n_wps - 1):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        all_t.append(np.linspace(t0, t1, samples, endpoint=False))
    all_t = np.concatenate(all_t)

    v_all = cs_vel(all_t)
    a_all = cs_acc(all_t)
    q_all = cs(all_t)

    vel_viol = np.max(np.abs(v_all) / MAX_VEL[None, :]) - 1.0
    acc_viol = np.max(np.abs(a_all) / MAX_ACC[None, :]) - 1.0
    jl_lo_viol = np.max(joint_limits[:, 0] - q_all)
    jl_hi_viol = np.max(q_all - joint_limits[:, 1])

    return max(vel_viol, acc_viol, jl_lo_viol, jl_hi_viol)


def _fast_min_time(wps, time_ratios, joint_limits, lo, hi, iterations=30, samples=20):
    """Fast binary search for min time without collision."""
    n_wps = len(wps)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        ts = _build_timestamps(mid, n_wps, time_ratios)
        viol = _max_violation(mid, wps, ts, joint_limits, samples=samples)
        if viol <= 0:
            hi = mid
        else:
            lo = mid
    return hi


def _binary_search_time(wps, time_ratios, joint_limits, robot_id, obs_id, joint_idxs,
                        lo, hi, iterations=24, samples=SAMPLES_PER_SEG, check_collision=True):
    n_wps = len(wps)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        ts = _build_timestamps(mid, n_wps, time_ratios)
        if _is_feasible(mid, wps, ts, joint_limits, robot_id, obs_id, joint_idxs,
                        samples=samples, check_collision=check_collision):
            hi = mid
        else:
            lo = mid
    return hi


def _compute_trapezoidal_time_ratios(n_total):
    """Compute time ratios from bottleneck joint's trapezoidal profile."""
    dq = np.abs(Q_GOAL - Q_START)

    per_joint_times = []
    for j in range(7):
        v_max = MAX_VEL[j]; a_max = MAX_ACC[j]; d = dq[j]
        if d < 1e-10:
            per_joint_times.append(0.0)
            continue
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel**2
        if 2 * d_accel <= d:
            per_joint_times.append(d / v_max + v_max / a_max)
        else:
            per_joint_times.append(2.0 * np.sqrt(d / a_max))

    T_min = max(per_joint_times)
    bottleneck = int(np.argmax(per_joint_times))

    d = dq[bottleneck]
    v_max = MAX_VEL[bottleneck]
    a_max = MAX_ACC[bottleneck]
    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * t_accel**2

    pos_alphas = np.linspace(0, 1, n_total)

    if 2 * d_accel <= d:
        T = d / v_max + v_max / a_max
        t_a = t_accel

        time_ratios = np.zeros(n_total)
        for i, alpha in enumerate(pos_alphas):
            pos = alpha * d
            if pos <= d_accel:
                t = np.sqrt(2 * pos / a_max)
            elif pos <= d - d_accel:
                t = t_a + (pos - d_accel) / v_max
            else:
                pos_from_end = d - pos
                t_from_end = np.sqrt(2 * max(0, pos_from_end) / a_max)
                t = T - t_from_end
            time_ratios[i] = t / T
    else:
        T = 2.0 * np.sqrt(d / a_max)
        d_mid = d / 2
        time_ratios = np.zeros(n_total)
        for i, alpha in enumerate(pos_alphas):
            pos = alpha * d
            if pos <= d_mid:
                t = np.sqrt(2 * pos / a_max)
            else:
                pos_from_end = d - pos
                t_from_end = np.sqrt(2 * max(0, pos_from_end) / a_max)
                t = T - t_from_end
            time_ratios[i] = t / T

    time_ratios[0] = 0.0
    time_ratios[-1] = 1.0
    return time_ratios[1:-1], T_min


def _compute_multijoint_time_ratios(n_total, T_total):
    """Compute time ratios that satisfy all joints simultaneously."""
    dq = np.abs(Q_GOAL - Q_START)
    pos_alphas = np.linspace(0, 1, n_total)

    time_ratios = np.zeros(n_total)
    for i, alpha in enumerate(pos_alphas):
        max_time = 0.0
        for j in range(7):
            d = dq[j]
            if d < 1e-10:
                continue
            target_pos = alpha * d
            v_max = MAX_VEL[j]
            a_max = MAX_ACC[j]

            t_accel = min(v_max / a_max, T_total / 2)
            v_peak = a_max * t_accel
            d_accel = 0.5 * a_max * t_accel**2
            d_total_possible = 2 * d_accel + v_peak * (T_total - 2 * t_accel)

            if d_total_possible < 1e-10:
                continue
            scale = d / d_total_possible

            lo_t, hi_t = 0.0, T_total
            for _ in range(20):
                mid_t = 0.5 * (lo_t + hi_t)
                if mid_t <= t_accel:
                    pos = 0.5 * a_max * mid_t**2
                elif mid_t <= T_total - t_accel:
                    pos = d_accel + v_peak * (mid_t - t_accel)
                else:
                    dt_from_end = T_total - mid_t
                    pos = d_total_possible - 0.5 * a_max * dt_from_end**2
                pos *= scale
                if pos < target_pos:
                    lo_t = mid_t
                else:
                    hi_t = mid_t

            t_for_joint = 0.5 * (lo_t + hi_t)
            max_time = max(max_time, t_for_joint)

        time_ratios[i] = max_time / T_total if T_total > 0 else alpha

    time_ratios[0] = 0.0
    time_ratios[-1] = 1.0
    for i in range(1, n_total):
        if time_ratios[i] <= time_ratios[i-1]:
            time_ratios[i] = time_ratios[i-1] + 1e-6
    time_ratios = time_ratios / time_ratios[-1]
    return time_ratios[1:-1]


def _compute_optimal_wps(n_total, T_total):
    """Compute waypoints at trapezoidal profile positions sampled at uniform time."""
    dq = Q_GOAL - Q_START
    abs_dq = np.abs(dq)

    time_fracs = np.linspace(0, 1, n_total)
    wps = np.zeros((n_total, 7))

    for j in range(7):
        d = abs_dq[j]
        if d < 1e-10:
            wps[:, j] = Q_START[j]
            continue

        v_max = MAX_VEL[j]
        a_max = MAX_ACC[j]
        sign = np.sign(dq[j])

        disc = (a_max * T_total)**2 - 4 * a_max * d
        if disc >= 0:
            v_peak = (a_max * T_total - np.sqrt(disc)) / 2.0
            v_peak = min(v_peak, v_max)
        else:
            v_peak = a_max * T_total / 2.0
            v_peak = min(v_peak, v_max)

        if v_peak < 1e-10:
            wps[:, j] = Q_START[j]
            continue

        t_accel = v_peak / a_max
        d_accel = 0.5 * a_max * t_accel**2

        for i, tf in enumerate(time_fracs):
            t = tf * T_total
            if t <= t_accel:
                pos = 0.5 * a_max * t**2
            elif t <= T_total - t_accel:
                pos = d_accel + v_peak * (t - t_accel)
            else:
                dt = T_total - t
                pos = d - 0.5 * a_max * max(0, dt)**2
            pos = max(0, min(pos, d))
            wps[i, j] = Q_START[j] + sign * pos

    wps[0] = Q_START
    wps[-1] = Q_GOAL
    return wps


def _try_candidate(wps, ratios, n_total, joint_limits, robot_id, obs_id, joint_idxs,
                   theoretical_min, current_best):
    """Try a candidate configuration and return (time, wps, ratios) or None."""
    lo_nc = theoretical_min * 0.85
    hi_nc = min(3.0, current_best)
    if lo_nc >= hi_nc:
        return None
    nc_time = _fast_min_time(wps, ratios, joint_limits, lo_nc, hi_nc,
                             iterations=25, samples=20)
    if nc_time >= current_best:
        return None

    # Find feasible time with collision
    found = False
    hi_t = None
    for mult in [1.0, 1.005, 1.01, 1.02, 1.04, 1.08, 1.15, 1.3, 1.6, 2.0]:
        t_try = nc_time * mult
        if t_try >= current_best:
            break
        ts = _build_timestamps(t_try, n_total, ratios)
        if _is_feasible(t_try, wps, ts, joint_limits, robot_id, obs_id, joint_idxs,
                        samples=6, check_collision=True):
            found = True
            hi_t = t_try
            break

    if not found:
        return None

    lo_t = nc_time * 0.99
    result_t = _binary_search_time(wps, ratios, joint_limits, robot_id, obs_id,
                                   joint_idxs, lo_t, hi_t, iterations=18, samples=10)

    # Verify with full samples
    ts = _build_timestamps(result_t, n_total, ratios)
    if _is_feasible(result_t, wps, ts, joint_limits, robot_id, obs_id, joint_idxs,
                    samples=SAMPLES_PER_SEG, check_collision=True):
        return result_t, wps, ratios
    else:
        result_t2 = _binary_search_time(wps, ratios, joint_limits, robot_id, obs_id,
                                        joint_idxs, result_t, result_t * 1.08,
                                        iterations=14, samples=SAMPLES_PER_SEG)
        ts2 = _build_timestamps(result_t2, n_total, ratios)
        if _is_feasible(result_t2, wps, ts2, joint_limits, robot_id, obs_id, joint_idxs,
                        samples=SAMPLES_PER_SEG, check_collision=True):
            return result_t2, wps, ratios
    return None


def _max_violation_full(total_time, wps, timestamps, joint_limits, samples=20):
    """Return detailed violation info for optimization."""
    n_wps = len(wps)
    cs = CubicSpline(timestamps, wps, bc_type="clamped")
    cs_vel = cs.derivative(1)
    cs_acc = cs.derivative(2)

    all_t = []
    for seg in range(n_wps - 1):
        t0 = float(timestamps[seg])
        t1 = float(timestamps[seg + 1])
        all_t.append(np.linspace(t0, t1, samples, endpoint=False))
    all_t = np.concatenate(all_t)

    v_all = cs_vel(all_t)
    a_all = cs_acc(all_t)
    q_all = cs(all_t)

    vel_viol = np.max(np.abs(v_all) / MAX_VEL[None, :]) - 1.0
    acc_viol = np.max(np.abs(a_all) / MAX_ACC[None, :]) - 1.0
    jl_lo_viol = np.max(joint_limits[:, 0] - q_all)
    jl_hi_viol = np.max(q_all - joint_limits[:, 1])

    return max(vel_viol, acc_viol, jl_lo_viol, jl_hi_viol)


def solve() -> tuple[list[list[float]], list[float]]:
    physics = p.connect(p.DIRECT)
    start_time = time_mod.time()
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

        dq = np.abs(Q_GOAL - Q_START)

        # Compute theoretical minimum
        per_joint_min = []
        for j in range(7):
            v = MAX_VEL[j]; a = MAX_ACC[j]; d = dq[j]
            if d < 1e-10:
                per_joint_min.append(0.0)
                continue
            t_a = v / a
            d_a = 0.5 * a * t_a**2
            if 2 * d_a <= d:
                per_joint_min.append(d / v + v / a)
            else:
                per_joint_min.append(2.0 * np.sqrt(d / a))
        theoretical_min = max(per_joint_min)

        best_time = float('inf')
        best_wps = None
        best_time_ratios = None

        def update_best(t, wps, ratios):
            nonlocal best_time, best_wps, best_time_ratios
            if t < best_time:
                best_time = t
                best_wps = wps.copy() if isinstance(wps, np.ndarray) else np.array(wps)
                best_time_ratios = ratios.copy() if isinstance(ratios, np.ndarray) else np.array(ratios)

        def time_left():
            return max(0, 110 - (time_mod.time() - start_time))

        # ================================================================
        # Strategy 1: Time-parameterized waypoints with uniform time ratios
        # ================================================================
        for n_total in [12, 17, 22, 32, 42, 52, 72, 102, 152]:
            if time_left() < 10:
                break
            for T_mult in [1.0, 1.005, 1.01, 1.02]:
                T_try = theoretical_min * T_mult
                wps_tp = _compute_optimal_wps(n_total, T_try)
                ratios_tp = np.linspace(0, 1, n_total)[1:-1]
                result = _try_candidate(wps_tp, ratios_tp, n_total, joint_limits,
                                        robot_id, obs_id, joint_idxs, theoretical_min, best_time)
                if result is not None:
                    update_best(result[0], result[1], result[2])

        # ================================================================
        # Strategy 2: Straight line + trapezoidal/multijoint time ratios
        # ================================================================
        for n_total in [22, 32, 42, 52, 72, 102, 152]:
            if time_left() < 10:
                break
            pos_alphas = np.linspace(0, 1, n_total)
            wps = np.array([(1 - a) * Q_START + a * Q_GOAL for a in pos_alphas])

            trap_ratios, _ = _compute_trapezoidal_time_ratios(n_total)
            result = _try_candidate(wps, trap_ratios, n_total, joint_limits,
                                    robot_id, obs_id, joint_idxs, theoretical_min, best_time)
            if result is not None:
                update_best(result[0], result[1], result[2])

            mj_ratios = _compute_multijoint_time_ratios(n_total, theoretical_min)
            result = _try_candidate(wps, mj_ratios, n_total, joint_limits,
                                    robot_id, obs_id, joint_idxs, theoretical_min, best_time)
            if result is not None:
                update_best(result[0], result[1], result[2])

        # ================================================================
        # Strategy 3: Hybrid - time-param waypoints + trapezoidal time ratios
        # ================================================================
        for n_total in [22, 42, 72, 102]:
            if time_left() < 10:
                break
            wps_tp = _compute_optimal_wps(n_total, theoretical_min)
            trap_ratios, _ = _compute_trapezoidal_time_ratios(n_total)
            result = _try_candidate(wps_tp, trap_ratios, n_total, joint_limits,
                                    robot_id, obs_id, joint_idxs, theoretical_min, best_time)
            if result is not None:
                update_best(result[0], result[1], result[2])

        # ================================================================
        # Strategy 4: Small waypoint count with joint optimization
        # For 3-8 waypoints, optimize waypoint positions + time ratios jointly
        # ================================================================
        for n_interior in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
            if time_left() < 15:
                break
            n_total = n_interior + 2

            # Start from time-parameterized waypoints
            wps_init = _compute_optimal_wps(n_total, theoretical_min * 1.01)
            ratios_init = np.linspace(0, 1, n_total)[1:-1]

            # Also try trapezoidal ratios
            trap_r, _ = _compute_trapezoidal_time_ratios(n_total)

            for ratios_try in [ratios_init, trap_r]:
                result = _try_candidate(wps_init, ratios_try, n_total, joint_limits,
                                        robot_id, obs_id, joint_idxs, theoretical_min, best_time)
                if result is not None:
                    update_best(result[0], result[1], result[2])

        # ================================================================
        # Phase 2: Optimize segment durations for best config
        # ================================================================
        if best_wps is not None and best_time_ratios is not None and len(best_wps) > 3 and time_left() > 15:
            n_total = len(best_wps)
            saved_best = best_time
            saved_wps = best_wps.copy()

            all_r = np.concatenate([[0.0], np.sort(best_time_ratios), [1.0]])
            seg_durations = np.diff(all_r)

            def opt_seg_dur_obj(log_dur):
                dur = np.exp(log_dur)
                dur = dur / dur.sum()
                cumsum = np.cumsum(dur)
                ratios = cumsum[:-1]
                lo_t = theoretical_min * 0.85
                hi_t = saved_best
                t = _fast_min_time(saved_wps, ratios, joint_limits, lo_t, hi_t,
                                   iterations=18, samples=18)
                return t

            try:
                x0_dur = np.log(seg_durations + 1e-10)
                result = minimize(opt_seg_dur_obj, x0_dur, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 0.0005, 'fatol': 0.0002,
                                           'adaptive': True})
                dur_opt = np.exp(result.x)
                dur_opt = dur_opt / dur_opt.sum()
                cumsum_opt = np.cumsum(dur_opt)
                opt_ratios = cumsum_opt[:-1]

                lo_t = theoretical_min * 0.85
                hi_t = best_time
                opt_t = _binary_search_time(saved_wps, opt_ratios, joint_limits, robot_id, obs_id,
                                            joint_idxs, lo_t, hi_t, iterations=25, samples=SAMPLES_PER_SEG)
                ts_opt = _build_timestamps(opt_t, n_total, opt_ratios)
                if _is_feasible(opt_t, saved_wps, ts_opt, joint_limits, robot_id, obs_id, joint_idxs,
                                samples=SAMPLES_PER_SEG):
                    update_best(opt_t, saved_wps, opt_ratios)
            except Exception:
                pass

        # ================================================================
        # Phase 3: Joint optimization of waypoint positions
        # Try to perturb waypoints to reduce minimum time
        # ================================================================
        if best_wps is not None and best_time_ratios is not None and time_left() > 20:
            n_total = len(best_wps)
            n_interior = n_total - 2

            if 1 <= n_interior <= 15:
                saved_best3 = best_time
                interior_wps = best_wps[1:-1].copy()
                curr_ratios = best_time_ratios.copy()

                def opt_wp_obj(delta):
                    perturbed = interior_wps + delta.reshape(n_interior, 7)
                    if np.any(perturbed < joint_limits[:, 0]) or np.any(perturbed > joint_limits[:, 1]):
                        return saved_best3 + 10.0
                    wps_new = np.vstack([Q_START, perturbed, Q_GOAL])
                    lo_t = theoretical_min * 0.85
                    hi_t = saved_best3
                    t = _fast_min_time(wps_new, curr_ratios, joint_limits, lo_t, hi_t,
                                       iterations=12, samples=12)
                    return t

                try:
                    x0 = np.zeros(n_interior * 7)
                    max_iter = min(3000, int(time_left() * 15))
                    result = minimize(opt_wp_obj, x0, method='Nelder-Mead',
                                      options={'maxiter': max_iter, 'xatol': 0.001, 'fatol': 0.0003,
                                               'adaptive': True})
                    delta_opt = result.x.reshape(n_interior, 7)
                    opt_interior = interior_wps + delta_opt
                    opt_wps = np.vstack([Q_START, opt_interior, Q_GOAL])

                    lo_t = theoretical_min * 0.85
                    hi_t = best_time
                    opt_t = _binary_search_time(opt_wps, curr_ratios, joint_limits, robot_id, obs_id,
                                                joint_idxs, lo_t, hi_t, iterations=22, samples=SAMPLES_PER_SEG)
                    ts_opt = _build_timestamps(opt_t, n_total, curr_ratios)
                    if _is_feasible(opt_t, opt_wps, ts_opt, joint_limits, robot_id, obs_id, joint_idxs,
                                    samples=SAMPLES_PER_SEG):
                        update_best(opt_t, opt_wps, curr_ratios)
                except Exception:
                    pass

        # ================================================================
        # Phase 4: For the best config, re-optimize time-param wps at best_time
        # ================================================================
        if best_time < float('inf') and time_left() > 10:
            for n_total in [32, 52, 72, 102, 152, 202]:
                if time_left() < 5:
                    break
                for T_mult in [0.98, 0.99, 1.0, 1.005]:
                    T_try = best_time * T_mult
                    if T_try < theoretical_min * 0.9:
                        continue
                    wps_tp = _compute_optimal_wps(n_total, T_try)
                    ratios_u = np.linspace(0, 1, n_total)[1:-1]
                    result = _try_candidate(wps_tp, ratios_u, n_total, joint_limits,
                                            robot_id, obs_id, joint_idxs, theoretical_min, best_time)
                    if result is not None:
                        update_best(result[0], result[1], result[2])

        # ================================================================
        # Phase 5: Alternating optimization - iterate between ratios and waypoints
        # ================================================================
        if best_wps is not None and time_left() > 15:
            for iteration in range(3):
                if time_left() < 8:
                    break
                n_total = len(best_wps)
                n_interior = n_total - 2
                improved = False

                # Optimize ratios
                if n_interior >= 1:
                    saved_best5 = best_time
                    saved_wps5 = best_wps.copy()
                    all_r = np.concatenate([[0.0], np.sort(best_time_ratios), [1.0]])
                    seg_dur = np.diff(all_r)

                    def opt_r5(log_d):
                        dur = np.exp(log_d)
                        dur = dur / dur.sum()
                        cs = np.cumsum(dur)
                        r = cs[:-1]
                        return _fast_min_time(saved_wps5, r, joint_limits,
                                              theoretical_min * 0.85, saved_best5,
                                              iterations=15, samples=15)
                    try:
                        res = minimize(opt_r5, np.log(seg_dur + 1e-10), method='Nelder-Mead',
                                       options={'maxiter': 1000, 'fatol': 0.0002, 'adaptive': True})
                        dur_opt = np.exp(res.x)
                        dur_opt = dur_opt / dur_opt.sum()
                        opt_r = np.cumsum(dur_opt)[:-1]
                        opt_t = _binary_search_time(saved_wps5, opt_r, joint_limits, robot_id, obs_id,
                                                    joint_idxs, theoretical_min * 0.85, best_time,
                                                    iterations=22, samples=SAMPLES_PER_SEG)
                        ts_chk = _build_timestamps(opt_t, n_total, opt_r)
                        if _is_feasible(opt_t, saved_wps5, ts_chk, joint_limits, robot_id, obs_id,
                                        joint_idxs, samples=SAMPLES_PER_SEG):
                            if opt_t < best_time:
                                update_best(opt_t, saved_wps5, opt_r)
                                improved = True
                    except Exception:
                        pass

                # Optimize waypoints
                if n_interior >= 1 and n_interior <= 15 and time_left() > 5:
                    saved_best5b = best_time
                    interior5 = best_wps[1:-1].copy()
                    ratios5 = best_time_ratios.copy()

                    def opt_w5(delta):
                        perturbed = interior5 + delta.reshape(n_interior, 7)
                        if np.any(perturbed < joint_limits[:, 0]) or np.any(perturbed > joint_limits[:, 1]):
                            return saved_best5b + 10.0
                        wps_n = np.vstack([Q_START, perturbed, Q_GOAL])
                        return _fast_min_time(wps_n, ratios5, joint_limits,
                                              theoretical_min * 0.85, saved_best5b,
                                              iterations=12, samples=12)
                    try:
                        res = minimize(opt_w5, np.zeros(n_interior * 7), method='Nelder-Mead',
                                       options={'maxiter': 1500, 'fatol': 0.0003, 'adaptive': True})
                        d_opt = res.x.reshape(n_interior, 7)
                        wps_opt = np.vstack([Q_START, interior5 + d_opt, Q_GOAL])
                        opt_t = _binary_search_time(wps_opt, ratios5, joint_limits, robot_id, obs_id,
                                                    joint_idxs, theoretical_min * 0.85, best_time,
                                                    iterations=22, samples=SAMPLES_PER_SEG)
                        ts_chk = _build_timestamps(opt_t, n_total, ratios5)
                        if _is_feasible(opt_t, wps_opt, ts_chk, joint_limits, robot_id, obs_id,
                                        joint_idxs, samples=SAMPLES_PER_SEG):
                            if opt_t < best_time:
                                update_best(opt_t, wps_opt, ratios5)
                                improved = True
                    except Exception:
                        pass

                if not improved:
                    break

        # Final precision binary search
        if best_wps is not None:
            n_total = len(best_wps)
            lo_t = best_time * 0.92
            hi_t = best_time
            best_time = _binary_search_time(best_wps, best_time_ratios, joint_limits, robot_id, obs_id,
                                            joint_idxs, lo_t, hi_t, iterations=35, samples=SAMPLES_PER_SEG)

        # Fallback
        if best_wps is None:
            Q_VIA = np.array([0.6, 0.1, 0.4, -1.8, 0.2, 0.9, 0.5], dtype=float)
            best_wps = np.vstack([Q_START, Q_VIA, Q_GOAL])
            best_time_ratios = np.array([0.48])
            best_time = 3.0

        n_total = len(best_wps)
        timestamps = _build_timestamps(best_time, n_total, best_time_ratios)
        waypoints = [w.tolist() for w in best_wps]
        return waypoints, timestamps.tolist()
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