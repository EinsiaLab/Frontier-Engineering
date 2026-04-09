# EVOLVE-BLOCK-START
"""
支持燃料补给飞船功能

任务目标：最大化运载质量（Payload）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar

# ==================== 常数定义 ====================
mu_e = 398600.0
mu_m = 4903.0
Re = 6378.0
Rm = 1737.0
LU = 384400.0
Ce_km_s = 3.0

mu_sys = mu_e + mu_m
mu = mu_m / mu_sys
TU = np.sqrt(LU**3 / mu_sys)
VU = LU / TU

Re_norm = Re / LU
Rm_norm = Rm / LU
h_LEO = 400.0 / LU
h_LLO = 100.0 / LU

M_dry = 10000.0
M_fuel_max = 15000.0
M_return_fuel = 100.0

VEC_RE = np.array([-mu, 0.0])
VEC_RM = np.array([1.0 - mu, 0.0])

# ==================== CR3BP动力学 ====================
def cr3bp(t, y):
    x, yy, vx, vy = y
    r1 = np.sqrt((x + mu)**2 + yy**2)
    r2 = np.sqrt((x - 1 + mu)**2 + yy**2)
    r1_3 = r1**3
    r2_3 = r2**3
    ax = 2*vy + x - (1-mu)*(x+mu)/r1_3 - mu*(x-1+mu)/r2_3
    ay = -2*vx + yy - (1-mu)*yy/r1_3 - mu*yy/r2_3
    return [vx, vy, ax, ay]

def propagate_circular(state, dt, t_old, center, gm):
    dr = state[0:2] - center
    r_mag = np.linalg.norm(dr)
    v_iner = state[2:4] + np.array([-dr[1], dr[0]])
    n = np.sqrt(gm / r_mag**3)
    cross = dr[0]*v_iner[1] - dr[1]*v_iner[0]
    d_theta = n * dt if cross >= 0 else -n * dt
    c, s = np.cos(d_theta), np.sin(d_theta)
    R = np.array([[c, -s], [s, c]])
    dr_new = R @ dr
    v_new = R @ v_iner
    state_new = np.empty(4)
    state_new[0:2] = dr_new + center
    state_new[2:4] = v_new - np.array([-dr_new[1], dr_new[0]])
    return state_new, t_old + dt

# ==================== 主优化循环 ====================
print("=" * 60)
print("地月转移轨道优化 - 最大化Payload")
print("=" * 60)

r_leo = h_LEO + Re_norm
v_circ_leo = np.sqrt((1 - mu) / r_leo)
target_moon = Rm_norm + h_LLO

best_payload = -1e30
best_params = None
best_data = None

# We'll do a grid search over dv1 and refine
for dv1_try in np.linspace(3.04, 3.12, 17):
    # For each dv1, find best departure angle
    def flyby_min_dist(th):
        v_dep = v_circ_leo + dv1_try
        pos = VEC_RE + np.array([r_leo * np.cos(th), r_leo * np.sin(th)])
        u_t = np.array([-np.sin(th), np.cos(th)])
        vel = v_dep * u_t - np.array([-pos[1], pos[0]])
        y0 = np.concatenate([pos, vel])
        sol = solve_ivp(cr3bp, [0, 5.0], y0, method='RK45', rtol=1e-10, atol=1e-10,
                        max_step=0.01)
        dists = np.sqrt((sol.y[0] - VEC_RM[0])**2 + (sol.y[1] - VEC_RM[1])**2)
        return abs(np.min(dists) - target_moon)

    # Coarse grid for theta
    best_th_local = -2.4
    best_err_local = 1e10
    for th_i in np.linspace(-2.7, -2.1, 25):
        try:
            err = flyby_min_dist(th_i)
            if err < best_err_local:
                best_err_local = err
                best_th_local = th_i
        except:
            pass

    # Refine theta
    try:
        res_th = minimize_scalar(flyby_min_dist,
                                 bounds=(best_th_local - 0.05, best_th_local + 0.05),
                                 method='bounded', options={'xatol': 1e-12})
        th1 = res_th.x
        if res_th.fun > 1e-4:
            continue
    except:
        continue

    # Execute TLI
    v_dep = v_circ_leo + dv1_try
    pos_1 = VEC_RE + np.array([r_leo * np.cos(th1), r_leo * np.sin(th1)])
    u_tan = np.array([-np.sin(th1), np.cos(th1)])
    vel_pre = v_circ_leo * u_tan - np.array([-pos_1[1], pos_1[0]])
    vel_post = v_dep * u_tan - np.array([-pos_1[1], pos_1[0]])
    dv1_vec = vel_post - vel_pre

    def ev_moon(t, y):
        return np.sqrt((y[0]-VEC_RM[0])**2 + (y[1]-VEC_RM[1])**2) - target_moon
    ev_moon.terminal = True

    try:
        sol_tli = solve_ivp(cr3bp, [0, 5.0], np.concatenate([pos_1, vel_post]),
                            method='DOP853', rtol=1e-12, atol=1e-12,
                            events=ev_moon, max_step=0.01)
        if sol_tli.t_events[0].size == 0:
            continue
    except:
        continue

    t_arr_M = sol_tli.t_events[0][0]
    state_arr_M = sol_tli.y_events[0][0]

    # LOI
    dr = state_arr_M[0:2] - VEC_RM
    r_act = np.linalg.norm(dr)
    v_circ_m = np.sqrt(mu / r_act)
    u_rad = dr / r_act
    u_tan_m = np.array([-u_rad[1], u_rad[0]])
    if np.dot(state_arr_M[2:4] + np.array([-dr[1], dr[0]]), u_tan_m) < 0:
        u_tan_m = -u_tan_m
    vel_loi = v_circ_m * u_tan_m - np.array([-dr[1], dr[0]])
    dv2_vec = vel_loi - state_arr_M[2:4]
    dv2_mag = np.linalg.norm(dv2_vec)

    state_loi = state_arr_M.copy()
    state_loi[2:4] = vel_loi

    # Lunar stay
    dt_stay_days = 3.5
    dt_stay = dt_stay_days * 86400 / TU
    state_pre_tei, t_dep = propagate_circular(state_loi, dt_stay, t_arr_M, VEC_RM, mu)

    # TEI optimization
    def compute_return(dv3_val):
        dr_d = state_pre_tei[0:2] - VEC_RM
        u_t_d = np.array([-dr_d[1], dr_d[0]]) / np.linalg.norm(dr_d)
        if np.dot(state_pre_tei[2:4] + np.array([-dr_d[1], dr_d[0]]), u_t_d) < 0:
            u_t_d = -u_t_d
        dv3_v = dv3_val * u_t_d
        sp = state_pre_tei.copy()
        sp[2:4] += dv3_v
        sol_r = solve_ivp(cr3bp, [0, 6.0], sp, method='DOP853', rtol=1e-12, atol=1e-12,
                          dense_output=True, max_step=0.02)
        ts = np.linspace(0.5, sol_r.t[-1], 3000)
        ds = np.array([np.linalg.norm(sol_r.sol(t)[0:2] - VEC_RE) for t in ts])
        idx = np.argmin(ds)
        tg = ts[idx]
        res2 = minimize_scalar(lambda t: np.linalg.norm(sol_r.sol(np.clip(t,0,sol_r.t[-1]))[0:2] - VEC_RE),
                               bounds=(max(0, tg-0.2), min(sol_r.t[-1], tg+0.2)),
                               method='bounded', options={'xatol': 1e-12})
        alt = (res2.fun - Re_norm) * LU
        return alt, res2.x, sol_r, dv3_v, sp

    try:
        res_dv3 = minimize_scalar(lambda d: abs(compute_return(d)[0]),
                                  bounds=(0.7, 0.9), method='bounded',
                                  options={'xatol': 1e-10})
        dv3_opt = res_dv3.x
        alt_f, t_peri, sol_tei, dv3_vec, state_post_tei = compute_return(dv3_opt)
    except:
        continue

    # Check total mission time
    t_arr_E = t_dep + t_peri
    total_days = t_arr_E * TU / 86400
    if total_days < 8 or total_days > 25:
        continue

    # C3 and mass budget
    x_rel = pos_1[0] + mu
    y_rel = pos_1[1]
    v_ix = vel_post[0] - y_rel
    v_iy = vel_post[1] + x_rel
    C3 = (v_ix**2 + v_iy**2) * VU**2 - 2*mu_e / (np.sqrt(x_rel**2 + y_rel**2) * LU)
    if C3 < -5 or C3 > 5:
        continue
    M0 = 25000 - 1000 * C3

    ratio_loi = np.exp(-(dv2_mag * VU) / Ce_km_s)
    ratio_tei = np.exp(-(dv3_opt * VU) / Ce_km_s)

    M_return_wet = M_dry + M_return_fuel
    Payload = M0 * ratio_loi - (M_return_wet / ratio_tei)
    Fuel_launch = M0 - M_dry - Payload

    if Fuel_launch > M_fuel_max:
        Fuel_launch = M_fuel_max
        Payload = M0 - M_dry - Fuel_launch

    if Payload <= 0:
        continue

    Fuel_after_loi = Fuel_launch - (M0 * (1 - ratio_loi))
    if Fuel_after_loi < 0:
        continue

    # Check return altitude reasonable
    if abs(alt_f) > 200:
        continue

    if Payload > best_payload:
        best_payload = Payload
        best_params = {
            'dv1': dv1_try, 'th1': th1, 'dv2_mag': dv2_mag, 'dv3': dv3_opt,
            'C3': C3, 'M0': M0, 'Payload': Payload, 'Fuel_launch': Fuel_launch,
            'Fuel_after_loi': Fuel_after_loi, 'alt_f': alt_f, 'total_days': total_days,
            't_arr_M': t_arr_M, 't_dep': t_dep, 't_arr_E': t_arr_E, 't_peri': t_peri,
            'pos_1': pos_1, 'vel_post': vel_post, 'vel_pre': vel_pre, 'dv1_vec': dv1_vec,
            'state_arr_M': state_arr_M, 'state_loi': state_loi, 'dv2_vec': dv2_vec,
            'state_pre_tei': state_pre_tei, 'state_post_tei': state_post_tei,
            'dv3_vec': dv3_vec, 'sol_tei': sol_tei, 'dt_stay_days': dt_stay_days,
        }
        print(f"  New best: dv1={dv1_try:.4f}, C3={C3:.4f}, M0={M0:.1f}, Payload={Payload:.2f}")

if best_params is None:
    raise RuntimeError("No valid trajectory found")

p = best_params
print(f"\nBest Payload = {p['Payload']:.2f} kg, C3={p['C3']:.4f}, total={p['total_days']:.2f} days")

# ==================== 生成results.txt ====================
data = []
t0 = 0.0

# Event 1: LEO pre-TLI
theta_check = 0.0
r_init_norm = Re_norm + h_LEO
pos_check = r_init_norm * np.array([np.cos(theta_check), np.sin(theta_check)]) + VEC_RE
v_init = np.sqrt((1 - mu) / r_init_norm)
vel_check = (v_init - r_init_norm) * np.array([-np.sin(theta_check), np.cos(theta_check)])
data.append([1, t0, pos_check[0], pos_check[1], vel_check[0], vel_check[1], 0, 0, p['Fuel_launch'], p['Payload']])

# Event 1: TLI post
data.append([1, t0, p['pos_1'][0], p['pos_1'][1], p['vel_post'][0], p['vel_post'][1],
             p['dv1_vec'][0], p['dv1_vec'][1], p['Fuel_launch'], p['Payload']])

# Event 0: Transfer
data.append([0, t0, p['pos_1'][0], p['pos_1'][1], p['vel_post'][0], p['vel_post'][1], 0, 0, p['Fuel_launch'], p['Payload']])
data.append([0, p['t_arr_M'], p['state_arr_M'][0], p['state_arr_M'][1], p['state_arr_M'][2], p['state_arr_M'][3], 0, 0, p['Fuel_launch'], p['Payload']])

# Event -1: LOI
data.append([-1, p['t_arr_M'], p['state_arr_M'][0], p['state_arr_M'][1], p['state_arr_M'][2], p['state_arr_M'][3], 0, 0, p['Fuel_launch'], p['Payload']])
data.append([-1, p['t_arr_M'], p['state_loi'][0], p['state_loi'][1], p['state_loi'][2], p['state_loi'][3],
             p['dv2_vec'][0], p['dv2_vec'][1], p['Fuel_after_loi'], p['Payload']])

# Event 2: Arrive LLO
data.append([2, p['t_arr_M'], p['state_loi'][0], p['state_loi'][1], p['state_loi'][2], p['state_loi'][3], 0, 0, p['Fuel_after_loi'], p['Payload']])

# Event 3: Depart LLO
data.append([3, p['t_dep'], p['state_pre_tei'][0], p['state_pre_tei'][1], p['state_pre_tei'][2], p['state_pre_tei'][3], 0, 0, p['Fuel_after_loi'], 0.0])

# Event -1: TEI
data.append([-1, p['t_dep'], p['state_pre_tei'][0], p['state_pre_tei'][1], p['state_pre_tei'][2], p['state_pre_tei'][3], 0, 0, p['Fuel_after_loi'], 0.0])
data.append([-1, p['t_dep'], p['state_post_tei'][0], p['state_post_tei'][1], p['state_post_tei'][2], p['state_post_tei'][3],
             p['dv3_vec'][0], p['dv3_vec'][1], M_return_fuel, 0.0])

# Event 0: Return transfer
t_safe = 0.1
state_safe = p['sol_tei'].sol(t_safe)
data.append([0, p['t_dep'], p['state_post_tei'][0], p['state_post_tei'][1], p['state_post_tei'][2], p['state_post_tei'][3], 0, 0, M_return_fuel, 0.0])
data.append([0, p['t_dep'] + t_safe, state_safe[0], state_safe[1], state_safe[2], state_safe[3], 0, 0, M_return_fuel, 0.0])

# Event 4: Earth arrival
state_final = p['sol_tei'].sol(p['t_peri'])
data.append([4, p['t_arr_E'], state_final[0], state_final[1], state_final[2], state_final[3], 0, 0, M_return_fuel, 0.0])

data = np.array(data)
np.savetxt('results.txt', data, fmt=['%d'] + ['%.12e']*9, delimiter='\t')

print("=" * 60)
print(f"✓ Payload: {p['Payload']:.2f} kg")
print(f"✓ Mission duration: {p['total_days']:.2f} days")
print("=" * 60)
# EVOLVE-BLOCK-END


# 固定的入口函数（不被演化）
def run_mission():
    """运行月球任务并返回生成results.txt文件的路径"""
    # 上面的代码会生成results.txt
    return "results.txt"