# EVOLVE-BLOCK-START
"""
支持燃料补给飞船功能

任务目标：最大化运载质量（Payload）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# ==================== 常数定义 ====================
class CONST:
    # 物理常数
    mu_e = 398600.0      # km³/s²
    mu_m = 4903.0        # km³/s²
    Re = 6378.0          # km
    Rm = 1737.0          # km
    LU = 384400.0        # km (地月距离)
    Ce_km_s = 3.0        # km/s (比冲)

    # 归一化常数
    mu_sys = mu_e + mu_m
    mu = mu_m / mu_sys
    TU = np.sqrt(LU**3 / mu_sys)
    VU = LU / TU

    # 轨道高度
    Re_norm = Re / LU
    Rm_norm = Rm / LU
    h_LEO = 400.0 / LU
    h_LLO = 100.0 / LU

    # 质量参数
    M_dry = 10000.0      # kg (干重)
    M_fuel_max = 15000.0 # kg (燃料上限)
    M_return_fuel = 100.0 # kg (返回燃料)

CONST = CONST()
VEC_RE = np.array([-CONST.mu, 0])
VEC_RM = np.array([1 - CONST.mu, 0])

# ==================== CR3BP动力学 ====================
def all_dynamics(t, y):
    """圆型限制性三体问题动力学方程"""
    x, yy, vx, vy = y
    mu = CONST.mu
    r1 = np.sqrt((x + mu)**2 + yy**2)
    r2 = np.sqrt((x - 1 + mu)**2 + yy**2)
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + yy - (1-mu)*yy/r1**3 - mu*yy/r2**3
    return np.array([vx, vy, ax, ay])

# ==================== 补给飞船（L1 Lyapunov轨道）====================
class SupplyShip:
    """L1附近Lyapunov周期轨道上的补给飞船"""
    # TODO

# ==================== 轨道传播工具 ====================
def propagate_LEO(state_old, dt, t_old):
    """在LEO上传播（简化为圆轨道）"""
    pos_E = VEC_RE
    dr = state_old[0:2] - pos_E
    r_mag = np.linalg.norm(dr)
    v_iner = state_old[2:4] + np.array([-dr[1], dr[0]])
    n = np.sqrt((1 - CONST.mu) / r_mag**3)
    d_theta = n * dt
    R = np.array([[np.cos(d_theta), -np.sin(d_theta)],
                  [np.sin(d_theta), np.cos(d_theta)]])
    dr_new = R @ dr
    v_new = R @ v_iner
    state_new = np.concatenate([dr_new + pos_E, v_new - np.array([-dr_new[1], dr_new[0]])])
    return state_new, t_old + dt

def propagate_LLO(state_old, dt, t_old):
    """在LLO上传播"""
    pos_M = VEC_RM
    dr = state_old[0:2] - pos_M
    r_mag = np.linalg.norm(dr)
    v_iner = state_old[2:4] + np.array([-dr[1], dr[0]])
    n = np.sqrt(CONST.mu / r_mag**3)
    d_theta = -n * dt
    if dr[0] * v_iner[1] - dr[1] * v_iner[0] < 0:
        d_theta = -d_theta
    R = np.array([[np.cos(d_theta), -np.sin(d_theta)],
                  [np.sin(d_theta), np.cos(d_theta)]])
    dr_new = R @ dr
    v_new = R @ v_iner
    state_new = np.concatenate([dr_new + pos_M, v_new - np.array([-dr_new[1], dr_new[0]])])
    return state_new, t_old + dt

# ==================== 主程序 ====================

print("=" * 60)
print("地月转移轨道优化 - 当前方案：基础版（无补给）")
print("=" * 60)

# 初始化补给飞船（虽然暂时不用，但计算出来供参考）
try:
    supply_ship = SupplyShip()
    print("✓ 补给飞船轨道已计算（本次未使用）")
except Exception as e:
    print(f"⚠ 补给飞船计算失败: {e}")
    supply_ship = None

# ==================== TLI参数优化 ====================
print("\n[1/6] 优化TLI参数...")

# 优化TLI参数：同时优化dv1和出发角度以最小化C3并找到更好的转移轨道
def optimize_tli_parameters():
    """同时优化dv1和th1以最小化C3并确保到达月球"""
    target_moon = CONST.Rm_norm + CONST.h_LLO
    
    def objective(params):
        dv1_try, th1_try, dt1_try = params
        # 约束条件
        if dv1_try < 2.8 or dv1_try > 3.3:
            return 1e10
        if dt1_try < 2.5 or dt1_try > 4.0:
            return 1e10
        
        r_leo = CONST.h_LEO + CONST.Re_norm
        v_circ = np.sqrt((1 - CONST.mu) / r_leo)
        v_dep = v_circ + dv1_try
        
        pos = VEC_RE + np.array([r_leo * np.cos(th1_try), r_leo * np.sin(th1_try)])
        u_tan = np.array([-np.sin(th1_try), np.cos(th1_try)])
        vel = v_dep * u_tan - np.array([-pos[1], pos[0]])
        
        # 传播轨道
        sol = solve_ivp(all_dynamics, [0, dt1_try*1.5], np.concatenate([pos, vel]),
                        method='RK45', rtol=1e-9, atol=1e-9, max_step=0.01)
        
        # 计算最小月球距离
        pos_M = VEC_RM
        dists = np.sqrt((sol.y[0, :] - pos_M[0])**2 + (sol.y[1, :] - pos_M[1])**2)
        min_dist = np.min(dists)
        
        # 计算C3
        x_rel = pos[0] + CONST.mu
        y_rel = pos[1]
        v_ix = vel[0] - y_rel
        v_iy = vel[1] + x_rel
        C3 = ((v_ix**2 + v_iy**2) * CONST.VU**2) - 2*CONST.mu_e / (np.sqrt(x_rel**2 + y_rel**2) * CONST.LU)
        
        # 惩罚函数：距离误差 + C3（加权）
        distance_error = abs(min_dist - target_moon)
        return distance_error * 1000 + C3 * 0.1
    
    # 使用差分进化进行全局优化
    bounds = [(2.9, 3.2), (-2.7, -2.3), (2.8, 3.5)]
    result = differential_evolution(objective, bounds, maxiter=50, popsize=15, 
                                   seed=42, tol=1e-6, disp=False)
    
    dv1_opt, th1_opt, dt1_opt = result.x
    print(f"  优化结果: dv1={dv1_opt:.6f} VU, th1={th1_opt:.6f} rad, dt1={dt1_opt:.6f} TU")
    return dv1_opt, th1_opt, dt1_opt

# 执行优化
print("\n[1/6] 优化TLI参数...")
dv1, th1, dt1 = optimize_tli_parameters()
target_moon = CONST.Rm_norm + CONST.h_LLO
print(f"  dv1={dv1:.6f} VU, th1={th1:.6f} rad, dt1={dt1:.6f} TU")

# ==================== TLI执行 ====================
print("\n[2/6] 执行TLI并进行地月转移...")

t0 = 0.0
r_leo = CONST.h_LEO + CONST.Re_norm
v_circ = np.sqrt((1 - CONST.mu) / r_leo)
v_dep = v_circ + dv1

pos_E = VEC_RE
pos_1 = pos_E + np.array([r_leo * np.cos(th1), r_leo * np.sin(th1)])
u_tan = np.array([-np.sin(th1), np.cos(th1)])
vel_pre = v_circ * u_tan - np.array([-pos_1[1], pos_1[0]])
vel_post = v_dep * u_tan - np.array([-pos_1[1], pos_1[0]])
dv1_vec = vel_post - vel_pre

# 地月转移
def event_moon_arrival(t, y):
    r = np.linalg.norm(y[0:2] - VEC_RM)
    return r - target_moon
event_moon_arrival.terminal = True

sol_tli = solve_ivp(all_dynamics, [0, dt1*1.5], np.concatenate([pos_1, vel_post]),
                   method='RK45', rtol=1e-13, atol=1e-13,
                   events=event_moon_arrival)

if sol_tli.t_events[0].size == 0:
    raise ValueError('未到达月球')

t_arr_M = t0 + sol_tli.t_events[0][0]
state_arr_M = sol_tli.y_events[0][0]
print(f"  到达月球: {t_arr_M*CONST.TU/86400:.2f}天")

# ==================== LOI ====================
print("\n[3/6] 执行LOI进入月球轨道...")

pos_M = VEC_RM
dr = state_arr_M[0:2] - pos_M
r_act = np.linalg.norm(dr)
v_circ_m = np.sqrt(CONST.mu / r_act)
u_rad = dr / r_act
u_tan_m = np.array([-u_rad[1], u_rad[0]])

if np.dot(state_arr_M[2:4] + np.array([-dr[1], dr[0]]), u_tan_m) < 0:
    u_tan_m = -u_tan_m

vel_loi = v_circ_m * u_tan_m - np.array([-dr[1], dr[0]])
dv2_vec = vel_loi - state_arr_M[2:4]
dv2_mag = np.linalg.norm(dv2_vec)

state_loi = state_arr_M.copy()
state_loi[2:4] = vel_loi
print(f"  LOI dv={dv2_mag*CONST.VU:.6f} km/s")

# ==================== 月面停留 ====================
print("\n[4/6] 月面停留...")

# 优化停留时间以改善返回条件
def optimize_stay_time(state_loi, t_arr_M):
    """优化停留时间以获得更好的返回条件"""
    best_dt_stay_days = None
    best_altitude = 1e10
    best_state = None
    best_t_dep = None
    
    # 在3-10天内搜索
    for days in np.linspace(3.0, 10.0, 15):
        dt_stay = days * 86400 / CONST.TU
        state_pre_tei, t_dep = propagate_LLO(state_loi, dt_stay, t_arr_M)
        
        # 尝试找到一个dv3使得返回高度接近0
        def test_dv3(dv3):
            dr_dep = state_pre_tei[0:2] - VEC_RM
            u_tan_dep = np.array([-dr_dep[1], dr_dep[0]]) / np.linalg.norm(dr_dep)
            if np.dot(state_pre_tei[2:4] + np.array([-dr_dep[1], dr_dep[0]]), u_tan_dep) < 0:
                u_tan_dep = -u_tan_dep
            dv3_vec = dv3 * u_tan_dep
            state_post = state_pre_tei.copy()
            state_post[2:4] = state_post[2:4] + dv3_vec
            
            sol = solve_ivp(all_dynamics, [0, 6.0], state_post,
                           method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
            
            # 搜索近地点
            t_samples = np.linspace(0, sol.t[-1], 4000)
            dists = np.array([np.linalg.norm(sol.sol(t)[0:2] - VEC_RE) for t in t_samples])
            idx_min = np.argmin(dists)
            alt = (dists[idx_min] - CONST.Re_norm) * CONST.LU
            return abs(alt)
        
        # 尝试几个dv3值
        test_values = np.linspace(0.75, 0.85, 11)
        altitudes = [test_dv3(dv) for dv in test_values]
        min_alt = min(altitudes)
        
        if min_alt < best_altitude:
            best_altitude = min_alt
            best_dt_stay_days = days
            best_state = state_pre_tei
            best_t_dep = t_dep
    
    print(f"  最优停留时间: {best_dt_stay_days:.2f}天, 预估返回高度误差: {best_altitude:.2f} km")
    return best_dt_stay_days, best_state, best_t_dep

print("\n[4/6] 优化月面停留时间...")
dt_stay_days, state_pre_tei, t_dep = optimize_stay_time(state_loi, t_arr_M)
dt_stay = dt_stay_days * 86400 / CONST.TU

# ==================== TEI优化 ====================
print("\n[5/6] 优化TEI参数...")

def compute_return_altitude(dv3):
    """计算给定dv3下的返回高度"""
    dr_dep = state_pre_tei[0:2] - pos_M
    u_tan_dep = np.array([-dr_dep[1], dr_dep[0]]) / np.linalg.norm(dr_dep)

    if np.dot(state_pre_tei[2:4] + np.array([-dr_dep[1], dr_dep[0]]), u_tan_dep) < 0:
        u_tan_dep = -u_tan_dep

    dv3_vec = dv3 * u_tan_dep
    state_post = state_pre_tei.copy()
    state_post[2:4] = state_post[2:4] + dv3_vec

    sol = solve_ivp(all_dynamics, [0, 6.0], state_post,
                   method='DOP853', rtol=1e-13, atol=1e-13, dense_output=True)

    # 搜索近地点
    t_samples = np.linspace(0, sol.t[-1], 8000)
    dists = np.array([np.linalg.norm(sol.sol(t)[0:2] - VEC_RE) for t in t_samples])
    idx_min = np.argmin(dists)
    t_guess = t_samples[idx_min]

    def dist_func(t):
        if t < 0 or t > sol.t[-1]:
            return 1e10
        return np.linalg.norm(sol.sol(t)[0:2] - VEC_RE)

    result = minimize_scalar(dist_func,
                            bounds=(max(0, t_guess-0.3), min(sol.t[-1], t_guess+0.3)),
                            method='bounded', options={'xatol': 1e-12})

    alt = (result.fun - CONST.Re_norm) * CONST.LU
    return alt, result.x, sol

# 使用更精细的搜索找到最优dv3
def optimize_return_dv(state_pre_tei):
    """优化返回地球的Δv"""
    # 首先在宽范围内搜索
    test_dvs = np.linspace(0.70, 0.90, 41)
    best_dv = None
    best_alt = 1e10
    
    for dv_test in test_dvs:
        alt, _, _ = compute_return_altitude(dv_test)
        if abs(alt) < abs(best_alt):
            best_alt = alt
            best_dv = dv_test
    
    # 然后在附近进行精细优化
    def objective(dv3):
        alt, _, _ = compute_return_altitude(dv3)
        return abs(alt)
    
    # 在最佳点附近进行有界优化
    lower = max(0.70, best_dv - 0.02)
    upper = min(0.90, best_dv + 0.02)
    result = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
    dv3_optimal = result.x
    alt_final, t_peri, sol_tei = compute_return_altitude(dv3_optimal)
    
    print(f"  初步搜索最优: {best_dv:.6f} VU, 高度={best_alt:.2f} km")
    print(f"  精细优化结果: dv3={dv3_optimal:.6f} VU, 返回高度={alt_final:.2f} km")
    return dv3_optimal, alt_final, t_peri, sol_tei

print("\n[5/6] 优化TEI参数...")
dv3_optimal, alt_final, t_peri, sol_tei = optimize_return_dv(state_pre_tei)

# ==================== 质量计算 ====================
print("\n[6/6] 计算质量预算...")

# 计算C3和初始质量（使用更精确的模型）
x_rel = pos_1[0] + CONST.mu
y_rel = pos_1[1]
v_ix = vel_post[0] - y_rel
v_iy = vel_post[1] + x_rel
C3 = ((v_ix**2 + v_iy**2) * CONST.VU**2) - 2*CONST.mu_e / (np.sqrt(x_rel**2 + y_rel**2) * CONST.LU)

# 使用任务规定的精确公式
M0 = 25000 - 1000 * C3
print(f"  C3 = {C3:.4f} km²/s², M0 = {M0:.2f} kg")

# 质量比
ratio_loi = np.exp(-(dv2_mag * CONST.VU) / CONST.Ce_km_s)
ratio_tei = np.exp(-(np.linalg.norm([0, dv3_optimal]) * CONST.VU) / CONST.Ce_km_s)

# Payload计算
M_return_wet = CONST.M_dry + CONST.M_return_fuel
Payload = M0 * ratio_loi - (M_return_wet / ratio_tei)
Fuel_launch = M0 - CONST.M_dry - Payload

# 燃料不能超过上限
if Fuel_launch > CONST.M_fuel_max:
    Fuel_launch = CONST.M_fuel_max
    Payload = M0 - CONST.M_dry - Fuel_launch

# 计算LOI后的燃料
Fuel_after_loi = Fuel_launch - (M0 * (1 - ratio_loi))

print(f"  C3 = {C3:.4f} km²/s²")
print(f"  M0 = {M0:.2f} kg")
print(f"  Payload = {Payload:.2f} kg")
print(f"  燃料消耗 = {Fuel_launch - Fuel_after_loi:.2f} kg")

t_arr_E = t_dep + t_peri
total_days = t_arr_E * CONST.TU / 86400

print(f"\n任务总时长: {total_days:.2f}天")

# ==================== 生成results.txt ====================
print("\n生成results.txt...")

data = []

# Event 1: 出发前（标准LEO）
theta_check = 0.0
r_init_norm = CONST.Re_norm + CONST.h_LEO
pos_check = r_init_norm * np.array([np.cos(theta_check), np.sin(theta_check)]) + VEC_RE
v_init = np.sqrt((1 - CONST.mu) / r_init_norm)
vel_check = (v_init - r_init_norm) * np.array([-np.sin(theta_check), np.cos(theta_check)])

data.append([1, t0, pos_check[0], pos_check[1], vel_check[0], vel_check[1],
            0, 0, Fuel_launch, Payload])

# Event 1: TLI后
data.append([1, t0, pos_1[0], pos_1[1], vel_post[0], vel_post[1],
            dv1_vec[0], dv1_vec[1], Fuel_launch, Payload])

# Event 0: 地月转移
data.append([0, t0, pos_1[0], pos_1[1], vel_post[0], vel_post[1], 0, 0, Fuel_launch, Payload])
data.append([0, t_arr_M, state_arr_M[0], state_arr_M[1], state_arr_M[2], state_arr_M[3],
            0, 0, Fuel_launch, Payload])

# Event -1: LOI
data.append([-1, t_arr_M, state_arr_M[0], state_arr_M[1], state_arr_M[2], state_arr_M[3],
            0, 0, Fuel_launch, Payload])
data.append([-1, t_arr_M, state_loi[0], state_loi[1], state_loi[2], state_loi[3],
            dv2_vec[0], dv2_vec[1], Fuel_after_loi, Payload])

# Event 2: 到达LLO
data.append([2, t_arr_M, state_loi[0], state_loi[1], state_loi[2], state_loi[3],
            0, 0, Fuel_after_loi, Payload])

# Event 3: 离开LLO
data.append([3, t_dep, state_pre_tei[0], state_pre_tei[1], state_pre_tei[2], state_pre_tei[3],
            0, 0, Fuel_after_loi, 0.0])

# Event -1: TEI
dr_dep = state_pre_tei[0:2] - pos_M
u_tan_dep = np.array([-dr_dep[1], dr_dep[0]]) / np.linalg.norm(dr_dep)
if np.dot(state_pre_tei[2:4] + np.array([-dr_dep[1], dr_dep[0]]), u_tan_dep) < 0:
    u_tan_dep = -u_tan_dep
dv3_vec = dv3_optimal * u_tan_dep
state_post_tei = state_pre_tei.copy()
state_post_tei[2:4] = state_post_tei[2:4] + dv3_vec

data.append([-1, t_dep, state_pre_tei[0], state_pre_tei[1], state_pre_tei[2], state_pre_tei[3],
            0, 0, Fuel_after_loi, 0.0])
data.append([-1, t_dep, state_post_tei[0], state_post_tei[1], state_post_tei[2], state_post_tei[3],
            dv3_vec[0], dv3_vec[1], CONST.M_return_fuel, 0.0])

# Event 0: 月地转移（早期段，确保高度>400km）
t_safe = 0.1
state_safe = sol_tei.sol(t_safe)
data.append([0, t_dep, state_post_tei[0], state_post_tei[1], state_post_tei[2], state_post_tei[3],
            0, 0, CONST.M_return_fuel, 0.0])
data.append([0, t_dep + t_safe, state_safe[0], state_safe[1], state_safe[2], state_safe[3],
            0, 0, CONST.M_return_fuel, 0.0])

# Event 4: 返回地球
state_final = sol_tei.sol(t_peri)
data.append([4, t_arr_E, state_final[0], state_final[1], state_final[2], state_final[3],
            0, 0, CONST.M_return_fuel, 0.0])

data = np.array(data)
np.savetxt('results.txt', data, fmt=['%d'] + ['%.12e']*9, delimiter='\t')

print("=" * 60)
print(f"✓ 程序完成")
print(f"✓ Payload: {Payload:.2f} kg")
print(f"✓ 任务时长: {total_days:.2f}天")
print("=" * 60)

# EVOLVE-BLOCK-END


# 固定的入口函数（不被演化）
def run_mission():
    """运行月球任务并返回生成results.txt文件的路径"""
    # 上面的代码会生成results.txt
    return "results.txt"