import json
import sys
import numpy as np

# ==========================================
# 物理与医疗常数定义
# ==========================================
C_CTV = np.array([0.0, 0.0, 50.0])   # 肿瘤中心坐标
R_CTV = 15.0                         # 肿瘤半径
C_OAR = np.array([0.0, 20.0, 60.0])  # 脑干(危及器官)中心坐标
R_OAR = 10.0                         # 脑干半径

SIGMA_XY = 5.0                       # 横向散射展宽
SIGMA_Z = 3.0                        # 纵向布拉格峰展宽
D_RX = 60.0                          # 肿瘤处方剂量
D_LIMIT = 20.0                       # 脑干耐受极值

LAMBDA_OAR = 10.0                    # OAR 过量惩罚权重
LAMBDA_W = 0.05                      # 束流权重惩罚

# ==========================================
# 预先构建 3D 采样网格 (加速计算)
# ==========================================
# 在包含 CTV 和 OAR 的空间内建立 2mm 精度的体素网格
x_range = np.arange(-20.0, 22.0, 2.0)
y_range = np.arange(-20.0, 36.0, 2.0)
z_range = np.arange(30.0, 76.0, 2.0)

# X, Y, Z 的 shape 将为 (len(x), len(y), len(z))
X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

# 预先计算出处于 CTV 和 OAR 内部的体素掩码 (Boolean Masks)
MASK_CTV = (X - C_CTV[0])**2 + (Y - C_CTV[1])**2 + (Z - C_CTV[2])**2 <= R_CTV**2
MASK_OAR = (X - C_OAR[0])**2 + (Y - C_OAR[1])**2 + (Z - C_OAR[2])**2 <= R_OAR**2

def evaluate_plan(plan_data):
    """
    主评估函数：计算三维网格上的剂量分布并打分。
    """
    spots = plan_data.get("spots", [])
    
    if not spots:
        raise ValueError("The 'spots' list is empty.")
    if len(spots) > 100:
        raise ValueError("The number of spots exceeds the maximum limit of 100.")
        
    # 初始化整个三维空间的剂量分布为 0
    total_dose_grid = np.zeros_like(X, dtype=np.float64)
    total_weight = 0.0
    
    # 叠加每一束质子射线的剂量核
    for spot in spots:
        x_j = float(spot.get('x', 0.0))
        y_j = float(spot.get('y', 0.0))
        z_j = float(spot.get('z', 0.0))
        w_j = max(0.0, float(spot.get('w', 0.0))) 
        
        total_weight += w_j
        
        if w_j == 0:
            continue
            
        # 1. 计算横向衰减项 (XY plane)
        dist_sq_xy = (X - x_j)**2 + (Y - y_j)**2
        term_xy = np.exp(-dist_sq_xy / (2.0 * SIGMA_XY**2))
        
        # 2. 计算纵向深度项 (Z axis) - 布拉格峰物理模型
        dist_sq_z = (Z - z_j)**2
        term_z = 0.2 + 0.8 * np.exp(-dist_sq_z / (2.0 * SIGMA_Z**2))
        
        # 质子停靠后 (Z > z_j) 剂量急剧降为 0
        term_z[Z > z_j] = 0.0
        
        # 3. 剂量叠加
        total_dose_grid += w_j * term_xy * term_z

    # ==========================================
    # 计算评估指标 (Metrics)
    # ==========================================
    # 提取靶区和危及器官内的体素剂量
    dose_ctv = total_dose_grid[MASK_CTV]
    dose_oar = total_dose_grid[MASK_OAR]
    
    # 1. CTV 处方剂量覆盖惩罚 (均方误差)
    if len(dose_ctv) > 0:
        p_ctv = np.mean((dose_ctv - D_RX)**2)
    else:
        p_ctv = 10000.0 # 理论上不会发生，网格必然覆盖CTV
        
    # 2. OAR 过量辐射惩罚 (只惩罚超过限制的部分)
    if len(dose_oar) > 0:
        overdose = np.maximum(0.0, dose_oar - D_LIMIT)
        p_oar = np.mean(overdose**2)
    else:
        p_oar = 0.0
        
    # 3. 机器跳数惩罚
    p_w = total_weight
    
    # 计算最终得分
    final_score = 100.0 - (p_ctv + LAMBDA_OAR * p_oar + LAMBDA_W * p_w)
    
    return {
        "score": float(final_score),
        "status": "success",
        "metrics": {
            "ctv_mse": float(p_ctv),
            "oar_overdose_penalty": float(p_oar),
            "total_weight": float(total_weight)
        }
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "score": 0.0, 
            "status": "error", 
            "message": "Usage: python evaluator.py <path_to_plan.json>"
        }))
        sys.exit(1)
        
    plan_file = sys.argv[1]
    
    try:
        with open(plan_file, 'r') as f:
            plan_data = json.load(f)
            
        result = evaluate_plan(plan_data)
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "score": 0.0, 
            "status": "error", 
            "message": f"Evaluation failed: {str(e)}"
        }
        print(json.dumps(error_result))