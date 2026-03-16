import json
import numpy as np
import os

def generate_baseline_plan():
    """
    生成一个基于启发式规则的基线质子治疗计划。
    策略：在肿瘤内部均匀布点，但严格避开靠近脑干的危险区域。
    """
    spots = []
    
    # CTV (肿瘤) 参数
    c_ctv = np.array([0.0, 0.0, 50.0])
    r_ctv = 15.0
    
    # OAR (脑干) 参数
    c_oar = np.array([0.0, 20.0, 60.0])
    r_oar = 10.0
    
    # 在肿瘤内部生成候选网格点 (步长为 6mm，确保点数不超过 100)
    xs = np.arange(-12, 13, 6)
    ys = np.arange(-12, 13, 6)
    zs = np.arange(38, 63, 6)
    
    for x in xs:
        for y in ys:
            for z in zs:
                pos = np.array([x, y, z])
                
                # 1. 检查是否在肿瘤深处 (稍微向内收缩边界，半径设为 13)
                if np.linalg.norm(pos - c_ctv) <= 13.0:
                    
                    # 2. 检查距离脑干的距离
                    dist_to_oar = np.linalg.norm(pos - c_oar)
                    
                    # 如果距离脑干太近 (半径 10 + 6mm 的安全缓冲区)，则放弃该束斑
                    if dist_to_oar < 16.0:
                        continue
                        
                    # 3. 添加安全的束斑，赋予初始均匀权重 4.5
                    spots.append({
                        "x": float(round(x, 2)),
                        "y": float(round(y, 2)),
                        "z": float(round(z, 2)),
                        "w": 4.5
                    })
    
    # 按照任务要求，截断至最多 100 个点
    spots = spots[:100]
    
    # 组装 JSON 结构
    plan_data = {
        "spots": spots
    }
    
    # 确保保存路径正确
    output_path = os.path.join(os.path.dirname(__file__), 'plan.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(plan_data, f, indent=2)
        
    print(f"Successfully generated baseline plan with {len(spots)} spots.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_baseline_plan()