# 任务：调强质子治疗 (IMPT) 剂量权重优化 (Proton Therapy Dose Optimization)

## 1. 物理与医疗工程背景
质子治疗是目前世界上最先进、最昂贵的放射治疗技术之一。传统 X 射线在穿透人体时会持续释放能量，对沿途的健康组织造成广泛损伤。而高能质子束具有独特的“布拉格峰（Bragg Peak）”物理特性：质子在穿透组织的前段释放极少能量（坪区），而在达到特定深度时瞬间释放出绝大部分能量，随后剂量急剧下降为零。

**真实的工程与医疗挑战**：
为了覆盖三维立体的肿瘤，现代质子加速器采用“笔形束扫描（Pencil Beam Scanning, PBS）”技术 。加速器通过改变磁场和质子初始能量，将成百上千个微小的质子“点（Spots）”打在肿瘤的三维网格上。
1. **肿瘤控制（Target Coverage）**：肿瘤区域（CTV）必须接收到足够且均匀的处方剂量（Prescription Dose），以彻底杀灭癌细胞。
2. **危及器官保护（OAR Sparing）**：肿瘤周围通常紧贴着极度敏感的健康器官（Organ at Risk, 比如脑干）。一旦这些器官的受照剂量超过容忍阈值，将导致瘫痪、失明等不可逆的医疗事故。
3. **机器跳数成本（Monitor Units Cost）**：为了延长几亿美元加速器的寿命并提高患者周转率，治疗方案应尽量用更少的总质子束流（权重之和）来实现目标。

本任务要求 AI Agent 在复杂的三维解剖结构中，挑选质子笔形束的投射点并优化它们的束流权重（Spot Weights），在“杀灭肿瘤”和“保护脑干”之间找到最佳的数学平衡。

## 2. 物理模型与环境设定
我们将患者脑部简化为一个三维坐标系（单位：**mm**）。质子束统一从正上方（$z=0$ 的皮肤表面）向下（$Z$ 轴正方向）照射。
* **临床靶区 (CTV, 肿瘤)**：一个中心位于 $(0, 0, 50)$，半径为 **15 mm** 的球体。
* **危及器官 (OAR, 脑干)**：一个中心位于 $(0, 20, 60)$，半径为 **10 mm** 的球体。（注意：OAR 与 CTV 距离极近，极易受到误伤）。

**智能体动作 (Agent Action)**：
Agent 需要输出一个包含 $N$ 个质子束斑（Spots）的列表（最大允许 $N = 100$ 个）。每个束斑 $j$ 由 4 个参数定义：$(x_j, y_j, z_j, w_j)$。
* $(x_j, y_j, z_j)$：质子束的布拉格峰（能量最高点）在三维空间中的物理停靠坐标。
* $w_j$：该质子束的权重（流强强度），要求 $w_j \ge 0$。

## 3. 目标函数 (Objective Function)
空间中任意一点 $P = (x, y, z)$ 接收到的总剂量 $D(P)$ 是所有质子束在该点产生的剂量叠加：
$$D(P) = \sum_{j=1}^{N} w_j \cdot K(P, S_j)$$

其中，$K(P, S_j)$ 是单束质子（停靠在 $S_j=(x_j,y_j,z_j)$）在点 $P$ 产生的剂量核函数，模拟了布拉格峰现象：
$$K(P, S_j) = \exp\left(-\frac{(x-x_j)^2 + (y-y_j)^2}{2\sigma_{xy}^2}\right) \cdot B(z, z_j)$$

轴向深度沉积 $B(z, z_j)$ 的计算规则为：
* 如果 $z \le z_j$（质子行进途中及到达峰值）：$B(z, z_j) = 0.2 + 0.8 \cdot \exp\left(-\frac{(z-z_j)^2}{2\sigma_z^2}\right)$。其中 **0.2** 代表坪区（Plateau）造成的路过剂量。
* 如果 $z > z_j$（质子停止后）：$B(z, z_j) = 0.0$。
常数设定：横向散射展宽 $\sigma_{xy} = 5.0$，纵向布拉格峰展宽 $\sigma_z = 3.0$。

**最终得分 $J(W)$ 将最大化以下公式（即最小化各项惩罚）**：
$$J(W) = 100.0 - \left( P_{\text{CTV}} + \lambda_{\text{OAR}} P_{\text{OAR}} + \lambda_{w} P_{w} \right)$$
* **CTV 覆盖惩罚**：$P_{\text{CTV}} = \frac{1}{V_{\text{CTV}}} \sum_{P \in \text{CTV}} (D(P) - D_{\text{rx}})^2$。目标处方剂量 $D_{\text{rx}} = 60.0$ Gy。
* **OAR 过量惩罚**：$P_{\text{OAR}} = \frac{1}{V_{\text{OAR}}} \sum_{P \in \text{OAR}} \max(0, D(P) - D_{\text{limit}})^2$。脑干最大耐受限制 $D_{\text{limit}} = 20.0$ Gy。
* **机器损耗惩罚**：$P_{w} = \sum w_j$（鼓励用最少的射线量）。
* **权重**：$\lambda_{\text{OAR}} = 10.0$（保护健康器官极其重要），$\lambda_{w} = 0.05$。

## 4. 智能体交互规范 (I/O Specification)

### 4.1 输出格式 (Agent's Action)
Agent 必须生成一个 `plan.json` 文件，至多包含 100 个 spots。坐标尽量覆盖在 CTV 区域附近。
```json
{
  "spots": [
    {"x": 0.0, "y": 0.0, "z": 50.0, "w": 25.5},
    {"x": 5.0, "y": -5.0, "z": 52.0, "w": 18.2}
  ]
}

### 4.2 环境反馈 （Environment Feedback）
{
  "score": 68.5,
  "status": "success",
  "metrics": {
    "ctv_mse": 15.2,
    "oar_overdose_penalty": 2.5,
    "total_weight": 43.7
  }
}