# Engineering Bench Case 2

## 问题背景

2025 集成电路计算机辅助设计(EDA)——物理设计优化

## 1. 题目背景

随着半导体工艺迈入纳米级（如7nm及以下），超大规模集成电路（VLSI）的设计复杂度呈指数级增长。在物理设计的后期阶段（Late-stage Placement Optimization），时序（Timing）、功耗（Power）和面积（Area）的权衡变得极为严苛。传统的基于CPU的启发式算法在处理包含数百万个节点的离散组合系统时，往往陷入局部最优，难以在有限时间内找到全局最优解。
为了突破这一瓶颈，利用现代高性能计算（HPC）技术（如GPU加速、可微编程）进行电路优化成为工业界和学术界的研究热点。本任务“增量布局优化”为背景，要求在地月级（Million-gate）规模的芯片网表上，利用逻辑门尺寸调整、缓冲器插入和单元移动等手段，在严格的物理约束下修复时序违例并降低静态功耗。

## 2. 题目描述

为了模拟真实的工业场景，任务给定一个已经完成布局布线（Place & Route）但处于时序不满足或功耗过高状态的芯片设计。参赛者需要开发一个优化程序，对电路进行“微创手术”。

### 2.1 任务主要流程

- 加载设计：读取基于工业标准格式（Verilog, DEF, LEF, Liberty）的电路网表和物理库。
- 增量优化：在保持电路逻辑功能不变的前提下，通过调整门尺寸、插入缓冲器或移动单元位置，优化电路的PPA（Power, Performance, Area）指标。
- 合法化与输出：确保所有修改在物理上是合法的（无重叠、对齐网格），并输出最终的布局文件（DEF）及工程变更单（ECO）。

### 2.2 流程细节补充说明

- A. 逻辑门尺寸调整 (Gate Sizing)：
  可以将网表中的实例（Instance）替换为库中功能相同但驱动能力不同（如从 X1 变为 X4）的单元。限制：新旧单元必须功能兼容（Footprint Compatible），且来自同一个逻辑库。
- B. 缓冲器插入 (Buffer Insertion)：
  允许在连线上插入缓冲器以解耦长导线的电容负载或修复时序。限制：插入的缓冲器必须有物理空间放置，且不能引起严重的局部拥塞。
- C. 单元重定位 (Cell Relocation)：
  允许在局部范围内微调单元的坐标 $(x, y)$。注意：尺寸调整通常会引起面积变化，进而导致周围单元重叠。必须通过移动周围单元（多米诺骨牌效应）来消除重叠，即实现“合法化”（Legalization）。

## 3. 设计环境与输入定义

工艺节点：采用 ASAP7 PDK（7nm 预测工艺库），包含真实的多阈值电压（RVT, LVT, SLVT）单元。

输入文件：

- Verilog Netlist (.v)：电路的逻辑连接关系（骨架）。
- DEF (.def)：电路的物理布局信息，包含所有单元的坐标、方向和放置状态（肉体）。
- Liberty Library (.lib)：标准单元的时序和功耗模型（物理法则）。
- Parasitics (.spef)：互连线的寄生电阻电容参数。

## 4. 物理模型与计算公式

本任务涉及静态时序分析（STA）和功耗计算，需遵循以下物理模型。

### 4.1 延迟模型 (Delay Model)

电路延迟由门延迟和互连线延迟组成。

互连线延迟：使用 Elmore 延迟模型。对于电阻为 $R$、电容为 $C$ 的导线，驱动负载 $C_{load}$ 的延迟近似为：

$$
\tau = R \cdot (\frac{C}{2} + C_{load})
$$

门延迟 (NLDM)：基于非线性延迟模型（NLDM），通过二维查找表计算。门延迟 $D_{gate}$ 和输出转换时间 $S_{out}$ 为输入转换时间 $S_{in}$ 和负载电容 $C_{out}$ 的函数：

$$
D_{gate} = f(C_{out}, S_{in})
$$

$$
S_{out} = g(C_{out}, S_{in})
$$

### 4.2 功耗模型 (Power Model)

重点关注静态漏功耗（Leakage Power），计算公式如下：

$$
P_{total} \approx P_{leakage} = \sum_{i \in Cells} I_{leak}(Cell_i, State_i) \cdot V_{DD}
$$

其中 $I_{leak}$ 与晶体管阈值电压 $V_{th}$ 呈指数关系。优化目标是用高 $V_{th}$（低功耗、慢）的门替换低 $V_{th}$（高功耗、快）的门。

### 4.3 物理位移 (Displacement)

增量优化带来的平均位移量定义为：

$$
D = \sum_{i \in MovedCells} (|x_i - x_{i,initial}| + |y_i - y_{i,initial}|)
$$

## 5. 评分标准 (Cost Function)

基准测试的评分由收益项 (Reward) 和 惩罚项 (Penalty) 组成。

$$
Score = Reward - Penalty
$$

### 5.1 收益计算

收益基于相对于初始基准线（Baseline）的优化幅度。

$$
Reward = 1000 \cdot (0.6 \cdot \frac{\Delta TNS}{|TNS_{base}|} + 0.3 \cdot \frac{\Delta Lkg}{P_{base}} + 0.1 \cdot \frac{\Delta Area}{A_{base}})
$$

TNS (Total Negative Slack)：总负时序裕量，代表电路时序违例的总和（主要优化目标）。
Lkg (Leakage)：总漏功耗。
Area：芯片总面积。

### 5.2 修正评分公式 (ICCAD 2025)

为了限制过度的布局变动，实际评分引入了位移惩罚项：

$$
S_{final} = S_{reward} - \lambda_{disp} \cdot \sum Dist(cell_i)
$$

## 6. 约束条件

- 时序约束：建立时间 (Setup Time)：$AT_{data} + T_{setup} \le T_{clk} + T_{skew}$。优化后的 WNS (Worst Negative Slack) 和 TNS 必须得到改善或至少不恶化。
- 物理合法性 (Legalization)：所有标准单元必须放置在预定义的行（Row）上。单元坐标必须是 Site 宽度的整数倍。单元之间严禁重叠。
- 设计规则检查 (DRC)：Max Capacitance：节点的负载电容不能超过库定义的上限。Max Slew：信号转换时间不能超过上限。惩罚：每发生一次 DRC 违例，扣除 50分。
- 运行时间限制：优化过程必须在规定的时间预算内完成（例如对应规模电路需在1小时内完成），超时将导致无法获得完整分数。

## 7. 结果文件格式

为了验证计算结果的正确性，参赛程序需输出以下标准格式文件。

### 7.1 输出文件

- Solution DEF (.def)：包含优化后所有单元最终位置和连接关系的完整布局文件。
- ECO Changelist：记录所有网表变更（Resize, Insert Buffer, Move）的文本列表，用于重放验证。

### 7.2 验证指标说明

评分脚本将基于 OpenROAD/OpenSTA 工具产生的报告进行解析，主要关注以下指标：

| 指标 (Metric) | 单位 | 说明 | 期望目标 |
| --- | --- | --- | --- |
| WNS | ns | 最差负时序裕量 | $> -0.100$ ns |
| TNS | ns | 总负时序裕量 | 消除 98% 以上 |
| Leakage Power | mW | 静态漏功耗 | 降低 8-10% |
| Avg Displacement | $\mu m$ | 平均单元位移 | 尽可能小 (< 5 $\mu m$) |
| DRC Violations | Count | 设计规则违例数 | 0 |

注意：如果输出的 DEF 文件无法被 OpenROAD 正确解析，或存在未修复的物理重叠（Illegal Placement），该测试用例得分将直接记为 0分。
