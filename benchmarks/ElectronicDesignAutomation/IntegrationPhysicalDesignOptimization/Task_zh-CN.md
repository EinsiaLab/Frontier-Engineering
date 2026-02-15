# 增量布局优化：超越详细布局的门级尺寸调整、缓冲器插入与单元重定位同步优化

**作者：** Yi-Chen Lu, Rongjian Liang, Wen-Hao Liu, and Haoxing (Mark) Ren  
**所属机构：** NVIDIA

## 0. 修订历史
* 2025-09-03：更新运行时间评分标准
* 2025-07-30：修正少量拼写错误
* 2025-07-23：更新提交及输出格式要求
* 2025-06-21：更新环境配置；评分细则说明
* 2025-05-06：环境配置更新
* 2025-05-05：完善评分细则说明

## 1. 引言
增量布局优化是现代物理设计（PD）中的关键阶段，其目标是精炼初始布局后的功耗、性能和面积（PPA）指标。然而，传统的增量布局流程本质上是迭代式的，依赖于多轮物理优化（包括门级尺寸调整、缓冲器插入和单元重定位），且每轮之后都需进行静态时序分析（STA）来验证 PPA 的改进。这一过程不仅耗时，而且由于每项新的变换都必须兼顾先前的修改和约束，导致其本质上是次优的。例如，解决一条关键路径的优化可能会恶化另一条路径或违反合法性约束。因此，迫切需要一种更具整体性、系统性的全局 PPA 优化方法。

本次竞赛专门针对这些挑战，鼓励参赛者超越局部的、临时的调整。竞赛邀请参赛者采用整合了门级尺寸调整（Gate Sizing）、缓冲器插入（Buffering）和单元重定位（Cell Relocation）的统一方法，而非局限于孤立的物理调整。通过集成这些技术，本次竞赛旨在释放比传统增量流程更高的设计质量，从而在时序、功耗和线长之间实现更好的权衡。重要的是，最终的布局方案必须在实现全局改进的同时，严格遵守布局合法性和设计规则。

为了应对现实世界设计的规模并进行大规模全局 PPA 优化，强烈建议（但不强制）参赛者探索现代机器学习（ML）基础设施（如 PyTorch [1]），利用其通过 GPU 加速反向传播提供的梯度优化能力。近年来，此类技术在处理大规模 PD 优化问题中展现了巨大潜力 [2][3][4][5]。虽然 GPU 或 ML 的使用是可选的，但参赛者应意识到，创造性地应用这些技术可能在方案质量和运行时间上获得显著优势。然而，核心重点依然在于通过任何高效手段实现超越传统工具的最佳 PPA 改进。

## 2. 竞赛目标
本次竞赛的目标是通过协调以下方法来增强初始种子布局的 PPA：
1. **单元重定位：** 实现有效且干扰最小（即位移最小）的全局单元移动（包括标准单元和宏单元）。
2. **门级尺寸调整：** 选择备选的库单元变体（例如不同的驱动强度、面积或阈值电压（VT）类型）以实现最佳的时序-功耗平衡。
3. **缓冲器插入：** 在必要位置引入缓冲器（包括反相器对），以最小的面积开销降低关键路径或高扇出净线的延迟。

请注意，对种子布局的所有修改必须产生合法的物理设计。即：单元必须位于有效位点上，不得重叠，且 IO 端口位置必须保持不变。此外还将进行网表功能性检查。

## 3. 问题定义
### 3.1 输入格式
参赛者将获得解决增量布局优化挑战所需的所有文件。设计数据采用标准 EDA 文件格式：
* **Verilog 网表 (.v)：** 定义逻辑连接性的门级网表。
* **DEF 文件 (.def)：** 初始布局的完整设计交换格式文件。
* **Bookshelf 文件 [6]：**
    * `.nodes`：设计实例列表及其尺寸。
    * `.nets`：引脚间的连接信息。
    * `.pl`：种子布局的单元坐标。
    * `.scl`：定义版图规划（如核心区域和布局栅格）。
* **ASAP7 库文件：** 本次竞赛使用 ASAP7 库，提供：
    * `.lib`：包含转换时间/延迟/功耗查找表的时序和功耗库。
    * `.lef`：描述物理尺寸的技术和单元 LEF 文件。

竞赛将提供一个示例设计集（包括完整的 Bookshelf 格式设计和相应的 ASAP7 库文件）作为参考。

### 3.2 输出格式
参赛者需提交更新后的 DEF 格式布局文件。主要输出包括：
1. 更新后的 **.def 文件**，反映应用单元重定位、尺寸调整和缓冲后的最终方案。
2. 一份 **ECO 变更列表 (Changelist)**，按顺序详述尺寸调整和缓冲变换：
    * **a. 尺寸调整命令：**
        * `size_cell <cellName> <libCellName>`
        * *说明：* `<cellName>` 必须存在于原始网表中；`<libCellName>` 必须存在于库中；新单元必须与原单元功能相同。
    * **b. 缓冲命令：**
        * `insert_buffer <{一个或多个负载引脚名称}> <库缓冲器名称> <新缓冲器实例名> <新缓冲净线名>`
        * `insert_buffer -inverter_pairs {一个或多个负载引脚名称} <库反相器名称> {<新反相器实例名列表>} {<新净线名列表>}`
        * *说明：* 遵循 Synopsys PrimeTime 格式 [7]。缓冲包括插入缓冲器和反相器对。新实例名必须与更新后的 .pl 文件匹配。对于反相器对，需指定每个反相器的名称及新创建的净线名称。

参赛者必须提交一份变更列表文件（可以为空）。命令将自上而下处理，后执行的变换会覆盖之前的。任何无效的 ECO 命令将被跳过。

输出必须满足以下标准：
* **合法布局：** 单元必须位于有效位点且无重叠。
* **固定 IO 端口：** IO 端口及任何固定的宏单元/单元位置不得改变。
* **库一致性：** 输出文件应与输入网表及库约束完全一致。

### 3.3 评分与评估指标
提交的作品将根据 PPA 改进情况进行综合评分，同时对过度的单元位移和低效的运行时间进行惩罚。总分 $S$ 的计算组件如下：

**PPA 改进 (P)：**
* **时序改进 ($TNS_{norm}$):**
  $$TNS_{norm}=\frac{TNS_{solution}-TNS_{seed}}{|TNS_{seed}|}$$
* **功耗降低 ($Power_{norm}$):**
  $$Power_{norm} = \frac{Power_{solution} - Power_{seed}}{Power_{seed}}$$
* **线长减少 ($WL_{norm}$):**
  $$WL_{norm}=\frac{WL_{solution}-WL_{seed}}{WL_{seed}}$$

综合 PPA 改进得分 $P$ 计算如下：
$$P=\alpha \cdot TNS_{norm} + \beta \cdot Power_{norm} + \gamma \cdot WL_{norm}$$
其中权重因子 $\alpha, \beta, \gamma$ 将根据具体测试用例而定。PPA 指标将由 OpenROAD [8] 验证。

**平均位移惩罚 (D)：**
为防止过度的设计扰动，计算每个单元的平均曼哈顿位移（以有效位点为单位）。
* $D$ = 每个单元的平均曼哈顿位移（相对于种子布局进行归一化）。

**运行时间效率 (R)：**
算法的总运行时间将针对每个测试用例的预定义参考值进行归一化。

### 3.4 每个设计的最终得分计算 (S)
最终综合得分 $S$ 定义为：
$$S=1000 \times P - 50 \times D - 30 \times R$$

所有 $P, D, R$ 均可视为相对于种子指标的“改进百分比”。如前所述，GPU 加速技术是竞赛鼓励的方向，因此运行时间表现是最终评估的关键因素。

## 4. 评估环境
所有提交将在基于提供的 Dockerfile 构建的容器中运行（Ubuntu 20.04）：
* **CPU:** 8x Intel Xeon vCPUs
* **RAM:** 32 GB
* **GPU:** 1 NVIDIA A100 (40 GB HBM2)
* **CUDA:** 11.8
* **环境:** 通过 `lagrange_env.yaml` 安装的 Python 3.9 (Mambaforge)

## 5. 提交与输出格式
提交名为 `solution.tar.gz` 的压缩包，解压后目录结构如下：
```text
solution/ 
├── setup_environment.sh  <-- 首先执行（安装依赖）
├── run.sh                 <-- 其次执行（运行方案）
├── <其他文件夹/文件/二进制等>
└── testcases/            <-- (由主办方提供)

```

`run.sh` 需接受四个参数：
`./run.sh <design_name> <TNS_weight, α> <power_weight, β> <WL_weight, γ>`

脚本需在 `solution/` 目录下生成：

* `<design_name>.sol.def`
* `<design_name>.sol.changelist`

## 6. 参考文献

[1] Imambi, Sagar, Kolla Bhanu Prakash, and G. R. Kanagachidambaresan. "PyTorch." Programming with TensorFlow: solution for edge computing applications (2021): 87-104. 
[2] Du, Yufan, et al. "Fusion of Global Placement and Gate Sizing with Differentiable Optimization." ICCAD. 2024. 
[3] Guo, Zizheng, and Yibo Lin. "Differentiable-timing-driven global placement." Proceedings of the 59th ACM/IEEE Design Automation Conference. 2022. 
[4] Lin, Yibo, et al. "Dreamplace: Deep learning toolkit-enabled gpu acceleration for modern visi placement." Proceedings of the 56th ACM/IEEE Annual Design Automation Conference 2019. 
[5] Lu, Yi-Chen, et al. "INSTA: An Ultra-Fast, Differentiable, Statistical Static Timing Analysis Engine for Industrial Physical Design Applications" Proceedings of the 62th ACM/IEEE Annual Design Automation Conference 2025. [6] Bookshelf format reference: http://vlsicad.eecs.umich.edu/BK/ISPD06bench/BookshelfFormat.txt 
[7] PrimeTime User Guide, Advanced Timing Analysis. V-2018.03, Synopsys Online Documentation 
[8] Ajayi, Tutu, and David Blaauw. "OpenROAD: Toward a self-driving, open-source digital layout implementation tool chain." Proceedings of Government Microcircuit Applications and Critical Technology Conference. 2019. 
