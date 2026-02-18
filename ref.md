# Frontier-Eng 建题注意事项 & 美赛 → Bench 转换指南

> 本文档记录了在创建 ISCSO 2015 / 2023 结构优化 benchmark 过程中的经验教训，以及如何将美赛（MCM/ICM）题目转化为本框架可用的 benchmark 格式。

---

## 一、框架核心概念

### 1.1 `combined_score`：唯一排名指标

OpenEvolve 框架通过 `metrics["combined_score"]` 对所有候选程序排序，**越大越好**。

```python
# 在 algo.py 中最终取分逻辑：
score = metrics.get("combined_score", metrics.get("score", None))
```

因此：

| 问题类型 | combined_score 计算 | 说明 |
|----------|---------------------|------|
| **最大化**（如 MLL 运载质量） | `combined_score = payload` | 直接用原始值 |
| **最小化**（如 ISCSO 结构重量） | `combined_score = -weight` | 取负，使"越轻→越好→越大" |
| **不可行 / 非法解** | `combined_score = 0.0`（最大化）或 `-1e18`（最小化） | 必须远差于任何合法解 |

> ⚠️ **常见坑**：最小化问题忘记取负号，导致越差的解反而排名越高。ISCSO 早期版本就犯过这个错误。

### 1.2 `valid` 标志

- `valid = 1.0`：解通过所有约束检验
- `valid = 0.0`：不可行或出错

### 1.3 文件管线

```
scripts/init.py          ← OpenEvolve 演化的入口（EVOLVE-BLOCK-START/END 标记）
    ↓ 执行
submission.json / results.txt  ← 候选程序输出的结果文件
    ↓ 读取
verification/evaluator.py      ← 评分脚本，返回 EvaluationResult(metrics, artifacts)
```

---

## 二、建题清单（Checklist）

建一个新 benchmark 需要完成以下组件。每一项都有对应的格式要求。

### 2.1 目录结构

```
benchmarks/<Domain>/<TaskName>/
├── Task.md                        # 完整题目描述（数学公式、变量定义、约束、目标函数）
├── Task_zh-CN.md                  # 中文版（可选但建议）
├── README.md / README_zh-CN.md    # 导航文档：文件结构、如何运行
├── references/
│   ├── problem_data.json          # 所有问题参数的机器可读 JSON
│   └── (可选) 手册、论文 PDF
├── verification/
│   ├── evaluator.py               # 评分核心逻辑
│   ├── (可选) fem_xxx.py 等求解器
│   ├── requirements.txt
│   └── docker/Dockerfile
├── scripts/
│   └── init.py                    # OpenEvolve 演化入口 = baseline 的副本
├── baseline/
│   ├── solution.py                # 参考解法（与 init.py 内容一致）
│   └── result_log.txt             # 基线运行结果记录
```

### 2.2 init.py / solution.py 格式

必须满足以下要求：

1. **顶层执行**：不能包装在 `if __name__ == "__main__"` 或 `main()` 里。OpenEvolve 需要直接 `exec` 或 `subprocess.run` 这个文件。
2. **EVOLVE-BLOCK 标记**：文件以 `# EVOLVE-BLOCK-START` 开头、`# EVOLVE-BLOCK-END` 结尾。LLM 只会修改这两个标记之间的代码。
3. **输出文件**：在当前工作目录生成 `submission.json`（或 `results.txt`，取决于 evaluator 的约定）。
4. **自包含**：所有需要的 FEM 求解器、物理模型等应该**内联**在文件中，不要依赖 `import fem_truss2d` 这种外部模块（因为 OpenEvolve 只会修改这个文件，不会搬运其他模块）。
5. **可用 pip 包**：`numpy`, `scipy`, `json` 等标准包可以正常 import。

### 2.3 evaluator.py 格式

```python
def evaluate(program_path: str, *, repo_root: Path | None = None) -> Any:
    """
    1. 在临时目录中 subprocess.run 候选程序
    2. 读取输出文件 (submission.json)
    3. 加载 problem_data.json
    4. 调用求解器/检验器
    5. 计算 metrics dict
    6. 返回 EvaluationResult(metrics, artifacts)
    """
```

关键点：

- **临时目录隔离**：每次评估在独立 `tempfile.mkdtemp()` 中运行，评估结束后 `shutil.rmtree`。
- **timeout**：`subprocess.run(..., timeout=600)` 防止候选程序死循环。
- **problem_data.json 需要拷贝到工作目录**：因为候选程序可能用相对路径 `Path("references/problem_data.json")` 来读取。
- **返回值**：优先尝试 `from openevolve.evaluation_result import EvaluationResult`；如果 import 失败则退回普通 dict。

### 2.4 frontier_eval 注册

```
frontier_eval/tasks/<task_name>/
├── __init__.py
├── task.py                 # Task 子类：initial_program_path() + evaluate_program()
└── evaluator/
    ├── __init__.py
    └── evaluate.py          # 桥接到 benchmarks/.../verification/evaluator.py
```

还需要在以下位置注册：

- `frontier_eval/registry_tasks.py`
- `frontier_eval/conf/task/<task_name>.yaml`

---

## 三、ISCSO 建题过程中踩过的坑

### 3.1 FEM 求解器必须独立验证

不要假设代码生成的 FEM 求解器是正确的。必须用已知解析解或参考解进行交叉验证：

- 用**单杆桁架**验证应力 = F/A
- 用**三角桁架**验证已知解析位移
- 用**对称结构 + 对称荷载**验证位移对称性

### 3.2 约束限值需要可行性标定

问题定义中的约束（应力限值、位移限值）需要先验证可行性：

- 先用**全最大截面积**跑一遍，确认在这种"材料最多"的情况下约束可以被满足
- 如果全最大截面都不可行，说明约束太紧（ISCSO 2023 的 `displacement_limit` 最初设为 5.0mm 对于 40m 塔来说太紧，后来改为 10.0mm）

### 3.3 solution_vector 维度必须明确

`problem_data.json` 中必须有明确的 `"dimension"` 字段。evaluator 第一步就检查 `len(x) == expected_dim`。

### 3.4 combined_score 方向

最小化问题一定要取负号：`combined_score = -weight`。不可行解要赋值为极小值 `-1e18`，而不是 `0.0`。

---

## 四、美赛 → Bench 转换指南

### 4.1 核心挑战：美赛缺乏单一可量化指标

美赛（MCM/ICM）题目的典型特征：

- **开放式建模**：没有给定数学公式，需要自己建模
- **多目标**：往往有多个互相矛盾的优化目标
- **定性约束**："合理""安全""可持续"等模糊表述
- **评判主观**：原始竞赛通过人工阅卷打分

本框架要求的是：**一个标量 `combined_score`，越大越好，且由 evaluator.py 自动计算**。

### 4.2 转换方法论

#### 步骤 1：从美赛题目中提取核心物理 / 数学模型

美赛题目通常包含一个或多个可数学化的子问题。选取其中**最具工程价值、约束最明确**的子问题。

**示例**：

| 美赛题目 | 可提取的核心模型 |
|----------|-----------------|
| 交通流优化 | 给定路网拓扑，最小化平均通行时间 |
| 传染病传播 | 给定 SIR 模型参数，最小化总感染人数 |
| 水资源分配 | 给定供需网络，最大化满足率或最小化调水成本 |
| 森林火灾消防 | 给定地形、风向，最小化扑灭时间或烧毁面积 |

#### 步骤 2：固化问题参数为 `problem_data.json`

美赛题目往往只给出模糊的场景描述。你需要：

1. **选择具体数据源**：从公开数据集、论文或合理假设中获取参数
2. **定义完整的输入空间**：节点、边、约束、材料属性等
3. **写死所有参数到 JSON**：不允许候选程序自行假设参数

```json
{
  "problem_id": "traffic_flow_2024",
  "description": "Minimize average travel time in a 50-node urban road network",
  "dimension": 50,
  "nodes": [...],
  "edges": [...],
  "demand_matrix": [...],
  "constraints": {
    "max_capacity_per_lane": 1800,
    "speed_limit_kmh": 60
  },
  "variable_bounds": {
    "signal_green_min": 10,
    "signal_green_max": 120
  }
}
```

#### 步骤 3：构造可量化的目标函数

这是最关键的一步。将美赛的多个评判标准**融合为一个标量**：

**方法 A：加权求和**

```
combined_score = -(w1 * cost + w2 * time + w3 * violation_penalty)
```

适合多个目标可以用同一量纲度量的情况。权重需要在 `problem_data.json` 中固定。

**方法 B：字典序 / 层次目标**

```
if not feasible:
    combined_score = -1e18
elif constraint_violation > tolerance:
    combined_score = -1e12 - total_violation
else:
    combined_score = -primary_objective  # 或 +primary_objective（最大化时）
```

先保证可行性，再优化主目标。这在工程问题中最常见。

**方法 C：约束转罚函数**

将软约束通过二次罚函数加入目标：

```python
raw_score = compute_raw_objective(x)
penalty = sum(max(g_i(x), 0)**2 for g_i in constraints) * PENALTY_COEFF
combined_score = raw_score - penalty  # 最大化
# 或
combined_score = -(raw_score + penalty)  # 最小化
```

#### 步骤 4：构建确定性验证器（evaluator）

美赛评分是主观的，但本框架的 evaluator 必须是**确定性的、可复现的**。关键要求：

1. **给定相同输入，永远返回相同分数**（不能有随机成分）
2. **物理约束要有明确数值判据**：不是"大致合理"，而是 `|stress| <= 250 MPa`
3. **验证独立于求解**：evaluator 中的求解器不依赖候选程序的代码，而是独立实现

#### 步骤 5：编写 baseline 解法

baseline 不需要是最优解，但需要：

1. **证明问题是可行的**：至少能找到一个满足所有约束的解
2. **给出 combined_score 的量级参考**：让 LLM 知道一个"还行"的分数大概是多少
3. **使用常见优化算法**：`scipy.optimize.differential_evolution`、遗传算法等。确保 LLM 有改进空间

### 4.3 美赛转换实操示例

假设美赛题目：*"为某城市设计公共自行车站点布局，使市民出行便利性最大化"*

**原始题目**（模糊）→ **本框架格式**（精确）：

| 原始描述 | 框架化定义 |
|----------|-----------|
| "某城市" | 具体的 10km×10km 网格，500 个候选站点位置已固定在 JSON |
| "公共自行车" | 每个站点容量 10-50 辆，总预算 200 个站点 |
| "出行便利性" | `combined_score = -平均步行距离（米）`，基于 10000 个需求点 |
| "合理布局" | 约束：任何需求点到最近站点 ≤ 800m，否则 penalty |

```json
{
  "problem_id": "bike_station_2024",
  "dimension": 500,
  "candidate_locations": [[x1,y1], [x2,y2], ...],
  "demand_points": [[dx1,dy1,w1], ...],
  "constraints": {
    "max_stations": 200,
    "max_distance_m": 800,
    "capacity_range": [10, 50]
  }
}
```

设计变量：长度 500 的向量，`x[i] > 0` 表示在候选位置 i 放置站点（容量 = x[i]），`x[i] = 0` 不放置。

```python
# evaluator 核心
active = x > 0
if sum(active) > max_stations:
    return {"combined_score": -1e18, "valid": 0.0}

avg_dist = compute_avg_distance(demand_points, candidate_locations[active])
max_dist = compute_max_min_distance(demand_points, candidate_locations[active])

if max_dist > 800:
    penalty = (max_dist - 800) ** 2 * 1e3
else:
    penalty = 0.0

metrics["combined_score"] = -(avg_dist + penalty)
```

### 4.4 美赛转换时的常见陷阱

| 陷阱 | 说明 | 解决方法 |
|------|------|---------|
| **指标不唯一** | 多目标无法统一排序 | 选最核心的 1 个主目标，其余转约束或罚函数 |
| **数据不完整** | 美赛只给模糊场景 | 自行构造或引用公开数据集，写死到 JSON |
| **评分不确定** | 同一输入 evaluator 给不同分数 | 消除所有随机性（固定种子或用解析公式） |
| **问题太大** | 候选程序跑不完 | 缩小规模（如 50 节点而非 5000），设合理 timeout |
| **约束太松** | 任何解都可行，没有区分度 | 收紧约束或增加约束维度 |
| **约束太紧** | 全最大资源都不可行 | 标定后放宽限值 |
| **baseline 太弱** | 随机解都能拿高分 | baseline 要用认真的优化算法跑出有意义的分 |
| **baseline 太强** | LLM 无法超越 | 限制 baseline 的迭代次数 / 种群大小 |

### 4.5 关于 `combined_score` 设计的黄金准则

1. **单调性**：对于主目标，更好的工程方案一定对应更高的 `combined_score`
2. **可行性壁垒**：不可行解的 `combined_score` 必须严格低于任何可行解
3. **灵敏度**：两个略有差异的方案应该得到不同的分数（避免量化为同一值）
4. **有界性**：分数不应该趋于 +∞（如果原始目标无界，做适当缩放）
5. **物理意义**：分数最好有工程含义（如 `-weight_kg`：绝对值就是结构重量公斤数）

---

## 五、Quick Reference：各题型 combined_score 模板

```python
# ==================== 最大化类（payload, throughput, accuracy）====================
if feasible:
    metrics["combined_score"] = float(objective_value)  # 正值，越大越好
else:
    metrics["combined_score"] = 0.0
    
# ==================== 最小化类（weight, cost, time, error）======================
if feasible:
    metrics["combined_score"] = -float(objective_value)  # 取负，越小→越好→越大
else:
    metrics["combined_score"] = -1e18                    # 极差

# ==================== 性能类（speedup, compression ratio）======================
if valid:
    metrics["combined_score"] = float(speedup_ratio)     # >1 表示更快
else:
    metrics["combined_score"] = 0.0

# ==================== 多目标加权 ================================================
if feasible:
    primary = -cost                          # 最小化成本
    bonus = +quality * 0.1                   # 最大化质量（权重较低）
    metrics["combined_score"] = primary + bonus
else:
    metrics["combined_score"] = -1e18
```

---

## 六、总结

将美赛题目转换为本框架的核心思路：

```
开放式建模题  →  固化参数(JSON)  →  确定性评分(evaluator)  →  标量指标(combined_score)
      ↓               ↓                    ↓                        ↓
   选子问题       公开数据集          独立求解器             取负/加权/罚函数
```

关键原则：**宁可简化模型、缩小规模，也要保证评分的确定性和可复现性。**  美赛的灵魂在于"建模创意"，但 benchmark 的灵魂在于"评分公正"。这两者之间的桥梁，就是你定义的 `problem_data.json` + `evaluator.py`。

