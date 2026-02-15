# AGENTS.md

## 目标
本文件用于指导在本仓库中基于 `reliable_sim/` 内容创建 Frontier-Eng 新题目，重点保证任务定义清晰、输入输出可验证、评测可复现。

## 已确认的数据集要求（Frontier-Eng）
新题目应遵循仓库根目录 `README.md` 的任务提交结构与核心要素：

- 必须有任务文档：`Task.md`（含背景、输入、输出、评测指标）。
- 必须有运行导航：`README.md`（如何运行、文件说明）。
- 必须有可执行验证程序：`verification/evaluator.py`（或等价入口）。
- 可选：`baseline/solution.py` 作为参考解。

建议目录（示例）：

```text
benchmarks/<Domain>/<TaskName>/
├── README.md
├── Task.md
├── references/
├── verification/
│   └── evaluator.py
└── baseline/
    └── solution.py
```

## 已确认的 ReliableSim 真实接口（以代码为准）
`reliable_sim/` 当前可稳定依赖的是 `code_linear.py` + `sampler.py`。

1. 主要输入（函数级）
- `HammingCode(r=3..8, decoder=...)`
- `simulate(noise_std, sampler, batch_size, num_samples, scale_factor, fix_tx, ...)`
- `simulate_variance_controlled(noise_std, target_std, max_samples, sampler, batch_size, ...)`

2. 主要输出（函数返回）
- `simulate(...) -> (errors_log, weights_log, err_ratio)`
- `simulate_variance_controlled(...) -> (errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`

3. 结果解释
- `errors_log` 与 `weights_log` 在对数域。
- 常用误码率对数：`err_rate_log = errors_log - weights_log`。
- `err_ratio` 是抽样中观测到的错误比例（非重要性加权真值）。

4. 代码现状注意
- `reliable_sim/run_experiment.py`、`reliable_sim/plot_general.py` 依赖 `test_general.py`，但该文件当前不在仓库中。
- 出题和评测时不要依赖 `test_general.py`；直接调用 `code_linear.py` 与 `sampler.py`。

## ReliableSim 题目的推荐输入输出结构（用于 Task.md）
为避免歧义，题面应固定一个文件 I/O 协议。

1. 输入（建议 `input.json`）
- `code_type`: `"hamming"`
- `r`: 整数，范围 `[3, 8]`
- `decoder`: `"binary" | "ORBGRAND" | "SGRAND" | "chase"`
- `sampler`: `"naive" | "bessel" | "sym"`
- `sigma`: 浮点噪声标准差
- `num_samples`: 整数
- `batch_size`: 整数
- `fix_tx`: 布尔
- 可选方差控制字段：`target_std`, `max_samples`

2. 输出（建议 `results.json`）
- 必需字段：
  - `err_rate_log`（`errors_log - weights_log`）
  - `err_ratio`
  - `exec_time`
- 若为方差控制任务，额外要求：
  - `actual_samples`
  - `actual_std`
  - `converged`

3. 评测建议
- 主分数：`combined_score = -err_rate_log`（误码率越低越好）与 `exec_time` 的加权组合。
- 需设定硬约束：最大运行时、参数合法范围、输出字段完整性。

## HighReliableSimulation 专项约束（当前任务）
- 评测入口固定调用 `MySampler.simulate_variance_controlled(...)`，不使用普通 `simulate(...)` 路径。
- 译码器固定为 `chase`，并使用 `t=3` 配置（由评测器写死）。
- 当前评测常量已冻结为 `sigma=0.268`，配套参数见 `Task.md` 与 `calibration.md`。
- 最终冻结参数应至少包含：`sigma*`、`r0`、`epsilon`、`target_std`、`max_samples`、`batch_size`。

## 在本仓库中新增 ReliableSim 任务时的执行清单
1. 在 `benchmarks/` 下创建新领域或子任务目录，并补齐 `README.md` 与 `Task.md`。
2. 在 `Task.md` 中明确：
- 输入文件名与 JSON 字段；
- 输出文件名与 JSON 字段；
- 计分公式与失败条件（超时/字段缺失/非法参数）。
3. 编写 `verification/evaluator.py`：
- 读取 `input.json`；
- 调用参赛程序；
- 校验 `results.json`；
- 返回统一指标（至少含 `combined_score`）。
4. 提供 `baseline/solution.py` 生成可通过的最小解。
5. 如需接入 `frontier_eval`：
- 新建 `frontier_eval/tasks/<task_name>/task.py`；
- 实现 `initial_program_path()` 与 `evaluate_program()`。

## 输出文档风格
- 优先中文，术语首次出现可附英文。
- 避免只写“优化性能”，必须给出可计算的输入输出与评分定义。
