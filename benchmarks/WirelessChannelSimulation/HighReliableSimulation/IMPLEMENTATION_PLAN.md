# HighReliableSimulation 实施计划（当前版本）

## 1. 目标
- 在 `benchmarks/WirelessChannelSimulation/HighReliableSimulation` 提供可运行、可评测、可复现的通信仿真任务。
- 题目固定为：Hamming(127,120) + `chase(t=3)`，选手实现 `MySampler` 并通过 `simulate_variance_controlled` 估计 BER。

## 2. 已落地内容
- 任务文档：`Task.md`、`README.md`、`USAGE.md`
- 代码骨架：
  - 初始程序：`scripts/init.py`（NaiveSampler）
  - baseline：`baseline/solution.py`（BesselSampler）
  - 评测器：`eval/evaluator.py`
  - 标定脚本：`scripts/calibrate_sigma.py`
- `frontier_eval` 接入：
  - `frontier_eval/tasks/wireless_channel_simulation/task.py`
  - `frontier_eval/registry_tasks.py`
  - `frontier_eval/conf/task/high_reliable_simulation.yaml`

## 3. 当前冻结评测配置
- `sigma = 0.268`
- `decoder = chase(t=3)`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`

## 4. 评分规则
- `e = |log(r / r0)|`
- 若 `e < epsilon`：
  - `combined_score = t0 / (t * e + 1e-6)`
- 否则：
  - `combined_score = 0`

其中：
- `r` 为选手估计 BER（由 `errors_log - weights_log` 还原）
- `t` 为选手评测耗时中位数

## 5. 验收状态
- `scripts/init.py`：稳定 0 分（符合弱基线预期）
- `baseline/solution.py`：可获得非零分，但存在采样统计波动
- 通过 `epsilon=0.8` 缓解波动导致的误判

## 6. 下一步优化项
1. 提升 `baseline` 稳定性（降低 `error_log_ratio` 波动）。
2. 增加一次高预算复标定，确认 `sigma/r0/t0` 的长期稳定性。
3. 在具备 `omegaconf/hydra` 环境时，补一轮 `frontier_eval` CLI 端到端回归。

