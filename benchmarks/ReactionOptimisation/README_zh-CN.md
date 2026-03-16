# ReactionOptimisation 任务集

这个目录把 4 个 SUMMIT 反应优化 benchmark 包装成了既能单独运行、也能通过 `frontier_eval` unified task 调用的任务集合。

下面所有命令都使用仓库内相对路径，不再依赖任何绝对路径。

## 子任务

| 任务目录 | 核心目标 | 说明 |
|---|---|---|
| `snar_multiobjective` | 最大化 `sty`、最小化 `e_factor` | 连续流 SnAr Pareto 优化 |
| `mit_case1_mixed` | 最大化 `y` | 连续变量 + 离散催化剂的混合优化 |
| `reizman_suzuki_pareto` | 最大化 `yld`、最小化 `ton` | 催化剂/工艺联合 Pareto 优化 |
| `dtlz2_pareto` | 逼近 DTLZ2 的 Pareto 前沿 | 合成多目标参考任务 |

## 目录结构

每个任务目录包含：

- `Task.md`：英文完整任务说明
- `Task_zh-CN.md`：中文完整任务说明
- `README.md`：英文导航文档
- `README_zh-CN.md`：中文导航文档
- `task.py`：benchmark 构造、采样辅助和打分逻辑
- `baseline/solution.py`：baseline 求解器，也是 unified task 中可编辑的候选文件
- `verification/reference.py`：更强的 SUMMIT 参考解
- `verification/evaluate.py`：评测候选解并与参考解对比
- `frontier_eval/`：`python -m frontier_eval` 使用的 unified-task 元数据

## 环境配置

已验证的配置：

- `summit` 环境：用于 direct verification 和 unified task 的实际评测
- `frontier-eval-2` 环境：用于运行 `python -m frontier_eval`

在仓库根目录执行示例：

```bash
conda create -n summit python=3.9
conda create -n frontier-eval-2 python=3.12

conda activate summit
python -m pip install -r benchmarks/ReactionOptimisation/requirements.txt

conda activate frontier-eval-2
python -m pip install -r frontier_eval/requirements.txt
```

如果你想只用一个环境，也可以把这两个 `requirements.txt` 都安装到同一个 env 中。

## Direct Verification

通用命令：

```bash
conda run -n summit python benchmarks/ReactionOptimisation/<task>/verification/evaluate.py
```

已验证命令：

```bash
conda run -n summit python benchmarks/ReactionOptimisation/snar_multiobjective/verification/evaluate.py
conda run -n summit python benchmarks/ReactionOptimisation/mit_case1_mixed/verification/evaluate.py
conda run -n summit python benchmarks/ReactionOptimisation/reizman_suzuki_pareto/verification/evaluate.py
conda run -n summit python benchmarks/ReactionOptimisation/dtlz2_pareto/verification/evaluate.py
```

在已验证环境中的实测耗时：

| 任务 | Direct Verification 耗时 | 说明 |
|---|---|---|
| `snar_multiobjective` | ~`122s` | 参考解会跑多个标量化权重 |
| `mit_case1_mixed` | ~`106s` | 这组任务里最快 |
| `reizman_suzuki_pareto` | ~`112s` | 包含催化剂筛选和固定催化剂 SOBO |
| `dtlz2_pareto` | ~`160s` | 这组任务里最慢的 direct verifier |

`verification/evaluate.py` 会在默认 3 个 seed 上同时运行 baseline 和 reference，因此它本来就不是秒级 smoke test。

## frontier_eval（Unified）

4 个子任务都已经通过 `task=unified` 接入，每个任务目录下都有对应的 `frontier_eval/` 元数据。

通用命令：

```bash
conda run -n frontier-eval-2 python -m frontier_eval \
  task=unified \
  task.benchmark=ReactionOptimisation/<task> \
  task.runtime.conda_env=summit \
  algorithm=openevolve \
  algorithm.iterations=0
```

可直接复制的单行示例：

```bash
conda run -n frontier-eval-2 python -m frontier_eval task=unified task.benchmark=ReactionOptimisation/snar_multiobjective task.runtime.conda_env=summit algorithm=openevolve algorithm.iterations=0
```

已验证的 `task.benchmark`：

| 任务 | `task.benchmark` | `algorithm.iterations=0` 实测耗时 | 说明 |
|---|---|---|---|
| `snar_multiobjective` | `ReactionOptimisation/snar_multiobjective` | ~`137s` | `iterations=0` 仍会执行一次完整评测 |
| `mit_case1_mixed` | `ReactionOptimisation/mit_case1_mixed` | ~`61s` | unified quick run 中最短 |
| `reizman_suzuki_pareto` | `ReactionOptimisation/reizman_suzuki_pareto` | ~`130s` | 评测里仍会计算参考解 |
| `dtlz2_pareto` | `ReactionOptimisation/dtlz2_pareto` | ~`107s` | 默认 `300s` 超时内可以完成 |

补充说明：

- `algorithm.iterations=0` 是框架适配验证，但它依然会对 `baseline/solution.py` 跑一次完整 benchmark 评测。
- 在已验证环境中，这 4 个任务都能在默认 `300s` evaluator timeout 下完成。
- 如果机器更慢，`snar_multiobjective` 和 `reizman_suzuki_pareto` 可以把 `algorithm.oe.evaluator.timeout` 提高到 `600`。
- 如果失败的 run 里显示 `runtime_conda_env=frontier-eval-2`，而不是 `summit`，说明 shell 没有正确解析 `task.runtime.conda_env=summit` 这个 override。最常见的原因是在 `task.benchmark=...` 后面误加了一个 `>`，或者多行命令复制时断掉了。此时请直接使用上面的单行命令重试。
- 可以通过 run 目录下的 `.hydra/overrides.yaml` 确认 override 是否真的生效，里面应当出现 `task.runtime.conda_env=summit`。

## 当前 Baseline 与 Reference

- `snar_multiobjective`
  baseline：自适应随机标量化 + 围绕当前标量化最优点做局部扰动
  reference：按多个权重拆预算运行 SUMMIT `SOBO`
- `mit_case1_mixed`
  baseline：混合随机 + 局部搜索
  reference：直接在混合变量域上运行 SUMMIT `SOBO`
- `reizman_suzuki_pareto`
  baseline：对催化剂和连续条件做纯随机搜索
  reference：先筛催化剂，再做固定催化剂 SUMMIT `SOBO`
- `dtlz2_pareto`
  baseline：随机标量化 + 局部扰动
  reference：按多个标量化权重拆预算运行 SUMMIT `SOBO`
