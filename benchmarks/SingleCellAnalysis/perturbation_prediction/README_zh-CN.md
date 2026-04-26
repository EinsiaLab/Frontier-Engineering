# Perturbation Prediction（扰动响应预测）

该任务基于 OpenProblems Bio 的真实公开数据与评测规范构建：

- Task 仓库：`openproblems-bio/task_perturbation_prediction`
- Benchmark 页面：https://openproblems.bio/benchmarks/perturbation_prediction
- NeurIPS 2023 / Kaggle 比赛：https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

数据来自公开的 OpenProblems `openproblems-data`（S3），评测脚本复现了核心指标（row-wise correlation / error）。

## 目录结构

- `baseline/`：简单 baseline（输出 `prediction.h5ad`）
- `verification/`：数据下载与打分脚本
- `scripts/`：v2 任务集初始化辅助脚本
- `Task.md`：任务说明与 I/O 规范

## 快速开始

本题属于当前 v2 任务集，使用 `.venvs/frontier-v2-extra` 作为本地运行环境，并且现在也支持 benchmark-local `task=unified`。

它的正式复现路径仍然是：

1. 下载 / 缓存公开数据
2. 生成预测结果
3. 运行 scorer

先下载数据：

```bash
bash scripts/data/fetch_perturbation_prediction.sh
```

生成 baseline 预测：

```bash
.venvs/frontier-v2-extra/bin/python benchmarks/SingleCellAnalysis/perturbation_prediction/baseline/run_mean_across_compounds.py \
  --output prediction.h5ad
```

评测预测结果：

```bash
.venvs/frontier-v2-extra/bin/python benchmarks/SingleCellAnalysis/perturbation_prediction/verification/evaluate_perturbation_prediction.py \
  --prediction prediction.h5ad
```

Unified smoke 命令：

```bash
bash scripts/run_v2_unified.sh SingleCellAnalysis/perturbation_prediction \
  algorithm=openevolve \
  algorithm.iterations=0
```
