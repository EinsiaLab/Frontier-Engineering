# 自适应光学 A1：受约束 DM 控制

该任务聚焦于形变镜（DM）在电压约束下的单步控制优化。

## 任务意义

AO 中常见控制是 `u = R @ s`。
实际硬件要求电压有上下限，所以通常会做 `clip`。
但“先无约束求解，再强行 clip”一般不是最优解。

本任务要求 agent 在严格电压约束下提升补偿质量。

## 目录结构

```text
task1_constrained_dm_control/
  baseline/
    controller.py                  # agent 修改目标
  verification/
    evaluate.py                    # valid 校验 + baseline/reference 对比
    reference_controller.py        # 更优参考实现
    outputs/                       # 运行后生成
  README.md
  README_zh-CN.md
  Task.md
  Task_zh-CN.md
```

## 环境依赖

- Python：`3.10+`（已验证解释器：`/data_storage/chihh2311/.conda/envs/aotools/bin/python`）
- Baseline 候选实现运行依赖：`numpy`
- Verification 评测依赖：`numpy`、`matplotlib`、仓库内本地 `aotools` 包
- 任务特定 oracle 依赖：`scipy`（`verification/reference_controller.py` 使用 `scipy.optimize.lsq_linear`）
- 建议在仓库根目录一次安装：`python -m pip install -r benchmarks/Optics/requirements.txt`

## 运行方式

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_constrained_dm_control
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

指定候选实现：

```bash
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py \
  --candidate /绝对路径/controller.py
```

## 输出文件

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

`metrics.json` 会在相同随机种子和场景下对比候选 baseline 与 reference。

## Baseline 与 Oracle 约束

- Baseline 目标（`baseline/controller.py`）应保持轻量（`numpy` + 提供矩阵）。
- Reference/oracle 使用第三方 SciPy 有界最小二乘（`scipy.optimize.lsq_linear`）。
- 当前配置为 `v3_delay_and_model_mismatch`（观测延迟 + 执行器滞后 + 模型失配）。
- 这种分离是刻意设计，用于避免“几行手写代码即可追平”的弱对比。
