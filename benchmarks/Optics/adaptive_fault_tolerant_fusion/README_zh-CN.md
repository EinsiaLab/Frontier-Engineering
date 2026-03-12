# 自适应光学 A4：故障容忍的多传感器融合

该任务聚焦于多传感器异常场景下的鲁棒控制。

## 任务意义

多 WFS 系统中，如果某一路出现失准或突发噪声，
简单平均融合会把异常直接传递到 DM 命令，导致补偿质量下降。

本任务要求先进行鲁棒融合，再输出控制命令。

## 目录结构

```text
task4_fault_tolerant_fusion/
  baseline/
    controller.py
  verification/
    evaluate.py
    reference_controller.py
    outputs/
  README.md
  README_zh-CN.md
  Task.md
  Task_zh-CN.md
```

## 环境依赖

- Python：`3.10+`（已验证解释器：`/data_storage/chihh2311/.conda/envs/aotools/bin/python`）
- Baseline 候选实现运行依赖：`numpy`
- Verification 评测依赖：`numpy`、`matplotlib`、仓库内本地 `aotools` 包
- 任务特定 oracle 依赖：`scikit-learn`（`verification/reference_controller.py` 使用 `sklearn.ensemble.IsolationForest`）
- 建议在仓库根目录一次安装：`python -m pip install -r benchmarks/Optics/requirements.txt`

## 运行方式

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_fault_tolerant_fusion
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

## 输出

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

## Baseline 与 Oracle 约束

- Baseline 目标（`baseline/controller.py`）刻意保持简单，不应依赖重型第三方包。
- Reference/oracle 使用第三方 `scikit-learn` `IsolationForest` + inlier 加权融合。
- 当前配置为 `v3_fault_stress`（故障主导的多传感器压力测试）。
- 验证场景采用重故障压力测试（每个样本多路传感器污染），确保对比差距具备工程意义。
