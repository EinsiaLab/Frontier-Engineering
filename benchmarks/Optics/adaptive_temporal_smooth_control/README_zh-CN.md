# 自适应光学 A2：时间平滑控制

该任务聚焦于 **时序 AO 控制** 中“补偿质量 + 命令平滑”折中优化。

## 任务意义

真实系统中命令抖动会带来：
- 执行器磨损，
- 镜面振动风险，
- 闭环鲁棒性下降。

逐帧独立控制可能瞬时误差低，但时间轨迹不平滑。
本任务强调工程可用的综合目标。

## 目录结构

```text
task2_temporal_smooth_control/
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
- 任务特定 oracle 依赖：无（reference 为解析控制器，不依赖额外第三方求解器）
- 建议在仓库根目录一次安装：`python -m pip install -r benchmarks/Optics/requirements.txt`

## 运行方式

```bash
cd /DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/Optics/adaptive_temporal_smooth_control
/data_storage/chihh2311/.conda/envs/aotools/bin/python verification/evaluate.py
```

## 输出

- `verification/outputs/metrics.json`
- `verification/outputs/metrics_comparison.png`
- `verification/outputs/example_visualization.png`

## Baseline 与 Oracle 约束

- Baseline 目标是 `baseline/controller.py`，应避免重型第三方求解器。
- Reference/oracle 是带延迟补偿的解析平滑控制器（仍不依赖第三方优化器）。
- 当前配置为 `v3_delay_and_model_mismatch`，包含观测延迟与执行器约束。
