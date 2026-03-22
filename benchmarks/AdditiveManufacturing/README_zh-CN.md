# Additive Manufacturing

该领域收录的是被改造成 Frontier-Engineering benchmark 的增材制造优化任务，这些任务尽量基于真实工艺数据、真实刀路或真实仿真工作流。

## 领域概览

增材制造优化通常需要同时权衡：

- 成形质量与缺陷规避；
- 热过程一致性与熔池稳定性；
- 物理可行性约束；
- 制造效率与实验成本。

## 任务列表

- `DiffSimThermalControl/`
  - 改编自 `differentiable-simulation-am` 中公开发布的真实构建几何与刀路；
  - 直接使用上游的 `0.k` 与 `toolpath.crs` 文件；
  - 将公开 case 重构为一个可复现的 `frontier_eval` 热过程控制 benchmark。

## 快速运行

```bash
python benchmarks/AdditiveManufacturing/DiffSimThermalControl/verification/evaluator.py \
  benchmarks/AdditiveManufacturing/DiffSimThermalControl/scripts/init.py
```

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=AdditiveManufacturing/DiffSimThermalControl \
  task.runtime.conda_env=<your_env> \
  algorithm.iterations=0
```