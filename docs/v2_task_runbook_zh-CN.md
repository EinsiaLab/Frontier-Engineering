# V2 任务集运行手册

本文档记录仓库主线当前的 v2 任务集运行方式，要求从全新 clone 出发即可复现，不依赖外部个人笔记或私有辅助目录。

## 环境映射

| 任务 | 环境 | 状态 | 备注 |
|---|---|---|---|
| `MaterialEngineering/MicrowaveAbsorberDesign` | `.venvs/frontier-v2-extra` | verified | direct baseline 与 unified smoke 均已通过。 |
| `ParticlePhysics/MuonTomography` | `.venvs/frontier-v2-extra` | verified | direct baseline 与 unified v2 已通过。 |
| `ParticlePhysics/PETScannerOptimization` | `.venvs/frontier-v2-extra` | verified | direct baseline 与 unified smoke 已通过；evaluator 已加严 ring schema 校验。 |
| `ParticlePhysics/ProtonTherapyPlanning` | `.venvs/frontier-v2-extra` | verified | 主线已补 benchmark-local unified 元数据。 |
| `SingleCellAnalysis/perturbation_prediction` | `.venvs/frontier-v2-extra` | verified | 仍保留 fetch + baseline + scorer 路径，同时主线已补 unified 元数据。 |
| `CommunicationEngineering/LDPCErrorFloor` | `.venvs/frontier-v2-extra` | hardened | evaluator 已改为 evaluator-owned 统计链路。 |
| `CommunicationEngineering/PMDSimulation` | `.venvs/frontier-v2-extra` | hardened | evaluator 已改为 evaluator-owned 统计链路。 |
| `CommunicationEngineering/RayleighFadingBER` | `.venvs/frontier-v2-extra` | hardened | evaluator 已改为 evaluator-owned 统计链路。 |
| `ReactionOptimisation/dtlz2_pareto` | `.venvs/frontier-v2-summit-compat` | verified | 需要兼容环境。 |
| `MolecularMechanics/weighted_parameter_coverage` | `.venvs/openff-dev` | verified | OpenFF 特殊运行时，不是 uv-only。 |
| `MolecularMechanics/diverse_conformer_portfolio` | `.venvs/openff-dev` | verified | OpenFF 特殊运行时，不是 uv-only。 |
| `MolecularMechanics/torsion_profile_fitting` | `.venvs/openff-dev` | verified | OpenFF 特殊运行时，不是 uv-only。 |
| `Optics/adaptive_constrained_dm_control` | `.venvs/frontier-v2-optics` | verified | unified v2 已通过。 |
| `Optics/adaptive_energy_aware_control` | `.venvs/frontier-v2-optics` | verified | unified v2 已通过。 |
| `Optics/phase_weighted_multispot_single_plane` | `.venvs/frontier-v2-optics` | verified | 依赖主机 `libGL.so.1` 与 OpenCV。 |
| `Optics/phase_large_scale_weighted_spot_array` | `.venvs/frontier-v2-optics` | verified | 依赖主机 `libGL.so.1` 与 OpenCV。 |

## 统一与特殊路径说明

当前 v2 任务分成两类：

- `unified`：通过 benchmark-local `frontier_eval/` 元数据接入 `task=unified`
- `special-case`：属于 v2 任务集，但仍保留额外的非-unified 正式运行路径

当前 special-case 任务只有：

- `SingleCellAnalysis/perturbation_prediction`

它已经支持 unified，但仍保留 fetch + baseline + scorer 的数据导向复现路径。

## 常用命令

### Unified 任务

```bash
bash scripts/run_v2_unified.sh MaterialEngineering/MicrowaveAbsorberDesign algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh ParticlePhysics/MuonTomography algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh ParticlePhysics/PETScannerOptimization algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh ParticlePhysics/ProtonTherapyPlanning algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh CommunicationEngineering/LDPCErrorFloor algorithm=openevolve algorithm.iterations=0 algorithm.oe.evaluator.timeout=60
bash scripts/run_v2_unified.sh CommunicationEngineering/PMDSimulation algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh CommunicationEngineering/RayleighFadingBER algorithm=openevolve algorithm.iterations=0
bash scripts/run_v2_unified.sh ReactionOptimisation/dtlz2_pareto task.runtime.python_path=uv-env:frontier-v2-summit-compat algorithm=openevolve algorithm.iterations=0
```

### Special-case 任务

`perturbation_prediction`：

```bash
bash scripts/data/fetch_perturbation_prediction.sh
bash scripts/run_perturbation_prediction_baseline.sh
```

其 unified smoke 命令：

```bash
bash scripts/run_v2_unified.sh SingleCellAnalysis/perturbation_prediction \
  algorithm=openevolve \
  algorithm.iterations=0
```
