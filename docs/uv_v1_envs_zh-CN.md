# Frontier-Engineering v1 的 uv 环境矩阵

结论先说清楚：

- 之前那版 `uv` 方案没有覆盖 `v1` 的全部任务环境
- 原因不是遗漏了几个包这么简单，而是 `v1` 本身就包含多套互相独立的 runtime
- 用 `uv` 管理是可行的，但应该按任务族拆成多个虚拟环境，而不是试图塞进一个 `.venv`

## 1. 先创建哪些环境

仓库现在提供了一个脚本：

```bash
bash scripts/setup_uv_envs.sh driver v1-general v1-optics v1-gpu v1-power v1-summit v1-sustaindc v1-kernel v1-openff v1-diffsim v1-singlecell-denoising
```

默认会把环境放到：

```text
.uv-envs/
```

## 2. 环境与任务覆盖关系

| uv 环境 | Python | 覆盖任务 |
| --- | --- | --- |
| `driver` | 3.12 | `frontier_eval` 驱动本身；也可直接跑 `Cryptographic/*`、`ComputerSystems/MallocLab` |
| `v1-general` | 3.12 | `CommunicationEngineering/*`、`EnergyStorage/*`、`ParticlePhysics/*`、`QuantumComputing/*`、`InventoryOptimization/*`、`PyPortfolioOpt/*`、`JobShop/*`、`StructuralOptimization/*`、`SingleCellAnalysis/predict_modality`、`SingleCellAnalysis/perturbation_prediction`、`WirelessChannelSimulation/HighReliableSimulation`、`Robotics/DynamicObstacleAvoidanceNavigation`、`Robotics/PIDTuning`、`Robotics/UAVInspectionCoverageWithWind`、`Robotics/CoFlyersVasarhelyiTuning`、`Aerodynamics/DawnAircraftDesignOptimization`、`ComputerSystems/DuckDBWorkloadOptimization` |
| `v1-optics` | 3.12 | `Optics/*` 全部 16 个任务 |
| `v1-gpu` | 3.11 | `Aerodynamics/CarAerodynamicsSensing`、`Robotics/QuadrupedGaitOptimization`、`Robotics/RobotArmCycleTimeOptimization` |
| `v1-power` | 3.12 | `PowerSystems/EV2GymSmartCharging` |
| `v1-summit` | 3.9 | `ReactionOptimisation/*` |
| `v1-sustaindc` | 3.10 | `SustainableDataCenterControl/hand_written_control` |
| `v1-kernel` | 3.12 | `KernelEngineering/FlashAttention`、`KernelEngineering/MLA`、`KernelEngineering/TriMul` |
| `v1-openff` | 3.11 | `MolecularMechanics/*` 的 Python 层依赖 |
| `v1-diffsim` | 3.12 | `AdditiveManufacturing/DiffSimThermalControl` |
| `v1-singlecell-denoising` | 3.12 | `SingleCellAnalysis/denoising_ttt` 的 Python 层依赖 |

不适合只靠一个 `uv` venv 解决的任务：

| 任务 | 额外说明 |
| --- | --- |
| `Astrodynamics/MannedLunarLanding` | 需要系统层 `octave`、`octave-signal`、`octave-control` |
| `EngDesign/*` | 仍以 Docker 流程为主 |
| `SingleCellAnalysis/denoising` | 仍以上游 build / Docker 工作流为主 |
| `MolecularMechanics/*` | 除 Python 包外，还需要 `rdkit`、`openmm`、`ambertools` 这类二进制依赖 |

补充说明：

- `v1-summit` 当前需要把 `setuptools` 固定在 `<81`，否则 `summit -> skorch` 这条链会因为缺少 `pkg_resources` 失败

## 2.1 当前实机验证状态

下表中的“已验证”指的是在当前机器上，实际执行过环境创建和代表性运行，而不是只做静态配置检查。

| 环境 / 任务链路 | 状态 | 已验证内容 | 备注 |
| --- | --- | --- | --- |
| `driver` | 已验证 | `uv` 创建环境成功；`task=smoke algorithm=openevolve algorithm.iterations=0` 成功；`Cryptographic/AES-128` 已验证可通过 `task.runtime.python_path=.uv-envs/driver/bin/python` 进入 unified 执行链路 | 当前默认是最小驱动环境，不再默认安装 `torch/CUDA`；若不显式关闭 conda 路径，unified 默认会回退到 `frontier-eval-2` |
| `v1-general` | 已验证 | 环境创建成功；`ComputerSystems/DuckDBWorkloadOptimization`、`EnergyStorage/BatteryFastChargingProfile`、`ParticlePhysics/MuonTomography`、`PyPortfolioOpt/robust_mvo_rebalance`、`CommunicationEngineering/PMDSimulation`、`PyPortfolioOpt/cvar_stress_control`、`QuantumComputing/task_01_routing_qftentangled`、`InventoryOptimization/tree_gsm_safety_stock` unified 成功 | 运行时使用 `task.runtime.python_path=.uv-envs/v1-general/bin/python`；补查发现 `JobShop/*` 还需要额外安装 `job-shop-lib`，之前脚本未包含 |
| `v1-summit` | 部分验证 | 环境创建成功；`ReactionOptimisation/mit_case1_mixed`、`ReactionOptimisation/snar_multiobjective`、`ReactionOptimisation/reizman_suzuki_pareto` unified 成功 | 需要 `setuptools<81`；`ReactionOptimisation/dtlz2_pareto` 仍受 `summit/sklearn` 私有接口兼容问题影响 |
| `v1-optics` | 阻塞 | 环境创建在镜像下载阶段失败 | 清华镜像下载 `nvidia-nccl-cu13` 超时；当前还没拿到可运行环境 |
| `v1-gpu` | 部分验证 | 已确认当前机器有 `A100 80GB`；已定位并修正 `torch==2.1.0` 需要 Python 3.11 的兼容问题；已重试镜像安装 | 仍卡在大体积 CUDA wheel 下载阶段，当前会话内未完成环境创建，也未跑代表性任务 |
| `v1-power` | 已验证 | 环境创建成功；`PowerSystems/EV2GymSmartCharging` unified 成功 | 需要 `setuptools<81`，否则 evaluator 里的 `pkg_resources` 导入会失败 |
| `v1-sustaindc` | 阻塞 | 已定位安装失败原因 | 需要 `UV_INDEX_STRATEGY=unsafe-best-match` 处理多索引解析；随后又卡在 `cchardet` 构建，缺少 `Python.h`，说明本机缺少 Python 3.10 开发头文件 |
| `v1-kernel` | 未验证 | 还没有在当前机器上实际创建环境并运行任务 | 依赖 GPU kernel 任务的大量 CUDA/Triton 依赖 |
| `v1-openff` | 阻塞 | 已定位安装失败原因 | 当前镜像下只看到 `openff-toolkit==0.18.0`，而该版本已被 yanked，依赖不可解；另外 runtime 仍需要 `rdkit` / `openmm` / `ambertools` |
| `v1-diffsim` | 已验证 | 环境创建成功；`AdditiveManufacturing/DiffSimThermalControl` unified 成功 | 当前链路可正常使用 |
| `v1-singlecell-denoising` | 未验证 | 还没有在当前机器上实际创建环境并运行任务 | `denoising_ttt` 只覆盖 Python 包侧；原始 `denoising` 仍以上游 build / Docker 为主 |
| `Astrodynamics/MannedLunarLanding` | 未验证 | 还没有用 `uv driver + octave` 组合重新实跑 | 额外需要系统层 `octave` / `octave-signal` / `octave-control` |
| `EngDesign/*` | 未验证 | 还没有按 `uv driver + Docker` 组合重新实跑 | 主路径仍是 Docker，不是单纯 Python 环境问题 |
| `SingleCellAnalysis/denoising` | 未验证 | 还没有按当前 `uv` 方案重新实跑 | 主路径仍是上游 build / Docker 工作流 |

## 2.2 仍未完成的任务清单

这里的“未完成”指的是：

- 还没有在当前机器上用 `uv` 方案把该任务实际跑通
- 或者环境创建被镜像 / 系统依赖 / 版本兼容问题阻塞

已经跑通的代表性任务：

- `ComputerSystems/DuckDBWorkloadOptimization`
- `CommunicationEngineering/PMDSimulation`
- `EnergyStorage/BatteryFastChargingProfile`
- `ParticlePhysics/MuonTomography`
- `QuantumComputing/task_01_routing_qftentangled`
- `InventoryOptimization/tree_gsm_safety_stock`
- `PyPortfolioOpt/robust_mvo_rebalance`
- `PyPortfolioOpt/cvar_stress_control`
- `ReactionOptimisation/mit_case1_mixed`
- `ReactionOptimisation/snar_multiobjective`
- `ReactionOptimisation/reizman_suzuki_pareto`
- `AdditiveManufacturing/DiffSimThermalControl`
- `PowerSystems/EV2GymSmartCharging`

仍未完成的任务如下。

| 任务组 | 具体任务 | 当前状态 | 主要阻塞 |
| --- | --- | --- | --- |
| `v1-general` 中尚未逐任务实跑的部分 | `CommunicationEngineering/LDPCErrorFloor`、`CommunicationEngineering/RayleighFadingBER`、`EnergyStorage/BatteryFastChargingSPMe`、`ParticlePhysics/ProtonTherapyPlanning`、`SingleCellAnalysis/predict_modality`、`SingleCellAnalysis/perturbation_prediction`、`StructuralOptimization/ISCSO2015`、`StructuralOptimization/ISCSO2023`、`StructuralOptimization/PyMOTOSIMPCompliance`、`StructuralOptimization/TopologyOptimization`、`WirelessChannelSimulation/HighReliableSimulation`、`Robotics/DynamicObstacleAvoidanceNavigation`、`Robotics/PIDTuning`、`Robotics/UAVInspectionCoverageWithWind`、`Robotics/CoFlyersVasarhelyiTuning`、`Aerodynamics/DawnAircraftDesignOptimization`、`PyPortfolioOpt/discrete_rebalance_mip`、`InventoryOptimization/general_meio`、`InventoryOptimization/joint_replenishment`、`InventoryOptimization/finite_horizon_dp`、`InventoryOptimization/disruption_eoqd`、`QuantumComputing/task_02_clifford_t_synthesis`、`QuantumComputing/task_03_cross_target_qaoa`、`ComputerSystems/MallocLab` | 环境已建成，但这些任务还没逐个实跑 | 还没有逐任务消耗时间去验证 |
| `v1-general` 中当前被依赖阻塞的部分 | `JobShop/abz`、`JobShop/ft`、`JobShop/la`、`JobShop/orb`、`JobShop/swv`、`JobShop/ta`、`JobShop/yn` | 已定位 | `benchmarks/JobShop/requirements.txt` 当前只装了 `ortools`，但 unified reference 还需要 `job-shop-lib`；已回补到 `scripts/setup_uv_envs.sh`，但镜像补装在当前会话内未完成 |
| `driver` / 系统依赖组合中当前被依赖阻塞的部分 | `Cryptographic/AES-128`、`Cryptographic/SHA-256`、`Cryptographic/SHA3-256` | 已部分定位 | 在 `uv driver + python_path` 模式下已能进入 unified 执行链路；`AES-128` 当前卡在 `verification/validate.cpp` 编译阶段缺少 `openssl/evp.h`，说明还需要系统层 OpenSSL 开发头文件（如 `libssl-dev`） |
| `v1-optics` | `Optics/*` 全部 16 个任务 | 未完成 | 清华镜像下载 `nvidia-nccl-cu13` 超时，环境没建完 |
| `v1-gpu` | `Aerodynamics/CarAerodynamicsSensing`、`Robotics/QuadrupedGaitOptimization`、`Robotics/RobotArmCycleTimeOptimization` | 未完成 | 已修正为 Python 3.11，但当前仍在 CUDA 大包下载阶段，未完成环境创建 |
| `v1-summit` 中尚未完成的部分 | `ReactionOptimisation/dtlz2_pareto` | 阻塞 | `summit` 依赖链仍会触发 `sklearn.utils.validation._check_fit_params` 私有接口兼容问题 |
| `v1-sustaindc` | `SustainableDataCenterControl/hand_written_control` | 未完成 | 需要 `UV_INDEX_STRATEGY=unsafe-best-match`；随后又卡在 `cchardet` 构建，缺 `Python.h` |
| `v1-kernel` | `KernelEngineering/FlashAttention`、`KernelEngineering/MLA`、`KernelEngineering/TriMul` | 未完成 | 还没有开始环境创建和任务实跑 |
| `v1-openff` | `MolecularMechanics/weighted_parameter_coverage`、`MolecularMechanics/diverse_conformer_portfolio`、`MolecularMechanics/torsion_profile_fitting` | 未完成 | 当前镜像下 `openff-toolkit` 依赖不可解，且 runtime 还需要 `rdkit` / `openmm` / `ambertools` |
| `v1-singlecell-denoising` | `SingleCellAnalysis/denoising_ttt` | 未完成 | 还没开始环境创建和任务实跑 |
| 系统依赖 / Docker 路线 | `Astrodynamics/MannedLunarLanding`、`SingleCellAnalysis/denoising`、`EngDesign/*` | 未完成 | 分别受 `octave`、上游 build / Docker、Docker workflow 约束 |

## 3. `uv` 下怎么运行 unified task

### 3.1 只用驱动环境

适用于 benchmark 依赖已经装进 `driver` 本身，或任务没有额外 runtime 的情况。

如果你在纯 `uv` 环境下运行 unified task，不要依赖默认的 `conda_env`，否则会回退到 `frontier-eval-2`。应显式指定：

```text
task.runtime.python_path=/absolute/path/to/.uv-envs/driver/bin/python
task.runtime.use_conda_run=false
```

例如运行 `Cryptographic/AES-128`：

```bash
UV_PROJECT_ENVIRONMENT=.uv-envs/driver uv run python -m frontier_eval \
  task=unified \
  task.benchmark=Cryptographic/AES-128 \
  task.runtime.python_path=/GenSIvePFS/users/hhchi/Frontier-Engineering/.uv-envs/driver/bin/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0
```

### 3.2 使用单独 runtime 环境

这是 `uv` 方式下最重要的模式。

不要再用：

```text
task.runtime.conda_env=...
```

而是改成：

```text
task.runtime.python_path=/absolute/path/to/.uv-envs/<env>/bin/python
task.runtime.use_conda_run=false
```

示例，运行 `ReactionOptimisation/mit_case1_mixed`：

```bash
UV_PROJECT_ENVIRONMENT=.uv-envs/driver uv run python -m frontier_eval \
  task=unified \
  task.benchmark=ReactionOptimisation/mit_case1_mixed \
  task.runtime.python_path=/GenSIvePFS/users/hhchi/Frontier-Engineering/.uv-envs/v1-summit/bin/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0
```

示例，运行 `Optics/phase_dammann_uniform_orders`：

```bash
UV_PROJECT_ENVIRONMENT=.uv-envs/driver uv run python -m frontier_eval \
  task=unified \
  task.benchmark=Optics/phase_dammann_uniform_orders \
  task.runtime.python_path=/GenSIvePFS/users/hhchi/Frontier-Engineering/.uv-envs/v1-optics/bin/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0
```

## 4. 推荐的最小组合

如果你只想覆盖昨天复现里最常用的一批任务，先建这几个：

```bash
bash scripts/setup_uv_envs.sh driver v1-general v1-optics v1-summit
```

如果你要覆盖几乎所有 `v1` Python runtime，再加上：

```bash
bash scripts/setup_uv_envs.sh v1-power v1-gpu v1-kernel v1-sustaindc v1-openff v1-diffsim v1-singlecell-denoising
```

## 5. 边界说明

这份矩阵解决的是“如何用 `uv` 管理 `v1` 的 Python 环境”。

它不声称解决以下问题：

- GPU 驱动、CUDA、容器运行时
- `octave` / `libGL.so.1` / `rdkit` / `openmm` 这类系统或二进制依赖
- 上游 benchmark 自己要求的 Docker / build 步骤

但对于仓库用户来说，这已经把最麻烦的一层统一了：

- 驱动环境固定用 `driver`
- benchmark runtime 明确映射到 `.uv-envs/<name>/bin/python`
- 不再依赖 `conda_env` 名称解析
