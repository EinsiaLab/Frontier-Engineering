# Frontier-Eng 环境 bring-up 与全量 Baseline 实测报告

测试时间: `2026-04-17`  
测试机器: `lenovo-06` - `2x NVIDIA A800 80GB PCIe`, `144 CPU`, `503 GiB RAM`, `linux-x86_64`  
测试Agent: Codex-5.4-xhigh

## 1. 结论

执行了全量 baseline sweep：

```bash
PYTHONNOUSERSITE=1 \
python scripts/run_full_baseline_validation.py \
  --output-root runs/full_baseline_validation_2026-04-17
```

实测结果：

- `74/74` 个 benchmark 都实际执行到了，`exit_code=0` 的任务数也是 `74`
- 原始全量 sweep: `valid=1` 为 `50`，`valid=0` 为 `24`
- 额外定点复测后，已确认有 `2` 个任务只是少装 task-local 依赖：
  - `ComputerSystems/DuckDBWorkloadOptimization`
  - `PowerSystems/EV2GymSmartCharging`
- 所以当前更准确的状态是：
  - 原始 sweep 通过数: `50/74`
  - 已确认可通过补环境恢复: `2`
  - 当前仍未解决的 blocker: `22`

实际：

- repo 主框架是可用的
- quickstart 没把环境分层、外部资产、task-local extras、权限条件讲清楚

## 2. quickstart 没给够的信息

这部分是本报告最重要的结论。

| 缺口 | 实测结论 | 直接影响 |
| --- | --- | --- |
| merged env 的覆盖范围没讲透 | `frontier-v1-main` 不等于所有映射到它的 benchmark 都 ready | `Optics/*`、GPU kernel、部分 task-local deps 都会踩坑 |
| task-local extras 信息分散 | `DuckDB` 和 `EV2Gym` 都需要额外装 verification 依赖，但 repo 入口文档没有把这件事提升到 quickstart 级别 | 全量 sweep 里直接变成 `valid=0` |
| `frontier-v1-summit` 没有稳定安装路径 | 本机复现 `pip install -r benchmarks/ReactionOptimisation/requirements.txt` 直接 `resolution-too-deep` | `ReactionOptimisation/*` 当前无法按 README 稳定 bring-up |
| 外部资产条件没收敛成 checklist | `dc-rl` 需要手动 clone + patch，`PhySense` 需要数据和 checkpoint，`EngDesign` 受 Docker 权限约束 | 新用户会把“缺资产”误判成“代码坏了” |

## 3. 实际覆盖了哪些环境

这次不是只测了一个默认 env，而是把全量 sweep 涉及到的运行环境都过了一遍。

| 环境 / 资产 | 作用 | 实测状态 | 关键结论 |
| --- | --- | --- | --- |
| `frontier-eval-2` | 默认 driver env | 已测 | `init.sh` 可创建；CPU 任务可跑；本机 GPU 不可用 |
| `frontier-v1-main` | 主 runtime env | 已测 | 基础 import 正常；但 `Optics/*` 不完整 |
| `frontier-v1-kernel` | GPU kernel runtime | 已测 | `torch 2.7.1+cu126` + `triton 3.3.1`，`3/3` kernel 任务通过 |
| `frontier-v1-summit` | `ReactionOptimisation/*` runtime | 已测 | env 存在但不可用；安装链复现 `resolution-too-deep` |
| `frontier-v1-sustaindc` | `SustainableDataCenterControl/*` runtime | 已测 | 需要外部 `dc-rl` clone + patch |
| `openff-dev` | `MolecularMechanics/*` runtime | 已测 | 需要额外 `openff-toolkit/openff-units/openff-utilities` |
| `Engi` | `AdditiveManufacturing/*` runtime | 已测 | `1/1` 通过 |
| `/tmp/fe_ext/dc-rl` | SustainDC 外部 repo | 已测 | 仓库 vendored 目录是空的 |
| `/tmp/fe_ext/PhySense` | CarAerodynamics 外部 repo | 已测 | 只 clone 代码不够，数据和 checkpoint 才是关键 |
| Docker daemon | EngDesign Docker mode | 已测 | 本机 `docker.sock` 无权限，只能走 local mode |

## 4. 实测卡点与影响

### 4.1 原始全量 sweep 的失败分类

| 分类 | 数量 | 说明 |
| --- | ---: | --- |
| `Optics/*` 依赖缺失 | 16 | `frontier-v1-main` 未装 `benchmarks/Optics/requirements.txt` |
| `ReactionOptimisation/*` 环境失效 | 4 | `frontier-v1-summit` 依赖解析失败 |
| `CarAerodynamicsSensing` 外部数据缺失 | 1 | 缺 `PhySense` 数据 / checkpoint / 参考点 |
| driver env 缺 `duckdb` | 1 | DuckDB verification requirements 未装 |
| driver env 缺 `ev2gym` | 1 | EV2Gym verification requirements 未装 |
| baseline timeout | 1 | `SingleCellAnalysis/predict_modality` 不是缺包，是预算不足 |

### 4.2 用户的 4 个卡点

1. `Optics/*` 不是“建好 `frontier-v1-main` 就能跑”。  
实测里 `16` 个 `Optics` 任务全部失败，直接原因是还缺 `benchmarks/Optics/requirements.txt`。

2. `ReactionOptimisation/*` 当前没有可靠 quickstart。  
本机复现：

```bash
PYTHONNOUSERSITE=1 \
conda run -n frontier-v1-summit python -m pip install \
  -r benchmarks/ReactionOptimisation/requirements.txt
```

直接报：

```text
error: resolution-too-deep
```

3. `DuckDB` 和 `EV2Gym` 都是“少一步安装就挂”的典型。  
这两个任务在原始 sweep 中失败，但补装各自 `verification/requirements.txt` 后都能转正。

4. `CarAerodynamicsSensing` 主要不是代码问题，而是外部资产没齐。  
只 clone `PhySense` 代码仓库不够，还需要数据、checkpoint 和 `references/car_surface_points.npy`。


## 5. 从当前 README 走到“尽量全量可跑”还缺哪些步骤

下面这些步骤，是“认真看现有 README”之后，实测仍然需要额外补上的信息。

### 5.1 最小成功路径

真正的轻量 smoke 应该是：

```bash
bash init.sh
conda activate frontier-eval-2
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

而不是只看到 `init.sh` 结束就默认认为“环境齐了”。

### 5.2 要走到 full sweep 前，至少还要知道这些

- 固定加：

```bash
export PYTHONNOUSERSITE=1
```

- `frontier-eval-2` 只是 driver env，不是所有 benchmark 的 runtime env
- 还需要准备：
  - `frontier-v1-main`
  - `frontier-v1-kernel`
  - `frontier-v1-summit`
  - `frontier-v1-sustaindc`
  - `openff-dev`
  - `Engi`
- `DuckDB` 需要额外安装：

```bash
conda run -n frontier-eval-2 python -m pip install \
  -r benchmarks/ComputerSystems/DuckDBWorkloadOptimization/verification/requirements.txt
```

- `EV2Gym` 需要额外安装：

```bash
conda run -n frontier-eval-2 python -m pip install \
  -r benchmarks/PowerSystems/EV2GymSmartCharging/verification/requirements.txt
```

- `Optics/*` 还要手动补 `benchmarks/Optics/requirements.txt` 到 `frontier-v1-main`
- `MolecularMechanics/*` 除了 README 里的 `rdkit/openmm/ambertools`，本机还需要再补：
  - `openff-toolkit`
  - `openff-units`
  - `openff-utilities`
- `SustainableDataCenterControl/*` 不能只依赖仓库里的 vendored 目录，实测必须：
  - 手动 clone `dc-rl`
  - checkout 到 `a92b475`
  - 应用 `sustaindc_optional_runtime.patch`
  - 导出 `SUSTAINDC_ROOT`
- `CarAerodynamicsSensing` 不能只 clone `PhySense` 仓库，数据和 checkpoint 必须另外准备
- `EngDesign` 在无 Docker 权限机器上应改用：

```bash
export ENGDESIGN_EVAL_MODE=local
```

- kernel 任务在本机应显式使用 `frontier-v1-kernel`
- `ReactionOptimisation/*` 当前仍缺一条可稳定复现的安装路径，现有 README 不足以让首次用户独立完成 bring-up

## 6. 定点复测结论

这些复测的目的，是区分“真 blocker”和“只是 quickstart 漏了一步安装”。

| Task | 操作 | 结果 |
| --- | --- | --- |
| `ComputerSystems/DuckDBWorkloadOptimization` | 补装 DuckDB verification requirements | 从 `valid=0` 变为 `valid=1` |
| `PowerSystems/EV2GymSmartCharging` | 补装 EV2Gym verification requirements | 从 `valid=0` 变为 `valid=1` |
| `SingleCellAnalysis/predict_modality` | 将 `algorithm.oe.evaluator.timeout` 提到 `600` | 仍然 `valid=0`，说明不是简单外层 timeout 太小 |
| `ReactionOptimisation/*` | 复现 `pip install -r benchmarks/ReactionOptimisation/requirements.txt` | 再次报 `resolution-too-deep` |


## 7. 附录：原始全量 74 benchmark 结果表

下表对应的是原始 full sweep 结果，不把后续 DuckDB / EV2Gym 的补装复测覆盖进去。

| Benchmark | Runtime env | valid | combined_score | Issue |
| --- | --- | ---: | ---: | --- |
| AdditiveManufacturing/DiffSimThermalControl | Engi | 1.0 | 0.460717 |  |
| Aerodynamics/CarAerodynamicsSensing | frontier-v1-main | 0.0 | -1e+18 | missing PhySense dataset/checkpoint artifacts |
| Aerodynamics/DawnAircraftDesignOptimization | frontier-eval-2(no-conda-run) | 1.0 | 0.741496 |  |
| Astrodynamics/MannedLunarLanding | frontier-eval-2(default) | 1.0 | 4577.44 |  |
| CommunicationEngineering/LDPCErrorFloor | frontier-eval-2(default) | 1.0 | 0.557225 |  |
| CommunicationEngineering/PMDSimulation | frontier-eval-2(default) | 1.0 | 8.82677 |  |
| CommunicationEngineering/RayleighFadingBER | frontier-eval-2(default) | 1.0 | 1577.27 |  |
| ComputerSystems/DuckDBWorkloadOptimization | frontier-eval-2(default) | 0.0 | -1e+18 | missing duckdb in frontier-eval-2 |
| ComputerSystems/MallocLab | frontier-eval-2(default) | 1.0 | 28 |  |
| Cryptographic/AES-128 | frontier-eval-2(default) | 1.0 | 9.16607 |  |
| Cryptographic/SHA-256 | frontier-eval-2(default) | 1.0 | 13.905 |  |
| Cryptographic/SHA3-256 | frontier-eval-2(default) | 1.0 | 22.0571 |  |
| EnergyStorage/BatteryFastChargingProfile | frontier-eval-2(default) | 1.0 | 71.2806 |  |
| EnergyStorage/BatteryFastChargingSPMe | frontier-eval-2(default) | 1.0 | 66.1636 |  |
| EngDesign | frontier-eval-2(default) | 1.0 | 1.35714 |  |
| InventoryOptimization/disruption_eoqd | frontier-v1-main | 1.0 | 0.36423 |  |
| InventoryOptimization/finite_horizon_dp | frontier-v1-main | 1.0 | 0.367322 |  |
| InventoryOptimization/general_meio | frontier-v1-main | 1.0 | 0.182532 |  |
| InventoryOptimization/joint_replenishment | frontier-v1-main | 1.0 | 0.303423 |  |
| InventoryOptimization/tree_gsm_safety_stock | frontier-v1-main | 1.0 | 0.38126 |  |
| JobShop/abz | frontier-v1-main | 1.0 | 80.5042 |  |
| JobShop/ft | frontier-v1-main | 1.0 | 80.3472 |  |
| JobShop/la | frontier-v1-main | 1.0 | 83.9445 |  |
| JobShop/orb | frontier-v1-main | 1.0 | 79.4454 |  |
| JobShop/swv | frontier-v1-main | 1.0 | 81.6325 |  |
| JobShop/ta | frontier-v1-main | 1.0 | 78.8 |  |
| JobShop/yn | frontier-v1-main | 1.0 | 76.8833 |  |
| KernelEngineering/FlashAttention | frontier-v1-kernel | 1.0 | 116.553 |  |
| KernelEngineering/MLA | frontier-v1-kernel | 1.0 | 0.756218 |  |
| KernelEngineering/TriMul | frontier-v1-kernel | 1.0 | 47.674 |  |
| MolecularMechanics/diverse_conformer_portfolio | openff-dev | 1.0 | 278.216 |  |
| MolecularMechanics/torsion_profile_fitting | openff-dev | 1.0 | 34.7442 |  |
| MolecularMechanics/weighted_parameter_coverage | openff-dev | 1.0 | 9.07776 |  |
| Optics/adaptive_constrained_dm_control | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/adaptive_energy_aware_control | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/adaptive_fault_tolerant_fusion | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/adaptive_temporal_smooth_control | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/fiber_dsp_mode_scheduling | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/fiber_guardband_spectrum_packing | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/fiber_mcs_power_scheduling | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/fiber_wdm_channel_power_allocation | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/holographic_multifocus_power_ratio | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/holographic_multiplane_focusing | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/holographic_multispectral_focusing | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/holographic_polarization_multiplexing | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/phase_dammann_uniform_orders | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/phase_fourier_pattern_holography | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/phase_large_scale_weighted_spot_array | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| Optics/phase_weighted_multispot_single_plane | frontier-v1-main | 0.0 | -1e+18 | frontier-v1-main missing benchmarks/Optics/requirements.txt deps |
| ParticlePhysics/MuonTomography | frontier-eval-2(default) | 1.0 | 199.32 |  |
| PowerSystems/EV2GymSmartCharging | frontier-eval-2(no-conda-run) | 0.0 | 0 | missing ev2gym in frontier-eval-2 |
| PyPortfolioOpt/cvar_stress_control | frontier-v1-main | 1.0 | 17.9402 |  |
| PyPortfolioOpt/discrete_rebalance_mip | frontier-v1-main | 1.0 | 37.4951 |  |
| PyPortfolioOpt/robust_mvo_rebalance | frontier-v1-main | 1.0 | 32.9804 |  |
| QuantumComputing/task_01_routing_qftentangled | frontier-v1-main | 1.0 | 0.208987 |  |
| QuantumComputing/task_02_clifford_t_synthesis | frontier-v1-main | 1.0 | 3.06162 |  |
| QuantumComputing/task_03_cross_target_qaoa | frontier-v1-main | 1.0 | 2.41491 |  |
| ReactionOptimisation/dtlz2_pareto | frontier-v1-summit | 0.0 | -1e+18 | frontier-v1-summit install failed; env missing numpy/summit stack |
| ReactionOptimisation/mit_case1_mixed | frontier-v1-summit | 0.0 | -1e+18 | frontier-v1-summit install failed; env missing numpy/summit stack |
| ReactionOptimisation/reizman_suzuki_pareto | frontier-v1-summit | 0.0 | -1e+18 | frontier-v1-summit install failed; env missing numpy/summit stack |
| ReactionOptimisation/snar_multiobjective | frontier-v1-summit | 0.0 | -1e+18 | frontier-v1-summit install failed; env missing numpy/summit stack |
| Robotics/CoFlyersVasarhelyiTuning | frontier-eval-2(no-conda-run) | 1.0 | 45.6286 |  |
| Robotics/DynamicObstacleAvoidanceNavigation | frontier-v1-main | 1.0 | 0.0722022 |  |
| Robotics/PIDTuning | frontier-v1-main | 1.0 | 0.0366268 |  |
| Robotics/QuadrupedGaitOptimization | frontier-v1-main | 1.0 | 0.0221543 |  |
| Robotics/RobotArmCycleTimeOptimization | frontier-v1-main | 1.0 | 0.292193 |  |
| Robotics/UAVInspectionCoverageWithWind | frontier-v1-main | 1.0 | 28.8519 |  |
| SingleCellAnalysis/predict_modality | frontier-v1-main | 0.0 | 0 | baseline timed out inside predict_modality evaluator budget |
| StructuralOptimization/ISCSO2015 | frontier-eval-2(default) | 1.0 | -5401.59 |  |
| StructuralOptimization/ISCSO2023 | frontier-eval-2(default) | 1.0 | -7.78132e+07 |  |
| StructuralOptimization/PyMOTOSIMPCompliance | frontier-eval-2(default) | 1.0 | 4.83481 |  |
| StructuralOptimization/TopologyOptimization | frontier-eval-2(default) | 1.0 | -195.915 |  |
| SustainableDataCenterControl/hand_written_control | frontier-v1-sustaindc | 1.0 | 8.3259 |  |
| WirelessChannelSimulation/HighReliableSimulation | frontier-eval-2(default) | 1.0 | 215.054 |  |
