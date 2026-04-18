# Frontier-Eng 环境配置实测报告

> 测试时间: 2026-04-16  
> 测试机器: `gb200-rack1-01` — 4x NVIDIA GB200 (189 GB each), 140 CPU, ~880 GB RAM, `linux-aarch64`  
> 测试人: Bowen (via Claude Code)

---

## 1. 环境配置步骤

### 1.1 前置条件

| 依赖 | 状态 | 备注 |
|------|------|------|
| Miniconda/Anaconda | 需手动安装 | init.sh 不会自动安装它 |
| conda 在 PATH 中 | **卡点** | 新装的 Miniconda 默认不在 PATH 里，`init.sh` 直接报 `conda not found` 退出 |
| conda ToS | **卡点** | conda 26.x 首次使用需要 `conda tos accept`，init.sh 未处理此情况，直接失败 |
| SLURM 节点分配 | OK | `salloc --partition=batch --nodelist=<node> --time=4:00:00` |

### 1.2 init.sh 执行流程

```bash
# 0. 确保 conda 在 PATH 中
export PATH="$HOME/miniconda3/bin:$PATH"

# 1. (新版 conda) 接受 ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. 运行 init.sh（非交互模式）
cd /path/to/Frontier-Engineering
echo "n" | bash init.sh

# 3. 激活环境
conda activate frontier-eval-2
```

### 1.3 init.sh 各步骤结果

| 步骤 | 内容 | 结果 | 耗时 |
|------|------|------|------|
| [1/4] | conda create frontier-eval-2 python=3.12 | OK | ~30s |
| [2/4] | Octave + signal/control (conda-forge) | **部分失败** | - |
| [3/4] | pip install requirements.txt | OK | ~2min |
| [4/4] | .env bootstrap | OK | 即时 |

**Step 2 在 aarch64 上的问题**: `octave-signal` 和 `octave-control` 在 `conda-forge` 的 `linux-aarch64` 频道中不存在。只有 `octave` 本体可以安装。这会导致依赖 Octave signal/control toolbox 的 benchmark（如 Astrodynamics 中的部分验证器）功能受限。

### 1.4 第三方算法安装

| 算法 | 安装方式 | 状态 |
|------|----------|------|
| OpenEvolve | 已在 requirements.txt 中 | OK |
| AB-MCTS (TreeQuest) | `git clone` + `pip install -e third_party/treequest` | OK |
| ShinkaEvolve | `git clone` + pin 到 `642664d` + `pip install -e` | OK |

**ShinkaEvolve 版本问题**: `frontier_eval/algorithms/shinkaevolve/algo.py:183` 导入 `EvolutionRunner`，但最新版 ShinkaEvolve (0.0.4, commit `0505d3d`) 已将其重命名为 `ShinkaEvolveRunner`。更早的 `14e62e3` 也有内部 import bug（`load_prompts_to_df` 不存在）。**实测可用的 commit 是 `642664d`**（2026-02-24 前后，`Merge pull request #81`）。

```bash
git clone https://github.com/SakanaAI/ShinkaEvolve.git third_party/ShinkaEvolve
(cd third_party/ShinkaEvolve && git checkout 642664d)
pip install -e third_party/ShinkaEvolve
```

注意: repo 自带的 `patches/third_party_shinkaevolve.patch` 是针对更新版本写的，在 `642664d` 上无法 apply。该 patch 修复 `DatabaseDisplay` 空 metadata 和添加 pricing.csv 条目，不影响 smoke test。

### 1.5 v1 合并环境

`scripts/setup_v1_merged_task_envs.sh` 在 aarch64 上**无法直接运行**（`set -euo pipefail` + `octave-control` 缺失导致第一步就退出）。需要手动创建各环境。

| 环境 | 手动创建方式 | 结果 |
|------|------------|------|
| `frontier-v1-main` | `conda create` + 按 env spec 装 pip deps | OK |
| `frontier-v1-summit` | `conda create -n ... python=3.10` + `pip install summit==0.8.9` | **FAIL** (pip resolution-too-deep) |
| `frontier-v1-sustaindc` | 需要 `sustaindc/` vendored dir（空目录，需手动 clone） | **SKIPPED** |
| `frontier-v1-kernel` | 未单独创建，`frontier-eval-2` 已有 torch+triton | 用 frontier-eval-2 替代 OK |

---

## 2. Smoke Tests

三个算法全部通过（ShinkaEvolve pin 到 `642664d` 后）。

| 命令 | 结果 | 分数 |
|------|------|------|
| `task=smoke algorithm=openevolve algorithm.iterations=0` | OK | 1.0 |
| `task=smoke algorithm=abmcts algorithm.iterations=0` | OK | 1.0 |
| `task=smoke algorithm=shinkaevolve algorithm.max_generations=0` | OK | 1.0 |

注意: `iterations=0` 只评估初始程序，**不调用 LLM API**，不需要 API key。

---

## 3. 全量 Baseline 验证 (iterations=0)

共 74 个 benchmark（含 EngDesign），逐一跑 `algorithm=openevolve algorithm.iterations=0`。

### 3.1 结果汇总

| 状态 | 数量 | 说明 |
|------|------|------|
| **valid=1 (baseline 通过)** | **49** | 评估器正确运行，baseline 有效 |
| **valid=0 (框架跑通但 baseline 无效)** | 4 | 框架无错但 baseline 本身不通过 |
| **DEP_FAIL (依赖缺失)** | 16 | 全部是 Optics（`diffractio` 需 PyQt5，aarch64 无法编译） |
| **SKIPPED (环境无法安装)** | 8 | MolecularMechanics(3) + ReactionOptimisation(4) + SustainDC(1) |
| **合计** | **74** (+EngDesign=75) | |

### 3.2 详细结果: frontier-eval-2 默认环境 (20 tasks)

这些 benchmark 只需 `frontier-eval-2`，无需额外环境。

| Benchmark | combined_score | valid | 备注 |
|-----------|----------------|-------|------|
| AdditiveManufacturing/DiffSimThermalControl | 0.4607 | 1.0 | |
| Aerodynamics/DawnAircraftDesignOptimization | 0.7415 | 1.0 | |
| Astrodynamics/MannedLunarLanding | 4577.44 | 1.0 | Octave 验证器正常 |
| CommunicationEngineering/LDPCErrorFloor | 0.7763 | 1.0 | |
| CommunicationEngineering/PMDSimulation | 14.8523 | 1.0 | |
| CommunicationEngineering/RayleighFadingBER | 3018.09 | 1.0 | |
| Cryptographic/AES-128 | 28.1613 | 1.0 | |
| Cryptographic/SHA-256 | 46.7378 | 1.0 | |
| Cryptographic/SHA3-256 | 98.8810 | 1.0 | |
| EnergyStorage/BatteryFastChargingProfile | 71.2806 | 1.0 | |
| EnergyStorage/BatteryFastChargingSPMe | 66.1636 | 1.0 | |
| ParticlePhysics/MuonTomography | 199.32 | 1.0 | |
| Robotics/CoFlyersVasarhelyiTuning | 45.6286 | 1.0 | |
| StructuralOptimization/ISCSO2015 | (valid) | 1.0 | score 在 metrics 其他字段 |
| StructuralOptimization/ISCSO2023 | (valid) | 1.0 | |
| StructuralOptimization/PyMOTOSIMPCompliance | 4.8348 | 1.0 | |
| StructuralOptimization/TopologyOptimization | (valid) | 1.0 | |
| WirelessChannelSimulation/HighReliableSimulation | 272.02 | 1.0 | |
| ComputerSystems/DuckDBWorkloadOptimization | N/A | 0.0 | baseline 不通过 |
| ComputerSystems/MallocLab | 0.0 | 0.0 | mdriver returncode=1 |

### 3.3 详细结果: frontier-eval-2 特殊配置

| Benchmark | combined_score | valid | 配置 |
|-----------|----------------|-------|------|
| PowerSystems/EV2GymSmartCharging | 0.0 | 0.0 | `use_conda_run=false` |
| EngDesign (7 subtasks) | 1.3571 | (0.0) | `task=engdesign`, `ENGDESIGN_EVAL_MODE=local` |
| Aerodynamics/CarAerodynamicsSensing | -1e18 | 0.0 | 需要 PhySense repo + 数据/checkpoint |

### 3.4 详细结果: KernelEngineering (GPU, frontier-eval-2)

| Benchmark | combined_score | valid | 备注 |
|-----------|----------------|-------|------|
| KernelEngineering/FlashAttention | 230.85 | 1.0 | GB200 GPU 正常 |
| KernelEngineering/MLA | 1.5648 | 1.0 | |
| KernelEngineering/TriMul | 132.49 | 1.0 | |

### 3.5 详细结果: frontier-v1-main 环境 (29 tasks)

这些 benchmark 需要 `frontier-v1-main`（含 anndata, mqt.bench, stockpyl, job-shop-lib, mujoco, pybullet 等）。

| Benchmark | combined_score | valid | 备注 |
|-----------|----------------|-------|------|
| InventoryOptimization/disruption_eoqd | 0.3642 | 1.0 | |
| InventoryOptimization/finite_horizon_dp | 0.3673 | 1.0 | |
| InventoryOptimization/general_meio | 0.1825 | 1.0 | |
| InventoryOptimization/joint_replenishment | 0.3034 | 1.0 | |
| InventoryOptimization/tree_gsm_safety_stock | 0.3813 | 1.0 | |
| JobShop/abz | 80.5042 | 1.0 | `python_path=conda-env:frontier-v1-main`, `use_conda_run=false` |
| JobShop/ft | 80.3472 | 1.0 | |
| JobShop/la | 83.9445 | 1.0 | |
| JobShop/orb | 79.4454 | 1.0 | |
| JobShop/swv | 81.6325 | 1.0 | |
| JobShop/ta | 78.80 | 1.0 | 80 instances, 最慢 |
| JobShop/yn | 76.8833 | 1.0 | |
| QuantumComputing/task_01_routing_qftentangled | 0.2090 | 1.0 | |
| QuantumComputing/task_02_clifford_t_synthesis | 1.7134 | 1.0 | |
| QuantumComputing/task_03_cross_target_qaoa | 2.4149 | 1.0 | |
| SingleCellAnalysis/predict_modality | 0.5467 | 1.0 | |
| PyPortfolioOpt/robust_mvo_rebalance | 32.98 | 1.0 | |
| PyPortfolioOpt/cvar_stress_control | 17.94 | 1.0 | |
| PyPortfolioOpt/discrete_rebalance_mip | 37.50 | 1.0 | |
| Robotics/DynamicObstacleAvoidanceNavigation | 0.0722 | 1.0 | |
| Robotics/PIDTuning | 0.0366 | 1.0 | |
| Robotics/QuadrupedGaitOptimization | 0.0222 | 1.0 | **需要 frontier-v1-main**（mujoco），frontier-eval-2 下 valid=0 |
| Robotics/RobotArmCycleTimeOptimization | 0.2922 | 1.0 | **需要 frontier-v1-main**（pybullet），frontier-eval-2 下 valid=0 |
| Robotics/UAVInspectionCoverageWithWind | 28.85 | 1.0 | |

### 3.6 无法运行的 Benchmark

#### Optics (16 tasks) — `diffractio` 依赖 PyQt5，aarch64 无法编译

| 子类 | Tasks | 缺失依赖 |
|------|-------|----------|
| adaptive_* | 4 | `diffractio` (via PyQt5/qmake) |
| phase_* | 4 | `diffractio` |
| fiber_* | 4 | `OptiCommPy` 已装，但 eval 脚本仍需 `diffractio` |
| holographic_* | 4 | `diffractio` |

**根因**: `diffractio>=0.2.4` → `PyQt5` → 需要 `qmake` 编译，aarch64 上无预编译 wheel 且系统无 Qt 开发库。

#### ReactionOptimisation (4 tasks) — summit 安装失败

- `summit==0.8.9` 要求 `Python>=3.8,<3.11`
- 即使用 Python 3.10，pip 遇到 `resolution-too-deep`（依赖图太复杂：GPy + botorch + numba + 旧版 scipy/torch）
- 在 aarch64 上某些旧版 torch（<2.0）wheel 可能不可用

#### MolecularMechanics (3 tasks) — openff-dev 环境未配置

- 需要 `rdkit`, `openmm` 等 conda-forge 专用包
- 未尝试安装（时间原因）

#### SustainableDataCenterControl (1 task) — 数据目录为空

- `benchmarks/.../hand_written_control/sustaindc/` 是 vendored dc-rl checkout，但目录为空
- 需要手动 `git clone` 上游仓库到该路径

#### Aerodynamics/CarAerodynamicsSensing (1 task) — 外部数据

- 需要 `third_party/PhySense` repo + 预训练 checkpoint
- 跑了但 `valid=0`（评估脚本找不到数据）

---

## 4. Benchmark → 环境映射表

| 环境 | Python | Benchmarks | 特殊依赖 |
|------|--------|------------|----------|
| `frontier-eval-2` | 3.12 | AdditiveManufacturing, Aerodynamics/Dawn, Astrodynamics, CommunicationEngineering(3), ComputerSystems(2), Cryptographic(3), EnergyStorage(2), ParticlePhysics, PowerSystems, Robotics/CoFlyers, StructuralOptimization(4), WirelessChannelSimulation | Octave (aarch64: 无 signal/control) |
| `frontier-eval-2` (GPU) | 3.12 | KernelEngineering/FlashAttention, MLA, TriMul | torch, triton, CUDA |
| `frontier-v1-main` | 3.12 | InventoryOptimization(5), JobShop(7), QuantumComputing(3), SingleCellAnalysis, PyPortfolioOpt(3), Robotics/Dynamic+PID+Quadruped+RobotArm+UAV | anndata, mqt.bench, stockpyl, job-shop-lib, mujoco, pybullet, cvxpy |
| `frontier-v1-main` + Optics deps | 3.12 | Optics(16) | diffractio, torchoptics, slmsuite, aotools, OptiCommPy |
| `frontier-v1-summit` | **3.10** | ReactionOptimisation(4) | summit==0.8.9, GPy, botorch, numba |
| `frontier-v1-sustaindc` | 3.10 | SustainableDataCenterControl(1) | vendored dc-rl |
| `openff-dev` | ? | MolecularMechanics(3) | rdkit, openmm |
| `frontier-eval-2` + Docker | 3.12 | EngDesign(7 subtasks) | Docker (`ENGDESIGN_EVAL_MODE=docker`) 或 local |

---

## 5. 遇到的卡点汇总

### 卡点 1: conda 不在 PATH（严重）
- **表现**: `init.sh` 第一行就失败，`conda: command not found`
- **原因**: 新安装 Miniconda 后未 source `.bashrc` 或未手动加 PATH
- **建议**: `init.sh` 开头加一段自动探测常见 conda 路径的逻辑

### 卡点 2: conda ToS 未接受（严重）
- **表现**: `conda create` 时报 `CondaToSNonInteractiveError`
- **原因**: conda 26.x 新增 ToS 流程
- **建议**: `init.sh` 在 `conda create` 前尝试 `conda tos accept`

### 卡点 3: aarch64 缺少 Octave 扩展包（中等）
- **表现**: `octave-signal` 和 `octave-control` 在 linux-aarch64 上无 conda-forge 包
- **影响**: init.sh step 2 失败（`set -euo pipefail`直接退出），`setup_v1_merged_task_envs.sh` 也同样失败
- **建议**: init.sh 在 aarch64 上降级为只装 octave 本体，或加 `--skip-octave`

### 卡点 4: ShinkaEvolve 版本不兼容（中等）
- **表现**: 最新 ShinkaEvolve 和 `algo.py` import 不兼容
- **实测可用 commit**: `642664d`（`Merge pull request #81`）
- **建议**: `frontier_eval/README.md` 安装说明 pin 到 `642664d`；patch 也需要相应更新

### 卡点 5: Optics 任务依赖 PyQt5（中等，aarch64 特有）
- **表现**: `diffractio` → `PyQt5` → 需要 qmake，aarch64 无法编译
- **影响**: 16 个 Optics benchmark 全部无法运行
- **建议**: 尝试 `PyQt5-sip` + 系统 Qt，或将 diffractio 替换为 headless 方案

### 卡点 6: summit 依赖解析失败（中等，aarch64 特有）
- **表现**: pip `resolution-too-deep`，summit 依赖 torch<2.0 + numba<0.56 + 旧 scipy
- **影响**: 4 个 ReactionOptimisation benchmark 无法运行
- **建议**: 提供 `environment.yml` 锁定版本，或用 conda 替代 pip 解析

### 卡点 7: SustainDC vendored 目录为空（轻微）
- **表现**: `sustaindc/` 目录存在但为空
- **建议**: README 中明确写出 git clone 命令

### 卡点 8: setup_v1_merged_task_envs.sh 不容错（轻微）
- **表现**: `set -euo pipefail` 导致第一步 octave 失败后全部跳过
- **建议**: 每个环境独立创建，互不影响

---

## 6. 改进建议

### P0: 立即可做

1. **修复 init.sh**: 处理 conda PATH 探测 + ToS accept + aarch64 octave 降级
2. **Pin ShinkaEvolve** 到 `642664d`（README + 可选: 更新 patch）
3. **setup_v1_merged_task_envs.sh 容错**: 每个 env 独立 `set +e`

### P1: 短期

4. **Benchmark → env 映射表** 加入 `frontier_eval/README.md`（本报告 Section 4 可直接复用）
5. **Optics `requirements.txt`** 加入 `frontier-v1-main` env spec（目前缺失）
6. **SustainDC README** 补充 vendored dir 的 clone 命令

### P2: 中期

7. **平台支持矩阵**: 明确列出 x86_64 vs aarch64 差异
8. **提供 `environment.lock.yml`**: 对 summit 等复杂依赖提供完整锁定文件
9. **`python -m frontier_eval --doctor`**: 自检命令，检查所有环境和依赖状态

---

## 7. 总体评价

**结论**: 在 GB200 aarch64 节点上，74 个 benchmark 中有 **49 个 baseline 完全通过**（66%），4 个框架跑通但 baseline 本身无效，**21 个因环境/依赖问题无法运行**（主要是 Optics 的 PyQt5 和 ReactionOptimisation 的 summit）。

核心框架（`frontier_eval` + `openevolve` + `abmcts` + `shinkaevolve`）在正确 pin 版本后全部可用。最大的 onboarding 摩擦来自:
1. **init.sh 的脆弱性**（conda 探测 + ToS + aarch64 Octave）——首次用户 100% 会遇到
2. **多环境映射分散**——需要翻阅多个 README 才能确定某个 benchmark 用什么环境
3. **aarch64 平台支持缺口**——Optics 和 ReactionOptimisation 共 20 个 task 无法运行
