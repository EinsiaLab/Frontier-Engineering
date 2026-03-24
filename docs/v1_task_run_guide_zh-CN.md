# README 中 `v1` 任务运行与环境核对

本文整理主仓库 [README.md](../README.md) 中标记为 `v1` 的任务，汇总：

- 对应 README/任务文档里的运行方式
- README 里是否写了资源或时间要求
- 这些任务在当前已有 Conda 环境中的建议运行环境
- 现有环境是否已经满足文档要求

## 约定

- 统一评测驱动环境统一按 `frontier-eval-2` 处理，也就是在这个环境里执行 `python -m frontier_eval`。
- 下文的“任务环境”指任务本地 evaluator 或 unified runtime 实际应该使用的环境。
- 除特别说明外，统一评测都可以套用这个模板：

```bash
conda run -n frontier-eval-2 python -m frontier_eval \
  task=unified \
  task.benchmark=<BENCHMARK_ID> \
  [task.runtime.conda_env=<TASK_ENV>] \
  algorithm.iterations=0
```

- `Malloc Lab` 的本地运行命令来自 [Task.md](../benchmarks/ComputerSystems/MallocLab/Task.md)，因为它自己的 README 只说明了要修改哪个文件，没有写完整运行命令。
- `ReactionOptimisation`、`InventoryOptimization`、`JobShop`、`Optics` 的部分时间说明来自各自领域 README，因为单任务 README 并不总是写典型耗时。

## 环境核对摘要

- `frontier-eval-2`：已核对 `numpy`、`scipy`；另外也能导入 `anndata`，并已安装 `octave 10.3.0`，适合大多数通用 Python evaluator，也可直接覆盖 `MannedLunarLanding` 的 Octave validator。
- `kernel`：当前直接 `import torch` 仍报 `libcusparseLt.so.0` 相关动态库错误，本文不再把它视为当前机器上的稳定 GPU 任务环境。
- `bio`：已核对 `numpy`、`scipy`、`anndata`。
- `mqt`：已核对 `mqt.bench`。
- `stock`：已核对 `stockpyl`、`numpy`、`scipy`。
- `summit`：直接 `import summit` 会碰到 sklearn 兼容问题，但任务代码自带 `shared/summit_compat.py`，实测可在补丁后导入 `SnarBenchmark`。
- `optics`：是当前最完整的 Optics/GPU 环境；已核对 `aotools`、`diffractio`、`torch`、`torchvision`、`timm`、`einops`、`dotwiz`、`PyYAML`、`torchoptics`、`optic`、`ortools`、`slmsuite`，也可覆盖 `CarAerodynamicsSensing`。
- `opticommpy`：已核对 `optic`，但缺 `ortools`；适合 `fiber_wdm_channel_power_allocation`，不适合需要 exact oracle 的 fiber 任务。
- `pyportfolioopt`：已核对 `cvxpy`、`pypfopt`、`highspy`、`osqp`、`scs`；缺 `ecos`，但参考求解器有 `SCS/OSQP` fallback，已实测可跑小例子。
- `jobshop`：已核对 `ortools`、`job_shop_lib`。
- `motion`：已核对 `numpy`、`scipy`、`mujoco`、`pybullet`、`pybullet_data`，可覆盖四足和机械臂两个任务。
- `sustaindc`：`verification/evaluate.py` 可导入；单独 `import jax` 会因缺 `jaxlib` 失败，但 `hand_written_control` 当前 evaluator 路径不依赖它。
- 系统工具：`docker`、`gcc`、`g++`、`make` 存在；系统层面未发现 `matlab`，但 `octave` 已安装在 `frontier-eval-2` 环境中。

## Astrodynamics

### `MannedLunarLanding`

- 本地运行：推荐直接用 `frontier-eval-2` 执行 `conda run -n frontier-eval-2 python scripts/init.py` 生成 `results.txt`，然后用 `conda run -n frontier-eval-2 octave --no-gui --quiet --eval "addpath('eval'); aerodynamics_check_octave_full;"` 做 Octave 校验。`error_checking_program.m` 更偏 MATLAB 路径；在 Octave 下建议优先使用 `aerodynamics_check_octave_full.m`。
- 统一运行：`BENCHMARK_ID=Astrodynamics/MannedLunarLanding`。
- 资源/时间：README 没写典型耗时；明确要求 MATLAB/Octave validator。
- 任务环境：`frontier-eval-2`（同时承担 unified driver 与 Octave 运行时）。
- 核对：已满足。`frontier-eval-2` 已安装 `octave 10.3.0`，并实测 `scripts/init.py + aerodynamics_check_octave_full.m` 可通过完整校验。

## ParticlePhysics

### `MuonTomography`

- 本地运行：`conda activate frontier-eval-2 && cd benchmarks/ParticlePhysics/MuonTomography && python baseline/solution.py && python verification/evaluator.py solution.json`
- 统一运行：`BENCHMARK_ID=ParticlePhysics/MuonTomography`
- 资源/时间：README 未写典型耗时；只写了 `verification/requirements.txt` 当前仅要求 `numpy>=1.24.0`。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

## KernelEngineering

### `MLA`

- 本地运行：`cd benchmarks/KernelEngineering/MLA/verification` 后可跑 `python eval.py test mla_tests.txt`、`python eval.py benchmark mla_bench.txt`、`python eval.py leaderboard mla_bench.txt`。
- 统一运行：`BENCHMARK_ID=KernelEngineering/MLA`，README 示例附加 `task.runtime.conda_env=kernel` 和 `algorithm.oe.evaluator.timeout=1800`。
- 资源/时间：GPU 任务；README 示例把 unified evaluator timeout 设为 `1800s`。
- 任务环境：README 示例是 `kernel`。
- 核对：README 需求层面仍是 `kernel`，但当前机器上的 `kernel` 环境未通过重新核对；`torch` 仍有 CUDA 动态库问题，因此暂不把它视为稳定可用。

### `TriMul`

- 本地运行：`cd benchmarks/KernelEngineering/TriMul/verification` 后可跑 `python eval.py test tri_tests.txt`、`python eval.py benchmark tri_bench.txt`、`python eval.py benchmark tri_bench.txt`（leaderboard 反复重检）。
- 统一运行：`BENCHMARK_ID=KernelEngineering/TriMul`，README 示例附加 `task.runtime.conda_env=kernel` 和 `algorithm.oe.evaluator.timeout=1800`。
- 资源/时间：GPU 任务；README 示例同样把 unified evaluator timeout 设为 `1800s`。
- 任务环境：README 示例是 `kernel`。
- 核对：README 需求层面仍是 `kernel`，但当前机器上的 `kernel` 环境未通过重新核对；`torch` 仍有 CUDA 动态库问题，因此暂不把它视为稳定可用。

### `FlashAttention`

- 本地运行：先 `pip install -r verification/requirements-gpumode.txt`，然后在 `benchmarks/KernelEngineering/FlashAttention/verification` 下依次跑 `python eval.py test flash_attn_tests.txt`、`python eval.py benchmark flash_attn_bench.txt`、`python eval.py leaderboard flash_attn_bench.txt`。
- 统一运行：`BENCHMARK_ID=KernelEngineering/FlashAttention`，README 示例附加 `task.runtime.conda_env=kernel`；当前机器若要直接运行，更推荐改成 `task.runtime.conda_env=optics`。
- 资源/时间：README 明确要求 CUDA-enabled PyTorch 和 Triton；属于 GPU 任务；未写 wall-clock。
- 任务环境：当前机器推荐 `optics`。
- 核对：`optics` 环境已实测通过 `flash_attn_tests.txt` correctness test；README 示例里的 `kernel` 环境当前本机未重验通过。

## SingleCellAnalysis

### `predict_modality`

- 本地运行：`python benchmarks/SingleCellAnalysis/predict_modality/baseline/run_mean_per_gene.py --output prediction.h5ad`，然后 `python benchmarks/SingleCellAnalysis/predict_modality/verification/evaluate_predict_modality.py --prediction prediction.h5ad`
- 统一运行：`BENCHMARK_ID=SingleCellAnalysis/predict_modality`
- 资源/时间：README 未写典型耗时；提到数据来自 OpenProblems S3。
- 任务环境：推荐 `bio`；`frontier-eval-2` 也已核对可导入 `anndata`。
- 核对：已满足。

## QuantumComputing

### `routing qftentangled`

- 本地运行：`cd benchmarks/QuantumComputing/task_01_routing_qftentangled && python verification/evaluate.py`；可加 `--artifact-dir` / `--json-out`。
- 统一运行：README 没写 unified 命令；主 README 中显示为 v1，目录为 `task_01_routing_qftentangled`。
- 资源/时间：README 未写时间或资源要求。
- 任务环境：`mqt`
- 核对：已满足，`mqt.bench` 可导入。

### `clifford t synthesis`

- 本地运行：`cd benchmarks/QuantumComputing/task_02_clifford_t_synthesis && python verification/evaluate.py`；可加 `--artifact-dir` / `--json-out`。
- 统一运行：README 没写 unified 命令；主 README 中显示为 v1，目录为 `task_02_clifford_t_synthesis`。
- 资源/时间：README 未写时间或资源要求。
- 任务环境：`mqt`
- 核对：已满足，`mqt.bench` 可导入。

### `cross target qaoa`

- 本地运行：`cd benchmarks/QuantumComputing/task_03_cross_target_qaoa && python verification/evaluate.py`；可加 `--artifact-dir` / `--json-out`。
- 统一运行：README 没写 unified 命令；主 README 中显示为 v1，目录为 `task_03_cross_target_qaoa`。
- 资源/时间：README 未写时间或资源要求。
- 任务环境：`mqt`
- 核对：已满足，`mqt.bench` 可导入。

## Cryptographic

### `AES-128 CTR`

- 本地运行：README 只给了 unified 入口，没有单独写本地验证命令。
- 统一运行：`BENCHMARK_ID=Cryptographic/AES-128`
- 资源/时间：README 未写典型耗时；任务本质上依赖 C/C++ 编译工具链。
- 任务环境：`frontier-eval-2` driver + 系统 `gcc/g++/make`
- 核对：已满足，系统编译工具存在。

### `SHA-256`

- 本地运行：README 只给了 unified 入口，没有单独写本地验证命令。
- 统一运行：`BENCHMARK_ID=Cryptographic/SHA-256`
- 资源/时间：README 未写典型耗时；任务本质上依赖 C/C++ 编译工具链。
- 任务环境：`frontier-eval-2` driver + 系统 `gcc/g++/make`
- 核对：已满足，系统编译工具存在。

### `SHA3-256`

- 本地运行：README 只给了 unified 入口，没有单独写本地验证命令。
- 统一运行：`BENCHMARK_ID=Cryptographic/SHA3-256`
- 资源/时间：README 未写典型耗时；任务本质上依赖 C/C++ 编译工具链。
- 任务环境：`frontier-eval-2` driver + 系统 `gcc/g++/make`
- 核对：已满足，系统编译工具存在。

## CommunicationEngineering

### `LDPCErrorFloor`

- 本地运行：`python benchmarks/CommunicationEngineering/LDPCErrorFloor/verification/evaluator.py benchmarks/CommunicationEngineering/LDPCErrorFloor/scripts/init.py`
- 统一运行：`BENCHMARK_ID=CommunicationEngineering/LDPCErrorFloor`，README 额外要求 `algorithm.oe.evaluator.timeout=60`
- 资源/时间：README 明确写了“takes a long time to run”，并把 unified timeout 提到 `60s`。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

### `PMDSimulation`

- 本地运行：`python benchmarks/CommunicationEngineering/PMDSimulation/verification/evaluator.py benchmarks/CommunicationEngineering/PMDSimulation/scripts/init.py`
- 统一运行：`BENCHMARK_ID=CommunicationEngineering/PMDSimulation`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

### `RayleighFadingBER`

- 本地运行：`python benchmarks/CommunicationEngineering/RayleighFadingBER/verification/evaluator.py benchmarks/CommunicationEngineering/RayleighFadingBER/scripts/init.py`
- 统一运行：`BENCHMARK_ID=CommunicationEngineering/RayleighFadingBER`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

## EnergyStorage

### `BatteryFastChargingProfile`

- 本地运行：`python benchmarks/EnergyStorage/BatteryFastChargingProfile/verification/evaluator.py benchmarks/EnergyStorage/BatteryFastChargingProfile/scripts/init.py`；也可附加 `--config references/battery_config.json`
- 统一运行：`BENCHMARK_ID=EnergyStorage/BatteryFastChargingProfile`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

### `BatteryFastChargingSPMe`

- 本地运行：`python benchmarks/EnergyStorage/BatteryFastChargingSPMe/verification/evaluator.py benchmarks/EnergyStorage/BatteryFastChargingSPMe/scripts/init.py`；也可附加 `--config references/battery_config.json`
- 统一运行：`BENCHMARK_ID=EnergyStorage/BatteryFastChargingSPMe`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

## SustainableDataCenterControl

### `hand_written_control`

- 本地运行：`conda run -n sustaindc python benchmarks/SustainableDataCenterControl/hand_written_control/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=SustainableDataCenterControl/hand_written_control`，附加 `task.runtime.conda_env=sustaindc`
- 资源/时间：README 给出直跑约 `19.8s`，`iterations=0` unified 约 `25.8s`，且在默认 `300s` timeout 内。
- 任务环境：`sustaindc`
- 核对：部分满足。当前 evaluator 可导入，但环境里的 `jax` 单独导入仍因缺 `jaxlib` 失败；对当前 `hand_written_control` 路径不构成已知阻塞。

## ReactionOptimisation

说明：三个 v1 子任务的时间和 unified 运行说明主要来自 [benchmarks/ReactionOptimisation/README.md](../benchmarks/ReactionOptimisation/README.md)。

### `snar_multiobjective`

- 本地运行：`conda run -n summit python benchmarks/ReactionOptimisation/snar_multiobjective/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=ReactionOptimisation/snar_multiobjective`，附加 `task.runtime.conda_env=summit`
- 资源/时间：领域 README 给出直跑约 `122s`，`iterations=0` unified 约 `137s`；慢 CPU 建议把 evaluator timeout 提到 `600s`。
- 任务环境：`summit`
- 核对：已满足。直接 `import summit` 会报 sklearn 兼容问题，但任务自带兼容补丁后已验证可导入 benchmark。

### `mit_case1_mixed`

- 本地运行：`conda run -n summit python benchmarks/ReactionOptimisation/mit_case1_mixed/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=ReactionOptimisation/mit_case1_mixed`，附加 `task.runtime.conda_env=summit`
- 资源/时间：领域 README 给出直跑约 `106s`，`iterations=0` unified 约 `61s`。
- 任务环境：`summit`
- 核对：已满足。直接 `import summit` 会报 sklearn 兼容问题，但任务自带兼容补丁后已验证可导入 benchmark。

### `reizman_suzuki_pareto`

- 本地运行：`conda run -n summit python benchmarks/ReactionOptimisation/reizman_suzuki_pareto/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=ReactionOptimisation/reizman_suzuki_pareto`，附加 `task.runtime.conda_env=summit`
- 资源/时间：领域 README 给出直跑约 `112s`，`iterations=0` unified 约 `130s`；慢 CPU 建议把 evaluator timeout 提到 `600s`。
- 任务环境：`summit`
- 核对：已满足。直接 `import summit` 会报 sklearn 兼容问题，但任务自带兼容补丁后已验证可导入 benchmark。

## Optics

说明：Optics v1 子任务的单任务 README 提供了本地运行方法；典型耗时主要来自 [benchmarks/Optics/README.md](../benchmarks/Optics/README.md)。当前最推荐的统一任务环境是 `optics`，因为它已经同时具备 `aotools`、`diffractio`、`torchoptics`、`optic`、`ortools`、`slmsuite` 等依赖。

### `adaptive_temporal_smooth_control`

- 本地运行：`cd benchmarks/Optics/adaptive_temporal_smooth_control && python verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=Optics/adaptive_temporal_smooth_control`
- 资源/时间：领域 README 给出 `adaptive_*` 典型单次评测约 `6-15s`。
- 任务环境：推荐 `optics`；`aotools` 也可。
- 核对：已满足。

### `adaptive_fault_tolerant_fusion`

- 本地运行：`cd benchmarks/Optics/adaptive_fault_tolerant_fusion && python verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=Optics/adaptive_fault_tolerant_fusion`
- 资源/时间：领域 README 给出 `adaptive_*` 典型单次评测约 `6-15s`。
- 任务环境：推荐 `optics`；`aotools` 也可。
- 核对：已满足。

### `phase_fourier_pattern_holography`

- 本地运行：`PYTHONPATH=. python benchmarks/Optics/phase_fourier_pattern_holography/baseline/init.py`，然后 `PYTHONPATH=. python benchmarks/Optics/phase_fourier_pattern_holography/verification/validate.py`
- 统一运行：`BENCHMARK_ID=Optics/phase_fourier_pattern_holography`
- 资源/时间：领域 README 给出 `phase_*` 典型单次评测约 `8-20s`。
- 任务环境：推荐 `optics`；`diffractio`/`torchoptics` 也都带有 `slmsuite`。
- 核对：已满足。

### `phase_dammann_uniform_orders`

- 本地运行：`PYTHONPATH=. python benchmarks/Optics/phase_dammann_uniform_orders/baseline/init.py`，然后 `PYTHONPATH=. python benchmarks/Optics/phase_dammann_uniform_orders/verification/validate.py`
- 统一运行：`BENCHMARK_ID=Optics/phase_dammann_uniform_orders`
- 资源/时间：领域 README 给出 `phase_*` 典型单次评测约 `8-20s`。
- 任务环境：推荐 `optics`；`diffractio` 也可。
- 核对：已满足。

### `fiber_wdm_channel_power_allocation`

- 本地运行：`python benchmarks/Optics/fiber_wdm_channel_power_allocation/verification/run_validation.py`
- 统一运行：`BENCHMARK_ID=Optics/fiber_wdm_channel_power_allocation`
- 资源/时间：领域 README 给出 `fiber_*` 典型单次评测约 `7-20s`。
- 任务环境：推荐 `optics`；`opticommpy` 也可。
- 核对：已满足。

### `fiber_mcs_power_scheduling`

- 本地运行：`python benchmarks/Optics/fiber_mcs_power_scheduling/verification/run_validation.py`
- 统一运行：`BENCHMARK_ID=Optics/fiber_mcs_power_scheduling`
- 资源/时间：领域 README 给出 `fiber_*` 典型单次评测约 `7-20s`；单任务 README 说明 `ortools` 只在 `--oracle-mode exact` 时需要，默认 `auto` 会 fallback。
- 任务环境：推荐 `optics`
- 核对：已满足。`opticommpy` 环境缺 `ortools`，所以如果想跑 exact oracle，不建议用 `opticommpy`。

### `fiber_guardband_spectrum_packing`

- 本地运行：`python benchmarks/Optics/fiber_guardband_spectrum_packing/verification/run_validation.py`
- 统一运行：`BENCHMARK_ID=Optics/fiber_guardband_spectrum_packing`
- 资源/时间：领域 README 给出 `fiber_*` 典型单次评测约 `7-20s`；单任务 README 说明 `ortools` 只在 `--oracle-mode exact_geometry` 时需要，默认 `auto` 会 fallback。
- 任务环境：推荐 `optics`
- 核对：已满足。`opticommpy` 环境缺 `ortools`，所以如果想跑 exact oracle，不建议用 `opticommpy`。

### `holographic_multifocus_power_ratio`

- 本地运行：`python benchmarks/Optics/holographic_multifocus_power_ratio/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=Optics/holographic_multifocus_power_ratio`
- 资源/时间：领域 README 给出 `holographic_*` 典型单次评测约 `170-260s`；慢 CPU 可能超过默认 `300s`，领域 README 建议必要时把 timeout 提到 `600s`。
- 任务环境：推荐 `optics`；`torchoptics` 也可。
- 核对：已满足。

### `holographic_multiplane_focusing`

- 本地运行：`python benchmarks/Optics/holographic_multiplane_focusing/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=Optics/holographic_multiplane_focusing`
- 资源/时间：领域 README 给出 `holographic_*` 典型单次评测约 `170-260s`；慢 CPU 可能超过默认 `300s`，领域 README 建议必要时把 timeout 提到 `600s`。
- 任务环境：推荐 `optics`；`torchoptics` 也可。
- 核对：已满足。

## ComputerSystems

### `Malloc Lab`

- 本地运行：`cd benchmarks/ComputerSystems/MallocLab/malloclab-handout && make && ./mdriver -V`
- 统一运行：当前仓库的 unified metadata 会走 `benchmarks/ComputerSystems/MallocLab/frontier_eval/run_eval.sh`，其中实际执行 `make clean && make && ./mdriver -V`。
- 资源/时间：README/Task 没写典型耗时；需要 C 编译工具链。
- 任务环境：`frontier-eval-2` driver + 系统 `make/gcc/g++`
- 核对：已满足。

## EngDesign

说明：主 README 的这一行对应 7 个 EngDesign 子任务；仓库里只有领域级 README，没有单独的子任务 README。

### `CY_03`

- 本地/统一运行：先按 README 准备 Docker 登录、构建 `engdesign-sim` 镜像；评测命令为 `task=engdesign`，而不是 `task=unified`。
- 统一运行：`export ENGDESIGN_EVAL_MODE=docker`、`export ENGDESIGN_DOCKER_IMAGE=engdesign-sim` 后执行 `conda run -n frontier-eval-2 python -m frontier_eval task=engdesign algorithm=openevolve algorithm.iterations=10`
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `WJ_01`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `XY_05`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `AM_02`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `AM_03`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `YJ_02`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

### `YJ_03`

- 本地/统一运行：同 `CY_03`。
- 统一运行：同 `CY_03`。
- 资源/时间：README 未写典型耗时；明确依赖 Docker。
- 任务环境：`frontier-eval-2` driver + Docker
- 核对：已满足，系统 `docker` 存在。

## InventoryOptimization

说明：这些单任务 README 里还保留了旧路径 `tasks/...`；当前仓库实际目录是 `benchmarks/InventoryOptimization/...`。下文统一按当前仓库路径写。

### `tree_gsm_safety_stock`

- 本地运行：`python benchmarks/InventoryOptimization/tree_gsm_safety_stock/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=InventoryOptimization/tree_gsm_safety_stock`，附加 `task.runtime.conda_env=stock`
- 资源/时间：领域 README 给出约 `3-4s`
- 任务环境：`stock`
- 核对：已满足。

### `general_meio`

- 本地运行：`python benchmarks/InventoryOptimization/general_meio/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=InventoryOptimization/general_meio`，附加 `task.runtime.conda_env=stock`
- 资源/时间：领域 README 给出约 `20-25s`
- 任务环境：`stock`
- 核对：已满足。

### `joint_replenishment`

- 本地运行：`python benchmarks/InventoryOptimization/joint_replenishment/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=InventoryOptimization/joint_replenishment`，附加 `task.runtime.conda_env=stock`
- 资源/时间：领域 README 给出约 `1s`
- 任务环境：`stock`
- 核对：已满足。

### `finite_horizon_dp`

- 本地运行：`python benchmarks/InventoryOptimization/finite_horizon_dp/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=InventoryOptimization/finite_horizon_dp`，附加 `task.runtime.conda_env=stock`
- 资源/时间：领域 README 给出约 `6s`
- 任务环境：`stock`
- 核对：已满足。

### `disruption_eoqd`

- 本地运行：`python benchmarks/InventoryOptimization/disruption_eoqd/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=InventoryOptimization/disruption_eoqd`，附加 `task.runtime.conda_env=stock`
- 资源/时间：领域 README 给出约 `3s`
- 任务环境：`stock`
- 核对：已满足。

## PyPortfolioOpt

### `robust_mvo_rebalance`

- 本地运行：`conda run -n pyportfolioopt python benchmarks/PyPortfolioOpt/robust_mvo_rebalance/verification/evaluate.py`
- 统一运行：`BENCHMARK_ID=PyPortfolioOpt/robust_mvo_rebalance`，附加 `task.runtime.conda_env=pyportfolioopt`
- 资源/时间：README 写明 `iterations=0` 一次通常约 `8-15s`。
- 任务环境：`pyportfolioopt`
- 核对：部分满足。`ecos` 缺失，不完全等价于 requirements 文件；但 `SCS/OSQP` fallback 仍在，参考求解器导入与小规模求解都已实测通过。

## JobShop

说明：JobShop README 的命令以 `JobShop/...` 为相对路径写法；当前仓库对应绝对目录是 `benchmarks/JobShop/...`。unified 路径推荐直接用 `task.runtime.python_path=/data_storage/chihh2311/.conda/envs/jobshop/bin/python`。

### `abz`

- 本地运行：`python benchmarks/JobShop/abz/verification/evaluate.py --max-instances 2 --reference-time-limit 5`
- 统一运行：`BENCHMARK_ID=JobShop/abz`，附加 `task.runtime.python_path=/data_storage/chihh2311/.conda/envs/jobshop/bin/python` 与 `task.runtime.use_conda_run=false`
- 资源/时间：领域 README 给出默认全实例粗略上界约 `50s+`
- 任务环境：`jobshop`
- 核对：已满足。

### `swv`

- 本地运行：`python benchmarks/JobShop/swv/verification/evaluate.py --max-instances 2 --reference-time-limit 5`
- 统一运行：`BENCHMARK_ID=JobShop/swv`，附加 `task.runtime.python_path=/data_storage/chihh2311/.conda/envs/jobshop/bin/python` 与 `task.runtime.use_conda_run=false`
- 资源/时间：领域 README 给出默认全实例粗略上界约 `200s+`
- 任务环境：`jobshop`
- 核对：已满足。

### `ta`

- 本地运行：`python benchmarks/JobShop/ta/verification/evaluate.py --max-instances 2 --reference-time-limit 5`
- 统一运行：`BENCHMARK_ID=JobShop/ta`，附加 `task.runtime.python_path=/data_storage/chihh2311/.conda/envs/jobshop/bin/python` 与 `task.runtime.use_conda_run=false`
- 资源/时间：领域 README 给出默认全实例粗略上界约 `800s+`
- 任务环境：`jobshop`
- 核对：已满足。

## StructuralOptimization

### `ISCSO2015`

- 本地运行：`python benchmarks/StructuralOptimization/ISCSO2015/scripts/init.py`，然后 `python benchmarks/StructuralOptimization/ISCSO2015/verification/evaluator.py benchmarks/StructuralOptimization/ISCSO2015/scripts/init.py`
- 统一运行：`BENCHMARK_ID=StructuralOptimization/ISCSO2015`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

### `ISCSO2023`

- 本地运行：`python benchmarks/StructuralOptimization/ISCSO2023/scripts/init.py`，然后 `python benchmarks/StructuralOptimization/ISCSO2023/verification/evaluator.py benchmarks/StructuralOptimization/ISCSO2023/scripts/init.py`
- 统一运行：`BENCHMARK_ID=StructuralOptimization/ISCSO2023`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

### `TopologyOptimization`

- 本地运行：先 `pip install -r benchmarks/StructuralOptimization/TopologyOptimization/verification/requirements.txt`，再 `cd benchmarks/StructuralOptimization/TopologyOptimization && python scripts/init.py`，最后 `python verification/evaluator.py scripts/init.py`
- 统一运行：`BENCHMARK_ID=StructuralOptimization/TopologyOptimization`
- 资源/时间：README 未写典型耗时。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

## Robotics

### `DynamicObstacleAvoidanceNavigation`

- 本地运行：`cd benchmarks/Robotics/DynamicObstacleAvoidanceNavigation && python baseline/solution.py && python verification/evaluator.py --submission submission.json`
- 统一运行：`BENCHMARK_ID=Robotics/DynamicObstacleAvoidanceNavigation`
- 资源/时间：README 未写典型耗时。
- 任务环境：推荐 `motion`；`frontier-eval-2` 也可满足当前 `numpy` 依赖。
- 核对：已满足。

### `QuadrupedGaitOptimization`

- 本地运行：当前机器可直接用 `motion` 执行：`conda run -n motion python baseline/solution.py`，然后 `conda run -n motion python verification/evaluator.py --submission submission.json`；README 也提供 Docker 路径。
- 统一运行：`BENCHMARK_ID=Robotics/QuadrupedGaitOptimization`，README 要求显式提供 `task.runtime.conda_env=<your_env>`
- 资源/时间：README 未写典型耗时；明确依赖 `mujoco`，并提供 Docker evaluator。
- 任务环境：`motion`
- 核对：已满足。`motion` 已安装 `mujoco`，并实测 baseline + evaluator 可跑通。

### `RobotArmCycleTimeOptimization`

- 本地运行：当前机器可直接用 `motion` 执行：`conda run -n motion python baseline/solution.py`，然后 `conda run -n motion python verification/evaluator.py --submission submission.json`；README 也提供 Docker 路径。
- 统一运行：`BENCHMARK_ID=Robotics/RobotArmCycleTimeOptimization`，README 要求显式提供 `task.runtime.conda_env=<your_env>`
- 资源/时间：README 未写典型耗时；明确依赖 `pybullet`，并提供 Docker evaluator。
- 任务环境：`motion`
- 核对：已满足。`motion` 已安装 `pybullet`/`pybullet_data`，并实测 baseline + evaluator 可跑通。

### `PIDTuning`

- 本地运行：`cd benchmarks/Robotics/PIDTuning && python scripts/init.py`；然后可用 `python verification/evaluator.py --submission submission.json` 或直接 `python verification/evaluator.py scripts/init.py`
- 统一运行：`BENCHMARK_ID=Robotics/PIDTuning`
- 资源/时间：README 未写典型耗时。
- 任务环境：推荐 `motion`；`frontier-eval-2` 也可满足当前 `numpy` 依赖。
- 核对：已满足。

### `UAVInspectionCoverageWithWind`

- 本地运行：`cd benchmarks/Robotics/UAVInspectionCoverageWithWind && python baseline/solution.py && python verification/evaluator.py --submission submission.json`
- 统一运行：`BENCHMARK_ID=Robotics/UAVInspectionCoverageWithWind`
- 资源/时间：README 未写典型耗时。
- 任务环境：推荐 `motion`；`frontier-eval-2` 也可满足当前 `numpy` 依赖。
- 核对：已满足。

## Aerodynamics

### `CarAerodynamicsSensing`

- 本地运行：当前机器已把 `third_party/PhySense`、数据集、预训练 checkpoint 与 `references/car_surface_points.npy` 放到 README 要求的固定路径；可直接执行 `conda run -n optics python references/extract_car_mesh.py --data-dir data/physense_car_data --output references/car_surface_points.npy`、`conda run -n optics python baseline/solution.py --output submission.json`，最后 `CUDA_VISIBLE_DEVICES=<GPU_ID> conda run -n optics python verification/evaluator.py --submission submission.json`。
- 统一运行：`BENCHMARK_ID=Aerodynamics/CarAerodynamicsSensing`。README 示例写的是 `task.runtime.conda_env=kernel`，但在当前机器上建议改用 `task.runtime.conda_env=optics`
- 资源/时间：README 明确要求 GPU；还要求外部 PhySense repo、数据集和模型 checkpoint；未写典型 wall-clock。
- 任务环境：`optics` + `third_party/PhySense` + 数据集 + checkpoint + GPU
- 核对：已满足。`third_party/PhySense` 已克隆到默认路径，`data/physense_car_data` 与 `data/physense_car_ckpt/physense_transolver_car_best_base.pth` 已就位，`references/car_surface_points.npy` 已生成，并实测 baseline + evaluator 可跑通（baseline smoke score `0.955921`）。

## WirelessChannelSimulation

### `HighReliableSimulation`

- 本地运行：`python benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/evaluator.py benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py`
- 统一运行：`BENCHMARK_ID=WirelessChannelSimulation/HighReliableSimulation`
- 资源/时间：README 未写典型耗时，只要求输出非零 `runtime_s` 和 `valid=1.0`。
- 任务环境：`frontier-eval-2`
- 核对：已满足。

## 最后结论

- 直接可用且依赖基本齐全的任务环境：`frontier-eval-2`、`bio`、`mqt`、`stock`、`jobshop`、`optics`、`motion`
- 需要说明但总体可用的环境：`summit`、`pyportfolioopt`、`sustaindc`
- 本次已补齐并实测可运行的原“未满足”任务：
  - `MannedLunarLanding`：`frontier-eval-2` + `octave`
  - `QuadrupedGaitOptimization`：`motion` + `mujoco`
  - `RobotArmCycleTimeOptimization`：`motion` + `pybullet`
  - `CarAerodynamicsSensing`：`optics` + `PhySense` + 数据集 + checkpoint + GPU
- 当前仍需额外注意的项：
  - `kernel`：当前机器上的 `torch`/CUDA 动态库异常，不建议直接作为稳定 GPU 任务环境
  - `EngDesign`：Docker 镜像构建与容器运行
