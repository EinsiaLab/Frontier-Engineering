# Frontier-Engineering 复现实验记录（2026-04-16）

## 1. 目标与边界

本次复现目标：

- 按仓库文档配置评测环境并跑通代表性评测。
- 优先使用镜像站。
- 不修改系统默认 Python（尤其不动 `/usr/bin/python`）。
- 如果遇到可疑问题，先记录并跳过，再继续能正常推进的部分。

本次实际采用的隔离工作目录：

- 环境与镜像配置目录：`/GenSIvePFS/users/hhchi/frontier_eng_repro`
- Frontier-Engineering 仓库：`/GenSIvePFS/users/hhchi/Frontier-Engineering`

## 2. 我是如何 follow 文档的

我先阅读了这些主文档：

- `README_zh-CN.md`
- `frontier_eval/README_zh-CN.md`
- `docs/v1_task_run_guide_zh-CN.md`
- 若干 benchmark README

按文档总结出的实际流程是：

1. 先准备 `frontier_eval` 驱动环境。
2. 再按各 benchmark README 决定 task runtime。
3. 先跑 smoke test，再跑 direct verification / unified task。
4. 对长任务按 README 中给出的耗时和 timeout 提示处理，不把“长”误判为“坏”。

## 3. 镜像与环境配置

为了不污染全局 conda 目录，也不依赖系统 Python，我在专用目录里创建了 prefix env，并额外写了镜像配置：

- conda 镜像：清华 `conda-forge`
- pip 镜像：清华 PyPI

实际文件：

- `/GenSIvePFS/users/hhchi/frontier_eng_repro/.condarc`
- `/GenSIvePFS/users/hhchi/frontier_eng_repro/pip.conf`

### 3.1 驱动环境

创建了独立驱动环境：

- Python 路径：`/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python`

安装内容：

- `python=3.12`
- `octave`
- `octave-signal`
- `octave-control`
- `hydra-core`
- `openevolve`
- `PyYAML`
- `joblib`
- `ulid-py`
- `numpy`
- `pandas`
- `anndata`
- `scipy`

说明：

- 文档里的 `frontier_eval/requirements.txt` 直接安装会把 `torch` 和 CUDA 13 相关大包一并拉下来。
- 在当前镜像速度下，这一步非常慢，而且对本次 CPU 复现的 smoke / 多个 CPU benchmark 并非入口必需。
- 因此我改为先安装能跑通 `python -m frontier_eval` 和本次所选 CPU benchmark 的最小依赖集合。

这不是“魔改仓库代码”，只是按实际依赖补齐一个可运行驱动环境。

### 3.2 `summit` 环境尝试

我也按 `ReactionOptimisation/README_zh-CN.md` 尝试创建了 `summit` runtime：

- Python 路径：`/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python`

安装 `benchmarks/ReactionOptimisation/requirements.txt` 时会进一步拉取：

- `torch==1.13.1`
- wheel 体积约 887MB

第一次尝试时，在当前镜像速度下这条线过慢，因此我先中止并记录，没有让整轮复现卡在下载阶段。

后续在网络更好的机器上继续完成了该环境安装，并确认：

- `torch`
- `botorch`
- `GPy`
- `summit`

相关依赖已经可用于 `ReactionOptimisation` 的大部分 direct verification 和 unified 评测。

补充说明：

- 直接 `import summit` 仍会因为 `sklearn` 私有接口变动报错。
- 但 `ReactionOptimisation` 仓库代码大部分路径已经自带 `shared/summit_compat.py` 兼容补丁，因此 4 个子任务里有 3 个已经成功跑通。

## 4. 实际运行结果

## 4.1 框架 smoke test

命令：

```bash
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

结果：

- 成功
- `combined_score=1.0`

说明驱动环境和 `frontier_eval` 主入口已经可用。

## 4.2 Direct verification 成功项

### `ParticlePhysics/MuonTomography`

- 命令按文档执行成功。
- 结果：`score=199.32012533144325`

### `EnergyStorage/BatteryFastChargingProfile`

- 命令按文档执行成功。
- 结果：`combined_score=71.28056205398363`

### `CommunicationEngineering/PMDSimulation`

- 命令按文档执行成功。
- 结果：`combined_score=5.025988745349662`

### `CommunicationEngineering/LDPCErrorFloor`

- 命令按文档执行成功。
- 结果：`combined_score=0.06334632157007855`
- 任务总耗时约 `190.66s`
- 运行期间出现一条 `RuntimeWarning: divide by zero encountered in arctanh`，但最终结果有效，未构成阻塞

### `ReactionOptimisation/mit_case1_mixed`

- direct verification 成功。
- 结果：baseline 平均分 `87.30816873120955`
- reference 平均分 `92.95037797323579`
- `score_gap=5.64220924202624`

### `ReactionOptimisation/reizman_suzuki_pareto`

- direct verification 成功。
- 结果：baseline 平均分 `63.52020293211535`
- reference 平均分 `81.78254388306357`
- `score_gap=18.262340950948214`

### `ReactionOptimisation/snar_multiobjective`

- direct verification 成功。
- 结果：baseline 平均分 `57.523363292468865`
- reference 平均分 `86.1320487066511`
- `score_gap=28.60868541418224`

## 4.3 Unified task 成功项

这里有一个关键现实问题：

- 如果直接照默认配置使用 `task.runtime.conda_env=frontier-eval-2`，在我这种“prefix env”场景下，unified evaluator 会很快失败，表现为 `benchmark_returncode=1`、`combined_score=-1e18`。
- 改成显式：
  - `task.runtime.python_path=/.../frontier-eval-2/bin/python`
  - `task.runtime.use_conda_run=false`
  之后即可正常运行。

成功结果如下。

### `CommunicationEngineering/PMDSimulation`

- unified 成功
- `combined_score=4.981656938964705`

### `ParticlePhysics/MuonTomography`

- unified 成功
- `combined_score=199.32012533144325`

### `EnergyStorage/BatteryFastChargingProfile`

- unified 成功
- `combined_score=71.28056205398363`

### `Cryptographic/AES-128`

- unified 成功
- 编译与验证均通过
- `combined_score=18.31127521501438`

### `CommunicationEngineering/LDPCErrorFloor`

- `algorithm.oe.evaluator.timeout=60` 时超时，失败
- `algorithm.oe.evaluator.timeout=300` 时成功
- 成功结果：`combined_score=0.0759089011315022`
- unified 本次运行总耗时约 `160.10s`

这个结果说明：

- 该任务确实是“任务本身慢”，不是死锁或坏掉。
- 对它使用偏紧的 timeout 很容易误判失败。

### `ReactionOptimisation/mit_case1_mixed`

- unified 成功
- 运行命令需要显式：
  - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python`
  - `task.runtime.use_conda_run=false`
- `combined_score=87.30816873120955`
- `reference_score_mean=92.95037797323579`
- `score_gap=5.64220924202624`
- 本次 unified 耗时约 `53.20s`

### `ReactionOptimisation/reizman_suzuki_pareto`

- unified 成功
- 同样使用显式 `task.runtime.python_path` 指向 `summit` prefix env
- `combined_score=63.52020293211535`
- `reference_score_mean=81.78254388306357`
- `score_gap=18.262340950948214`
- 本次 unified 耗时约 `129.82s`

### `ReactionOptimisation/snar_multiobjective`

- unified 成功
- 同样使用显式 `task.runtime.python_path` 指向 `summit` prefix env
- `algorithm.oe.evaluator.timeout=600`
- `combined_score=57.523363292468865`
- `reference_score_mean=86.1320487066511`
- `score_gap=28.60868541418224`
- 本次 unified 耗时约 `132.51s`

## 5. 我跳过了什么，为什么

### 5.1 GPU / CUDA 强依赖任务

状态：

- 例如 `KernelEngineering/*`
- 本轮未优先推进

原因：

- 文档和已有 `v1` 核对指南已经提示这些任务对环境更敏感
- 本轮目标是先验证 public repo 的 CPU 主流程、统一入口、benchmark 文档可跟随性

## 6. 这次复现中暴露出的实际问题

### 问题 1：`frontier_eval/requirements.txt` 对 CPU 复现过重

表现：

- 会直接拉 `torch` 以及 CUDA 13 相关大包
- 对只想跑 smoke / CPU benchmark 的用户非常不友好

影响：

- 首次上手时间明显变长
- 镜像慢时很容易误以为仓库安装有问题

建议：

- 拆分为至少两套依赖
- 例如：
  - `frontier_eval/requirements-core.txt`
  - `frontier_eval/requirements-gpu.txt`

### 问题 2：文档默认更偏向“按 conda 环境名运行”，对 prefix env 指引不够

表现：

- 我是按“专用目录 + prefix env”创建环境的
- unified 默认 `task.runtime.conda_env=frontier-eval-2` 在这个场景下会失败
- 改成 `task.runtime.python_path=/path/to/python` + `task.runtime.use_conda_run=false` 后即可恢复正常

影响：

- 对不想污染全局 env 的用户不够友好
- 用户会看到 benchmark 很快失败，但不容易第一时间意识到是 runtime 指定方式的问题

建议：

- 在 `frontier_eval/README_zh-CN.md` 里新增一个“小节：prefix env / 非命名 conda env 的运行方式”
- 直接给出可复制命令

### 问题 3：长任务的 timeout 建议还可以更保守

表现：

- `LDPCErrorFloor` direct 实测约 `190s`
- unified 在 `timeout=60` 下失败
- 提到 `300` 后成功

建议：

- README 中对这类任务给出更明确的建议 timeout
- 不要只写“takes a long time to run”，最好附：
  - direct 典型耗时
  - unified 典型耗时
  - 慢机器建议 timeout

### 问题 4：README 对“最小成功路径”不够集中

现在的信息是有的，但分散在：

- 根 README
- `frontier_eval/README`
- benchmark README
- `docs/v1_task_run_guide_zh-CN.md`

对第一次上手的人，还是需要自己再拼一遍。

建议：

- 增加一份“最低成本复现路径”文档
- 目标是让用户在 10 分钟内跑通：
  - smoke
  - 1 个 direct verifier
  - 1 个 unified task

## 7. 如果想让这个 public repo 更易于入手，建议从哪边开始做

如果只能优先做几件事，我建议顺序如下：

### 第一优先级：拆分依赖

先把 `frontier_eval` 的安装拆成：

- core / CPU
- GPU / vision / heavy

这是最能直接改善首次上手体验的改动。

### 第二优先级：给出 prefix env 示例

补一个明确示例：

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=ParticlePhysics/MuonTomography \
  task.runtime.python_path=/path/to/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0
```

这能避免不少“为什么 direct 能跑、unified 失败”的问题。

### 第三优先级：提供官方“快速验仓”任务列表

建议在 README 里直接列一个“推荐新用户先跑”的 3~5 个任务清单，例如：

- `smoke`
- `ParticlePhysics/MuonTomography`
- `EnergyStorage/BatteryFastChargingProfile`
- `CommunicationEngineering/PMDSimulation`
- `Cryptographic/AES-128`

这些任务覆盖面已经不错，而且上手成本相对可控。

### 第四优先级：把长任务的时间预期写清楚

对 `LDPCErrorFloor` 这种任务，最好把下面三件事直接写在命令旁边：

- 典型 wall-clock
- 推荐 timeout
- 低频监控即可，不要误判卡死

### 第五优先级：先修复已验证到的 evaluator 兼容性问题

这类问题会直接破坏“用户照文档运行即可成功”的预期，优先级应该高于新增更多任务说明。

## 8. 本轮结论

结论是：

- 仓库的核心主流程是可复现的。
- `frontier_eval` 入口、direct verification、unified task 都能在本机成功跑通。
- `ReactionOptimisation` 中 `v1` 子任务已完成 direct + unified 复现。
- 但对首次上手用户来说，环境安装和 runtime 选择仍有明显门槛，尤其是：
  - `torch` / CUDA 大依赖过重
  - unified 默认的 env 选择对 prefix env 不够友好
  - 长任务 timeout 建议还不够稳妥
  - 个别 evaluator 还存在实际兼容性缺口

如果在发布前想优先降低上手摩擦，我建议先处理：

1. 依赖拆分
2. prefix env 文档
3. 快速验仓路径
4. 长任务 timeout / 耗时表
5. 已知 evaluator bug 修复

## 9. 后续补充进展（ReactionOptimisation）

这部分是我在前述记录之后继续推进 `ReactionOptimisation/*` 的增量结果。

### 9.1 `summit` runtime 实际安装路径

之前我把 `ReactionOptimisation` 暂时跳过，核心原因是：

- `pip install -r benchmarks/ReactionOptimisation/requirements.txt` 会触发 `summit==0.8.9`
- 进一步又会拉 `torch==1.13.1`
- 在默认 PyPI 路径下，下载速度极慢，而且默认会往 CUDA 依赖链走

在这台网络更好的机器上，我最终采用了“尽量少改、但避开慢路径”的做法：

1. 保留原有 prefix env：
   - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit`
2. 先用 `mamba --offline` 把本地已缓存到 `miniforge3/pkgs` 的关键依赖装入这个 prefix env：
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `scipy`
   - `numba`
   - `llvmlite`
   - `pytorch=1.13.1`（CPU build）
   - `botorch`
   - `gpytorch`
   - `linear_operator`
   - `GPy`
   - `GPyOpt`
   - `pyro-ppl`
   - `h5py`
   - `skorch`
   - `pymoo`
3. 再从本地 wheel 安装 `summit` 本体：
   - `summit-0.8.9-py3-none-any.whl`

关键结论：

- `summit` 顶层裸导入仍会因 `sklearn` 兼容性报 `_check_fit_params` 错误。
- 但仓库自带的 `benchmarks/ReactionOptimisation/shared/summit_compat.py` 会在任务代码里先打补丁。
- 因此实际按仓库 benchmark 入口跑时是可用的，不需要改 benchmark 代码。

### 9.2 `mit_case1_mixed` 已跑通

#### Direct verification

命令：

```bash
cd /GenSIvePFS/users/hhchi/Frontier-Engineering
FRONTIER_ENGINEERING_ROOT=/GenSIvePFS/users/hhchi/Frontier-Engineering \
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python \
benchmarks/ReactionOptimisation/mit_case1_mixed/verification/evaluate.py
```

结果：

- 成功
- baseline `mean score = 87.30816873120955`
- reference `mean score = 92.95037797323579`
- `score_gap = 5.64220924202624`

额外观察：

- README 里给的 direct 参考耗时约 `106s`
- 但本机这次实测明显更慢，量级已经接近 `300s`
- 因此 README 里的时间预期对这台机器偏乐观

#### Unified

我继续沿用 prefix env 友好的运行方式：

```bash
cd /GenSIvePFS/users/hhchi/Frontier-Engineering
FRONTIER_ENGINEERING_ROOT=/GenSIvePFS/users/hhchi/Frontier-Engineering \
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=ReactionOptimisation/mit_case1_mixed \
  task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0
```

第一次结果：

- 失败
- 不是环境错误，而是超时
- evaluator 记录：
  - `runtime_s=295.1581`
  - `timeout_budget_s=300`
  - `timeout=1`
  - `combined_score=-1e18`

把 timeout 提高后再次运行：

```bash
cd /GenSIvePFS/users/hhchi/Frontier-Engineering
FRONTIER_ENGINEERING_ROOT=/GenSIvePFS/users/hhchi/Frontier-Engineering \
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=ReactionOptimisation/mit_case1_mixed \
  task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=0 \
  algorithm.oe.evaluator.timeout=600
```

第二次结果：

- 成功
- `combined_score=87.30816873120955`
- `candidate_score_std=7.7161`
- `reference_score_mean=92.9504`
- `score_gap=5.6422`
- `runtime_s=350.9775`

这说明：

- `ReactionOptimisation` 在本机的 unified 耗时比 README 给出的参考值高很多
- 至少 `mit_case1_mixed` 这项不应继续假设默认 `300s` timeout 足够稳妥
- 在本机上，更保守的建议是先用 `algorithm.oe.evaluator.timeout=600`

### 9.3 `snar_multiobjective` direct verification 已成功

命令：

```bash
cd /GenSIvePFS/users/hhchi/Frontier-Engineering
FRONTIER_ENGINEERING_ROOT=/GenSIvePFS/users/hhchi/Frontier-Engineering \
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/summit/bin/python \
benchmarks/ReactionOptimisation/snar_multiobjective/verification/evaluate.py
```

结果：

- 成功
- baseline `mean score = 57.523363292468865`
- reference `mean score = 86.1320487066511`
- `score_gap = 28.60868541418224`

这次运行过程中的关键观察：

- 进程长时间持续高 CPU 运行（约 `280%~290%`）
- 没有卡死迹象
- 实际 wall-clock 明显高于 README 中 direct 约 `122s` 的参考值

当前判断：

- 这组任务在本机上可以跑通，但时间预期需要整体上调
- 对 `ReactionOptimisation/*`，本机不适合继续假设 README 中给出的 `~60s ~160s` 是稳妥上界
- 如果继续跑 unified，建议优先把 evaluator timeout 提高到 `600s`

## 10. v1 任务当前完成情况总表

下面这个表是基于本轮复现截至当前时刻的实际进度整理的。

状态约定：

- `完成`：已经实际跑通并拿到结果
- `部分完成`：只完成了 direct / 只完成了环境核对 / 或已确认还差 timeout 等运行参数
- `未开始`：尚未进入实际 benchmark 运行阶段

| Domain | v1 Task | Direct Verification | Unified | 当前状态 | 备注 |
|---|---|---|---|---|---|
| Astrodynamics | `MannedLunarLanding` | 基线脚本完成 | 完成 | 完成 | 手工直接调 `octave` 会因默认启动路径缺失 `odeset` 失败，但 unified evaluator 会自动补齐 Octave 路径并成功通过 |
| ParticlePhysics | `MuonTomography` | 完成 | 完成 | 完成 | 已稳定跑通 |
| KernelEngineering | `MLA` | 完成 | 完成 | 完成 | 使用新建 `kernel-lite` prefix env 跑通；运行时 `torch 2.3.1+cu121`，`cuda_available=True` |
| KernelEngineering | `TriMul` | 完成 | 完成 | 完成 | 使用新建 `kernel-lite` prefix env 跑通；运行时 `torch 2.3.1+cu121`，`cuda_available=True` |
| KernelEngineering | `FlashAttention` | 完成 | 完成 | 完成 | 使用新建 `kernel-lite` prefix env 跑通；运行时 `torch 2.3.1+cu121`，`cuda_available=True` |
| SingleCellAnalysis | `predict_modality` | 完成 | 完成 | 完成 | 运行时会自动下载 OpenProblems 公共数据到本地 cache |
| QuantumComputing | `task_01_routing_qftentangled` | 完成 | 完成 | 完成 | 使用新建 `mqt` prefix env 跑通 |
| QuantumComputing | `task_02_clifford_t_synthesis` | 完成 | 完成 | 完成 | 使用新建 `mqt` prefix env 跑通 |
| QuantumComputing | `task_03_cross_target_qaoa` | 完成 | 完成 | 完成 | 使用新建 `mqt` prefix env 跑通 |
| Cryptographic | `AES-128` | 未单独记录 | 完成 | 完成 | 当前主要按 unified 验证 |
| Cryptographic | `SHA-256` | 未单独记录 | 完成 | 完成 | unified 成功，`validate_passed=10/10` |
| Cryptographic | `SHA3-256` | 未单独记录 | 完成 | 完成 | unified 成功，`validate_passed=10/10` |
| CommunicationEngineering | `LDPCErrorFloor` | 完成 | 完成 | 完成 | `timeout=300` 成功，`timeout=60` 超时 |
| CommunicationEngineering | `PMDSimulation` | 完成 | 完成 | 完成 | 已稳定跑通 |
| CommunicationEngineering | `RayleighFadingBER` | 完成 | 完成 | 完成 | 已稳定跑通 |
| EnergyStorage | `BatteryFastChargingProfile` | 完成 | 完成 | 完成 | 已稳定跑通 |
| EnergyStorage | `BatteryFastChargingSPMe` | 完成 | 完成 | 完成 | 已稳定跑通 |
| SustainableDataCenterControl | `hand_written_control` | 完成 | 完成 | 完成 | 已按 README 指引补齐 vendored `sustaindc/` 内容，并在 `sustaindc` prefix env 中 direct + unified 跑通 |
| ReactionOptimisation | `snar_multiobjective` | 完成 | 完成 | 完成 | unified 需 `timeout=600` |
| ReactionOptimisation | `mit_case1_mixed` | 完成 | 完成 | 完成 | 运行时需显式指向 `summit` prefix env |
| ReactionOptimisation | `reizman_suzuki_pareto` | 完成 | 完成 | 完成 | unified 建议 `timeout=600` |
| Optics | `adaptive_temporal_smooth_control` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `adaptive_fault_tolerant_fusion` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `phase_fourier_pattern_holography` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通；`slmsuite` 提示未装 `cupy`，自动回退到 numpy 路径，但结果有效 |
| Optics | `phase_dammann_uniform_orders` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `fiber_wdm_channel_power_allocation` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `fiber_mcs_power_scheduling` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `fiber_guardband_spectrum_packing` | 完成 | 完成 | 完成 | 使用新建 `optics-lite` prefix env 跑通 |
| Optics | `holographic_multifocus_power_ratio` | 完成 | 完成 | 完成 | 需在 `kernel-lite` 中补 `torchoptics/slmsuite/opencv-python-headless`，并通过 runtime env 隐藏 GPU |
| Optics | `holographic_multiplane_focusing` | 完成 | 完成 | 完成 | 需在 `kernel-lite` 中补 `torchoptics/slmsuite/opencv-python-headless`，并通过 runtime env 隐藏 GPU |
| InventoryOptimization | `tree_gsm_safety_stock` | 完成 | 完成 | 完成 | 使用新建 `stock-lite` prefix env 跑通 |
| InventoryOptimization | `general_meio` | 完成 | 完成 | 完成 | 使用新建 `stock-lite` prefix env 跑通；direct 有 `SmallSampleWarning` 但结果有效 |
| InventoryOptimization | `joint_replenishment` | 完成 | 完成 | 完成 | 使用新建 `stock-lite` prefix env 跑通 |
| InventoryOptimization | `finite_horizon_dp` | 完成 | 完成 | 完成 | 使用新建 `stock-lite` prefix env 跑通 |
| InventoryOptimization | `disruption_eoqd` | 完成 | 完成 | 完成 | 使用新建 `stock-lite` prefix env 跑通 |
| PyPortfolioOpt | `robust_mvo_rebalance` | 完成 | 完成 | 完成 | 使用新建 `portfolio-lite` prefix env 跑通 |
| JobShop | `abz` | 完成 | 完成 | 完成 | 使用新建 `jobshop-lite` prefix env 跑通 |
| JobShop | `swv` | 完成 | 完成 | 完成 | unified 需显式 `python_path` 指向 prefix env，并将 `timeout` 提到 `600` |
| JobShop | `ta` | 完成 | 完成 | 完成 | unified 需显式 `python_path` 指向 prefix env，并将 `timeout` 提到 `1200`；任务本身耗时明显偏长 |
| Robotics | `DynamicObstacleAvoidanceNavigation` | 完成 | 完成 | 完成 | 首次 unified 因 named env 解析失败，改为显式 `python_path` 后成功 |
| Robotics | `PIDTuning` | 完成 | 完成 | 完成 | 首次 unified 因 named env 解析失败，改为显式 `python_path` 后成功 |
| Robotics | `UAVInspectionCoverageWithWind` | 完成 | 完成 | 完成 | 首次 unified 因 named env 解析失败，改为显式 `python_path` 后成功 |
| WirelessChannelSimulation | `HighReliableSimulation` | 完成 | 完成 | 完成 | unified 需显式 `python_path` 指向 prefix env，不能依赖 named conda env |

### 10.1 目前已完成的 v1 任务

截至当前，已可明确记为“完成”的 v1 任务有：

- `Astrodynamics/MannedLunarLanding`
- `ParticlePhysics/MuonTomography`
- `KernelEngineering/MLA`
- `KernelEngineering/TriMul`
- `KernelEngineering/FlashAttention`
- `SustainableDataCenterControl/hand_written_control`
- `EnergyStorage/BatteryFastChargingProfile`
- `EnergyStorage/BatteryFastChargingSPMe`
- `CommunicationEngineering/PMDSimulation`
- `CommunicationEngineering/LDPCErrorFloor`
- `CommunicationEngineering/RayleighFadingBER`
- `Cryptographic/AES-128`
- `Cryptographic/SHA-256`
- `Cryptographic/SHA3-256`
- `ReactionOptimisation/mit_case1_mixed`
- `ReactionOptimisation/reizman_suzuki_pareto`
- `ReactionOptimisation/snar_multiobjective`
- `SingleCellAnalysis/predict_modality`
- `QuantumComputing/task_01_routing_qftentangled`
- `QuantumComputing/task_02_clifford_t_synthesis`
- `QuantumComputing/task_03_cross_target_qaoa`
- `Optics/adaptive_temporal_smooth_control`
- `Optics/adaptive_fault_tolerant_fusion`
- `Optics/phase_fourier_pattern_holography`
- `Optics/phase_dammann_uniform_orders`
- `Optics/fiber_wdm_channel_power_allocation`
- `Optics/fiber_mcs_power_scheduling`
- `Optics/fiber_guardband_spectrum_packing`
- `Optics/holographic_multifocus_power_ratio`
- `Optics/holographic_multiplane_focusing`
- `InventoryOptimization/tree_gsm_safety_stock`
- `InventoryOptimization/general_meio`
- `InventoryOptimization/joint_replenishment`
- `InventoryOptimization/finite_horizon_dp`
- `InventoryOptimization/disruption_eoqd`
- `PyPortfolioOpt/robust_mvo_rebalance`
- `JobShop/abz`
- `JobShop/swv`
- `JobShop/ta`
- `Robotics/DynamicObstacleAvoidanceNavigation`
- `Robotics/PIDTuning`
- `Robotics/UAVInspectionCoverageWithWind`
- `WirelessChannelSimulation/HighReliableSimulation`

### 10.2 目前部分完成的 v1 任务

截至当前，属于“部分完成”的 v1 任务有：

- 当前无

说明：

- 本轮新增跑通后，之前标记为“部分完成”的 `MannedLunarLanding` 与 `snar_multiobjective` 已转为“完成”。
- 本轮收口后，按当前 `v1` CPU batch config 核对，已不存在仍处于“部分完成”的任务。

## 11. 后续新增结果（本轮继续推进）

这一节记录的是在前文基础上继续推进后新增确认的结果，避免和最初记录混在一起。

### 11.1 新增 unified 成功项

- `ComputerSystems/MallocLab`
  - `combined_score=28.0`
  - `testcases_passed=6/11`
- `StructuralOptimization/ISCSO2015`
  - `combined_score=-5401.589001522704`
  - `valid=1`
  - `feasible=1`
- `StructuralOptimization/ISCSO2023`
  - `combined_score=-77813242.90462679`
  - `valid=1`
  - `feasible=1`
- `WirelessChannelSimulation/HighReliableSimulation`
  - 第一次 unified 尝试失败，原因不是 benchmark 本身，而是默认 `task.runtime.conda_env=frontier-eval-2` 会在当前机器上解析成一个并不存在的 named env：
    - `EnvironmentLocationNotFound: Not a conda environment: /GenSIvePFS/users/hhchi/miniforge3/envs/frontier-eval-2`
  - 改为显式：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python`
    - `task.runtime.use_conda_run=false`
    后 unified 成功
  - direct 成功：`combined_score=180.97383613710724`
  - unified 成功：`combined_score=180.63593987881632`
- `CommunicationEngineering/RayleighFadingBER`
  - direct 成功：`combined_score=504.2163236981388`
  - unified 成功：`combined_score=670.1902586624659`
- `SingleCellAnalysis/predict_modality`
  - direct 成功：`combined_score=0.5466578345573954`
  - unified 成功：`combined_score=0.5466578345573954`
- `EnergyStorage/BatteryFastChargingSPMe`
  - direct 成功：`combined_score=66.16356426696784`
  - unified 成功：`combined_score=66.16356426696784`
- `StructuralOptimization/TopologyOptimization`
  - direct 成功：`compliance=195.9152621065792`
  - unified 成功：`combined_score=-195.9152621065792`
- `StructuralOptimization/PyMOTOSIMPCompliance`
  - direct 成功：`combined_score=4.834812682779515`
  - unified 成功：`combined_score=4.834812682779515`
- `Cryptographic/SHA-256`
  - unified 成功：`combined_score=29.102860684132075`
  - `validate_passed=10/10`
- `Cryptographic/SHA3-256`
  - unified 成功：`combined_score=63.38620670145833`
  - `validate_passed=10/10`
- `Astrodynamics/MannedLunarLanding`
  - 基线脚本成功生成 `results.txt`
  - unified 成功：`combined_score=4577.437043`
  - `payload_kg=4577.437043`
  - `octave_returncode=0`
- `ComputerSystems/DuckDBWorkloadOptimization`
  - 先通过清华 PyPI 镜像补装 `duckdb>=1.1.0`
  - direct 成功：`combined_score=0.9997836413338042`
  - unified 成功：`combined_score=0.9985957018888292`
  - direct / unified 都会先构建 benchmark 数据，`data_build_s` 约 `22s`
- `QuantumComputing/task_01_routing_qftentangled`
  - direct 成功：`avg_candidate_score_0_to_3=0.20898716119828828`
  - unified 成功：`combined_score=0.20898716119828828`
- `QuantumComputing/task_02_clifford_t_synthesis`
  - direct 成功：`avg_candidate_score_0_to_3=1.6633069602871013`
  - unified 成功：`combined_score=1.6633069602871013`
  - 直接运行期间出现一条 `UserWarning: Trying to add QuantumRegister to a QuantumCircuit having a layout`
  - warning 不影响最终结果有效性
- `QuantumComputing/task_03_cross_target_qaoa`
  - direct 成功：`avg_candidate_score_0_to_3=2.4149139615375192`
  - unified 成功：`combined_score=2.4149139615375192`
- `Optics/adaptive_temporal_smooth_control`
  - direct 成功：`score_0_to_1_higher_is_better=0.31517132841504814`
  - unified 成功：`combined_score=0.31517132841504814`
- `Optics/adaptive_fault_tolerant_fusion`
  - direct 成功：`score_0_to_1_higher_is_better=0.3958695233765083`
  - unified 成功：`combined_score=0.3958695233765083`
- `Optics/phase_fourier_pattern_holography`
  - direct 成功：`baseline score_pct=32.64571443630872`
  - unified 成功：`combined_score=32.64571443630872`
  - `slmsuite` 运行时提示未安装 `cupy`，自动使用 `numpy`
  - 这只影响速度，不影响本次 direct / unified 的结果有效性
- `Optics/phase_dammann_uniform_orders`
  - direct 成功：`candidate score_pct=26.896904752419033`
  - unified 成功：`combined_score=26.896904752419033`
  - oracle 选择结果为 `scipy_differential_evolution`
- `Optics/fiber_wdm_channel_power_allocation`
  - direct 成功：`candidate score=0.3255243713068484`
  - unified 成功：`combined_score=0.3255243713068484`
  - baseline 可运行，但与 oracle 仍有明显差距：`score_gap_oracle_minus_candidate=0.33963721973005734`
- `Optics/fiber_mcs_power_scheduling`
  - direct 成功：`candidate score=0.3297323928738737`
  - unified 成功：`combined_score=0.3297323928738737`
  - direct 结果显示 baseline 的 `ber_pass_ratio=0.045454545454545456`，但 evaluator 返回 `valid=true`
- `Optics/fiber_guardband_spectrum_packing`
  - direct 成功：`candidate score=0.3860644257703081`
  - unified 成功：`combined_score=0.3860644257703081`
  - unified 单次评测耗时约 `17.94s`，属于任务本身正常偏慢，不是卡死
- `Optics/holographic_multifocus_power_ratio`
  - 最初 direct 在默认 auto-device 路径下失败：
    - `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`
  - 改为 `--device cpu` 后，又因 `slmsuite -> cv2` 依赖 `libGL.so.1` 失败
  - 在 `kernel-lite` 里补装 `opencv-python-headless` 后，CPU 路径 direct 成功：
    - `baseline score=0.3927324668462351`
    - `reference score=0.5486962092975242`
  - unified 需显式：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/kernel-lite/bin/python`
    - `task.runtime.use_conda_run=false`
    - `+task.runtime.env.CUDA_VISIBLE_DEVICES=''`
  - unified 成功：`combined_score=0.3927324668462351`
  - 当前判断：
    - benchmark 本身可复现
    - 但 README 仅写 `opencv-python` 时，在当前这类无图形库节点上会误导复现者
    - 且默认 auto-device 会触发 baseline / torchoptics 的设备不一致 bug
- `Optics/holographic_multiplane_focusing`
  - 初始问题模式与 `holographic_multifocus_power_ratio` 相同：
    - auto-device 下报 `cuda:0` / `cpu` 设备不一致
    - `--device cpu` 下若只装 `opencv-python`，会因 `libGL.so.1` 缺失失败
  - 在 `kernel-lite` 中补 `opencv-python-headless` 后，CPU 路径 direct 成功：
    - `baseline mean_score=0.3301705437635439`
    - `reference mean_score=0.4870032404223486`
  - unified 同样需要 `+task.runtime.env.CUDA_VISIBLE_DEVICES=''`
  - unified 成功：`combined_score=0.3301705437635439`
- `InventoryOptimization/tree_gsm_safety_stock`
  - direct 成功：`baseline score=0.3813`
  - unified 成功：`combined_score=0.38125997730251027`
- `InventoryOptimization/general_meio`
  - direct 成功：`baseline score=0.1825`
  - unified 成功：`combined_score=0.18253152886847146`
  - direct 运行时有一条 `SmallSampleWarning`
  - 当前判断：warning 来自 `stockpyl.sim.stats.sem(...)` 的样本量提示，不影响 direct / unified 返回有效分数
- `InventoryOptimization/joint_replenishment`
  - direct 成功：`baseline score=0.3034`
  - unified 成功：`combined_score=0.3034231848949367`
- `InventoryOptimization/finite_horizon_dp`
  - direct 成功：`baseline score=0.3673`
  - unified 成功：`combined_score=0.3673219124866723`
- `InventoryOptimization/disruption_eoqd`
  - direct 成功：`baseline score=0.3642`
  - unified 成功：`combined_score=0.36423022249600623`
- `PyPortfolioOpt/robust_mvo_rebalance`
  - 为避免污染已有环境，新建 prefix env：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/portfolio-lite`
  - 通过清华镜像安装：
    - `cvxpy`
    - `PyPortfolioOpt`
    - `highspy`
    - `ecos`
    - `osqp`
    - `scs`
  - direct 成功：
    - `baseline_average_score: 32.98/100`
  - unified 成功：
    - `combined_score=32.980438911095824`
- `JobShop/abz`
  - 为避免依赖一个并不存在的 named env，新建最小 prefix env：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/jobshop-lite`
  - 环境中补齐：
    - `ortools`
    - `job_shop_lib`
  - direct 成功：
    - average best-known baseline score `80.50`
    - average lower-bound baseline score `80.07`
  - unified 成功：
    - `combined_score=80.50421052827963`
- `JobShop/swv`
  - direct 成功：
    - average best-known baseline score `81.63`
    - average lower-bound baseline score `80.94`
    - average reference runtime `10.7042s`
  - unified 采用：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/jobshop-lite/bin/python`
    - `task.runtime.use_conda_run=false`
    - `algorithm.oe.evaluator.timeout=600`
  - unified 成功：
    - `combined_score=81.6324740519551`
    - `reference_runtime_avg_s=10.4277`
- `JobShop/ta`
  - direct 成功：
    - average best-known baseline score `78.80`
    - average lower-bound baseline score `78.33`
    - average reference runtime `10.3445s`
  - direct 覆盖 `80` 个实例，其中：
    - `reference failures: 10`
  - 当前判断：
    - 这些失败来自 reference solver 在 `ta71` 到 `ta80` 上未在时间预算内找到解
    - evaluator 仍然成功给出 baseline 分数，因此不构成 baseline 复现失败
  - unified 采用：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/jobshop-lite/bin/python`
    - `task.runtime.use_conda_run=false`
    - `algorithm.oe.evaluator.timeout=1200`
  - unified 成功：
    - `combined_score=78.79996227624356`
    - `reference_success_rate=0.6375`
    - `reference_failures=29`
- `Robotics/DynamicObstacleAvoidanceNavigation`
  - direct 成功：
    - `score=12.850000000000046`
    - `feasible=true`
  - 第一次 unified 沿用默认 named env 思路，快速失败
  - 改为显式：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python`
    - `task.runtime.use_conda_run=false`
    后 unified 成功：
    - `combined_score=0.07220216606498171`
- `Robotics/PIDTuning`
  - direct 成功：
    - `combined_score=0.036626766599899996`
  - 第一次 unified 同样因 prefix env / named env 不匹配而快速失败
  - 改为显式 `python_path` 后 unified 成功：
    - `combined_score=0.036626766599899996`
- `Robotics/UAVInspectionCoverageWithWind`
  - direct 成功：
    - `score=28.851886471062496`
    - `feasible=true`
  - 第一次 unified 同样因 prefix env / named env 不匹配而快速失败
  - 改为显式 `python_path` 后 unified 成功：
    - `combined_score=28.851886471062496`

### 11.1.2 新增 KernelEngineering 成功项

- `KernelEngineering/FlashAttention`
  - 为避免修改系统 `/usr/bin/python`，新建独立 prefix env：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/kernel-lite`
  - 当前可用状态：
    - `torch=2.3.1+cu121`
    - `triton=2.3.1`
    - `numpy=2.4.4`
    - `einops=0.8.2`
    - `torch.cuda.is_available() == True`
    - 设备：`NVIDIA A100-SXM4-80GB`
  - direct correctness 成功：`flash_attn_tests.txt` 三组 case 全部 `pass`
  - unified 成功：
    - `combined_score=110.93783853374799`
    - `geom_mean_ns=9014057.0`
    - `runtime_s=4.5746`

- `KernelEngineering/MLA`
  - direct correctness 成功：`mla_tests.txt` 四组 case 全部 `pass`
  - direct 运行末尾有一条：
    - `Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>`
    - `OSError: [Errno 9] Bad file descriptor`
  - 当前判断：
    - 这条异常出现在所有测试已经 `pass` 且 `check: pass` 之后
    - 更像 evaluator 对 `POPCORN_FD=1` 直接绑定 stdout 时的收尾噪声，不影响本次 correctness 结果有效性
  - unified 成功：
    - `combined_score=0.6006242822721272`
    - `geom_mean_ns=1664934351.6667`
    - `runtime_s=16.96`
    - 本次显式设置 `algorithm.oe.evaluator.timeout=1800`

- `KernelEngineering/TriMul`
  - direct correctness 成功：`tri_tests.txt` 共 `18` 组 case 全部 `pass`
  - unified 成功：
    - `combined_score=42.34594458530887`
    - `geom_mean_ns=23615012.2472`
    - `runtime_s=9.3709`
    - 本次显式设置 `algorithm.oe.evaluator.timeout=1800`
  - 额外发现：
    - `benchmarks/KernelEngineering/TriMul/README.md` 里写的是 `tri_test.txt`
    - 但仓库里真实文件名是 `verification/tri_tests.txt`
    - 这属于 README 示例命令的小错误，不是环境问题

### 11.1.3 新增 SustainableDataCenterControl 成功项

- `SustainableDataCenterControl/hand_written_control`
  - 之前的阻塞点不是 Python 环境，而是仓库里缺少 README 所描述的 vendored `hand_written_control/sustaindc/` 内容
  - 本轮按仓库 README 的指导补齐这部分内容：
    - 目标上游：`HewlettPackard/dc-rl`
    - 目标 commit：`a92b4755aca560e34a98d14028dda629eb968482`
    - 先尝试使用镜像：
      - `git clone https://githubfast.com/HewlettPackard/dc-rl.git .../hand_written_control/sustaindc`
    - 这一步在当前机器上没有正常收尾，只留下了不完整 `.git` 目录，没有形成可用工作树
    - 因此前面已通过镜像成功拉到本地的副本：
      - `/GenSIvePFS/users/hhchi/frontier_eng_repro/external/dc-rl-a92b475`
      - 被无 `.git` 地 vendoring 到：
      - `benchmarks/SustainableDataCenterControl/hand_written_control/sustaindc`
  - 补齐后已确认：
    - `sustaindc/sustaindc_env.py`
    - `sustaindc/requirements.txt`
    - `sustaindc/data`
    - `sustaindc/envs`
    - `sustaindc/utils`
    - 都已存在，满足 unified `copy_files.txt` 的假设
  - 也确认 vendored 内容已经包含 benchmark 需要的 patch：
    - `matplotlib` 相关导入已变为 optional
    - `Dashboard` 相关导入已变为 optional
  - direct 成功：
    - `Average score: 8.33 / 100.00`
    - 四个场景均完成，结构化结果已写入 `verification/last_eval.json`
  - unified 成功：
    - `combined_score=8.3433`
    - `runtime_s=13.4237`
    - 默认 `300s` timeout 已足够
  - 运行过程中出现四条：
    - `WARNING, using default values for chiller sizing...`
    - 但 direct / unified 都返回 `benchmark_returncode=0` 且结果有效

### 11.2 `MannedLunarLanding` 的一个易踩坑点

我手工直接运行下面这条命令时：

```bash
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/octave --no-gui --quiet --eval "addpath('eval'); aerodynamics_check_octave_full;"
```

会报：

- `odeset undefined`

这说明当前机器上的 Octave 默认启动搜索路径不完整，直接手工跑 validator 会误导复现者，以为 benchmark 本身坏了。

但 unified evaluator 自带了额外的 `addpath(genpath(...))` 补路径逻辑，所以：

- unified 实际是成功的
- 问题不在 benchmark baseline，而在当前环境里的 Octave 启动路径

这类差异建议后续在文档里专门说明，否则用户很容易在“手工验证失败、统一评测成功”之间困惑。

### 11.3 当前机器的真实环境边界

需要特别记录：`docs/v1_task_run_guide_zh-CN.md` 中有一些“已满足”表述，不完全对应当前这台机器的真实状态，更像是来自另一台已配置更多 env 的机器上下文。

当前这台机器我实际确认到的是：

- 已有并可用的 prefix env 只有：
  - `frontier-eval-2`
  - `summit`
- 当前机器没有可直接使用的这些专用 env：
  - `motion`
  - `optics`
  - `stock`
  - `pyportfolioopt`
  - `jobshop`
  - `mqt`
  - 以及其他文档里提到但本机未实际存在的环境
- 当前机器也没有 `docker`
- 当前 `frontier-eval-2` env 是可以继续小幅补轻量依赖的；例如本轮补装了：
  - `duckdb==1.5.2`
  - 使用命令：

```bash
PIP_CONFIG_FILE=/GenSIvePFS/users/hhchi/frontier_eng_repro/pip.conf \
/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/pip install 'duckdb>=1.1.0'
```

  - 实际下载源：
    - `https://pypi.tuna.tsinghua.edu.cn/simple`
  - 轮子体积：
    - `21.4 MB`
  - 本机这次实际下载耗时大约：
    - `1 分 47 秒`
- 本轮还新建并使用了两个 prefix env：
  - `mqt`
    - 用于 `QuantumComputing/*`
    - 通过清华 PyPI 安装了 `mqt.bench`
    - `mqt.bench` 同时拉起了 `qiskit`、`rustworkx`、`scikit-learn` 等依赖
  - `optics-lite`
    - 只为当前 3 个 `v1` Optics 任务安装最小依赖，而不是整套 `benchmarks/Optics/requirements.txt`
    - conda 安装：
      - `python=3.11`
      - `numpy`
      - `scipy`
      - `matplotlib`
      - `scikit-learn`
    - pip 补装：
      - `aotools`
      - `slmsuite`
    - 这样就足够覆盖：
      - `adaptive_temporal_smooth_control`
      - `adaptive_fault_tolerant_fusion`
      - `phase_fourier_pattern_holography`

因此下面这些方向目前仍属于“待补环境后再复现”而不是“已确认可在本机直接跑”：

- `EngDesign/*`
- `Optics/*`
- `Robotics/*`
- `PyPortfolioOpt/*`
- `InventoryOptimization/*`
- `JobShop/*`
- `QuantumComputing/*`

结论上，后续继续扩大覆盖面时，应该以“当前机器真实已有 env”为准，不应直接复用 `v1_task_run_guide_zh-CN.md` 里更乐观的环境结论。

### 11.4 本轮继续推进的实际过程记录

这一轮我没有只补“结果”，而是按剩余矩阵任务逐类推进，并把每一步的环境策略收敛成更稳定的 prefix-env 模式。

- 对剩余 `Optics`：
  - 先延续前一轮已建好的 `optics-lite`，补跑：
    - `phase_dammann_uniform_orders`
    - `fiber_wdm_channel_power_allocation`
    - `fiber_mcs_power_scheduling`
    - `fiber_guardband_spectrum_packing`
  - 这些任务 direct 先跑通后，再统一用：
    - `task.runtime.python_path=/.../optics-lite/bin/python`
    - `task.runtime.use_conda_run=false`
    跑 unified。
- 对两项 `holographic_*`：
  - 起初尝试直接复用已有 CUDA `torch` 的 `kernel-lite`
  - 先补装：
    - `matplotlib`
    - `slmsuite`
    - `torchoptics`
  - 这一步在清华镜像下总体可完成，但明显慢于轻量 CPU 包：
    - 例如 `opencv-python-4.13.0.92` 轮子 `72.9 MB`
    - 实测下载速率大约只有 `395 KB/s`
    - 但属于“慢”而不是“卡死”，所以没有频繁中断
  - 首次 direct 失败不是缺包，而是默认 auto-device 会把一部分张量放在 `cuda:0`、另一部分留在 CPU，触发 `torchoptics` 的设备不一致错误
  - 改为 `--device cpu` 后，又暴露出：
    - `ImportError: libGL.so.1`
  - 最终采用的无代码修改方案是：
    - 在 `kernel-lite` 中补装 `opencv-python-headless`
    - direct 显式使用 `--device cpu`
    - unified 显式传：
      - `+task.runtime.env.CUDA_VISIBLE_DEVICES=''`
  - 这样两项 holographic 才稳定 direct + unified 跑通
- 对 `InventoryOptimization/*`：
  - 没有继续等待“大而全”的 `frontier-v1-main`
  - 直接新建最小 prefix env：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/stock-lite`
  - conda 只装：
    - `python=3.11`
    - `pip`
    - `numpy`
    - `scipy`
  - pip 再补：
    - `stockpyl`
  - `stockpyl` 会拉一串文档相关依赖，resolver 输出比较长，但最终能完成安装
  - 随后 5 个子任务先 direct 全扫，再统一用显式 `python_path` 跑 unified
- 对 `PyPortfolioOpt/robust_mvo_rebalance`：
  - 同样新建最小 prefix env：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/portfolio-lite`
  - conda 先装基础：
    - `python=3.11`
    - `pip`
    - `numpy`
    - `scipy`
  - pip 通过清华镜像补：
    - `cvxpy`
    - `PyPortfolioOpt`
    - `highspy`
    - `ecos`
    - `osqp`
    - `scs`
  - 这条依赖线比预期顺利，主要包都有现成 wheel，没有遇到长时间编译
  - 任务 direct 与 unified 都顺利通过，没有再暴露额外环境问题
- 对 `JobShop/*`：
  - 没有沿用 batch config 里 `conda-env:frontier-v1-main` 的写法，因为当前机器并不存在对应 named env
  - 改为直接新建：
    - `/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/jobshop-lite`
  - 只补最小依赖：
    - `ortools`
    - `job_shop_lib`
  - `abz` 先完成 direct + unified，证明这条最小依赖线是成立的
  - `swv` 因 reference 每个实例本身就接近 `10s`，因此 unified 预先把 `algorithm.oe.evaluator.timeout` 提到 `600`
  - `swv` 最终 direct + unified 都成功，说明它属于“任务本身耗时明显偏长”，不是卡死
  - `ta` 也按同样思路启动：
    - direct 用 `jobshop-lite`
    - unified 用显式 `python_path`
    - timeout 提到 `1200`
  - `ta` 最终也 direct + unified 成功
  - 这条线上最值得记录的不是“失败”，而是耗时特征：
    - unified 单次评测墙钟时间约 `492.99s`
    - direct 会更久，而且几乎只在最后一次性输出汇总
  - 因此它非常容易被误判成“卡住”，但从实际结果看，这就是任务本身长
- 对 `Robotics/*`：
  - 这三项并不需要另建专用 env，直接复用驱动 env 即可
  - 但 unified 如果照 batch config 默认去找 named env，会快速失败
  - 因此统一改成：
    - `task.runtime.python_path=/GenSIvePFS/users/hhchi/frontier_eng_repro/envs/frontier-eval-2/bin/python`
    - `task.runtime.use_conda_run=false`
  - 这样 `DynamicObstacleAvoidanceNavigation`、`PIDTuning`、`UAVInspectionCoverageWithWind` 都已完成 direct + unified

### 11.5 本轮收口时的最终状态

按 `frontier_eval/conf/batch/v1_cpu_openevolve_p8_i100_gemini-3.1-pro-preview.yaml` 当前矩阵核对，这台机器上本轮已把其中全部 v1 任务跑到可记录结果。

最终没有遗留“未开始”或“部分完成”的 v1 CPU batch 任务。

但仍需继续关注文档与环境默认值的一致性问题。

这一轮执行下来，实践上得到几个可复用结论：

- 对 prefix env，统一优先使用：
  - `task.runtime.python_path=/绝对路径/bin/python`
  - `task.runtime.use_conda_run=false`
- 对无图形库的计算节点，README 中写 `opencv-python` 的任务，实际需要评估是否应改成：
  - `opencv-python-headless`
- 对默认会自动选 CUDA 的任务，如果 benchmark 自己没有妥善处理 device，一定要尝试：
  - `--device cpu`
  - 或 unified runtime env 中显式隐藏 GPU

## 12. 配环境过程是否顺利

整体判断：

- 不算“一路顺滑”，但也不是“到处都坏”
- 更准确地说，是主干可通，若干边缘处有明显的入手门槛和文档/配置错位

相对顺利的部分：

- 驱动环境可用
  - 用独立 prefix env 建 `frontier-eval-2` 后，`python -m frontier_eval task=smoke` 很快就能通
- 纯 CPU、小依赖任务扩展很顺
  - `InventoryOptimization`
  - `PyPortfolioOpt`
  - `JobShop`
  - `QuantumComputing`
  - 这几类都可以通过“最小 prefix env + 清华 pip/conda 镜像”稳定补齐
- 大部分 direct / unified 的主流程是稳定的
  - 一旦 runtime 选对、timeout 合理，许多任务可以直接复用同一套路

主要卡点：

- `torch` 和大轮子下载慢
  - 尤其是 `ReactionOptimisation` 早期安装链路上的 `torch==1.13.1`
  - 镜像并没有把这件事彻底变成“快”，只是从“完全不可接受”变成“可等待但要有耐心”
- named conda env 假设和真实环境不一致
- 仓库多处默认写法假设用户已经有一组预先命名好的 conda env
  - 但实际复现时，prefix env 更常见，也更安全
  - 这导致 unified 一开始会出现“配置看起来对，实际秒失败”的情况
- GPU / 图形依赖的真实节点差异
  - `opencv-python` 在无图形库节点上会踩 `libGL.so.1`
  - 某些任务默认 auto-device 会混用 `cuda:0` 和 `cpu`
  - 这类问题不改代码也能绕开，但第一次遇到时不够直观
- 第三方库兼容性
  - `summit` 和新版本 `scikit-learn` 的兼容问题会影响 `ReactionOptimisation`
- 仓库内容和 README 假设不总一致
  - `SustainableDataCenterControl/hand_written_control` 需要 vendored `sustaindc/`
  - `KernelEngineering/TriMul` README 示例文件名与仓库真实文件名不一致

我认为最奇怪、最容易让初次用户误判的卡点有 5 个：

1. unified 默认 `task.runtime.conda_env=...` 在 prefix env 场景下会快速失败，但日志表面上像 benchmark 挂了。
2. `docs/v1_task_run_guide_zh-CN.md` 混入了明显的“某台机器当时的环境结论”，不是一个稳定可移植的公开指南。
3. `JobShop/ta` 这类任务长时间无输出，但最后是成功的；如果不了解特性，很容易被当成卡死。
4. `holographic_*` 任务在当前节点上既有 device 选择问题，又有 `libGL` 问题，两个问题会串联出现。
5. v1 batch config 中存在指向私有代理域名的默认 `api_base`，这对 public repo 非常不合适。

## 13. 如果想让仓库更易于入手，建议先做什么

如果只能优先做少数几件事，我建议顺序如下：

### 13.1 先统一环境叙事

优先把“推荐环境模型”收敛成一套一致说法，例如：

- 方案 A：官方推荐 named env
- 方案 B：官方支持 prefix env

目前最大的摩擦不是“不会装包”，而是：

- 文档
- batch config
- task config
- 实际机器

这四者对环境名字和调用方式的假设并不完全一致。

最值得先补的是：

- 在主 README 放一张“driver env / task runtime env / unified runtime”的关系图
- 明确写出 prefix env 的官方模板命令
- 说明何时该用 `task.runtime.conda_env`，何时该改用 `task.runtime.python_path`

### 13.2 再清理 public-facing 文档

当前最应该先处理的是：

- `docs/v1_task_run_guide_zh-CN.md`

原因不是它“不能存在”，而是它现在更像：

- 某次内部核对记录
- 某台机器上的本地环境检查结论
- 混有绝对路径和个人环境假设的运行备忘

如果要公开，至少应拆成两层：

- `docs/v1_overview_zh-CN.md`
  - 只保留稳定信息：任务列表、依赖类别、典型耗时、是否需要 GPU/Docker/外部数据
- `docs/internal/` 或单独 issue/附件
  - 放本地核对记录、临时结论、具体机器上的运行经验

### 13.3 然后处理“跑一次就脏工作区”的问题

这件事对公开仓库体验影响非常大。

目前大量 benchmark 会把结果写到这些被 git 跟踪的目录或文件：

- `verification/outputs/*`
- `verification/artifacts/*`
- `verification/last_eval.json`
- 某些任务根目录下的生成文件，如 `plan.json`

这样用户只要运行一次 baseline/evaluator，仓库就会出现大量改动，体验很差，也容易污染提交。

更合理的做法通常是二选一：

- 把这些文件改成真正的示例产物，移到 `examples/` 或 `docs/assets/`
- 或把它们改成运行时产物，默认 `.gitignore`，评测时自动创建目录

### 13.4 再修掉最容易误导新人的配置默认值

建议优先修：

- batch config 中写死的私有 `api_base`
- README/guide 里写死的个人绝对路径
- 任务 README 里的错误文件名或过时命令

这类问题不一定会阻塞老用户，但对第一次接触的人影响最大。

### 13.5 最后再做 benchmark 级兼容性收口

发布前最值得单独过一遍的是：

- `SustainableDataCenterControl/hand_written_control`
- `Optics/holographic_*`

因为这些任务暴露出的不是“下载慢”，而是：

- evaluator 假设
- 元数据缺失
- 外部 vendored 内容
- 节点图形/GPU差异

它们更像会被外部用户反复撞到的首批问题。

## 14. 哪些文件或内容不太适合 public repo

这里分成两类：一类是我认为“应优先移除或改写”的；另一类是“可以保留，但需要更规范化”。

### 14.1 应优先移除、重写或迁移的位置

- `docs/v1_task_run_guide_zh-CN.md`
  - 这份文档混有明显的本地运行记录、环境核对痕迹和个人机器结论
  - 例如它写了本机专用路径：
    - `/data_storage/chihh2311/.conda/envs/jobshop/bin/python`
  - 也写了很多“当前机器已满足”的判断，这不适合作为 public repo 的稳定文档
- `frontier_eval/conf/batch/v1_cpu_openevolve_p8_i100_gemini-3.1-pro-preview.yaml`
  - `api_base` 默认值指向私有代理：
    - `https://litellm.nbdevenv.xiaoaojianghu.fun/v1`
  - public repo 不应把组织内代理当默认值
- `scripts/pr_review.py`
  - 同样写死了私有代理地址：
    - `https://litellm.nbdevenv.xiaoaojianghu.fun/v1/chat/completions`
  - 如果保留，至少应改成完全由环境变量驱动

### 14.2 不一定要删除，但目前不太符合公开仓库的规范

- 各 benchmark 下被跟踪的运行产物
  - 例如：
    - `benchmarks/Optics/*/verification/outputs/*`
    - `benchmarks/Optics/*/verification/artifacts/*`
    - `benchmarks/SustainableDataCenterControl/hand_written_control/verification/last_eval.json`
  - 问题不是“里面有秘密”，而是这些文件一运行就会变，导致仓库天然变脏
- `docs/patches/communicationengineering_followup_20260324.patch`
  - 这更像内部 follow-up patch 留档
  - 若是为了发布说明，建议转成 issue、PR 链接或 changelog 摘要
  - 如果只是临时工作痕迹，没必要放在 public docs 目录
- `docs/shinkaevolve_adapter_hardening_plan.md`
  - 内容本身并不敏感
  - 但它属于内部工程计划文档，不一定应该和面向外部用户的使用文档平铺在同一层
  - 更适合移到 `docs/dev/`、`docs/internal/` 或 RFC 目录

### 14.3 规范性上建议调整的地方

- `.gitignore` 目前没有覆盖大量运行时输出目录
  - 这会让 benchmark 评测天然制造 git diff
- 文档层次混杂
  - 当前 `docs/` 同时放了：
    - 用户指南
    - 复现实验记录
    - patch 文件
    - 内部 hardening 计划
  - 建议至少拆成：
    - `docs/user/`
    - `docs/dev/`
    - `docs/repro/`
    - `docs/archive/` 或 `docs/internal/`
- batch config 的命名带强烈实验痕迹
  - 如 `...gemini-3.1-pro-preview.yaml`
  - 对 public repo 来说可以保留，但更适合作为“示例实验矩阵”，不应给人一种它是唯一官方入口的感觉
- README 与真实命令/文件名不完全一致
  - 这类问题会显著放大首次上手成本
  - 发布前应至少跑一轮 README 命令校验

## 15. 发布前清理 Checklist

这一节给发布前最后一轮整理用，按优先级分成 `必须改`、`建议改`、`可后续改`。

### 15.1 必须改

- 清理私有代理默认值
  - `frontier_eval/conf/batch/v1_cpu_openevolve_p8_i100_gemini-3.1-pro-preview.yaml`
  - `scripts/pr_review.py`
  - 当前问题：
    - 写死了私有代理域名
  - 建议：
    - 改成完全由环境变量驱动
    - 或回退到公开默认值，再允许用户自行覆盖

- 处理 `docs/v1_task_run_guide_zh-CN.md`
  - 当前问题：
    - 混有本地运行记录
    - 混有个人绝对路径
    - 混有“当前机器已满足”的环境结论
  - 建议：
    - 从 public 使用文档中移出
    - 或拆成“稳定指南”和“内部/历史核对记录”两份文档

- 处理被 git 跟踪的运行产物
  - 典型路径：
    - `benchmarks/Optics/*/verification/outputs/*`
    - `benchmarks/Optics/*/verification/artifacts/*`
    - `benchmarks/SustainableDataCenterControl/hand_written_control/verification/last_eval.json`
  - 当前问题：
    - 用户一跑 benchmark，仓库就会变脏
  - 建议：
    - 这些文件若只是运行产物，应停止跟踪并加入 `.gitignore`
    - 如果确实想保留示例，应改放到专门的示例目录

- 校正 README 中已经确认错误的示例
  - `benchmarks/KernelEngineering/TriMul/README.md`
  - 当前问题：
    - 文档写的是 `tri_test.txt`
    - 仓库真实文件是 `verification/tri_tests.txt`

### 15.2 建议改

- 统一环境叙事
  - 相关位置：
    - `README.md`
    - `frontier_eval/README_zh-CN.md`
    - `frontier_eval/conf/task/unified.yaml`
    - `frontier_eval/conf/batch/v1_cpu_openevolve_p8_i100_gemini-3.1-pro-preview.yaml`
    - `scripts/setup_v1_merged_task_envs.sh`
  - 当前问题：
    - 文档、batch config、task config、脚本对 runtime env 的假设不完全一致
  - 建议：
    - 明确官方首选是 named env 还是 prefix env
    - 明确给出 prefix env 的标准 unified 模板
    - 写清楚何时用 `task.runtime.conda_env`
    - 写清楚何时改用 `task.runtime.python_path`

- 优化文档分层
  - 相关位置：
    - `docs/`
  - 当前问题：
    - 用户指南、复现记录、patch 留档、内部工程计划平铺在同一层
  - 建议：
    - 至少拆成 `docs/user/`、`docs/repro/`、`docs/dev/` 或类似结构

- 补充 `.gitignore`
  - 相关位置：
    - `.gitignore`
  - 当前问题：
    - 未覆盖大量 benchmark 运行输出
  - 建议：
    - 如果决定不再跟踪运行产物，就把对应输出路径统一忽略

- 发布前校验 README 命令
  - 相关位置：
    - 各 benchmark README / README_zh-CN.md
  - 当前问题：
    - 少数命令、文件名、环境说明与仓库真实状态不完全一致
  - 建议：
    - 跑一轮 README 命令 smoke check
    - 至少覆盖 v1 任务

### 15.3 可后续改

- 规范 batch config 的实验命名
  - 相关位置：
    - `frontier_eval/conf/batch/*.yaml`
  - 当前问题：
    - 文件名带较强实验痕迹，如模型名、preview 标记
  - 建议：
    - 区分“官方推荐矩阵”和“历史实验矩阵”

- 整理内部工程文档位置
  - 相关位置：
    - `docs/patches/communicationengineering_followup_20260324.patch`
    - `docs/shinkaevolve_adapter_hardening_plan.md`
  - 当前问题：
    - 内容本身不一定敏感，但更像内部工程资料，不太像外部用户文档
  - 建议：
    - 迁移到更明确的内部/开发者目录
    - 或转成 issue、PR、RFC 链接

- 为长任务补更明确的耗时提示
  - 相关位置：
    - `JobShop/*`
    - 其他长任务 README
  - 当前问题：
    - 用户容易把“长时间无输出”误判成卡死
  - 建议：
    - 在 README 里标典型 wall-clock
    - 写明推荐 timeout 和监控频率
