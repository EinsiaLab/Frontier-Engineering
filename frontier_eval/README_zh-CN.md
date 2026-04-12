# Frontier Eval Framework

`Frontier-Engineering` 的评测框架。

## 结构

- `frontier_eval/cli.py`: 评测主入口（`python -m frontier_eval`）
- `frontier_eval/tasks/`: 所有评测任务
- `frontier_eval/algorithms/`: 所有算法（目前支持接入 `abmcts`、`openevolve`、`shinkaevolve`）
- `frontier_eval/conf/`: Hydra 配置（task / algorithm / llm）

## 环境准备

推荐使用 conda。

最简单的方式是在仓库根目录执行：

```bash
bash init.sh
conda activate frontier-eval-2
```

手动安装方式（仅在无法使用 `init.sh` 时）：

```bash
conda create -n frontier-eval-2 python=3.12 -y
conda activate frontier-eval-2

# Octave + signal/control
conda install -c conda-forge octave octave-signal octave-control -y

pip install -r frontier_eval/requirements.txt
```

重要说明：

上面的安装步骤只会把 `frontier_eval` 框架本身准备好。仓库里的很多 benchmark 还需要单独的 runtime 环境、benchmark 本地依赖、`third_party/` 仓库，或者 Docker。

在运行具体 benchmark 之前，请始终先看：

1. `benchmarks/<Domain>/README*.md`
2. 如果该 task 还有自己的 README，再继续看 `benchmarks/<Domain>/<Task>/README*.md`

这些 benchmark README 才是 runtime 配置的最终依据。里面如果写了 `task.runtime.conda_env=...`、`task.runtime.python_path=...`、`task.runtime.use_conda_run=false` 等 override，请直接按文档带进运行命令。

仓库里的常见例子包括：

- `ReactionOptimisation` 使用 `summit` 作为 benchmark runtime。
- `MolecularMechanics` 使用 `openff-dev`。
- `SustainableDataCenterControl` 使用 `sustaindc`。
- `PyPortfolioOpt` 使用 `pyportfolioopt`。
- `QuantumComputing` 使用 `mqt`。
- `InventoryOptimization` 使用 `stock`。
- `JobShop` 使用显式 `task.runtime.python_path`。
- `EngDesign` 优先走 Docker，也支持本地回退。

可选 Assistant/Agent 配置见 **[docs/agent_setup_zh-CN.md](../docs/agent_setup_zh-CN.md)**；skill 源码位于 **`skill/source/`**。

关于 `third_party/`：

本仓库会把部分第三方/较大依赖放在 `third_party/` 下，但这些目录内容默认不随 git 提交（见 `.gitignore`）。因此如果你看到类似 `pip install -e third_party/...` 的命令，需要先把对应仓库 clone 到本地，例如：

```bash
mkdir -p third_party

# AB-MCTS / TreeQuest（使用 `algorithm=abmcts` 时必需）
git clone https://github.com/SakanaAI/treequest.git third_party/treequest

# CarAerodynamicsSensing / PhySense（评测该任务时必需）
git clone https://github.com/thuml/PhySense.git third_party/PhySense
```

可选（ShinkaEvolve）：

```bash
# 注意：PyPI 上的 `shinka` 不是 ShinkaEvolve

# 方式 A：本地 clone 到 `third_party/`（需要打补丁/调试时推荐）
git clone https://github.com/SakanaAI/ShinkaEvolve.git third_party/ShinkaEvolve
# Frontier-Engineering 补丁：修复 `DatabaseDisplay` 在 `program.metadata` 缺失时的崩溃，
# 并在价格表中补充 OpenRouter 模型 `qwen/qwen3-coder-next`。
git apply patches/third_party_shinkaevolve.patch
pip install -e third_party/ShinkaEvolve

# 方式 B：可编辑 VCS 安装（确保 `shinka.core` 可用）：
pip install -e "git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka"
```

使用自定义模型运行 `algorithm=shinkaevolve` 时，请注意：

- 即使你把 `OPENAI_API_BASE` / `llm.api_base` 指向了自托管或 OpenAI-compatible 接口，ShinkaEvolve 仍然会从 `third_party/ShinkaEvolve/shinka/llm/providers/pricing.csv` 里解析模型路由和计价元信息。
- 如果 `OPENAI_MODEL` / `llm.model` 不在这张表里，ShinkaEvolve 很容易报 `Model ... not supported.` 或 `Model ... not found in pricing data`。
- 如果你要修改 `pricing.csv`、`client.py` 或 `query.py`，请优先使用上面的方式 A（本地 checkout），不要只做 VCS 安装。
- 对于挂在 OpenAI-compatible 接口后的自定义模型，通常应该在 `third_party/ShinkaEvolve/shinka/llm/providers/pricing.csv` 里新增一行，并把 `provider` 设为 `openai`，因为真正会读取 `OPENAI_API_BASE` 的是 `openai` backend。

自托管 OpenAI-compatible 模型的 `pricing.csv` 示例：

```csv
my-model,openai,N/A,N/A,,,," False"," 0"," 0"
```

然后在运行环境里把模型名设成同一个值：

```bash
export OPENAI_API_BASE=http://<your-endpoint>/v1
export OPENAI_API_KEY=<your-key>
export OPENAI_MODEL=my-model
```

如果你的模型需要新的 provider 或非标准 API，仅修改 `pricing.csv` 还不够。请参考 `third_party/ShinkaEvolve/docs/support_local_llm.md`，同时修改 `shinka/llm/client.py` 和 `shinka/llm/query.py`。

可选（AB-MCTS / TreeQuest）：

```bash
# 依赖 `third_party/treequest`（见上面的 clone 说明）。
pip install -e third_party/treequest
# 可选（ABMCTS-M / 混合模型）：
pip install -e "third_party/treequest[abmcts-m]"
# 可选（树可视化）：
pip install -e "third_party/treequest[vis]"
```

环境变量（推荐用 `.env`）：

```bash
cp .env.example .env
# 编辑 .env，写入 OPENAI_API_KEY / OPENAI_API_BASE 等
```

运行 `python -m frontier_eval ...` 时会自动从当前目录向上查找并加载最近的 `.env`。

## 运行

```bash
python -m frontier_eval algorithm.iterations=10
```

快速自检（很快、无需额外 benchmark 依赖）：

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
python -m frontier_eval task=smoke algorithm=shinkaevolve algorithm.max_generations=0
python -m frontier_eval task=smoke algorithm=abmcts algorithm.iterations=0
```

## Unified 统一任务

使用 `task=unified` 可以通过 benchmark 目录下的元数据文件接入新评测，不再需要为每个 benchmark 手写 `frontier_eval/tasks/<task>/...`。

运行示例：

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=ComputerSystems/MallocLab \
  algorithm=openevolve \
  algorithm.iterations=10
```

EngDesign 示例（使用预设配置，本质仍是 unified）：

```bash
python -m frontier_eval \
  task=engdesign \
  algorithm=openevolve \
  algorithm.iterations=10
```

等价的显式 unified 命令：

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=EngDesign \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=10
```

EngDesign 运行环境说明（参考 `benchmarks/EngDesign/README.md`）：
- `benchmarks/EngDesign/frontier_eval/run_eval.sh` 在可用时优先使用 `docker`（`ENGDESIGN_EVAL_MODE=auto`）。
- 如需强制本地 Python 评测，可设置 `ENGDESIGN_EVAL_MODE=local`。

#### Benchmark 元数据目录约定

在 `benchmarks/<Domain>/<Task>/frontier_eval/` 下放置：

```text
initial_program.txt      # 必需：baseline 候选程序相对路径
candidate_destination.txt# 可选：候选程序在沙箱中的落盘位置
eval_command.txt         # 必需：评测命令模板
eval_cwd.txt             # 可选：评测命令工作目录（相对 benchmark 根）
agent_files.txt          # 可选：暴露给 agent 的上下文文件列表
copy_files.txt           # 可选：复制到临时沙箱的文件/目录列表
readonly_files.txt       # 可选：运行前后必须保持不变的文件/目录
artifact_files.txt       # 可选：由框架自动收集的输出文件/目录
constraints.txt          # 可选：约束/提示词文本
```

行列表类型的 `*.txt`（如 `initial_program.txt`、`candidate_destination.txt`、`agent_files.txt`、`artifact_files.txt` 等）规则：
- 每行一个相对路径
- 空行忽略
- 以 `#` 开头的行忽略

`eval_command.txt` 是原始 shell 命令（可多行）。

#### 各个元数据文件的含义

- `initial_program.txt`：演化起始程序路径（相对 benchmark 根目录）。
- `candidate_destination.txt`：每轮候选程序在沙箱 benchmark 中写入的位置。不配置时默认等于 `initial_program.txt`。
- `eval_command.txt`：评测命令模板。
- `eval_cwd.txt`：评测命令工作目录（沙箱内，相对 benchmark 根目录），`.` 表示 benchmark 根。
- `agent_files.txt`：会注入到 artifacts 给 LLM 参考的文件/目录列表。
- `copy_files.txt`：复制到评测沙箱的文件/目录列表。为空时默认复制整个 benchmark 目录。
- `readonly_files.txt`：评测前后做指纹校验的路径，变化即判为 invalid。
- `artifact_files.txt`：评测结束后由 unified 框架自动采集到 artifacts 的文件/目录（如日志、stdout/stderr 输出文件），避免用户自己写 artifacts 导出代码。
- `constraints.txt`：自由文本约束，会作为 artifacts 提供给 agent 上下文。

#### 占位符说明

安全占位符（已做 shell 转义，推荐优先使用）：
- `{python}`：运行评测的 Python 命令。
- `{candidate}`：候选程序在沙箱中的路径。
- `{benchmark}`：沙箱 benchmark 根目录。
- `{sandbox}`：本次评测临时目录根。
- `{repo_root}`：Frontier-Engineering 仓库根目录。
- `{benchmark_source}`：原始 benchmark 目录（非沙箱）。
- `{benchmark_id}`：规范化 benchmark 标识（例如 `ComputerSystems/MallocLab`）。

原始占位符（不做 shell 转义）：
- `{python_raw}`, `{candidate_raw}`, `{benchmark_raw}`, `{sandbox_raw}`, `{repo_root_raw}`, `{benchmark_source_raw}`, `{benchmark_id_raw}`。
- 仅在你明确自己处理引号/转义时使用。

示例：

```text
bash frontier_eval/run_eval.sh {python} {benchmark} {candidate}
```

默认会尝试读取你的评测命令产出：
- `metrics.json`：JSON 对象。Unified 会读取所有“可转成数值”的字段，不仅是 `combined_score` 和 `valid`。
- `artifacts.json`（可选）：JSON 对象，可放结构化附加信息。
- 对于简单任务，可不写 `artifacts.json`，改用 `artifact_files.txt` 让 unified 自动收集日志类产物。
- 缺少 `valid` 时，用命令返回码兜底（`0 -> 1`，非 `0 -> 0`）。
- 缺少 `combined_score` 时，用 `valid` 兜底（`valid > 0 -> 1`，否则 `0`）。
- 若评测命令返回非 0，Unified 会强制 `valid=0` 且 `combined_score=0`。

若输出路径不同，可运行时覆盖：

```bash
python -m frontier_eval task=unified \
  task.benchmark=MyDomain/MyTask \
  task.metrics_json=verification/out/metrics.json \
  task.artifacts_json=verification/out/artifacts.json
```

具体例子可参考 `benchmarks/ComputerSystems/MallocLab/frontier_eval`

### 评测环境选择

`unified` 支持环境参数传入：
- 默认 conda 环境：`frontier-eval-2`
- 覆盖环境名：`task.runtime.conda_env=<env_name>`
- 显式 Python 路径：`task.runtime.python_path=/path/to/python`

重要：

- `frontier-eval-2` 只是 unified 的默认兜底值。
- 仓库里有不少 benchmark 并不应该直接用这个默认 runtime。
- 运行前请先查 benchmark README，并优先使用其中写明的环境名、Python 路径或 Docker 运行方式。

示例：

```bash
python -m frontier_eval task=unified \
  task.benchmark=MyDomain/MyTask \
  task.runtime.conda_env=frontier-eval-2
```

## 批量评测

使用 batch runner（会为每个组合写入独立的 `run.output_dir`，并汇总到 `summary.jsonl`）：

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml
```

补测（只重跑部分 task）：

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml \
  --tasks denoising --tasks trimul
```

原地补测（在已有 batch 目录下补测；会先删除选中的 task 目录再重跑）：

```bash
python -m frontier_eval.batch --matrix runs/batch/<batch_id>/matrix_resolved.yaml \
  --in-place --tasks denoising
```

## 扩展方式（新增 task / algorithm）

- 新 benchmark 贡献的默认要求：直接使用 `task=unified` + benchmark 本地元数据文件（见上文）接入；新增 benchmark 的 PR 不应再默认新增 `frontier_eval/tasks/` 下的 Python task 代码。
- 仅在 unified 明确无法满足需求、且已先与维护者沟通例外方案时：实现 `frontier_eval/tasks/base.py` 的 `Task` 子类（`initial_program_path()` + `evaluate_program()`），并在 `frontier_eval/registry_tasks.py` 注册（或继续用 `frontier_eval/registry.py` 的 `get_task`）。
- 新增算法：实现 `frontier_eval/algorithms/base.py` 的 `Algorithm` 子类，并在 `frontier_eval/registry_algorithms.py` 注册。

## `v1` 合并任务环境

为减少有效 `v1` 任务池使用的 task runtime 环境数量，仓库提供了 `scripts/env_specs/` 下的声明式环境清单与一键构建脚本：

- `frontier-eval-2` 由 `scripts/env_specs/frontier-eval-2.yml` 统一管理为默认 driver 环境。
- 合并环境（`frontier-v1-main`、`frontier-v1-summit`、`frontier-v1-sustaindc`、`frontier-v1-kernel`）由仓库内 manifest 声明并构建，不再依赖克隆本地临时环境。
- 对于需要「直连解释器」而不是 `conda run` 的 `v1` 任务（当前主要是 `ReactionOptimisation/*` 与 `JobShop/*`），batch matrix 里使用可移植标记 `conda-env:<env-name>`，由 unified evaluator 在运行时解析为对应环境中的 Python 路径，因此不需要把机器本地前缀写进仓库。

当前 `v1` task runtime 合并结果为：

- `frontier-v1-main`：`SingleCellAnalysis/predict_modality`、`QuantumComputing/*`、`Optics/*`、`InventoryOptimization/*`、`PyPortfolioOpt/*`、`JobShop/*`、`Robotics/DynamicObstacleAvoidanceNavigation`、`Robotics/PIDTuning`、`Robotics/UAVInspectionCoverageWithWind`、`Robotics/QuadrupedGaitOptimization`、`Robotics/RobotArmCycleTimeOptimization`、`Aerodynamics/CarAerodynamicsSensing`、`KernelEngineering/FlashAttention`
- `frontier-v1-summit`：`ReactionOptimisation/*`
- `frontier-v1-sustaindc`：`SustainableDataCenterControl/*`
- `frontier-v1-kernel`：`KernelEngineering/MLA`、`KernelEngineering/TriMul`

如果某个历史 README 仍然写着旧环境名（例如 `mqt`、`stock`、`pyportfolioopt`、`jobshop` 等），对于当前 `v1` 批量运行，请优先以 `frontier_eval/conf/batch/` 下的 matrix 配置为准。

环境准备与验证脚本：

- 基于声明式清单初始化/更新合并环境（默认会在构建后执行 `iter=0` 验证）：`bash scripts/setup_v1_merged_task_envs.sh`
- 按 `iter=0` 验证合并环境：`DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<gpu_id> bash scripts/validate_v1_merged_task_envs.sh`
- 审计 benchmark 的 readonly 元数据覆盖：`python scripts/audit_unified_metadata_readonly.py [--strict]`

说明：

- 上述验证默认使用 `conda run -n frontier-eval-2 python` 作为 driver，也可以通过 `DRIVER_PY=/path/to/python` 显式覆盖；脚本会验证 CPU `v1`、GPU `v1`、`FlashAttention`、`MLA`、`TriMul`。
- `MuonTomography` 已列在 [TASK_DETAILS_zh-CN.md](../TASK_DETAILS_zh-CN.md) 中，但在评测器重构完成前暂不纳入 `v1` 批量矩阵。
- 已知限制：`KernelEngineering/TriMul` 的官方 full benchmark（`verification/tri_bench.txt`）在 24GB 级别 GPU 上可能受显存上限影响；这通常是 task 本身的显存边界问题，而不是 `frontier-v1-kernel` 环境缺依赖。
