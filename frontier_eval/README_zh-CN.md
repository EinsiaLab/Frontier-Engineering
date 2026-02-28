# Frontier Eval Framework

`Frontier-Engineering` 的评测框架。

## 结构

- `frontier_eval/cli.py`: 主程序入口（`python -m frontier_eval`）
- `frontier_eval/tasks/`: 所有评测任务
- `frontier_eval/algorithms/`: 所有算法（目前支持接入 `abmcts`、`openevolve`、`shinkaevolve`）
- `frontier_eval/conf/`: Hydra 配置（task / algorithm / llm）

## 环境准备

推荐使用 conda。

最简单的方式是在仓库根目录执行：

```bash
bash init.sh
conda activate frontier-eval
```

手动安装方式：

```bash
conda create -n frontier-eval python=3.12 -y
conda activate frontier-eval

# Octave + signal/control
conda install -c conda-forge octave octave-signal octave-control -y

pip install -r frontier_eval/requirements.txt
```

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
# 推荐用可编辑安装（确保 `shinka.core` 可用）：
pip install -e "git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka"
```

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

## 扩展方式（新增 task / algorithm）

- 大多数新 benchmark 推荐：直接使用 `task=unified` + benchmark 本地元数据文件（见上文），无需新增 Python task 代码。
- 仅在 unified 不满足需求时：实现 `frontier_eval/tasks/base.py` 的 `Task` 子类（`initial_program_path()` + `evaluate_program()`），并在 `frontier_eval/registry_tasks.py` 注册（或继续用 `frontier_eval/registry.py` 的 `get_task`）。
- 新增算法：实现 `frontier_eval/algorithms/base.py` 的 `Algorithm` 子类，并在 `frontier_eval/registry_algorithms.py` 注册。
