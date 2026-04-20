# Frontier-Eng 运行说明

本文区分：

- **冒烟**：几乎不跑演化，只验证调度/链路（`algorithm.iterations=0` 等）。
- **完整跑**：对真实任务做多轮演化/搜索（`iterations` 或等价参数 > 0），且任务子进程能真正执行校验。

更细的 Hydra 参数、算法差异、任务覆盖项以 [`frontier_eval/README.md`](frontier_eval/README.md) 为准。

---

## 一、环境已配好后：一键跑 v1 批量评测

维护中的 **v1** 任务集合在单个矩阵文件：**`frontier_eval/conf/batch/v1.yaml`**（当前 **47** 个任务）。模型与网关等从环境变量读取（约定同 `frontier_eval/conf/llm/openai_compatible.yaml`）：`OPENAI_API_BASE`、`OPENAI_MODEL`、`OPENAI_API_KEY` 等。

### 集成脚本（一键）

在仓库根目录，已装好 `conda` 且已执行过 `init.sh`、按需完成合并任务环境、配置好 `.env` 后：

```bash
bash scripts/run_v1_batch.sh
```

脚本会 `cd` 到仓库根、设置 `PYTHONNOUSERSITE=1`、默认用 **`conda run -n frontier-eval-2`** 调用 `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml`。  
额外参数会原样传给 `frontier_eval.batch`，例如：

```bash
bash scripts/run_v1_batch.sh --dry-run
bash scripts/run_v1_batch.sh --override algorithm.iterations=0
```

可选环境变量（与手动运行相同）：

- `CUDA_VISIBLE_DEVICES`：GPU 类任务
- `ENGDESIGN_EVAL_MODE`、`ENGDESIGN_DOCKER_IMAGE` 等：见 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md)
- `DRIVER_PY`：若不想用 `conda run`，可设为 `frontier-eval-2` 里 `python` 的绝对路径
- `DRIVER_ENV`：默认 `frontier-eval-2`
- `V1_MATRIX`：默认 `frontier_eval/conf/batch/v1.yaml`

### 等价的手动命令

```bash
export PYTHONNOUSERSITE=1
conda activate frontier-eval-2

python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

Windows PowerShell（未用 Git Bash 时，将 `python.exe` 路径改成你本机 `frontier-eval-2` 路径）：

```powershell
cd "你的\Frontier-Engineering\根目录"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
& "$env:USERPROFILE\.conda\envs\frontier-eval-2\python.exe" -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

**注意：**

- 默认 `algorithm.iterations=100`，**耗时长、API 费用高**。
- 矩阵内同时包含 CPU 与 GPU 类任务；GPU 相关任务请在运行前设置 **`CUDA_VISIBLE_DEVICES`**（按需）。
- **EngDesign** 需先按 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md) 设置 Docker 相关环境变量（例如 `ENGDESIGN_EVAL_MODE`、`ENGDESIGN_DOCKER_IMAGE`）。
- 若只需跑子集，可使用 `frontier_eval.batch --tasks ...` 或 `--exclude-tasks ...`，或调小 YAML 里的 `run.max_parallel`。

**矩阵自检（可选）**：在仓库根目录执行 `python scripts/debug_verify_v1_matrix.py`，会校验 `v1.yaml` 能否解析、`validate_v1_merged_task_envs.sh` 里用到的任务标签是否都存在，并将简要结果写入 `debug-e710d8.log`（可忽略或删除该文件）。

批量输出目录：`runs/batch/<run.name>/`（含 `summary.jsonl`）。若仅需 **iter=0** 验证合并环境，见 [`scripts/validate_v1_merged_task_envs.sh`](scripts/validate_v1_merged_task_envs.sh)。

---

## 二、共用前置：驱动环境 + 隔离

1. 仓库根目录（Git Bash / WSL；需已安装 conda）：

   ```bash
   bash init.sh
   conda activate frontier-eval-2
   ```

2. 完整跑之前建议设置（与仓库根 `README.md` 一致）：

   ```bash
   export PYTHONNOUSERSITE=1
   ```

   Windows PowerShell：

   ```powershell
   $env:PYTHONNOUSERSITE = "1"
   $env:PYTHONUTF8 = "1"
   $env:PYTHONIOENCODING = "utf-8"
   ```

3. 演化需要 LLM：复制并编辑 `.env`（`init.sh` 可能已生成）：

   ```bash
   cp .env.example .env
   # 至少配置 OPENAI_API_KEY；兼容网关可设 OPENAI_API_BASE、OPENAI_MODEL
   ```

---

## 三、冒烟（快速验证驱动）

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

或默认 unified + 某 benchmark、0 次迭代：

```bash
python -m frontier_eval algorithm.iterations=0
```

---

## 四、完整跑起来（非冒烟）

完整跑 = 驱动环境（`frontier-eval-2`）+ 该任务 runtime + 有效 `.env` + `iterations>0`。

- 合并任务环境：`bash scripts/setup_v1_merged_task_envs.sh`
- 阅读 `benchmarks/<Domain>/README*.md` 与各任务目录说明
- 单任务示例见 [`frontier_eval/README.md`](frontier_eval/README.md)「Unified task」

**最小完整演化闭环（`task=smoke`，不依赖重 benchmark）：**

```bash
cp .env.example .env
# 填写 OPENAI_API_KEY
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=3
```

未配置密钥时，`algorithm.iterations>0` 会报错：`Missing API key for OpenEvolve...`
