# Frontier-Eng 运行说明

英文：[run_en.md](run_en.md)

Hydra 参数、算法与任务覆盖以 [`frontier_eval/README.md`](frontier_eval/README.md) 为准。

---

## 一、v1 批量评测

v1 矩阵：**`frontier_eval/conf/batch/v1.yaml`**（**47** 个任务）。模型与网关从环境变量读取（约定同 `frontier_eval/conf/llm/openai_compatible.yaml`）：`OPENAI_API_BASE`、`OPENAI_MODEL`、`OPENAI_API_KEY` 等。

### 脚本

在仓库根目录，已执行 `init.sh`、装好合并任务环境、配置好 `.env` 后：

```bash
bash scripts/run_v1_batch.sh
```

脚本会 `cd` 到仓库根、设置 `PYTHONNOUSERSITE=1`，默认用 **`conda run -n frontier-eval-2`** 调用 `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml`。  
额外参数会传给 `frontier_eval.batch`，例如：

```bash
bash scripts/run_v1_batch.sh --dry-run
```

环境变量（与手动运行相同）：

- `CUDA_VISIBLE_DEVICES`：GPU 类任务
- `ENGDESIGN_EVAL_MODE`、`ENGDESIGN_DOCKER_IMAGE` 等：见 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md)
- `DRIVER_PY`：不用 `conda run` 时，设为 `frontier-eval-2` 里 `python` 的绝对路径
- `DRIVER_ENV`：默认 `frontier-eval-2`
- `V1_MATRIX`：默认 `frontier_eval/conf/batch/v1.yaml`

### 等价命令

```bash
export PYTHONNOUSERSITE=1
conda activate frontier-eval-2

python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

Windows PowerShell（把 `python.exe` 路径改成你本机 `frontier-eval-2`）：

```powershell
cd "你的\Frontier-Engineering\根目录"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
& "$env:USERPROFILE\.conda\envs\frontier-eval-2\python.exe" -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

**注意：**

- 默认 `algorithm.iterations=100`，耗时长、API 费用高。
- 矩阵里同时有 CPU 与 GPU 任务；GPU 任务运行前请设置 **`CUDA_VISIBLE_DEVICES`**（按需）。
- **EngDesign** 需按 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md) 配置 Docker 相关环境变量。
- 只跑子集可用 `frontier_eval.batch --tasks ...` 或 `--exclude-tasks ...`，或调小 YAML 里的 `run.max_parallel`。

可选：在仓库根执行 `python scripts/debug_verify_v1_matrix.py`，检查 `v1.yaml` 能否解析、校验脚本里用到的任务标签是否存在；简要结果写入 `debug-e710d8.log`（可删）。  
合并任务环境安装见 [`scripts/validate_v1_merged_task_envs.sh`](scripts/validate_v1_merged_task_envs.sh)。

批量输出目录：`runs/batch/<run.name>/`（含 `summary.jsonl`）。

---

## 二、环境与隔离

1. 仓库根（Git Bash / WSL；需已安装 conda）：

   ```bash
   bash init.sh
   conda activate frontier-eval-2
   ```

2. 全量跑前建议设置（与仓库根 `README.md` 一致）：

   ```bash
   export PYTHONNOUSERSITE=1
   ```

   Windows PowerShell：

   ```powershell
   $env:PYTHONNOUSERSITE = "1"
   $env:PYTHONUTF8 = "1"
   $env:PYTHONIOENCODING = "utf-8"
   ```

3. **API 密钥（`.env`）**：演化需要可调用的 LLM。若还没有 `.env`，从示例复制后编辑：

   ```bash
   cp .env.example .env
   ```

   在 `.env` 中**至少**填写 **`OPENAI_API_KEY`**。使用兼容 OpenAI 的第三方网关时，按需同时设置 **`OPENAI_API_BASE`**、**`OPENAI_MODEL`**（含义与 `frontier_eval/conf/llm/openai_compatible.yaml` 一致）。  
   不要将含真实密钥的 `.env` 提交到版本库。

   批量矩阵与单任务运行都会从环境/`.env` 读取上述变量；未正确配置时，`algorithm.iterations>0` 的演化通常会失败，例如：

   - `Missing API key for OpenEvolve...`
   - 或其它依赖 LLM 的算法在启动时提示缺少 key / 认证失败

---

## 三、单任务与非批量

单任务命令与说明见 [`frontier_eval/README.md`](frontier_eval/README.md)（Unified task 等）。  
合并任务环境：`bash scripts/setup_v1_merged_task_envs.sh`。各任务见 `benchmarks/<Domain>/README*.md`。
