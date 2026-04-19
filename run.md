# Frontier-Eng 运行说明

本文区分：

- **冒烟**：几乎不跑演化，只验证调度/链路（`algorithm.iterations=0` 等）。
- **完整跑**：对真实任务做多轮演化/搜索（`iterations` 或等价参数 > 0），且任务子进程能真正执行校验。

更细的 Hydra 参数、算法差异、任务覆盖项以 [`frontier_eval/README.md`](frontier_eval/README.md) 为准。

---

## 零、环境已配好后：一键跑 v1 批量评测

维护中的 **v1** 任务集合已合并为单个矩阵文件：

**`frontier_eval/conf/batch/v1.yaml`**

其中 `llms` 条目名为 **`v1`**，模型与网关从环境变量读取（与 `frontier_eval/conf/llm/openai_compatible.yaml` 一致）：

- `OPENAI_API_BASE`（可选）
- `OPENAI_MODEL`（可选，默认 `gpt-4o-mini`）
- `OPENAI_API_KEY`（通过 `api_key_env` 指定）

**在仓库根目录**（已 `conda activate frontier-eval-2`，并已按需完成 `init.sh`、合并任务环境、`scripts/setup_v1_merged_task_envs.sh`、`.env` 等）：

```bash
export PYTHONNOUSERSITE=1
conda activate frontier-eval-2

python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

**注意：**

- 默认 `algorithm.iterations=100`，**耗时长、API 费用高**。
- 矩阵内同时包含 CPU 与 GPU 类任务；GPU 相关任务请在运行前设置 **`CUDA_VISIBLE_DEVICES`**（按需）。
- **EngDesign** 需先按 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md) 设置 Docker 相关环境变量（例如 `ENGDESIGN_EVAL_MODE`、`ENGDESIGN_DOCKER_IMAGE`）。
- 若只需跑子集，可使用 `frontier_eval.batch --tasks ...` 或 `--exclude-tasks ...`，或调小 YAML 里的 `run.max_parallel`。

**矩阵自检（可选）**：在仓库根目录执行 `python scripts/debug_verify_v1_matrix.py`，会校验 `v1.yaml` 能否解析、`validate_v1_merged_task_envs.sh` 里用到的任务标签是否都存在，并将简要结果写入 `debug-e710d8.log`（可忽略或删除该文件）。

批量输出目录：`runs/batch/<run.name>/`（含 `summary.jsonl`）。若仅需 **iter=0** 验证合并环境，见 [`scripts/validate_v1_merged_task_envs.sh`](scripts/validate_v1_merged_task_envs.sh)。

---

## 一、共用前置：驱动环境 + 隔离

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

## 二、冒烟（快速验证驱动）

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

或默认 unified + 某 benchmark、0 次迭代：

```bash
python -m frontier_eval algorithm.iterations=0
```

---

## 三、完整跑起来（非冒烟）

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

---

## 四、怎样算「完整跑通」

- 冒烟：退出码 0 只说明调度侧 OK；子任务校验仍可能失败。
- 完整跑：`iterations>0` 下能持续评估，且基准 returncode/分数符合预期，并已按任务 README 处理 runtime 与资源。

---

## 五、一句话对照

| 目标 | 步骤概要 |
|------|----------|
| 冒烟 | `init.sh` → `activate` → `algorithm.iterations=0` 或 `task=smoke` |
| v1 全量 batch | `activate` → 配 `.env` → `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml` |
| 单任务完整跑 | `setup_v1_merged_task_envs`（按需）→ 读任务 README → `task=unified` + `task.benchmark=...` + `iterations>0` |
