# Frontier-Eng 运行说明

英文：[run.md](run.md)

框架级命令见 [`frontier_eval/README_zh-CN.md`](frontier_eval/README_zh-CN.md)。这份文档只讲发布版 `v1` 矩阵的实际运行方式。

## 一、先把环境准备好

在仓库根目录执行：

```bash
bash init.sh
bash scripts/setup_v1_merged_task_envs.sh
source .venvs/frontier-eval-2/bin/activate
```

这会准备出：

- `.venvs/frontier-eval-2`：driver 环境
- `.venvs/frontier-v1-main`：大多数 CPU 任务
- `.venvs/frontier-v1-summit`：`ReactionOptimisation/*`
- `.venvs/frontier-v1-sustaindc`：`SustainableDataCenterControl/*`
- `.venvs/frontier-v1-kernel`：kernel / GPU 任务

跑长任务前建议：

```bash
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
```

## 二、配置模型访问

需要跑优化过程时，准备 `.env`：

```bash
cp .env.example .env
```

至少设置：

- `OPENAI_API_KEY`
- 可选 `OPENAI_API_BASE`
- 可选 `OPENAI_MODEL`

如果只是做 baseline 验证，并且使用 `algorithm.iterations=0`，则**不需要** LLM key。

## 三、运行发布版 `v1` 矩阵

### 正常批量运行

```bash
bash scripts/run_v1_batch.sh
```

它本质上会用 `.venvs/frontier-eval-2` 里的解释器执行：

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

常见变体：

```bash
bash scripts/run_v1_batch.sh --dry-run
bash scripts/run_v1_batch.sh --tasks KernelEngineering/MLA
bash scripts/run_v1_batch.sh --exclude-tasks engdesign
```

### 只验证 baseline

不调用 LLM、只检查仓库自带 baseline 是否可评测：

```bash
bash scripts/validate_v1_merged_task_envs.sh
```

这个脚本会把 `algorithm.iterations=0` 带进矩阵，并分 CPU、GPU、kernel 三段去验证。

## 四、常用运行参数

- `CUDA_VISIBLE_DEVICES`：选择 GPU
- `GPU_DEVICES`：`scripts/validate_v1_merged_task_envs.sh` 使用的 GPU 编号
- `DRIVER_ENV`：默认 `frontier-eval-2`
- `DRIVER_PY`：如果不想用默认 driver，可直接指定 Python 路径
- `V1_MATRIX`：覆盖矩阵文件路径
- `ENGDESIGN_EVAL_MODE`、`ENGDESIGN_DOCKER_IMAGE`：见 [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md)

## 五、baseline sweep 能说明什么

baseline-only 验证很有价值，因为它能确认：

- Hydra 配置能否正常解析
- benchmark runtime 能否启动
- 仓库自带 baseline 能否被 evaluator 正常执行
- `metrics.json` / `artifacts.json` 这一整条链是否通畅

但它**不能**自动证明所有任务在一台全新机器上都完全自包含。部分任务仍然依赖外部数据、模型、Docker、Octave、CUDA 或 benchmark-local 资源。

## 六、输出目录

普通 batch 输出在：

```text
runs/batch/<run.name>/
```

baseline 验证默认写到：

```text
runs/batch_validation/
```

每个任务都有独立目录，汇总结果写入 `summary.jsonl`。
