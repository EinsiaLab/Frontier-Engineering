# Frontier Eval Framework

`Frontier-Engineering` 的评测框架。

## 结构

- `frontier_eval/cli.py`: 主程序入口（`python -m frontier_eval`）
- `frontier_eval/tasks/`: 所有评测任务
- `frontier_eval/algorithms/`: 所有算法（目前支持接入 `openevolve`、`shinkaevolve`）
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

可选（ShinkaEvolve）：

```bash
# 注意：PyPI 上的 `shinka` 不是 ShinkaEvolve
# 推荐用可编辑安装（确保 `shinka.core` 可用）：
pip install -e "git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka"
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
```

## 批量评测

使用 batch runner（会为每个组合写入独立的 `run.output_dir`，并汇总到 `summary.jsonl`）：

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml
```

## 扩展方式（新增 task / algorithm）

- 新增任务：实现 `frontier_eval/tasks/base.py` 的 `Task` 子类（`initial_program_path()` + `evaluate_program()`），并在 `frontier_eval/registry_tasks.py` 注册（或继续用 `frontier_eval/registry.py` 的 `get_task`）。
- 新增算法：实现 `frontier_eval/algorithms/base.py` 的 `Algorithm` 子类，并在 `frontier_eval/registry_algorithms.py` 注册。
