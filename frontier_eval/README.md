# Frontier Eval Framework

目标：为 `Frontier-Engineering` 提供一个结构清晰、可扩展的评测框架（task / algorithm 插件化 + Hydra 配置），并先跑通 `Astrodynamics/MannedLunarLanding` + `OpenEvolve` 的最小工作流。

## 结构

- `frontier_eval/cli.py`: Hydra 入口（`python -m frontier_eval`）
- `frontier_eval/tasks/`: 任务适配层（提供初始程序、OpenEvolve evaluator 等）
- `frontier_eval/algorithms/`: 算法适配层（目前先接入 `openevolve`）
- `frontier_eval/evaluators/`: OpenEvolve 所需的 `evaluate(program_path)` 文件
- `frontier_eval/conf/`: Hydra 配置（task / algorithm / llm）

## 环境准备（MannedLunarLanding + Octave）

推荐使用 conda（安装 Octave 与 Python 依赖；Python 依赖统一走 `requirements.txt`）：

```bash
conda create -n frontier-eval python=3.12 -y
conda activate frontier-eval

# Octave + signal/control
conda install -c conda-forge octave octave-signal octave-control -y

pip install -r frontier_eval/requirements.txt
```

环境变量（推荐用 `.env`）：

```bash
cp .env.example .env
# 编辑 .env，写入 OPENAI_API_KEY / OPENAI_API_BASE 等
```

运行 `python -m frontier_eval ...` 时会自动从当前目录向上查找并加载最近的 `.env`。

> 说明：conda-forge 的 `octave` 会依赖 `qscintilla2`（Qt 编辑器组件）。老版本 Octave（例如 6.4）对 `qscintilla2` 有更严格的版本约束，
> 在 `python=3.12` 环境里可能无法求解；不 pin 版本通常能让 conda 自动选到可用的 Octave（如 10.x）。
>
> 也可以用系统 Octave（apt）+ `pip install -r frontier_eval/requirements.txt`。

## 运行

### 1) 先跑通（不调用 LLM）

```bash
python -m frontier_eval algorithm.iterations=0
```

会在 `runs/manned_lunar_landing/openevolve/.../openevolve/best/` 写出 best program 与指标。
同时会额外输出：

- `.../openevolve/history/`：每一步的 `program.py` + `metrics.json` + `artifacts/`（run log 等）
- `.../openevolve/db/`：OpenEvolve 数据库快照（`programs/*.json` + `metadata.json`）
- `.../openevolve/evolution_trace.jsonl`：父/子程序的 trace（含代码、指标、prompts、artifacts）

### 2) 真正演化（需要 OpenAI-compatible API）

```bash
export OPENAI_API_KEY="YOUR_KEY"
# 可选：export OPENAI_API_BASE="https://api.openai.com/v1"

python -m frontier_eval algorithm.iterations=10 llm.model=gpt-4o-mini
```

## 扩展方式（新增 task / algorithm）

- 新增任务：实现 `frontier_eval/tasks/base.py` 的 `Task` 子类，并在 `frontier_eval/registry.py` 注册；同时提供一个对应的 evaluator 文件（给 OpenEvolve 调用）。
- 新增算法：实现 `frontier_eval/algorithms/base.py` 的 `Algorithm` 子类，并在 `frontier_eval/registry.py` 注册。
