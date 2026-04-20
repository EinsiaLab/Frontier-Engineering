# Frontier-Eng: how to run

Chinese: [run_zh-CN.md](run_zh-CN.md)

Hydra options, algorithms, and which tasks exist are documented in [`frontier_eval/README.md`](frontier_eval/README.md).

---

## 1. v1 batch evaluation

Matrix: **`frontier_eval/conf/batch/v1.yaml`** (**47** tasks). Model and gateway settings come from the environment (same idea as `frontier_eval/conf/llm/openai_compatible.yaml`): `OPENAI_API_BASE`, `OPENAI_MODEL`, `OPENAI_API_KEY`, etc.

### Script

From the repo root, after `init.sh`, merged task envs, and a configured `.env`:

```bash
bash scripts/run_v1_batch.sh
```

The script `cd`s to the repo root, sets `PYTHONNOUSERSITE=1`, and by default runs **`conda run -n frontier-eval-2`** with `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml`.  
Extra args are passed through to `frontier_eval.batch`, e.g.:

```bash
bash scripts/run_v1_batch.sh --dry-run
```

Environment variables (same as running by hand):

- `CUDA_VISIBLE_DEVICES`: GPU tasks
- `ENGDESIGN_EVAL_MODE`, `ENGDESIGN_DOCKER_IMAGE`, etc.: see [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md)
- `DRIVER_PY`: absolute path to `python` in `frontier-eval-2` if you do not use `conda run`
- `DRIVER_ENV`: defaults to `frontier-eval-2`
- `V1_MATRIX`: defaults to `frontier_eval/conf/batch/v1.yaml`

### Same thing without the script

```bash
export PYTHONNOUSERSITE=1
conda activate frontier-eval-2

python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

Windows PowerShell (adjust the path to your `frontier-eval-2`):

```powershell
cd "D:\path\to\Frontier-Engineering"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
& "$env:USERPROFILE\.conda\envs\frontier-eval-2\python.exe" -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

**Notes**

- Default `algorithm.iterations=100` is slow and costs API calls.
- The matrix mixes CPU and GPU tasks; set **`CUDA_VISIBLE_DEVICES`** for GPU tasks when needed.
- **EngDesign** needs Docker-related env vars as in [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md).
- Subsets: `frontier_eval.batch --tasks ...` or `--exclude-tasks ...`, or lower `run.max_parallel` in the YAML.

Optional: from the repo root, `python scripts/debug_verify_v1_matrix.py` checks that `v1.yaml` parses and that task tags used by the validation script exist; a short log goes to `debug-e710d8.log` (safe to delete).  
Merged env install: [`scripts/validate_v1_merged_task_envs.sh`](scripts/validate_v1_merged_task_envs.sh).

Batch output: `runs/batch/<run.name>/` (including `summary.jsonl`).

---

## 2. Environment and isolation

1. Repo root (Git Bash / WSL; conda installed):

   ```bash
   bash init.sh
   conda activate frontier-eval-2
   ```

2. Before long runs, same as root [`README.md`](README.md):

   ```bash
   export PYTHONNOUSERSITE=1
   ```

   Windows PowerShell:

   ```powershell
   $env:PYTHONNOUSERSITE = "1"
   $env:PYTHONUTF8 = "1"
   $env:PYTHONIOENCODING = "utf-8"
   ```

3. **API keys (`.env`)**: evolution needs a working LLM. If you do not have `.env` yet:

   ```bash
   cp .env.example .env
   ```

   Set at least **`OPENAI_API_KEY`** in `.env`. For OpenAI-compatible gateways, set **`OPENAI_API_BASE`** and **`OPENAI_MODEL`** as needed (same meaning as in `frontier_eval/conf/llm/openai_compatible.yaml`).  
   Do not commit a `.env` that contains real secrets.

   Batch and single-task runs read these from the environment / `.env`. If they are wrong or missing, runs with `algorithm.iterations>0` usually fail, e.g.:

   - `Missing API key for OpenEvolve...`
   - or other LLM-backed algorithms failing at startup with missing key / auth errors

---

## 3. Single tasks (not batch)

Commands and details: [`frontier_eval/README.md`](frontier_eval/README.md) (e.g. Unified task).  
Merged task envs: `bash scripts/setup_v1_merged_task_envs.sh`. Per-task docs: `benchmarks/<Domain>/README*.md`.
