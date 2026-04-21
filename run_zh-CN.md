# Frontier-Eng 运行说明

英文：[run.md](run.md)

框架级命令见 [`frontier_eval/README_zh-CN.md`](frontier_eval/README_zh-CN.md)。这份文档只讲发布版 `v1` 题集的实际运行方式。

## 0. 宿主机要求，以及仓库到底会自动化什么

这个仓库会自动化它自己管理的 Python 环境，但不会替你安装宿主机工具、GPU 驱动、Docker，或者下载大体积第三方 benchmark 资产。

如果你想完整跑 `v1` baseline sweep，默认应当具备：

| 条件 | 用途 | 如何检查 |
|---|---|---|
| Linux shell 环境、常规构建工具和可用外网 | 所有 setup 路径 | `python3 --version`、`git --version`、`curl --version` |
| NVIDIA GPU + 可用 CUDA runtime | `KernelEngineering/*`、`Aerodynamics/CarAerodynamicsSensing`、部分 Robotics 任务 | `nvidia-smi` 和一次成功的 CUDA PyTorch 导入 |
| Docker | `EngDesign` | `docker version` |
| Octave | `Astrodynamics/MannedLunarLanding` | `octave --version` |
| task-local 资产 / checkpoint / 第三方仓库 | 部分任务 | 以各任务 README 的说明为准 |

### 需要时怎样安装宿主机工具

下面的示例默认你用的是 Ubuntu 或其他 Debian 系 Linux。

如果你想直接让仓库帮你跑常见的 apt 安装流程，可以用：

```bash
bash scripts/bootstrap/install_host_deps.sh --octave
bash scripts/bootstrap/install_host_deps.sh --docker --configure-docker-group
```

#### Octave

```bash
sudo apt-get update
sudo apt-get install -y octave
octave --version
```

#### Docker

如果发行版自带的包已经够你本机评测使用，可以直接：

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
docker version
```

改完用户组后，重新开一个 shell 再继续。如果你需要更标准的官方安装路径，请参考 Docker 官方文档：[docs.docker.com/engine/install](https://docs.docker.com/engine/install/)。

#### CUDA / NVIDIA 栈

这个仓库不会替你安装 GPU 驱动或 CUDA。跑 GPU 任务前，至少先检查：

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

如果这些检查不过，先按你机器或集群的标准流程把 NVIDIA driver / CUDA 栈配好。

### 仍需单独准备的外部资产

仓库现在也提供了一个统一的 benchmark / algorithm 资产拉取入口：

```bash
python scripts/bootstrap/fetch_task_assets.py --list
python scripts/bootstrap/fetch_task_assets.py --target v1-baseline-assets
python scripts/bootstrap/fetch_task_assets.py --target shinkaevolve
python scripts/bootstrap/fetch_task_assets.py --target abmcts
```

其中 `v1-baseline-assets` 会覆盖目前已经自动化的 benchmark 资产 bundle，例如 `Aerodynamics/CarAerodynamicsSensing` 的 PhySense 代码/数据/权重，以及在 vendored tree 缺失时补 `SustainableDataCenterControl` 的 `dc-rl` checkout。

| 条件 | 受影响任务 | 去哪里看 |
|---|---|---|
| `dc-rl` checkout 和 SustainDC 资产 | `SustainableDataCenterControl` | `benchmarks/SustainableDataCenterControl/README_zh-CN.md` 和 `hand_written_control/README_zh-CN.md` |
| PhySense 数据、checkpoint 和参考点 | `Aerodynamics/CarAerodynamicsSensing` | `benchmarks/Aerodynamics/CarAerodynamicsSensing/README_zh-CN.md` |
| `openff-dev` runtime | `MolecularMechanics/*` | 先执行 `bash scripts/bootstrap/install_openff_dev.sh`，再看 `benchmarks/MolecularMechanics/README_zh-CN.md` |

## 一、先把环境准备好

在仓库根目录执行：

```bash
bash init.sh
bash scripts/env/setup_v1_task_envs.sh
source .venvs/frontier-eval-driver/bin/activate
```

这会准备出：

- `.venvs/frontier-eval-driver`：driver 环境
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

## 三、运行发布版 `v1` 题集

### 正常批量运行

```bash
bash scripts/batch/run_v1_batch.sh
```

它本质上会用 `.venvs/frontier-eval-driver` 里的解释器执行：

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

常见变体：

```bash
bash scripts/batch/run_v1_batch.sh --dry-run
bash scripts/batch/run_v1_batch.sh --tasks KernelEngineering/MLA
bash scripts/batch/run_v1_batch.sh --exclude-tasks engdesign
```

### 只验证 baseline

不调用 LLM、只检查仓库自带 baseline 是否可评测：

```bash
bash scripts/batch/validate_v1_task_envs.sh
```

这个脚本会按发布版 `v1` 题集对应的 batch 配置，把 `algorithm.iterations=0` 带进任务，并分 CPU、GPU、kernel 三段去验证。

要注意：它验证的是“在宿主机条件和外部资产都已经准备好的前提下，这些 baseline 能不能跑”。它并不意味着一台刚装完 `uv` 的全新机器就会自动 47/47 全绿。

## 四、常用运行参数

- `CUDA_VISIBLE_DEVICES`：选择 GPU
- `GPU_DEVICES`：`scripts/batch/validate_v1_task_envs.sh` 使用的 GPU 编号
- `DRIVER_ENV`：默认 `frontier-eval-driver`
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
