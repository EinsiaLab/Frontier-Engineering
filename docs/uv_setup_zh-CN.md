# 使用 uv 管理 Frontier-Engineering 环境

这份说明的目标不是把所有 benchmark 强行塞进一个超大环境，而是把仓库环境整理成：

1. 一个默认的 `uv` 驱动环境，用于运行 `frontier_eval`
2. 若干按任务族拆分的 benchmark runtime 环境
3. 少量 `uv` 无法管理的系统依赖，继续单独安装

如果你的目标是覆盖整个 `v1`，请直接看：

```text
docs/uv_v1_envs_zh-CN.md
```

## 1. 安装 uv

先确保本机已有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或按官方文档选择其他安装方式。

## 2. 创建默认驱动环境

在仓库根目录执行：

```bash
uv sync
```

默认会同步 `driver` group，并在仓库根目录创建 `.venv/`。

当前 `driver` 是最小可运行驱动环境，不再默认安装 `torch` / `torchvision` / CUDA 大包。
这样做的目的是让：

- `uv sync`
- `uv run python -m frontier_eval task=smoke ...`

能更快完成，并与仓库当前 CPU 复现实践保持一致。

如果你希望显式指定 Python 版本：

```bash
uv python pin 3.12
uv sync
```

同步完成后，可以直接这样运行：

```bash
uv run python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

或：

```bash
uv run frontier-eval task=smoke algorithm=openevolve algorithm.iterations=0
```

## 3. 按需安装 benchmark 依赖

### 当前 `pyproject.toml` 里的 group

如果你只想在同一个 `.venv` 上叠加一部分兼容依赖，可以继续使用 group：

```bash
uv sync --group driver --group v1-main
```

这个 group 目前覆盖仓库里一组相对容易共存的 Python 依赖，包括：

- `mqt.bench`
- `stockpyl`
- `job-shop-lib`
- `PyPortfolioOpt`
- `mujoco`
- `pybullet`

如果你确实需要把 `torch` 栈额外装进当前项目环境，可以使用：

```bash
uv sync --group driver --group driver-torch
```

### 更完整的 v1 方式

如果你要覆盖 `v1` 全量任务，不建议继续往一个 `.venv` 里叠包，而是按环境矩阵批量创建：

```bash
bash scripts/setup_uv_envs.sh driver v1-general v1-optics v1-gpu v1-power v1-summit v1-sustaindc v1-kernel v1-openff v1-diffsim v1-singlecell-denoising
```

完整覆盖关系见：

```text
docs/uv_v1_envs_zh-CN.md
```

## 4. 非 Python 依赖仍需单独安装

`uv` 只管理 Python 环境，不能替代系统包管理器或 Conda 的非 Python 包能力。

当前仓库里最典型的例子是：

- `octave`
- `octave-signal`
- `octave-control`

如果你要跑依赖这些组件的 verifier，仍需先在系统层安装它们，然后再用 `uv` 管理 Python 依赖。

换句话说，推荐模型是：

- `uv` 管 Python 包和虚拟环境
- 系统包管理器 / Conda 管 `octave` 这类非 Python 依赖

## 5. 运行建议

为了避免把不同 benchmark 的冲突依赖混到一个环境里，建议按用途拆分：

### 仅驱动框架

```bash
uv sync
```

### 驱动框架 + 一部分常用依赖

```bash
uv sync --group driver --group v1-main
```

### 按 v1 任务族批量创建独立环境

```bash
bash scripts/setup_uv_envs.sh driver v1-general v1-optics v1-summit
```

## 6. 当前这版 uv 改造的边界

这版改造现在解决四件事：

- 仓库有了顶层 `pyproject.toml`
- `frontier_eval` 可以作为项目脚本入口被 `uv run` 调用
- 常用 Python 依赖有了可复用的 `dependency-groups`
- `v1` 任务有了按 runtime 拆分的 `uv` 环境矩阵和建环境脚本

暂时没有做的事情：

- 没有把所有 benchmark 的 `requirements.txt` 全部无脑并进一个环境
- 没有把 GPU kernel 那些超重依赖默认装进 driver 环境
- 没有试图让 `uv` 接管 `octave` 等非 Python 组件

这也是更适合 public repo 的做法，因为它能显著降低首次安装成本。
