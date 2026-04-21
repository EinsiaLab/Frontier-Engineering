# SustainableDataCenterControl

这个 benchmark 目录现在作为一个任务集合，封装了基于 SustainDC 的数据中心控制子任务，并同时支持单独验证与 `frontier_eval` 的 unified task 运行。

下面所有命令都使用仓库相对路径，不依赖任何绝对路径，方便他人在 clone 后直接复现。

## 当前子任务

| 子任务目录 | 核心目标 | 说明 |
|---|---|---|
| `hand_written_control` | 为负载迁移、冷却控制和电池调度编写确定性控制策略 | 主要只修改 `baseline/solution.py`，评测时会与 noop 参考策略对比 |

## 环境准备

已验证的环境组合：

- `.venvs/frontier-v1-sustaindc`：用于 direct verification 和实际 benchmark 运行时
- `.venvs/frontier-eval-driver`：用于执行 `python -m frontier_eval`

从仓库根目录执行：

```bash
bash init.sh
RUN_VALIDATION=0 bash scripts/env/setup_v1_task_envs.sh
source .venvs/frontier-eval-driver/bin/activate
```

`benchmarks/SustainableDataCenterControl/requirements.txt` 当前会继续引用 `hand_written_control/sustaindc/requirements.txt` 里的上游依赖列表。

## 子任务索引与 Unified 运行命令

- `hand_written_control/`
  - direct verification：
    ```bash
    .venvs/frontier-v1-sustaindc/bin/python benchmarks/SustainableDataCenterControl/hand_written_control/verification/evaluate.py
    ```
  - unified 运行：
    ```bash
    python -m frontier_eval \
      task=unified \
      task.benchmark=SustainableDataCenterControl/hand_written_control \
      task.runtime.env_name=frontier-v1-sustaindc \
      algorithm=openevolve \
      algorithm.iterations=0
    ```
  - 已验证环境上的实测耗时：
    - direct verification：约 `19.8s`
    - unified `algorithm.iterations=0`：约 `25.8s`
  - 耗时说明：这个任务在当前验证环境上不属于长任务，两条命令都明显低于 unified 默认的 `300s` timeout。

## 备注

- 仓库中自带的 `hand_written_control/sustaindc/` 对应上游 `dc-rl` 的 `a92b475` commit。
- `hand_written_control/patches/sustaindc_optional_runtime.patch` 记录了 benchmark 兼容运行时所需的少量补丁。
- 更完整的任务说明和从全新上游 clone 开始的复现方法，请看 `hand_written_control/README.md`。
