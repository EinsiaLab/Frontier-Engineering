# Frontier-Engineering 复现实验简版结论（2026-04-16）

完整版记录见：

- `docs/repro_2026-04-16_zh-CN.md`

## 1. 一句话结论

这次复现不是“完全顺滑”，但主线是通的。按当前 `v1` CPU batch 配置，任务整体已经能跑到可记录结果；主要问题集中在环境叙事不统一、少数 benchmark 兼容性缺口，以及一些明显带内部运行痕迹的 public-facing 文件。

## 2. 配环境是否顺利

相对顺利的部分：

- `frontier_eval` 驱动环境可以用隔离 prefix env 稳定搭起来
- 多数 CPU 任务都能通过“最小依赖 env + 镜像源”快速补齐
- direct / unified 的复用套路比较明确

主要卡点：

- 大包下载慢，尤其是 `torch`
- 仓库默认更偏向 named conda env，而实际复现时 prefix env 更稳妥
- GPU / 图形相关任务在无图形库节点上会踩 `libGL.so.1`
- `summit` 与 `scikit-learn` 的兼容问题会影响 `ReactionOptimisation`
- 一些任务需要仓库外内容或额外元数据，文档没有把这件事讲透

最容易让初次用户误判的问题：

1. `JobShop/ta` 这类任务长时间无输出，但实际上是正常长任务。
2. unified 在 prefix env 场景下若沿用 `task.runtime.conda_env`，会很快失败，看起来像 benchmark 坏了。
3. `holographic_*` 任务会连续暴露 device 选择和 `opencv/libGL` 两类问题。

## 3. 还没完全漂亮收口的点

从 `v1` 范围看，这一轮已经收口。

剩下更值得发布前继续修的，是文档、环境默认值和运行产物管理这些“上手体验”问题，而不是 `v1` 主矩阵本身。

## 4. 让 repo 更易于入手，建议先做什么

建议优先顺序：

1. 统一环境模型
   - 明确官方支持 named env 还是 prefix env
   - 给出一套标准 unified 模板
2. 重写 `v1` 入口文档
   - 把稳定指南和本地运行记录拆开
3. 处理运行产物
   - 不要让用户一跑 benchmark 就把仓库跑脏
4. 清理 public 默认值
   - 去掉私有代理域名、个人路径、过时命令
5. 单独收口少数 benchmark 兼容性问题
   - 优先处理已经在 `v1` 复现过程中暴露出来的 evaluator / runtime 边界问题

## 5. 不太适合出现在 public repo 的内容

优先处理：

- `docs/v1_task_run_guide_zh-CN.md`
  - 更像本地运行/环境核对记录，且混有个人绝对路径
- `frontier_eval/conf/batch/v1_cpu_openevolve_p8_i100_gemini-3.1-pro-preview.yaml`
  - 默认 `api_base` 指向私有代理域名
- `scripts/pr_review.py`
  - 同样写死了私有代理地址

建议规范化处理：

- `benchmarks/Optics/*/verification/outputs/*`
- `benchmarks/Optics/*/verification/artifacts/*`
- `benchmarks/SustainableDataCenterControl/hand_written_control/verification/last_eval.json`

这些文件不是“泄密”，但它们是运行产物，当前被 git 跟踪，导致仓库一运行就变脏。

## 6. 发布前最值得先修的规范问题

- 去掉私有代理和个人路径
- 把内部计划文档、patch 留档、用户文档分层
- 补 `.gitignore` 或移动运行产物位置
- 校验 README 示例命令和真实文件名是否一致
