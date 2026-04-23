# PyPortfolioOpt

该领域聚焦于贴近实盘的投资组合再平衡与配置优化任务。
任务设定与求解流程依赖
[PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) 项目。

## 依赖说明

在仓库根目录安装依赖：

```bash
pip install -r benchmarks/PyPortfolioOpt/requirements.txt
pip install PyPortfolioOpt
```

## 三个子任务的共性

- 都是在现实约束下优化组合决策（暴露、换手、预算、交易摩擦等）。
- 都使用固定随机种子生成评测样本，评测可复现。
- 都以参考优化器结果作为上界进行归一化评分（理论上界分数为 `100`）。

## 三个子任务的差异

- `robust_mvo_rebalance`：鲁棒均值-方差目标，含行业/因子/换手约束（凸优化）。
- `cvar_stress_control`：基于情景收益的尾部风险最小化，兼顾收益门槛与暴露约束（CVaR 优化）。
- `discrete_rebalance_mip`：整数手数下单，含交易费与换手金额上限（混合整数规划）。

## 任务难度（工程实践视角）

- `robust_mvo_rebalance`：中等
- `cvar_stress_control`：中等偏高
- `discrete_rebalance_mip`：高（整数优化，耗时波动更大）

## 子任务索引与 Unified 运行命令

- `robust_mvo_rebalance/`
  - unified 运行：
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/robust_mvo_rebalance \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - `iterations=0` 典型耗时：约 8-15 秒。

- `cvar_stress_control/`
  - unified 运行：
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/cvar_stress_control \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - `iterations=0` 典型耗时：约 9-18 秒。

- `discrete_rebalance_mip/`
  - unified 运行：
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/discrete_rebalance_mip \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - `iterations=0` 典型耗时：约 8-20 秒；在较慢 CPU 上可能更久。

当 `algorithm.iterations > 0` 时，总耗时会随迭代轮数近似线性增长。
