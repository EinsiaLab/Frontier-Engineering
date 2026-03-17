# 交流最优潮流（5-bus DC 子集）

在满足 DC 潮流约束与机组/电压限值下最小化总发电成本。本 benchmark 使用内嵌 5 母线算例，无需外部 MATPOWER 数据即可运行。

## Source

| 项目 | 说明 |
|------|------|
| **基准库** | [power-grid-lib/pglib-opf](https://github.com/power-grid-lib/pglib-opf) — 388+ ★，数据 CC-BY-4.0 |
| **形式** | 本任务为 5-bus DC-OPF 子集，便于自包含评测 |

## 难度

**3 / 5** — DC-OPF 为凸（QP），需构造 B 矩阵与边界；扩展至 AC-OPF 为非凸。

## 评分

`combined_score = min(1.0, HUMAN_BEST_COST / total_generation_cost)`，HUMAN_BEST_COST = 26.0。
