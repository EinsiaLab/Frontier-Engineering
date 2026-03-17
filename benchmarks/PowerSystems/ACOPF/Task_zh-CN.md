# Task: 交流最优潮流（5-bus DC-OPF）

## 任务名

`acopf`

## 说明

实现 `solve(instance)`，在给定 5-bus DC-OPF 算例下返回可行发电方案并使总发电成本最小。输出为 `total_cost` 或可推算总成本的字典。

## 评分

`combined_score = min(1.0, HUMAN_BEST_COST / total_cost)`，HUMAN_BEST_COST = 26.0。无效解得 0 分。
