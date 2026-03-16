# 带容量约束的车辆路径问题 (CVRP) — Augerat A-n32-k5

## 概述

给定一个配送中心（depot）和 **31 个客户**（含坐标与需求量），为一批相同车辆（容量 = 100）
规划配送路线，使每个客户恰好被访问一次、每辆车不超载，且**总行驶距离最小**。

基准实例：**Augerat A-n32-k5**（32 节点，最少 5 辆车，EUC_2D 距离），已知最优解距离为 **784**。

## 来源

| 字段       | 详情 |
|------------|------|
| Benchmark  | CVRPLIB — Augerat et al. set A (1995) |
| 求解器库   | [PyVRP](https://github.com/PyVRP/PyVRP)（>1 000 ★）|
| 实例链接   | http://vrp.atd-lab.inf.puc-rio.br/ |
| 许可证     | 学术 / 公共域实例 |

## 输入 / 输出

**输入**（传给 `solve()` 的 `instance` 字典）：
- `coords` — 32 个 `(x, y)` 整数元组；索引 0 为配送中心。
- `demands` — 32 个整数需求量；`demands[0] = 0`（配送中心）。
- `capacity` — 车辆容量（100）。

**输出**：`list[list[int]]` — 每条路线为一个客户索引列表（1 起始），不含配送中心。

## 评分

```
score = min(1.0, 784 / total_distance)
```

- `valid = 1`：所有路线满足容量约束且每个客户恰好出现一次。
- `combined_score = score`（越大越好；最大值 = 1.0 为最优解）。

## 人类最优

| 指标       | 数值 |
|------------|------|
| 已知最优解 | 784  |
| 来源       | Augerat et al. (1995)，已被多个精确求解器验证 |

基线（最近邻贪心）约得 900–950，对应 score ≈ 0.83–0.87。

## 快速开始

```bash
cd benchmarks/Transportation/CapacitatedVehicleRouting
pip install -r verification/requirements.txt
python verification/evaluate.py
```
