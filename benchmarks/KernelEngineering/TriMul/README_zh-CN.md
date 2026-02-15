# 三角形乘法

此任务源自 https://www.gpumode.com/leaderboard/496?tab=rankings

TriMul 参考实现位于 baseline/reference.py 这里是基础实现，同时也是数值正确性的标准
baseline/solution.py 是 test-time-training 所提供的实现
agent 可以基于 baseline/submission.py 进行修改，这是待优化的模板版本
baseline/util.py 提供公共工具
评测入口位于 verification/eval.py
verification/eval-profile.py 是带细粒度计时诊断版，用于定位时间花在哪
verification/requirements-gpumode.txt 提供所需依赖