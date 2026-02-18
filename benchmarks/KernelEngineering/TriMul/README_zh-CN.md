# 三角形乘法

此任务源自 https://www.gpumode.com/leaderboard/496?tab=rankings

TriMul 参考实现位于 `baseline/reference.py` 这里是基础实现，同时也是数值正确性的标准
`baseline/solution.py` 是 `test-time-training` 所提供的实现
agent 可以基于 `baseline/submission.py` 进行修改，这是待优化的模板版本
`baseline/util.py` 提供公共工具
评测入口位于 `verification/eval.py`
`verification/eval-profile.py` 是带细粒度计时诊断版，用于定位时间花在哪
`verification/requirements-gpumode.txt` 提供所需依赖

## 运行方式

```
cd benchmarks/KernelEngineering/TriMul/verification

# 只检验正确性
exec 3>tri_test.log POPCORN_FD=3 python eval.py test tri_test.txt

# 每个case计时，只做一次初始正确性检查，后续主要测试速度
exec 3>tri_bench.log POPCORN_FD=3 python eval.py benchmark tri_bench.txt

# 每个case计时，会在循环中反复换 seed 并重新做正确性校验，更严格
exec 3>tri_leaderboard.log POPCORN_FD=3 python eval.py benchmark tri_bench.txt
```

上述代码会使用`submission.custom_kernel` 进行评测，您可以选择将`benchmarks/KernelEngineering/TriMul/baseline/submission.py`替换为您的代码，或者将 `benchmarks/KernelEngineering/TriMul/verification/eval.py` 中所有的 `from baseline.submission import custom_kernel` 替换为从您指定的代码中 import