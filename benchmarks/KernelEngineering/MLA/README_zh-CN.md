# MLA 解码内核

此任务源自 https://www.gpumode.com/leaderboard/463?tab=rankings

MLA 参考实现位于 baseline/reference.py 这里是基础实现，同时也是数值正确性的标准
baseline/mla_code_1/2/3.py 是 test-time-training 所提供的实现
agent 可以基于 baseline/submission.py 进行修改，这是待优化的模板版本
baseline/util.py 提供公共工具
评测入口位于 verification/eval.py
verification/requirements-gpumode.txt 提供所需依赖

## 运行方式

```
cd benchmarks/KernelEngineering/MLA/verification

# 只检验正确性
POPCORN_FD=1 python eval.py test mla_tests.txt

# 每个case计时，只做一次初始正确性检查，后续主要测试速度
POPCORN_FD=1 python eval.py benchmark mla_bench.txt

# 只运行最后一个例子，会在循环中反复recheck，更严格
POPCORN_FD=1 python eval.py leaderboard mla_bench.txt
```

上述代码会使用`submission.custom_kernel` 进行评测，您可以选择将`benchmarks/KernelEngineering/MLA/baseline/submission.py`替换为您的代码，或者将 `benchmarks/KernelEngineering/MLA/verification/eval.py` 中所有的 `from baseline.submission import custom_kernel` 替换为从您指定的代码中 import