# MLA 解码内核

此任务源自 https://www.gpumode.com/leaderboard/463?tab=rankings

MLA 参考实现位于 baseline/reference.py 这里是基础实现，同时也是数值正确性的标准
baseline/mla_code_1/2/3.py 是 test-time-training 所提供的实现
agent 可以基于 baseline/submission.py 进行修改，这是待优化的模板版本
baseline/util.py 提供公共工具
评测入口位于 verification/eval.py
verification/requirements-gpumode.txt 提供所需依赖