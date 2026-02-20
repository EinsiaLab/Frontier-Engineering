# 无线信道仿真（WirelessChannelSimulation）

本领域聚焦数字通信在随机信道下的端到端可靠性评估问题，典型目标是估计极低误码率（BER, Bit Error Rate），并在可控计算预算内保证估计结果可验证、可复现。

与常规“直接蒙特卡洛”不同，当 BER 低到 1e-6 甚至更低时，朴素采样往往需要天量样本才能观测到足够错误事件。本领域任务通常会引入重要性采样（Importance Sampling）与方差控制（Variance Control）等技术，在不改变物理模型与译码器的前提下提升估计效率。

## 子任务索引

- `HighReliableSimulation/`：高可靠通信场景下的 BER 估计。要求实现自定义采样器 `MySampler`，并通过固定评测入口在冻结参数下完成方差受控的误码率估计。
