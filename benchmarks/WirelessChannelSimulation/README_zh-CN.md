# 无线信道仿真（WirelessChannelSimulation）

本领域聚焦噪声信道下数字通信系统的可靠性评估。
当前任务关注稀有错误场景下的 BER 估计。

## 子任务

- `HighReliableSimulation/`：实现自定义 `MySampler`，在固定评测配置下对 AWGN 信道中的 Hamming(127,120) 进行方差受控 BER 估计。
