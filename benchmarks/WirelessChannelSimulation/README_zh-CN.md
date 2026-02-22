# 无线信道仿真（WirelessChannelSimulation）

本领域聚焦数字通信在随机信道下的端到端可靠性评估问题，典型目标是估计极低误码率（BER, Bit Error Rate），并在可控计算预算内保证估计结果可验证、可复现。

与常规“直接蒙特卡洛”不同，当 BER 低到 1e-6 甚至更低时，朴素采样往往需要天量样本才能观测到足够错误事件。本领域任务通常会引入重要性采样（Importance Sampling）与方差控制（Variance Control）等技术，在不改变物理模型与译码器的前提下提升估计效率。

## 问题描述

我们考虑线性纠错码（linear error-correcting code）的译码可靠性评估。设发送码字为 $\mathbf{c}$、接收信号为
$$
\mathbf{y}=\mathbf{c}+\mathbf{n},
$$
其中噪声 $\mathbf{n}$ 为加性白高斯噪声（AWGN），各维独立同分布 $\mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$。本文档默认采用反相二进制相移键控（antipodal BPSK）调制，因此 $\mathbf{c}\in\{-1,+1\}^n$；通常将“全零信息字”对应的发送码字记为 $\mathbf{c}=[-1,-1,\ldots,-1]$。给定确定性译码器 $\hat{\mathbf{c}}=\mathrm{Dec}(\mathbf{y})$，我们关心的误差率可定义为码字错误率（也常称帧错误率，FER/BLER）：
$$
P_{\mathrm{err}}=\Pr(\hat{\mathbf{c}}\neq \mathbf{c}).
$$
由于线性码与 BPSK 的对称性，所有码字的错误率相同，因此分析时可不失一般性地固定 $\mathbf{c}=[-1,\ldots,-1]$。

进一步地，译码器是确定性的：每个噪声向量 $\mathbf{n}$ 都对应一个唯一的译码输出，从而噪声空间可被划分为不同码字的“译码区域”。记 $A_{\mathbf{c}'}$ 为会被译码为 $\mathbf{c}'$ 的噪声集合，则错误区域为 $A_{\mathrm{err}}=\bigcup_{\mathbf{c}'\neq \mathbf{c}}A_{\mathbf{c}'}$。令 $f(\mathbf{n})$ 为高斯噪声的概率密度函数，则
$$
P_{\mathrm{err}}=\int_{A_{\mathrm{err}}} f(\mathbf{n})\,\mathrm{d}\mathbf{n}.
$$

## 重要性采样

在极低误差率场景（例如 $10^{-6}$ 甚至更低）下，直接蒙特卡洛（Monte Carlo）方法用 $\mathbf{n}_i\sim f$ 的独立样本估计
$$
\hat P_{\mathrm{err}}=\frac{1}{N}\sum_{i=1}^N \mathbf{1}_{\{\mathbf{n}_i\in A_{\mathrm{err}}\}},
$$
其方差为
$$
\mathrm{Var}(\hat P_{\mathrm{err}})=\frac{1}{N}P_{\mathrm{err}}(1-P_{\mathrm{err}})\approx \frac{1}{N}P_{\mathrm{err}},
$$
导致为了“看到足够多的错误事件”通常需要近似 $O(1/P_{\mathrm{err}})$ 的样本量，计算开销难以承受。

重要性采样（Importance Sampling）是一类经典的方差缩减（variance reduction）技术，其核心是将噪声的采样分布从真实分布 $f(\mathbf{n})$ 改为更“偏向错误区域”的提议分布 $g(\mathbf{n})$，再用似然比做无偏校正。对任意 $g(\mathbf{n})$（其支持集需覆盖 $A_{\mathrm{err}}$）有恒等式
$$
P_{\mathrm{err}}=\int_{A_{\mathrm{err}}} f(\mathbf{n})\,\mathrm{d}\mathbf{n}
=\int_{A_{\mathrm{err}}}\frac{f(\mathbf{n})}{g(\mathbf{n})}g(\mathbf{n})\,\mathrm{d}\mathbf{n}.
$$
因此可采样 $\mathbf{n}_i\sim g$ 并用权重 $w_i=\frac{f(\mathbf{n}_i)}{g(\mathbf{n}_i)}$ 做加权平均，得到无偏估计
$$
\hat P_{\mathrm{err}}^{\mathrm{IS}}=\frac{1}{N}\sum_{i=1}^N w_i\,\mathbf{1}_{\{\mathbf{n}_i\in A_{\mathrm{err}}\}}.
$$
理论上使方差最小的最优提议分布满足
$$
g^*(\mathbf{n})\propto f(\mathbf{n})\mathbf{1}_{\{\mathbf{n}\in A_{\mathrm{err}}\}},
$$
但由于错误区域 $A_{\mathrm{err}}$ 难以解析刻画且 $P_{\mathrm{err}}$ 未知，该最优分布不可直接实现。本领域的基准任务因此固定物理信道模型与译码器，仅允许参赛者设计可实现的采样器/提议分布（并在高可靠任务中进一步施加方差控制约束），以在可控计算预算内稳定、可复现地估计极低误差率。

## 现有研究方向

针对“用重要性采样估计译码器错误率”的问题，Bucklew 等人 \cite{bucklew2003monte} 提出了一种具有解析特征的构造思路，在高维情形下尤其具有启发性。为便于讨论，先给出如下定义。

**定义（最近错误距离、最近错误位置与最近邻码字）**  
对某线性码的确定性译码器，定义其**最近错误距离（nearest error distance）**为导致译码错误的最小噪声范数：
$$
d=\inf_{\mathbf{n}\in A_{\mathrm{err}}}\|\mathbf{n}\|.
$$
定义其（对全零码字）**最近错误位置集合（nearest error positions）**为：
$$
\mathcal{M}=\{\mathbf{n}\in \partial A_{\mathrm{err}}:\|\mathbf{n}\|=d\},
$$
其中 $\partial A_{\mathrm{err}}$ 是错误区域 $A_{\mathrm{err}}$ 的边界。这里仅需考虑边界点：对任意错误区域内部点，其任意邻域都包含更靠近原点（范数更小）的错误点。

另一方面，记 $\mathcal{C}$ 为所有合法码字的集合，则全零码字 $\mathbf{c}$ 的**最近邻码字集合（nearest neighbors）**为与其欧氏距离最小的合法码字：
$$
\mathcal{N}=\left\{\mathbf{c}':\|\mathbf{c}'-\mathbf{c}\|=\inf_{\mathbf{c}''\in \mathcal{C}\backslash\{\mathbf{c}\}}\|\mathbf{c}''-\mathbf{c}\|\right\}.
$$

Bucklew 方法的核心观察是：在高维高斯噪声下，错误区域 $A_{\mathrm{err}}$ 中的概率质量主要集中在“离原点最近”的那一部分附近。设空间维度为 $n$，根据大偏差（large deviation）理论可得到上界形式
$$
\limsup_{n\to\infty}\frac{1}{n}\log P_{\mathrm{err}}\le -\frac{d^2}{2\sigma^2}.
$$
据此，该方法倾向于只在这些最近错误位置附近进行采样，从而显著提高稀有错误事件的采样效率。

该思路可概括为以下三种典型情形：

1. 若只有一个最近错误位置 $\boldsymbol{\mu}$，则用以 $\boldsymbol{\mu}$ 为中心、方差为 $\sigma^2$ 的高斯分布作为提议分布：
   $$
   g(\mathbf{n})=\frac{1}{(2\pi\sigma^2)^{n/2}}\exp\left(-\frac{\|\mathbf{n}-\boldsymbol{\mu}\|^2}{2\sigma^2}\right).
   $$
2. 若最近错误位置有限多个 $\mathcal{M}=\{\boldsymbol{\mu}_1,\boldsymbol{\mu}_2,\ldots,\boldsymbol{\mu}_M\}$，则用以这些位置为中心的高斯混合分布：
   $$
   g(\mathbf{n})=\frac{1}{M}\sum_{j=1}^{M}\frac{1}{(2\pi\sigma^2)^{n/2}}\exp\left(-\frac{\|\mathbf{n}-\boldsymbol{\mu}_j\|^2}{2\sigma^2}\right).
   $$
3. 若最近错误位置无穷多或其分布复杂、难以枚举，则可用“球面均匀分布噪声 + 高斯扰动”的近似，其概率密度函数可由贝塞尔函数（Bessel functions）刻画。

在高维编码系统中，由于最近错误位置数量常呈组合爆炸，上述第 3 种情形往往更符合实际。但测度集中（concentration of measure）现象会带来根本性限制：当 $n\to\infty$ 时，超球体“薄壳”的体积占比趋于 1（$V_{\text{shell}}/V_{\text{ball}}\to 1$），使得球面上的最近错误位置相对稀疏、导致“在球面附近采样”的效率随维度升高而下降。
该局限性会在本领域的具体子任务与实验对比中体现。本项目提供的初始代码即是用这种思想实现的一个基于贝塞尔函数的采样器（见 `runtime/sampler.py` 中的 `BesselSampler` 类）。

第二种思路是目前的研究重点，因为它可以极大的提高采样效率。然而，这依赖于对于码和译码器的具体结构的深入理解，才能找到那些最近错误位置并构造相应的提议分布。
对于一般的线性码，最近错误位置与最近邻码字之间存在密切联系：如果是极大似然/最近邻译码，每个最近邻码字 $\mathbf{c}'$ 都对应一个最近错误位置 $\boldsymbol{\mu}=\frac{\mathbf{c}'+\mathbf{c}}{2}$（即连接全零码字与该最近邻码字的线段中点）；考虑到优秀的译码器都是尽可能类似于极大似然译码的，大部分译码器的最近错误位置也会集中在这些最近邻码字的附近。
因此，寻找最近邻码字是构造高效重要性采样分布的关键步骤之一。
然而，对于一般的线性码，最近邻码字的数量可能非常大（例如 Hamming(127,120) 码的最近邻码字数量为 127），且没有已知的解析表达式或高效算法来枚举这些最近邻码字，这使得基于最近邻码字构造提议分布的方法具有挑战性。
需要对码的代数或几何结构有非常深刻的认识才能高效地找到这些最近邻码字，并基于它们构造高效的提议分布。
因此，虽然基于最近邻码字的采样方法在理论上非常有吸引力，但在实际应用中可能面临较大的挑战，尤其是对于高维线性码而言。

## 子任务索引

- `HighReliableSimulation/`：高可靠通信场景下的 BER 估计。要求实现自定义采样器 `MySampler`，并通过固定评测入口在冻结参数下完成方差受控的误码率估计。
