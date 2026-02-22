# Wireless Channel Simulation (WirelessChannelSimulation)

This domain studies end-to-end reliability evaluation for digital communication systems under stochastic channels. A typical goal is to estimate extremely low BER (Bit Error Rate) with reproducible and verifiable results under limited compute budgets.

Unlike direct Monte Carlo, when BER is around 1e-6 or lower, naive sampling usually needs too many samples to observe enough error events. Tasks in this domain therefore use importance sampling and variance-control techniques to improve estimation efficiency without changing the physical channel model or decoder.

## Problem Setup

We consider decoding reliability for linear error-correcting codes. Let transmitted codeword be $\mathbf{c}$ and received signal be
$$
\mathbf{y}=\mathbf{c}+\mathbf{n},
$$
where noise $\mathbf{n}$ is AWGN with i.i.d. components $\mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$. We assume antipodal BPSK, so $\mathbf{c}\in\{-1,+1\}^n$. Usually, the all-zero information word maps to $\mathbf{c}=[-1,-1,\ldots,-1]$. For deterministic decoder $\hat{\mathbf{c}}=\mathrm{Dec}(\mathbf{y})$, the error rate is
$$
P_{\mathrm{err}}=\Pr(\hat{\mathbf{c}}\neq \mathbf{c}).
$$
By linear-code symmetry with BPSK, all codewords have equal error probability, so fixing $\mathbf{c}=[-1,\ldots,-1]$ is without loss of generality.

Because decoding is deterministic, each noise vector maps to a unique decoded codeword. Noise space is partitioned into decoding regions. Let $A_{\mathbf{c}'}$ be the region decoded as $\mathbf{c}'$, then the error region is
$$
A_{\mathrm{err}}=\bigcup_{\mathbf{c}'\neq \mathbf{c}}A_{\mathbf{c}'}.
$$
With Gaussian pdf $f(\mathbf{n})$,
$$
P_{\mathrm{err}}=\int_{A_{\mathrm{err}}} f(\mathbf{n})\,\mathrm{d}\mathbf{n}.
$$

## Importance Sampling

At very low BER (e.g. $10^{-6}$ or lower), direct Monte Carlo with $\mathbf{n}_i\sim f$ uses
$$
\hat P_{\mathrm{err}}=\frac{1}{N}\sum_{i=1}^N \mathbf{1}_{\{\mathbf{n}_i\in A_{\mathrm{err}}\}},
$$
with variance
$$
\mathrm{Var}(\hat P_{\mathrm{err}})=\frac{1}{N}P_{\mathrm{err}}(1-P_{\mathrm{err}})\approx \frac{1}{N}P_{\mathrm{err}}.
$$
So roughly $O(1/P_{\mathrm{err}})$ samples are needed, which is often too expensive.

Importance sampling replaces true noise distribution $f(\mathbf{n})$ with proposal $g(\mathbf{n})$ that biases toward error regions, then corrects by likelihood ratio. For any $g(\mathbf{n})$ whose support covers $A_{\mathrm{err}}$,
$$
P_{\mathrm{err}}=\int_{A_{\mathrm{err}}} f(\mathbf{n})\,\mathrm{d}\mathbf{n}
=\int_{A_{\mathrm{err}}}\frac{f(\mathbf{n})}{g(\mathbf{n})}g(\mathbf{n})\,\mathrm{d}\mathbf{n}.
$$
Thus with $\mathbf{n}_i\sim g$ and weights $w_i=f(\mathbf{n}_i)/g(\mathbf{n}_i)$,
$$
\hat P_{\mathrm{err}}^{\mathrm{IS}}=\frac{1}{N}\sum_{i=1}^N w_i\,\mathbf{1}_{\{\mathbf{n}_i\in A_{\mathrm{err}}\}}
$$
is unbiased.

The minimum-variance ideal proposal is
$$
g^*(\mathbf{n})\propto f(\mathbf{n})\mathbf{1}_{\{\mathbf{n}\in A_{\mathrm{err}}\}},
$$
but it is not directly usable because $A_{\mathrm{err}}$ is hard to characterize and $P_{\mathrm{err}}$ is unknown.

## Existing Research Direction

For decoder error-rate estimation with importance sampling, Bucklew et al. \cite{bucklew2003monte} provide an influential framework, especially in high dimensions.

Define nearest error distance:
$$
d=\inf_{\mathbf{n}\in A_{\mathrm{err}}}\|\mathbf{n}\|.
$$
Define nearest error positions:
$$
\mathcal{M}=\{\mathbf{n}\in \partial A_{\mathrm{err}}:\|\mathbf{n}\|=d\}.
$$
Define nearest-neighbor codeword set for transmitted codeword $\mathbf{c}$:
$$
\mathcal{N}=\left\{\mathbf{c}':\|\mathbf{c}'-\mathbf{c}\|=\inf_{\mathbf{c}''\in \mathcal{C}\backslash\{\mathbf{c}\}}\|\mathbf{c}''-\mathbf{c}\|\right\}.
$$

Their key observation is that in high-dimensional Gaussian noise, most error-region probability mass is concentrated near the closest part to the origin. A large-deviation upper bound is
$$
\limsup_{n\to\infty}\frac{1}{n}\log P_{\mathrm{err}}\le -\frac{d^2}{2\sigma^2}.
$$

Three typical proposal constructions are:

1. Single nearest error position: a Gaussian centered at $\boldsymbol{\mu}$.
2. Finite nearest set: a Gaussian mixture centered at $\{\boldsymbol{\mu}_1,\ldots,\boldsymbol{\mu}_M\}$.
3. Infinite/complex nearest set: sphere-uniform perturbation plus Gaussian noise, often involving Bessel-function forms.

In high-dimensional coding systems, case 3 is often practical due to combinatorial growth of nearest error positions. However, concentration of measure also limits it: as $n\to\infty$, shell volume dominates ball volume, making nearest positions sparse on the sphere and reducing sampling efficiency.

The provided starter implementation follows this direction using a Bessel-based sampler (`runtime/sampler.py`, class `BesselSampler`).

A major active direction is nearest-neighbor-based proposals, since they can greatly improve sampling efficiency. But this depends on deep structural understanding of both code and decoder to identify nearest error positions. For many linear codes, nearest-neighbor counts can be large (e.g., 127 for Hamming(127,120)), and efficient exact enumeration is hard. That makes practical construction of high-quality proposals challenging in high dimensions.

## Subtask Index

- `HighReliableSimulation/`: BER estimation under high-reliability settings. Requires implementing custom `MySampler` and running variance-controlled evaluation under frozen settings.
