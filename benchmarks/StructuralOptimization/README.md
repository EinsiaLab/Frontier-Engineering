# Structural Optimization

This domain covers structural engineering optimization problems derived from the **International Student Competition in Structural Optimization (ISCSO)**, organized by [Bright Optimizer](http://www.brightoptimizer.com/).

Structural optimization is a core discipline in civil, aerospace, and mechanical engineering, aiming to find the optimal design of load-bearing structures that minimizes material usage (weight) while satisfying safety constraints (stress and displacement limits).

## Tasks

| Task | Description | Dimension | Type |
| :--- | :--- | :---: | :--- |
| `ISCSO2015` | 45-bar 2D truss size + shape optimization | 54 | Continuous, constrained, multi-load-case |
| `ISCSO2023` | 284-member 3D truss sizing optimization | 284 | Continuous, constrained, multi-load-case |

## Why These Problems Are Suitable for Frontier-Engineering

| Feature | ISCSO 2015 | ISCSO 2023 |
| :--- | :--- | :--- |
| High-dimensional continuous variables | Medium (54-D) | High (284-D) |
| Real physical model (FEM) | Yes | Yes |
| Deterministic evaluation | Yes | Yes |
| Multi-load-case constraints | Yes (2 cases) | Yes (3 cases) |
| Non-convex feasible region | Yes | Yes |
| Industrial relevance | Yes | Yes |

These benchmarks serve as:

- Black-box constrained optimization benchmarks
- Agent + FEM simulation interaction benchmarks
- Automated algorithm design benchmarks
- LLM + numerical simulation benchmarks

