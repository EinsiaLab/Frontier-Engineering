# HighReliableSimulation

Navigation document for this task.

## Goal

Implement `MySampler` (inherits `SamplerBase`) and provide `simulate_variance_controlled(...)` to estimate BER for Hamming(127,120) over AWGN under fixed evaluator settings.

## Files

- `Task.md`: task contract and scoring rules (English).
- `Task_zh-CN.md`: Chinese version of task contract.
- `scripts/init.py`: minimal runnable starter.
- `baseline/solution.py`: baseline implementation.
- `runtime/`: task runtime components.
- `eval/evaluator.py`: evaluator entry.

## Quick Run

Run from the task directory:

```bash
cd benchmarks/WirelessChannelSimulation/HighReliableSimulation && python eval/evaluator.py scripts/init.py
```
