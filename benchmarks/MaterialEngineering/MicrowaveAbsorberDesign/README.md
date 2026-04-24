# MicrowaveAbsorberDesign

A benchmark for optimizing a single-layer microwave absorber in the X-band (8-12 GHz).

## Overview

The task requires designing a single-layer absorber backed by a perfect electrical conductor. The optimizer must choose absorber thickness and the volume fractions of a matrix, a dielectric filler, and a magnetic filler to maximize absorption performance while limiting thickness, density, and cost.

## Quick Start

```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

The official score is `combined_score`, computed by the evaluator from the reflection-loss curve and engineering proxy terms. See [Task.md](./Task.md) for details.
