# MicrowaveAbsorberDesign

A benchmark for optimizing single-layer microwave absorber design in the X-band (8–12 GHz).

## Overview

The task requires designing a single-layer microwave absorbing material backed by a perfect electrical conductor (PEC). The optimizer must choose the absorber thickness and volume fractions of three material components (matrix, dielectric filler, magnetic filler) to maximize absorption performance while minimizing thickness, weight, and cost.

## File Structure

```
MicrowaveAbsorberDesign/
├── README.md                          # This file (navigation)
├── Task.md                            # Detailed task definition
├── references/
│   ├── material_db.json               # Predefined material property database
│   └── problem_config.json            # Benchmark configuration and scoring weights
├── verification/
│   ├── evaluator.py                   # Official evaluator (ground truth)
│   └── requirements.txt               # Python dependencies
├── scripts/
│   └── init.py                        # Minimal valid initialization (agent evolution target)
└── baseline/
    ├── solution.py                    # Random-search baseline
    └── result_log.txt                 # Baseline execution log
```

## Quick Start

```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

## Evaluation

The evaluator uses **transmission line theory** to compute reflection loss (RL) over the X-band frequency grid, then derives a **normalized** combined score balancing electromagnetic performance against physical/economic constraints. See `Task.md` for full details.

**Scoring**: `combined_score` (higher is better). All metrics are min-max normalized to [0, 1] before weighting.
