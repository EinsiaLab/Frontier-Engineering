# NanoCarbonAbsorberOptimization

A **mixed-variable** benchmark for optimizing nano-carbon type and content in Nd-BaM composites for broadband microwave absorption (2–18 GHz).

## What Makes This Task Different

Unlike the other MaterialEngineering tasks (pure continuous optimization), this task combines:
- **Discrete variable**: carbon material type (CNTs / GO / OLC)
- **Continuous variables**: carbon content (1-10%) and thickness (1.5-5 mm)

This mixed categorical+continuous optimization reflects a real engineering decision: which carbon material to use, and how much.

## Quick Start
```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

## Reference
Feng et al., *J Mater Sci: Mater Eng* 2024, 19:49.
