# LightweightBroadbandAbsorber

Lightweight broadband CNTs@Nd-BaM/PE microwave absorber optimization (8.2–18 GHz).

## Key Features
- 4 material components with competing weight/performance trade-offs
- Minimum EAB hard constraint (>= 4.0 GHz)
- Density penalty is the dominant penalty term (weight 0.5)

## Quick Start
```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

## Unified Run

```bash
bash scripts/run_v2_unified.sh MaterialEngineering/LightweightBroadbandAbsorber \
  algorithm=openevolve \
  algorithm.iterations=0
```

## Reference
Wang et al., *Materials* 2024, 17, 3433.
