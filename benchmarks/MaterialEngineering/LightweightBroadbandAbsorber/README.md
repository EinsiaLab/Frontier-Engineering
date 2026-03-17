# LightweightBroadbandAbsorber

A benchmark for optimizing lightweight broadband CNTs@Nd-BaM/PE microwave absorber design across 8.2–18 GHz.

## Overview

Based on the CNTs@Nd₀.₁₅-BaM/PE composite system (Wang et al., *Materials* 2024), this task optimizes a single-layer absorber for broadband performance with a lightweight focus:
- **4 material components**: PE matrix, Nd-BaM ferrite, CNTs, hollow Nd-BaM microspheres
- **8.2–18 GHz** frequency range (standard waveguide VNA band)
- **Hard constraint**: EAB ≥ 4.0 GHz (infeasible otherwise)
- **Dominant density penalty** (w=0.5) for lightweight solutions

## Quick Start
```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```
