# LightweightBroadbandAbsorber

<<<<<<< Updated upstream
A benchmark for optimizing lightweight broadband CNTs@Nd-BaM/PE microwave absorber design across 8.2–18 GHz.

## Overview

Based on the CNTs@Nd₀.₁₅-BaM/PE composite system (Wang et al., *Materials* 2024), this task optimizes a single-layer absorber for broadband performance with a lightweight focus:
- **4 material components**: PE matrix, Nd-BaM ferrite, CNTs, hollow Nd-BaM microspheres
- **8.2–18 GHz** frequency range (standard waveguide VNA band)
- **Hard constraint**: EAB ≥ 4.0 GHz (infeasible otherwise)
- **Dominant density penalty** (w=0.5) for lightweight solutions
=======
A benchmark for optimizing lightweight broadband microwave absorber design across 2–18 GHz.

## Overview

This task extends MicrowaveAbsorberDesign to a broadband scenario with a **lightweight focus**:
- **Frequency range**: 2–18 GHz (vs 8–12 GHz)
- **4 material components** including a lightweight magnetic filler option
- **Minimum bandwidth constraint**: EAB ≥ 4.0 GHz (infeasible otherwise)
- **Dominant density penalty** (w=0.5) to incentivize lightweight solutions

## File Structure
```
LightweightBroadbandAbsorber/
├── README.md
├── Task.md
├── references/
│   ├── material_db.json
│   └── problem_config.json
├── verification/
│   ├── evaluator.py
│   └── requirements.txt
├── scripts/
│   └── init.py
└── baseline/
    ├── solution.py
    └── result_log.txt
```
>>>>>>> Stashed changes

## Quick Start
```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```
