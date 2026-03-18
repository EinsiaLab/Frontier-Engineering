# PyMOTO SIMP Compliance

This benchmark targets topology optimization for a 2D cantilever-like beam with a fixed material budget.
The baseline is implemented with the official pyMOTO-style pipeline:

- density filtering
- SIMP interpolation
- finite-element stiffness assembly
- linear solve
- compliance minimization with OC or MMA

## File Structure

```text
PyMOTOSIMPCompliance/
├── README.md
├── Task.md
├── references/
│   └── problem_config.json
├── scripts/
│   └── init.py
├── verification/
│   ├── evaluator.py
│   └── requirements.txt
└── frontier_eval/
    ├── initial_program.txt
    ├── candidate_destination.txt
    ├── eval_command.txt
    ├── eval_cwd.txt
    ├── constraints.txt
    ├── agent_files.txt
    ├── copy_files.txt
    ├── readonly_files.txt
    └── artifact_files.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r verification/requirements.txt
```

### 2. Run Baseline Candidate

```bash
cd benchmarks/StructuralOptimization/PyMOTOSIMPCompliance
python scripts/init.py
# outputs temp/submission.json
```

### 3. Evaluate Candidate

```bash
cd benchmarks/StructuralOptimization/PyMOTOSIMPCompliance
python verification/evaluator.py scripts/init.py
```

## Submission Format

The candidate program must write `temp/submission.json`:

```json
{
  "benchmark_id": "pymoto_simp_compliance",
  "nelx": 120,
  "nely": 40,
  "density_vector": [0.5, 0.5, 0.5],
  "compliance": 123.45,
  "volume_fraction": 0.5,
  "feasible": true
}
```

Evaluator-required field:

- `density_vector` (flattened length `nelx * nely`)

Extra fields are accepted, but scoring uses independent evaluator computation.

## Task Summary

- **Task name (frontier_eval config key)**: `pymoto_simp_compliance`
- **Benchmark path**: `StructuralOptimization/PyMOTOSIMPCompliance`
- **Mesh**: `120 x 40` (4800 design variables)
- **Objective**: minimize compliance
- **Constraint**: `mean(density) <= volfrac` with evaluator tolerance
- **Volume fraction**: `0.5`
- **SIMP penalization**: `3.0`
- **Filter radius**: `2.0`
- **Material**: `E0=1.0`, `Emin=1e-9`, `nu=0.3`

## Scoring

The evaluator computes compliance from submitted density and applies:

- feasible: `combined_score = baseline_uniform_compliance / compliance`
- infeasible: `combined_score = 0`, `valid = 0`

Where `baseline_uniform_compliance` is the compliance of the uniform density field (`density = volfrac`).
Higher score is better.

## Run with frontier_eval

```bash
python -m frontier_eval \
  task=pymoto_simp_compliance \
  algorithm=openevolve \
  algorithm.iterations=10
```

This task is integrated through the unified task interface.

