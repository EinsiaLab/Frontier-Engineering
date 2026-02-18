# ISCSO 2015 — 45-Bar 2D Truss Size + Shape Optimization

This folder contains the task specification, a Python-based FEM evaluator, and a baseline optimization solution for the ISCSO 2015 structural optimization benchmark.

## Key Files and Roles

- `Task.md` / `Task_zh-CN.md`
  - Full problem specification: background, mathematical formulation, physical model, constraints, and I/O format.

- `references/problem_data.json`
  - Complete problem data: node coordinates, bar connectivity, material properties, load cases, support conditions, and constraint limits.

- `verification/evaluator.py`
  - **[Core]** Evaluation script entry point. Runs a candidate program, reads `submission.json`, performs FEM analysis, checks constraints, and returns a score.

- `verification/fem_truss2d.py`
  - Pure Python 2D truss FEM solver using the Direct Stiffness Method. Dependencies: `numpy` only.

- `verification/requirements.txt`
  - Python dependencies for the evaluation environment.

- `verification/docker/Dockerfile`
  - Containerized evaluation environment for reproducibility.

- `baseline/differential_evolution.py`
  - Reference optimization script using `scipy.optimize.differential_evolution`. Produces `submission.json`.

## Quick Start

### 1. Run the Baseline Solution

```bash
cd benchmarks/StructuralOptimization/ISCSO2015
python baseline/differential_evolution.py
```

This produces `submission.json` in the current directory.

### 2. Evaluate a Submission

```bash
python verification/evaluator.py baseline/differential_evolution.py
```

Or evaluate a pre-existing `submission.json` directly:

```bash
python verification/evaluator.py --test submission.json
```

### 3. Run with Docker

```bash
cd verification/docker
docker build -t iscso2015-eval .
docker run -v $(pwd)/../../:/workspace iscso2015-eval python /workspace/baseline/differential_evolution.py
```

## Typical Workflow

1. Write or modify an optimization script that outputs `submission.json`.
2. Run the evaluator to check feasibility and score.
3. Iterate to improve the objective (minimize weight) while maintaining feasibility.

## Scoring

- **Feasible solutions**: Score = structural weight (kg). Lower is better.
- **Infeasible solutions**: Score = +Infinity.
- All stress and displacement constraints must be satisfied across all load cases.

