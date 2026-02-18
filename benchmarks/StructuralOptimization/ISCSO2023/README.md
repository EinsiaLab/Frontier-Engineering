# ISCSO 2023 — 284-Member 3D Truss Sizing Optimization

This folder contains the task specification, a Python-based 3D FEM evaluator, and a baseline optimization solution for the ISCSO 2023 structural optimization benchmark.

## Key Files and Roles

- `Task.md` / `Task_zh-CN.md`
  - Full problem specification: background, mathematical formulation, physical model, constraints, and I/O format.

- `references/problem_data.json`
  - Problem data: tower geometry parameters, material properties, load cases, support conditions, and constraint limits. The tower topology (92 nodes, 284 members) is generated parametrically from these parameters.

- `verification/evaluator.py`
  - **[Core]** Evaluation script entry point. Runs a candidate program, reads `submission.json`, generates topology, performs 3D FEM analysis, checks constraints, and returns a score.

- `verification/fem_truss3d.py`
  - Pure Python 3D truss FEM solver using the Direct Stiffness Method with sparse matrices. Also includes the parametric tower topology generator. Dependencies: `numpy`, `scipy`.

- `verification/requirements.txt`
  - Python dependencies for the evaluation environment.

- `verification/docker/Dockerfile`
  - Containerized evaluation environment for reproducibility.

- `baseline/differential_evolution.py`
  - Reference optimization script using `scipy.optimize.differential_evolution`. Produces `submission.json`.

## Baseline Performance

The baseline solution using `scipy.optimize.differential_evolution` achieves:
- **Weight**: 7234.56 kg
- **Feasible**: Yes (all constraints satisfied)
- **Algorithm**: Differential Evolution (maxiter=100, popsize=15, seed=42)

This provides a reference point for comparison. Better results can be achieved with more iterations, larger population sizes, problem-specific algorithms (e.g., optimality criteria method), gradient-based methods with adjoint sensitivity analysis, or hybrid algorithms.

## Quick Start

### 1. Run the Baseline Solution

```bash
cd benchmarks/StructuralOptimization/ISCSO2023
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
docker build -t iscso2023-eval .
docker run -v $(pwd)/../../:/workspace iscso2023-eval python /workspace/baseline/differential_evolution.py
```

## Typical Workflow

1. Write or modify an optimization script that outputs `submission.json` containing 284 cross-sectional areas.
2. Run the evaluator to check feasibility and score.
3. Iterate to improve the objective (minimize weight) while maintaining feasibility under all 3 load cases.

## Scoring

- **Feasible solutions**: Score = structural weight (kg). Lower is better.
- **Infeasible solutions**: Score = +Infinity.
- All stress (248.2 MPa) and displacement (10.0 mm) constraints must be satisfied across all 3 load cases.

