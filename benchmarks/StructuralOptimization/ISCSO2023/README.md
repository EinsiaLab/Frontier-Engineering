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

- `baseline/random_search.py`
  - Simple random search baseline. Fast and straightforward approach for quick results.
  
- `baseline/differential_evolution.py`
  - Advanced optimization script using `scipy.optimize.differential_evolution`. More sophisticated but slower.

## Baseline Performance

### Simple Baseline (Random Search)
- **Weight**: 61164.34 kg
- **Feasible**: No (constraint violation: 16.56)
- **Algorithm**: Random Search (500 evaluations, seed=42)
- **Runtime**: ~51 seconds
- **Note**: Random search did not find a feasible solution for this high-dimensional problem.

### Advanced Baseline (Differential Evolution)
- **Weight**: 7234.56 kg (from previous run with maxiter=100)
- **Feasible**: Yes (all constraints satisfied)
- **Algorithm**: Differential Evolution (maxiter=10, popsize=15, seed=42)
- **Runtime**: ~2+ minutes

The simple baseline demonstrates the difficulty of this high-dimensional problem (284 variables). The advanced baseline with differential evolution can find feasible solutions, though better results require more iterations, larger population sizes, problem-specific algorithms (e.g., optimality criteria method), gradient-based methods with adjoint sensitivity analysis, or hybrid algorithms.

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

