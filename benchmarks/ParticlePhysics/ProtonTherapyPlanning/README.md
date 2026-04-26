# Particle Physics: IMPT Dose Weight Optimization

English | [简体中文](./README_zh-CN.md)

## 1. Task Overview

This task (Proton Therapy Planning Optimization) is a premier optimization problem in the **Particle Physics and Medical Engineering** domain within the `Frontier-Eng` benchmark.

Proton therapy utilizes the unique "Bragg Peak" physical characteristic of high-energy proton beams—releasing very little energy initially and instantaneously releasing the vast majority at a specific depth—to achieve targeted "detonations" on tumors. This task requires the AI Agent to optimize the 3D spatial stopping points and irradiation weights of proton pencil beams under extremely strict medical safety constraints.

> **Core Challenge**: The Clinical Target Volume (CTV, tumor) is often located immediately adjacent to extremely sensitive Organs at Risk (OAR, e.g., the brainstem). The Agent must use precise 3D dose kernel superposition calculations to find a highly challenging Pareto optimal solution between "maximizing tumor prescription dose coverage" and "ensuring the brainstem dose remains within safe limits."

For detailed physical and mathematical models, objective functions, and I/O formats designed for the Agent, please refer to the core task document: [Task.md](./Task.md).

## 2. Local Run

For the current v2 task set, this task uses `.venvs/frontier-v2-extra` for direct local execution:

```bash
cd benchmarks/ParticlePhysics/ProtonTherapyPlanning
../../../.venvs/frontier-v2-extra/bin/python baseline/solution.py
../../../.venvs/frontier-v2-extra/bin/python verification/evaluator.py plan.json
```

`verification/requirements.txt` currently only requires `numpy>=1.24.0`.

The baseline above has been verified in this repository with the following result:

```json
{"score": -2685.8873258471367, "status": "success", "metrics": {"ctv_mse": 2779.3623258471366, "oar_overdose_penalty": 0.0, "total_weight": 130.5}}
```

## 3. Run with `frontier_eval`

This task is now integrated through benchmark-local `task=unified` metadata on the mainline v2 workflow.

From the repository root, the standard compatibility check is:

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/ProtonTherapyPlanning \
  algorithm=openevolve \
  algorithm.iterations=0
```

If you want to run the equivalent explicit `frontier_eval` command:

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=ParticlePhysics/ProtonTherapyPlanning \
  algorithm=openevolve \
  algorithm.iterations=0
```

## 4. Evaluation Metrics

`evaluator.py` outputs the results in a standard JSON format:
* `score`: The final comprehensive score (higher is better).
* `metrics`: Contains internal details, such as `ctv_mse` (Mean Squared Error of tumor dose, lower is better), `oar_overdose_penalty` (penalty for OAR overdose), and `total_weight` (total beam current consumed).
