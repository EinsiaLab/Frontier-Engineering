# Particle Physics: PET Scanner Geometry and Cost Pareto Optimization

English | [简体中文](./README_zh-CN.md)

## 1. Task Overview

This task (PET Scanner Geometry Optimization) is a premier optimization problem in the **Particle Physics and Medical Engineering** domain within the `Frontier-Eng` benchmark.

Positron Emission Tomography (PET) utilizes the physical phenomenon of antimatter annihilation—when a positron emitted by a radioactive tracer collides with an electron, it converts entirely into energy, releasing two 511 keV gamma rays at exactly 180 degrees. This task requires the AI Agent to optimize the geometric dimensions of the scanner's detector ring under extremely strict financial and physical constraints.



> **Core Challenge**: High-energy 511 keV gamma rays are difficult to stop, necessitating thick and expensive LYSO scintillation crystals. The Agent must use 3D geometric and exponential attenuation calculations to find a highly challenging Pareto optimal solution among "maximizing system sensitivity," "minimizing spatial Parallax Error (Depth of Interaction effect)," and "ensuring the total crystal volume does not exceed the multi-million-dollar budget."

For detailed physical and mathematical models, objective functions, and I/O formats designed for the Agent, please refer to the core task document: [Task.md](./Task.md).
For academic references regarding the Reality Gap, see: [reference/references.txt](./reference/references.txt).

## 2. Local Run

After preparing the `frontier-eval-2` environment, you can run the benchmark directly from the task directory:

```bash
conda activate frontier-eval-2
cd benchmarks/ParticlePhysics/PETScannerOptimization
python baseline/solution.py
python verification/evaluator.py solution.json
```

`verification/requirements.txt` currently only requires `numpy>=1.24.0`.

The baseline above has been verified in this repository with the following result:

```json
{"status": "success", "score": 73.80681701043795, "metrics": {"volume_mm3": 5089380.098815464, "sensitivity_factor": 0.1637684467863431, "resolution_gamma": 6.4031242374328485, "cost_penalty": 0.0}}
```

## 3. Run with `frontier_eval`

This task is registered in `frontier_eval` as `pet_scanner_optimization`.

From the repository root, the standard compatibility check is:

```bash
conda activate frontier-eval-2
python -m frontier_eval task=pet_scanner_optimization algorithm=openevolve algorithm.iterations=0
```

After completing the framework-level `.env` or model configuration described in [frontier_eval/README.md](../../../frontier_eval/README.md), you can start a real search by increasing `algorithm.iterations`, for example:

```bash
conda activate frontier-eval-2
python -m frontier_eval task=pet_scanner_optimization algorithm=openevolve algorithm.iterations=10
```

## 4. Evaluation Metrics

`evaluator.py` outputs the results in a standard JSON format:
* `score`: The final comprehensive score (higher is better).
* `metrics`: Contains internal details, such as `volume_mm3` (total crystal volume consumed), `sensitivity_factor` (efficiency of photon capture, higher is better), `resolution_gamma` (spatial resolution degradation, lower is better), and `cost_penalty` (penalty for exceeding the volume budget).