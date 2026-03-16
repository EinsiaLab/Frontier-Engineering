# Particle Physics: IMPT Dose Weight Optimization

English | [简体中文](./README_zh-CN.md)

## 1. Task Overview

This task (Proton Therapy Planning Optimization) is a premier optimization problem in the **Particle Physics and Medical Engineering** domain within the `Frontier-Eng` benchmark.

Proton therapy utilizes the unique "Bragg Peak" physical characteristic of high-energy proton beams—releasing very little energy initially and instantaneously releasing the vast majority at a specific depth—to achieve targeted "detonations" on tumors. This task requires the AI Agent to optimize the 3D spatial stopping points and irradiation weights of proton pencil beams under extremely strict medical safety constraints.

> **Core Challenge**: The Clinical Target Volume (CTV, tumor) is often located immediately adjacent to extremely sensitive Organs at Risk (OAR, e.g., the brainstem). The Agent must use precise 3D dose kernel superposition calculations to find a highly challenging Pareto optimal solution between "maximizing tumor prescription dose coverage" and "ensuring the brainstem dose remains within safe limits."

For detailed physical and mathematical models, objective functions, and I/O formats designed for the Agent, please refer to the core task document: [Task.md](./Task.md).

## 2. File Structure

```text
ProtonTherapyPlanning/
├── README.md                        # This navigation document
├── README_zh-CN.md                  # Navigation document (Chinese version)
├── Task.md                          # [Core] Agent task description & physical model
├── Task_zh-CN.md                    # [Core] Agent task description (Chinese version)
├── references/                      # Domain reference materials
│   └── constants.json               # Constants for prescription doses, coordinates, etc.
├── verification/                    # Verification and scoring system
│   ├── evaluator.py                 # Core scoring Python script (computes 3D dose superposition)
│   ├── requirements.txt             # Environment dependencies (numpy)
│   └── docker/                      
│       └── Dockerfile               # Containerization configuration
└── baseline/                        # Baseline solution and reference code
    └── solution.py                  # Reference code to generate an initial baseline solution
```

## 3. Quick Start

You can run the verification script for this task using either a local Python environment or Docker. The script reads a JSON file containing proton beam coordinates and weights, computes the dose on a 3D grid, and outputs the final score.

### Method 1: Run Locally via Python

1. **Install Dependencies**:
   Ensure `numpy` is installed in your environment.
   ```bash
   cd verification
   pip install -r requirements.txt
   ```

2. **Generate Baseline Solution (Optional)**:
   Run the baseline code to generate a simulated Agent output file named `plan.json`.
   ```bash
   cd ../baseline
   python solution.py
   ```

3. **Run the Evaluator**:
   Pass the path of the generated JSON file to the evaluator.
   ```bash
   cd ../verification
   python evaluator.py ../baseline/plan.json
   ```

### Method 2: Run via Docker (Recommended)

To ensure absolute consistency of the evaluation environment, using Docker is highly recommended.

1. **Build the Image**:
   Execute the build command in the `verification/docker` directory.
   ```bash
   cd verification/docker
   docker build -t frontier-proton-eval -f Dockerfile ..
   ```

2. **Run the Container for Evaluation**:
   Mount your local solution file into the container for scoring.
   ```bash
   docker run --rm -v $(pwd)/../../baseline/plan.json:/app/plan.json frontier-proton-eval /app/plan.json
   ```

## 4. Evaluation Metrics

`evaluator.py` outputs the results in a standard JSON format:
* `score`: The final comprehensive score (higher is better).
* `metrics`: Contains internal details, such as `ctv_mse` (Mean Squared Error of tumor dose, lower is better), `oar_overdose_penalty` (penalty for OAR overdose), and `total_weight` (total beam current consumed).