# Frontier-Eng: Large-Scale Engineering Optimization Benchmark for AI Agents

English | [简体中文](README_zh-CN.md)

**Frontier-Eng** is a benchmark designed to evaluate the ability of AI Agents to solve **open-ended optimization problems** in real-world **engineering domains**.

Unlike existing benchmarks that focus on Computer Science (CS) or purely abstract mathematical problems, Frontier-Eng focuses on engineering challenges with actual **economic benefits** and **physical constraints**. It is expected to cover multiple fields such as aerospace, civil engineering, EDA, bioengineering, and more.

## Runtime Environment Notes

`frontier_eval/requirements.txt` only sets up the evaluation framework itself. It does **not** mean every benchmark can run inside the same environment.

Before running any specific benchmark, always read the corresponding environment instructions in:

- `benchmarks/<Domain>/README*.md`
- `benchmarks/<Domain>/<Task>/README*.md` if the task has its own README

Many benchmark families require their own runtime environment, extra `requirements.txt`, extra `third_party/` checkouts, or Docker-based execution. When a benchmark README documents runtime overrides such as `task.runtime.conda_env=...`, `task.runtime.python_path=...`, or `task.runtime.use_conda_run=false`, treat the benchmark README as the source of truth and copy those overrides into your run command.

Examples already in this repository include `ReactionOptimisation` (`summit`), `MolecularMechanics` (`openff-dev`), `SustainableDataCenterControl` (`sustaindc`), `PyPortfolioOpt` (`pyportfolioopt`), `QuantumComputing` (`mqt`), `InventoryOptimization` (`stock`), `JobShop` (custom `python_path`), and `EngDesign` (Docker / local mode).

Project-local agent skills are bundled under `skill/`. Run `python -m frontier_eval skill` to choose interactively, or use `python -m frontier_eval skill evaluator codex` for a direct install.

### `v1` Merged Task Environments

To reduce the number of runtime environments used by the effective `v1` task pool without breaking existing setups, the repository now uses the following convention:

- `frontier-eval-2` remains the evaluation-framework / driver environment and is left unchanged.
- Existing task environments such as `bio`, `mqt`, `optics`, `stock`, `pyportfolioopt`, `motion`, `jobshop`, `summit`, `sustaindc`, and `kernel` are preserved and not overwritten.
- New merged task environments are created under whichever environment prefix the current `conda` installation manages, with default names `frontier-v1-main`, `frontier-v1-summit`, `frontier-v1-sustaindc`, and `frontier-v1-kernel`.
- For `v1` tasks that need a direct interpreter instead of `conda run` (currently `ReactionOptimisation/*` and `JobShop/*`), the batch matrices use the portable marker `conda-env:<env-name>`. The unified evaluator resolves that marker to the target env's Python executable at runtime, so repository files stay machine-independent.

Current `v1` runtime consolidation:

- `frontier-v1-main`: `SingleCellAnalysis/predict_modality`, `QuantumComputing/*`, `Optics/*`, `InventoryOptimization/*`, `PyPortfolioOpt/*`, `JobShop/*`, `Robotics/DynamicObstacleAvoidanceNavigation`, `Robotics/PIDTuning`, `Robotics/UAVInspectionCoverageWithWind`, `Robotics/QuadrupedGaitOptimization`, `Robotics/RobotArmCycleTimeOptimization`, `Aerodynamics/CarAerodynamicsSensing`, `KernelEngineering/FlashAttention`
- `frontier-v1-summit`: `ReactionOptimisation/*`
- `frontier-v1-sustaindc`: `SustainableDataCenterControl/*`
- `frontier-v1-kernel`: `KernelEngineering/MLA`, `KernelEngineering/TriMul`

If an older benchmark README still mentions legacy env names such as `mqt`, `stock`, `pyportfolioopt`, or `jobshop`, prefer the batch matrix files under `frontier_eval/conf/batch/` as the source of truth for current `v1` runs.

Setup and validation scripts:

- Initialize merged envs: `bash scripts/setup_v1_merged_task_envs.sh`
- Validate merged envs with `iter=0`: `DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<gpu_id> bash scripts/validate_v1_merged_task_envs.sh`

Notes:

- The validation script uses `conda run -n frontier-eval-2 python` as the default driver, and can also be overridden with `DRIVER_PY=/path/to/python`. It checks CPU `v1`, GPU `v1`, `FlashAttention`, `MLA`, and `TriMul`.
- `MuonTomography` remains excluded from the current effective `v1` pool as described later in this README.
- Known caveat: the official `KernelEngineering/TriMul` full benchmark (`verification/tri_bench.txt`) may still be VRAM-limited on 24GB-class GPUs; this is typically a task-level memory-bound issue rather than a missing dependency in `frontier-v1-kernel`.

## 🎯 Motivation

Current AI4Research evaluation systems have the following limitations:

1. **Limited Evaluation Methods**: Most adopt 0/1 binary evaluation or closed-interval rubrics, failing to effectively measure an Agent's ability to perform **iterative optimization** through interaction in an open world.
2. **Domain Limitations**: Existing benchmarks are mostly confined to the CS domain (e.g., code generation) or highly abstract real problems into math problems, stripping away real-world complexity and preventing Agents from utilizing rich external knowledge and tools.
3. **Metric Bias**: Traditional computational metrics focus on model average performance, whereas for engineering optimization problems, we should focus more on the **Peak Performance** a model can achieve on a single problem through exploration mechanisms.

**Frontier-Eng** aims to evaluate the ability of Agents to solve problems with practical value across a wide range of engineering disciplines by providing rich context and tool support.

## 🤝 Contribution Guidelines

We need the power of the community to expand the coverage of the Benchmark. We welcome the submission of new engineering problems via Pull Requests (PR). If you wish to contribute, please follow the standards and processes below:

> **AI-Assisted Contributions**: We welcome contributions created with the assistance of AI tools. If you're using an agent to help with your contribution, run `python -m frontier_eval skill` and install `Contributor`, or use `skill/source/frontier-contributor/SKILL.md` directly. **However, please do not over-rely on AI tools or leave the process entirely to AI**. Human review and supervision are essential to ensure quality and correctness.

### Sample Requirements

1. **Reality Gap**: Must be close to reality, considering real-world influencing factors, not purely abstract mathematics.
2. **Economic Value**: The problem should have clear engineering or economic value upon solution.
3. **Verifiability**: Must provide an executable verification program (Docker preferred) capable of completing the evaluation within an acceptable time.

### Submission Format

Each Task should contain the following file structure:

```text
<Domain_Name>/                       # Level 1 Directory: Domain Name (e.g., Astrodynamics)
├── README.md                        # [Required] Domain Overview (Default entry, EN or CN): Background & sub-task index
├── README_zh-CN.md                  # [Optional] Domain Overview (Chinese version. Used only if README.md is in English)
├── <Task_Name_A>/                   # Level 2 Directory: Specific Task Name (e.g., MannedLunarLanding)
│   ├── README.md                    # [Required] Navigation Doc: File structure, how to run & quick start
│   ├── README_zh-CN.md              # [Optional] Navigation Doc (Chinese version)
│   ├── Task.md                      # [Required] Task Detail Doc: Core doc including background, physical model, I/O definitions
│   ├── Task_zh-CN.md                # [Optional] Task Detail Doc (Chinese version)
│   ├── references/                  # References Directory
│   │   ├── constants.json           # Physical constants, simulation parameters, etc.
│   │   └── manuals.pdf              # Domain knowledge manual, physical equations, or constraints docs
│   ├── frontier_eval/               # [Required] Unified-task metadata for Frontier Eval onboarding
│   │   ├── initial_program.txt      # Initial editable program path (relative to task root)
│   │   ├── eval_command.txt         # Evaluation command template used by `task=unified`
│   │   ├── agent_files.txt          # Context files exposed to the agent
│   │   ├── artifact_files.txt       # Output files/logs to collect after evaluation
│   │   └── constraints.txt          # Optional task-specific constraints/instructions
│   ├── verification/                # Verification & Scoring System
│   │   ├── evaluator.py             # [Core] Scoring script entry point
│   │   ├── requirements.txt         # Dependencies required for the scoring environment
│   │   └── docker/                  # Environment containerization configuration
│   │       └── Dockerfile           # Ensures consistency of the evaluation environment
│   └── baseline/                    # [Optional] Baseline Solution / Example Code
│       ├── solution.py              # Reference code implementation
│       └── result_log.txt           # Execution log or scoring result of the reference code
└── <Task_Name_B>/                   # Another task under this domain
    └── ...
```

> The above directory structure serves only as a reference template. Contributors may adjust the file organization based on specific circumstances, provided that all core elements (e.g., background, input/output, evaluation metrics) are included. Additionally, there are no restrictions on the programming language and format of the verification code.
>
> New benchmark contributions must be onboarded through the unified task format. In practice, this means adding benchmark-local metadata under `<Task_Name>/frontier_eval/` and validating the task with `task=unified`. Adding a new custom task under `frontier_eval/tasks/<task>/...` is an exception path that should only be used when the unified format is demonstrably insufficient and the maintainer team has agreed on the exception first. See `frontier_eval/README.md` for the full unified metadata schema.

### Submission Guidelines

1. Keep test commands as short as possible (ideally single-line commands). Testing is mandatory before submission!

  1. `python verification/evaluator.py scripts/init.py` # Run under benchmark, using `verification/evaluator.py` as the evaluation entry point. The target of the test, i.e., the target of agent evolution, is `scripts/init.py`.
  2. `python -m frontier_eval task=unified task.benchmark=<Domain_Name>/<Task_Name> algorithm.iterations=0` # Framework compatibility verification for new benchmark contributions. Please document the exact unified benchmark id and any required runtime overrides (for example `task.runtime.conda_env=...`) in the README, and explicitly call out any benchmark-specific environment setup (extra envs, Docker, `third_party/`, custom `python_path`, etc.).

2. Please avoid files containing private information, such as: `.env`, API keys, IDE configurations (`.vscode/`), temporary files (`*.log`, `temp/`, `__pycache__`, and personal test scripts). Also, please check that the submitted content does not contain absolute paths to avoid reproducibility issues and privacy leaks.

3. **EVOLVE-BLOCK Markers (Required for ShinkaEvolve / ABMCTS)**: The file evolved by the agent (e.g., `scripts/init.py`, or language-specific baselines like `malloclab-handout/mm.c`) must include `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers to define the *only* editable region.
   - Keep the marker lines intact, and keep all code outside the markers read-only (CLI/I/O contracts, constraint checks, evaluator glue, etc.).
   - Use the correct comment style for your language:
     - Python: `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`
     - C/C++/CUDA/Rust/Swift: `// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END`

### Contribution Process

We adopt the standard GitHub collaboration flow:

1. **Fork this Repository**: Click the "Fork" button in the top right corner to copy the project to your GitHub account.
2. **Create Branch**:
* Clone your Fork locally.
* Create a new branch for development, recommended naming format: `feat/<Domain>/<TaskName>` (e.g., `feat/Astrodynamics/MarsLanding`).


3. **Add/Modify Content**:
* Add your engineering problem files following the submission format above.
* Ensure all necessary explanatory documentation and verification code are included.


4. **Local Test**: Run `evaluator.py` or build the Docker image to ensure the evaluation logic is correct and runs normally.
5. **Submit Pull Request (PR)**:
* Push changes to your remote Fork.
* Initiate a Pull Request to the `main` branch of this repository.
* **PR Description**: Please briefly explain the background, source, and how to run the verification code for the Task.


6. **Code Review**:
* **Agent Review**: After submitting the PR, an **AI Agent** will first conduct an automated preliminary review (including code standards, basic logic verification, etc.) and may propose modifications directly in the PR.
* **Maintainer Review**: After the Agent review passes, **maintainers** will conduct a final re-check. Once confirmed correct, your contribution will be merged.



---

> 💡 If this is your first contribution or you have questions about the directory structure, feel free to submit an Issue for discussion first.

## 📊 Task Progress & Planning

The full benchmark coverage table and planning notes are now maintained in [TASK_PROGRESS.md](TASK_PROGRESS.md).

That file keeps the complete task list, status, contributor / reviewer metadata, and the current `v1` inclusion notes, including the temporary exclusion note for `MuonTomography`.

## 📦 `best_code_only`

`best_code_only/` is a root-level snapshot of the final global best code extracted for each available experiment / algorithm / model / task combination.

Structure:

```text
best_code_only/
└── <experiment>/
    └── <algorithm>/
        └── <model>/
            └── <task>/
                └── <code-file>
```

Usage notes:

- Check `best_code_only/coverage.json` first if you want to confirm coverage for a specific experiment, algorithm, or model.
- Open the task directory directly to get the final best code file for that combination. Example paths:
  - `best_code_only/experiment1/openevolve/gpt-5.4/Astrodynamics_MannedLunarLanding/`
  - `best_code_only/experiment2/shinkaevolve/claude-opus-4.6/KernelEngineering_TriMul/`
- Filenames keep the original source suffix and task-local naming, so you may see `.py`, `.c`, `.cpp`, or other benchmark-specific filenames.

## 🧪 Evaluation Framework
An initial integration between some evaluation algorithms and benchmarks has been implemented. The core implementation is located in `./frontier_eval`. For usage instructions, see the [Evaluation README](frontier_eval/README.md). Note: some optional algorithms/benchmarks require extra repos under `third_party/` (local clones); the Evaluation README documents how to set them up.

## 💬 Join the Community
Welcome to our developer community! Whether you want to discuss new engineering problem concepts, find task collaborators, or encounter technical issues during your contribution, you can always communicate with us in the group.

* 🟢 **Feishu (Lark)**: [Click here to join our Feishu discussion group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c)

* 🔜 **Discord**: [Click here to join our Discord community](https://discord.gg/hxeVhZNN)

<!-- AI_GENERATED -->
