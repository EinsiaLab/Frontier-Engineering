# Contributing to Frontier-Eng

[简体中文](CONTRIBUTING_zh-CN.md)

We need the power of the community to expand the coverage of the Benchmark. We welcome the submission of new engineering problems via Pull Requests (PR). If you wish to contribute, please follow the standards and processes below.

> **AI-Assisted Contributions**: We welcome contributions created with the assistance of AI tools. If you're using an agent, point it at `skill/source/frontier-contributor/` (see `SKILL.md` there). **However, please do not over-rely on AI tools or leave the process entirely to AI**. Human review and supervision are essential to ensure quality and correctness.

## Sample Requirements

1. **Reality Gap**: Must be close to reality, considering real-world influencing factors, not purely abstract mathematics.
2. **Economic Value**: The problem should have clear engineering or economic value upon solution.
3. **Verifiability**: Must provide an executable verification program (Docker preferred) capable of completing the evaluation within an acceptable time.

## Submission Format

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

## Submission Guidelines

1. Keep test commands as short as possible (ideally single-line commands). Testing is mandatory before submission!

   1. `python verification/evaluator.py scripts/init.py` — Run under benchmark, using `verification/evaluator.py` as the evaluation entry point. The target of the test, i.e., the target of agent evolution, is `scripts/init.py`.
   2. `python -m frontier_eval task=unified task.benchmark=<Domain_Name>/<Task_Name> algorithm.iterations=0` — Framework compatibility verification for new benchmark contributions. Please document the exact unified benchmark id and any required runtime overrides (for example `task.runtime.conda_env=...`) in the README, and explicitly call out any benchmark-specific environment setup (extra envs, Docker, `third_party/`, custom `python_path`, etc.).

2. Please avoid files containing private information, such as: `.env`, API keys, IDE configurations (`.vscode/`), temporary files (`*.log`, `temp/`, `__pycache__`, and personal test scripts). Also, please check that the submitted content does not contain absolute paths to avoid reproducibility issues and privacy leaks.

3. **EVOLVE-BLOCK Markers (Required for ShinkaEvolve / ABMCTS)**: The file evolved by the agent (e.g., `scripts/init.py`, or language-specific baselines like `malloclab-handout/mm.c`) must include `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers to define the *only* editable region.
   - Keep the marker lines intact, and keep all code outside the markers read-only (CLI/I/O contracts, constraint checks, evaluator glue, etc.).
   - Use the correct comment style for your language:
     - Python: `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`
     - C/C++/CUDA/Rust/Swift: `// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END`

## Contribution Process

We adopt the standard GitHub collaboration flow:

1. **Fork this Repository**: Click the "Fork" button in the top right corner to copy the project to your GitHub account.
2. **Create Branch**:
   - Clone your Fork locally.
   - Create a new branch for development, recommended naming format: `feat/<Domain>/<TaskName>` (e.g., `feat/Astrodynamics/MarsLanding`).
3. **Add/Modify Content**:
   - Add your engineering problem files following the submission format above.
   - Ensure all necessary explanatory documentation and verification code are included.
4. **Local Test**: Run `evaluator.py` or build the Docker image to ensure the evaluation logic is correct and runs normally.
5. **Submit Pull Request (PR)**:
   - Push changes to your remote Fork.
   - Initiate a Pull Request to the `main` branch of this repository.
   - **PR Description**: Please briefly explain the background, source, and how to run the verification code for the Task.
6. **Code Review**:
   - **Agent Review**: After submitting the PR, an **AI Agent** will first conduct an automated preliminary review (including code standards, basic logic verification, etc.) and may propose modifications directly in the PR.
   - **Maintainer Review**: After the Agent review passes, **maintainers** will conduct a final re-check. Once confirmed correct, your contribution will be merged.

---

> If this is your first contribution or you have questions about the directory structure, feel free to submit an Issue for discussion first.
