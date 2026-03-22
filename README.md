# Frontier-Eng: Large-Scale Engineering Optimization Benchmark for AI Agents

English | [简体中文](README_zh-CN.md)

**Frontier-Eng** is a benchmark designed to evaluate the ability of AI Agents to solve **open-ended optimization problems** in real-world **engineering domains**.

Unlike existing benchmarks that focus on Computer Science (CS) or purely abstract mathematical problems, Frontier-Eng focuses on engineering challenges with actual **economic benefits** and **physical constraints**. It is expected to cover multiple fields such as aerospace, civil engineering, EDA, bioengineering, and more.

## 🎯 Motivation

Current AI4Research evaluation systems have the following limitations:

1. **Limited Evaluation Methods**: Most adopt 0/1 binary evaluation or closed-interval rubrics, failing to effectively measure an Agent's ability to perform **iterative optimization** through interaction in an open world.
2. **Domain Limitations**: Existing benchmarks are mostly confined to the CS domain (e.g., code generation) or highly abstract real problems into math problems, stripping away real-world complexity and preventing Agents from utilizing rich external knowledge and tools.
3. **Metric Bias**: Traditional computational metrics focus on model average performance, whereas for engineering optimization problems, we should focus more on the **Peak Performance** a model can achieve on a single problem through exploration mechanisms.

**Frontier-Eng** aims to evaluate the ability of Agents to solve problems with practical value across a wide range of engineering disciplines by providing rich context and tool support.

## 🤝 Contribution Guidelines

We need the power of the community to expand the coverage of the Benchmark. We welcome the submission of new engineering problems via Pull Requests (PR). If you wish to contribute, please follow the standards and processes below:

> **AI-Assisted Contributions**: We welcome contributions created with the assistance of AI tools. If you're using an AI assistant to help with your contribution, we recommend providing the prompt guide from this repository (`AGENT.md`) to ensure your AI assistant follows our standards and requirements. **However, please do not over-rely on AI tools or leave the process entirely to AI**. Human review and supervision are essential to ensure quality and correctness.

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
  2. `python -m frontier_eval task=unified task.benchmark=<Domain_Name>/<Task_Name> algorithm.iterations=0` # Framework compatibility verification for new benchmark contributions. Please document the exact unified benchmark id and any required runtime overrides (for example `task.runtime.conda_env=...`) in the README.

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

The table below lists the current coverage of domain tasks in the Benchmark. We welcome not only code contributions but also ideas for challenging new engineering problems from the community.

<table>
  <thead>
    <tr>
      <th>Domain</th>
      <th>Task Name</th>
      <th>Status</th>
      <th>Contributor</th>
      <th>Reviewer</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Astrodynamics</b></td>
      <td><code>MannedLunarLanding</code></td>
      <td>Completed</td>
      <td>@jdp22</td>
      <td>@jdp22</td>
      <td>Lunar soft landing trajectory optimization</td>
    </tr>
    <tr>
      <td rowspan="2"><b>ParticlePhysics</b></td>
      <td><code>MuonTomography</code></td>
      <td>Completed</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>Muon detector placement optimization under flux, budget, and excavation constraints</td>
    </tr>
    <tr>
      <td><code>ProtonTherapyPlanning</code></td>
      <td>Completed</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>IMPT dose weight optimization under tumor coverage, OAR safety, and beam cost constraints</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Kernel Engineering</b></td>
      <td><code>MLA</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode</td>
    </tr>
    <tr>
      <td><code>TriMul</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode</td>
    </tr>
    <tr>
      <td><code>FlashAttention</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>Optimize a causal scaled dot-product attention forward kernel for GPU execution</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Single Cell Analysis</b></td>
      <td><code>denoising</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Open Problems in Single-Cell Analysis</td>
    </tr>
    <tr>
      <td><code>perturbation_prediction</code></td>
      <td>Completed</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>NeurIPS 2023 scPerturb</td>
    </tr>
    <tr>
      <td><code>predict_modality</code></td>
      <td>Completed</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>NeurIPS 2021, RNA→ADT</td>
    </tr>
    <tr>
      <td rowspan="3"><b>QuantumComputing</b></td>
      <td><code>routing qftentangled</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Routing-Oriented Optimization</td>
    </tr>
    <tr>
      <td><code>clifford t synthesis</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Clifford+T Synthesis Optimization</td>
    </tr>
    <tr>
      <td><code>cross target qaoa</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Cross-Target Robust Optimization</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Cryptographic</b></td>
      <td><code>AES-128 CTR</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Advanced Encryption Standard, 128-bit key, Counter mode</td>
    </tr>
    <tr>
      <td><code>SHA-256</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 256-bit</td>
    </tr>
    <tr>
      <td><code>SHA3-256</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 3 256-bit</td>
    </tr>
    <tr>
      <td rowspan="3"><b>CommunicationEngineering</b></td>
      <td><code>LDPCErrorFloor</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>LDPC code error floor estimation using importance sampling for trapping sets</td>
    </tr>
    <tr>
      <td><code>PMDSimulation</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>Polarization Mode Dispersion simulation with importance sampling for rare outage events</td>
    </tr>
    <tr>
      <td><code>RayleighFadingBER</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>BER analysis under Rayleigh fading with importance sampling for deep fade events</td>
    </tr>
    <tr>
      <td rowspan="2"><b>EnergyStorage</b></td>
      <td><code>BatteryFastChargingProfile</code></td>
      <td>Completed</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>Fast-charge current-profile optimization for a lithium-ion cell under voltage, thermal, and degradation constraints</td>
    </tr>
    <tr>
      <td><code>BatteryFastChargingSPMe</code></td>
      <td>Completed</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>Staged fast-charge optimization under a reduced SPMe-T-Aging style electrochemical, thermal, plating, and aging model</td>
    </tr>
    <tr>
      <td><b>SustainableDataCenterControl</b></td>
      <td><code>hand_written_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>SustainDC joint control benchmark for load shifting, cooling, and battery dispatch through the unified evaluation pipeline</td>
    </tr>
    <tr>
      <td rowspan="4"><b>ReactionOptimisation</b></td>
      <td><code>snar_multiobjective</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Continuous-flow SnAr reaction optimization with a Pareto trade-off between productivity and waste</td>
    </tr>
    <tr>
      <td><code>mit_case1_mixed</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Mixed-variable reaction yield maximization with continuous process settings and a categorical catalyst</td>
    </tr>
    <tr>
      <td><code>reizman_suzuki_pareto</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Reizman Suzuki emulator Pareto optimization over catalyst choice and operating conditions</td>
    </tr>
    <tr>
      <td><code>dtlz2_pareto</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>DTLZ2 Pareto-front approximation task integrated through the unified evaluation pipeline</td>
    </tr>
    <tr>
      <td rowspan="3"><b>MolecularMechanics</b></td>
      <td><code>weighted_parameter_coverage</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Rare force-field parameter coverage under a molecule budget</td>
    </tr>
    <tr>
      <td><code>diverse_conformer_portfolio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Low-energy, high-diversity conformer portfolio selection</td>
    </tr>
    <tr>
      <td><code>torsion_profile_fitting</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Force-field torsion-scale fitting against target energy profiles</td>
    </tr>
    <tr>
      <td rowspan="16"><b>Optics</b></td>
      <td><code>adaptive_constrained_dm_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Constrained deformable mirror control</td>
    </tr>
    <tr>
      <td><code>adaptive_temporal_smooth_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Temporal smoothness versus correction quality</td>
    </tr>
    <tr>
      <td><code>adaptive_energy_aware_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Energy-aware adaptive optics control</td>
    </tr>
    <tr>
      <td><code>adaptive_fault_tolerant_fusion</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Fault-tolerant multi-WFS fusion</td>
    </tr>
    <tr>
      <td><code>phase_weighted_multispot_single_plane</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Single-plane weighted multispot phase DOE</td>
    </tr>
    <tr>
      <td><code>phase_fourier_pattern_holography</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Fourier pattern holography</td>
    </tr>
    <tr>
      <td><code>phase_dammann_uniform_orders</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dammann grating uniform diffraction orders</td>
    </tr>
    <tr>
      <td><code>phase_large_scale_weighted_spot_array</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Large-scale weighted spot array synthesis</td>
    </tr>
    <tr>
      <td><code>fiber_wdm_channel_power_allocation</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>WDM channel and launch power allocation</td>
    </tr>
    <tr>
      <td><code>fiber_mcs_power_scheduling</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Joint MCS and power scheduling</td>
    </tr>
    <tr>
      <td><code>fiber_dsp_mode_scheduling</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Receiver DSP mode scheduling</td>
    </tr>
    <tr>
      <td><code>fiber_guardband_spectrum_packing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Spectrum packing with guard-band constraints</td>
    </tr>
    <tr>
      <td><code>holographic_multifocus_power_ratio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-focus power ratio control</td>
    </tr>
    <tr>
      <td><code>holographic_multiplane_focusing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-plane holographic focusing</td>
    </tr>
    <tr>
      <td><code>holographic_multispectral_focusing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multispectral holographic focusing</td>
    </tr>
    <tr>
      <td><code>holographic_polarization_multiplexing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Polarization-multiplexed holography</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Computer Systems</b></td>
      <td><code>Malloc Lab</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dynamic memory allocation</td>
    </tr>
    <tr>
      <td><code>DuckDBWorkloadOptimization</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Index/materialized-view selection and query rewriting optimization on official DuckDB workloads</td>
    </tr>
    <tr>
      <td><b>EngDesign</b></td>
      <td><code>CY_03, WJ_01, XY_05, AM_02, AM_03, YJ_02, YJ_03</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td><a href="https://github.com/AGI4Engineering/EngDesign.git">EngDesign</a></td>
    </tr>
    <tr>
      <td rowspan="5"><b>InventoryOptimization</b></td>
      <td><code>tree_gsm_safety_stock</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Tree-structured multi-echelon safety-stock placement (GSM)</td>
    </tr>
    <tr>
      <td><code>general_meio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>General-topology MEIO with simulation-based objective</td>
    </tr>
    <tr>
      <td><code>joint_replenishment</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-SKU joint replenishment with shared setup cost</td>
    </tr>
    <tr>
      <td><code>finite_horizon_dp</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Finite-horizon stochastic inventory control via time-varying policy</td>
    </tr>
    <tr>
      <td><code>disruption_eoqd</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>EOQ lot-sizing optimization under supply disruptions</td>
    </tr>
    <tr>
      <td rowspan="3"><b>PyPortfolioOpt</b></td>
      <td><code>robust_mvo_rebalance</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Robust mean-variance rebalancing with sector/factor/turnover constraints</td>
    </tr>
    <tr>
      <td><code>cvar_stress_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>CVaR stress-controlled portfolio allocation under return and exposure constraints</td>
    </tr>
    <tr>
      <td><code>discrete_rebalance_mip</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Discrete lot-constrained rebalancing with mixed-integer optimization</td>
    </tr>
    <tr>
      <td rowspan="7"><b>JobShop</b></td>
      <td><code>abz</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP ABZ family (Adams, Balas, Zawack 1988)</td>
    </tr>
    <tr>
      <td><code>ft</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP FT family (Fisher and Thompson 1963)</td>
    </tr>
    <tr>
      <td><code>la</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP LA family (Lawrence 1984)</td>
    </tr>
    <tr>
      <td><code>orb</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP ORB family (Applegate and Cook 1991)</td>
    </tr>
    <tr>
      <td><code>swv</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP SWV family (Storer, Wu, Vaccari 1992)</td>
    </tr>
    <tr>
      <td><code>ta</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP TA family (Taillard 1993)</td>
    </tr>
    <tr>
      <td><code>yn</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP YN family (Yamada and Nakano 1992)</td>
    </tr>
    <tr>
      <td rowspan="4"><b>StructuralOptimization</b></td>
      <td><code>ISCSO2015</code></td>
      <td>Completed</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>45-bar 2D truss size + shape</td>
    </tr>
    <tr>
      <td><code>ISCSO2023</code></td>
      <td>Completed</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>284-member 3D truss sizing</td>
    </tr>
    <tr>
      <td><code>TopologyOptimization</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>MBB beam 2D topology optimization (SIMP), Continuous, volume-constrained, compliance minimization</td>
    </tr>
    <tr>
      <td><code>PyMOTOSIMPCompliance</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>pyMOTO-based 2D beam topology optimization (SIMP + OC/MMA) under a volume-fraction constraint</td>
    </tr>
    <tr>
      <td rowspan="6"><b>Robotics</b></td>
      <td><code>DynamicObstacleAvoidanceNavigation</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Navigate a differential-drive robot from start to goal</td>
    </tr>
    <tr>
      <td><code>QuadrupedGaitOptimization</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Maximize the forward locomotion speed of a quadruped robot by optimizing 8 gait parameters</td>
    </tr>
    <tr>
      <td><code>RobotArmCycleTimeOptimization</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Minimize the motion time of a 7-DOF KUKA LBR iiwa arm moving from a start to a goal configuration, collision-free</td>
    </tr>
    <tr>
      <td><code>PIDTuning</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>Tune a cascaded PID controller for a 2D quadrotor across multiple flight scenarios</td>
    </tr>
    <tr>
      <td><code>UAVInspectionCoverageWithWind</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@ahydchh</td>
      <td>UAV inspection under wind field disturbance</td>
    </tr>
    <tr>
      <td><code>CoFlyersVasarhelyiTuning</code></td>
      <td>In Progress</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Tune the original CoFlyers Vasarhelyi flocking parameters</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Aerodynamics</b></td>
      <td><code>CarAerodynamicsSensing</code></td>
      <td>Completed</td>
      <td>@LeiDQ, @llltttwww</td>
      <td>@llltttwww</td>
      <td>Sensor placement on 3D car surface for pressure field reconstruction</td>
    </tr>
    <tr>
      <td><code>DawnAircraftDesignOptimization</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Jointly optimize wing, fuselage, and propulsion variables under cruise/endurance/payload constraints to minimize total aircraft mass</td>
    </tr>
    <tr>
      <td><b>WirelessChannelSimulation</b></td>
      <td><code>HighReliableSimulation</code></td>
      <td>Completed</td>
      <td>@tonyhaohan</td>
      <td>@yks23, @ahydchh</td>
      <td>BER estimation with importance sampling for Hamming(127,120)</td>
    </tr>
    <tr>
      <td><b>PowerSystems</b></td>
      <td><code>EV2GymSmartCharging</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Upstream-aligned EV smart charging</td>
    </tr>
    <tr>
      <td><b>AdditiveManufacturing</b></td>
      <td><code>DiffSimThermalControl</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Study process optimization in additive manufacturing using differentiable simulation</td>
    </tr>
  </tbody>
</table>

> 💡 **Have an idea for a new engineering problem?**
> Even if you cannot provide complete verification code for now, we highly welcome you to share good **Task concepts**!
> Please create an Issue detailing the **real-world background** and **engineering value** of the problem. After discussion and confirmation, we will add it to the table above to rally community power to solve it together.

## 🧪 Evaluation Framework
An initial integration between some evaluation algorithms and benchmarks has been implemented. The core implementation is located in `./frontier_eval`. For usage instructions, see the [Evaluation README](frontier_eval/README.md). Note: some optional algorithms/benchmarks require extra repos under `third_party/` (local clones); the Evaluation README documents how to set them up.

## 💬 Join the Community
Welcome to our developer community! Whether you want to discuss new engineering problem concepts, find task collaborators, or encounter technical issues during your contribution, you can always communicate with us in the group.

* 🟢 **Feishu (Lark)**: [Click here to join our Feishu discussion group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c)

* 🔜 **Discord**: [Click here to join our Discord community](https://discord.gg/hxeVhZNN)

