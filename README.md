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

### Submission Guidelines

1. Keep test commands as short as possible (ideally single-line commands). Testing is mandatory before submission!

  1. `python verification/evaluator.py scripts/init.py` # Run under benchmark, using `verification/evaluator.py` as the evaluation entry point. The target of the test, i.e., the target of agent evolution, is `scripts/init.py`.
  2. `python -m frontier_eval task=<task_name> algorithm.iterations=0` # Framework compatibility verification. Note: Please specify the `task_name` registered in the README.

2. Please avoid files containing private information, such as: `.env`, API keys, IDE configurations (`.vscode/`), temporary files (`*.log`, `temp/`, `__pycache__`, and personal test scripts). Also, please check that the submitted content does not contain absolute paths to avoid reproducibility issues and privacy leaks.

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
      <td><b>ElectronicDesignAutomation</b></td>
      <td><code>IntegrationPhysicalDesignOptimization</code></td>
      <td>In Development</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Chip macro placement optimization</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Kernel Engineering</b></td>
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
      <td><b>Computer Systems</b></td>
      <td><code>Malloc Lab</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dynamic memory allocation</td>
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
      <td rowspan="2"><b>StructuralOptimization</b></td>
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
      <td><b>Aerodynamics</b></td>
      <td><code>CarAerodynamicsSensing</code></td>
      <td>Completed</td>
      <td>@LeiDQ, @llltttwww</td>
      <td>@llltttwww</td>
      <td>Sensor placement on 3D car surface for pressure field reconstruction</td>
    </tr>
    <tr>
      <td><b>WirelessChannelSimulation</b></td>
      <td><code>HighReliableSimulation</code></td>
      <td>Completed</td>
      <td>@tonyhaohan</td>
      <td>@yks23, @ahydchh</td>
      <td>BER estimation with importance sampling for Hamming(127,120)</td>
    </tr>
  </tbody>
</table>

> 💡 **Have an idea for a new engineering problem?**
> Even if you cannot provide complete verification code for now, we highly welcome you to share good **Task concepts**!
> Please create an Issue detailing the **real-world background** and **engineering value** of the problem. After discussion and confirmation, we will add it to the table above to rally community power to solve it together.

## 🧪 Evaluation Framework
An initial integration between some evaluation algorithms and benchmarks has been implemented. The core implementation is located in `./frontier_eval`. For usage instructions, see the [Evaluation README](frontier_eval/README.md).

## 💬 Join the Community
Welcome to our developer community! Whether you want to discuss new engineering problem concepts, find task collaborators, or encounter technical issues during your contribution, you can always communicate with us in the group.

* 🟢 **Feishu (Lark)**: [Click here to join our Feishu discussion group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=a1cuff9f-347a-43ce-8825-79c2a38038c6)

* 🔜 **Discord / Slack**: (Preparing, coming soon...)
