# Background

Current correctness evaluations for benchmarks in the AI4Research field are mostly in a binary (0/1) format. Even rubric-based evaluation methods result in a closed interval, limiting the assessment of an agent's capability in optimization problems. Recent benchmarks like ALE-Bench, MLE-Bench, and FrontierCS focus on testing open-ended answers to maximally assess an Agent's ability to perform iterative optimization on open questions in an interactive manner. However, existing benchmarks mostly focus on the CS domain, which is narrow in scope and has limited practical application value; or they highly abstract simple practical problems into mathematical problems, where agents cannot fully utilize their own knowledge and the internet; existing benchmark calculation metrics focus on average model performance, but open-ended optimization problems should encourage model performance on the mechanisms of a single problem.

# Solutions

1. Addressing the limitation of practical application value: Shift the focus domain from CS to broader Engineering (Eng), explore within real-world scenarios, and increase the number of problems.
2. Addressing highly abstract problems: Provide rich context for the agent, allowing the agent to invoke external tools to assist in problem-solving.
3. Addressing the issue of evaluation metrics: Propose a new metric for assessment during the exploration process.

# Expected Goals

1. Propose a large-scale benchmark (hundreds of problems) covering most fields of engineering, becoming an industry standard, with problems holding economic value.

# Example Samples

1. **Aerospace Dynamics Optimization Problems:** Easy to verify, have a real-world background, large search space, require complex reference documentation, and are highly difficult.
2. **Truss Construction:** Can reference many classic bridge structures, etc.
Similar to the game "Poly Bridge".
3. **Non-CUDA Kernel Optimization:** Few reference materials, but an important computing scenario.

# Sample Requirements

1. Small gap with real-world problems; must consider influencing factors that may exist in reality.
2. Engineering problems with certain economic benefits.
3. Must be able to write a corresponding verification program for the sample, and evaluation must be possible within an acceptable time frame.
4. Prioritize problems in fields you are familiar with to ensure the rationality of the sample settings as much as possible.
5. Samples must include the following content:
    1. **Problem Background:** Find corresponding real-world examples as a basis.
    2. **Problem Description:** Includes the task flow, requires explanation of detailed information, involved data should reference real cases as much as possible, can design basic tasks as well as Bonuses, while stipulating scoring standards and explaining them in detail.
    3. **Reference Information:** Basic information needed for solving (e.g., parameter constants and equations) or constraints.
    4. **Data Format:** Clear input/output data formats; reference examples should be provided for explanation.
    5. **Verification Program:** Provide corresponding verification code and the environment configuration needed for evaluation; provide Docker if possible.
    6. **(Optional)** Provide a basic solution to the problem and provide evaluation results as a reference.



# Related Work

1. **FrontierCS: Evolving Challenges for Evolving Intelligences**
Problems where the optimal solution is unknown, but the quality of the solution can be evaluated. However, it focuses on TCS (Theoretical Computer Science).
2. **CYBERGYM: EVALUATING AI AGENTS’ REAL-WORLD CYBERSECURITY CAPABILITIES AT SCALE**
[https://arxiv.org/abs/2506.02548](https://arxiv.org/abs/2506.02548) A very small domain, not necessarily having economic value.
3. **ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering**
40 problems, few in number, small scale; economic viability is not guaranteed; lacks practical significance, cannot verify the Agent's ability to use tools.
4. **PT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling**
Highly abstract mathematical optimization problems.
[https://arxiv.org/pdf/2601.19924](https://arxiv.org/pdf/2601.19924)
5. **MLE-Bench: MLE-BENCH: EVALUATING MACHINE LEARNING AGENTS ON MACHINE LEARNING ENGINEERING**
75 ML problems, scraped from Kaggle, limits the resources the agent can use; the analysis of this resource part is very valuable for reference.
6. **LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics**
An evaluation method suitable for reference.
7. **ICCAD CAD Contest** (International Conference on Computer-Aided Design Contest), utilizing modern HPC technologies (such as GPU acceleration, differentiable programming) to perform multi-objective optimization on discrete combinatorial systems containing millions of nodes at nanometer process nodes. [https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_iccad25_contest.pdf](https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_iccad25_contest.pdf)
8. **Review of Data-Driven Process Monitoring and Fault Diagnosis** [https://www.mdpi.com/2227-9717/12/2/251](https://www.mdpi.com/2227-9717/12/2/251)
9. **ISCSO 2025** is an international student structural optimization competition. Its main goal is to encourage undergraduate and graduate students to solve engineering optimization problems. [https://www.brightoptimizer.com/](https://www.brightoptimizer.com/)
10. **ISPD24 Contest: GPU/ML-Enhanced Large Scale Global Routing** [https://liangrj2014.github.io/ISPD24_contest/](https://liangrj2014.github.io/ISPD24_contest/)
11. **The America’s cup of rocket science** Interplanetary trajectory design [https://sophia.estec.esa.int/gtoc_portal/](https://sophia.estec.esa.int/gtoc_portal/)
12. **The world's largest and most influential synthetic biology competition:** iGEM's core philosophy lies in the engineering cycle of "Design-Build-Test-Learn", requiring participants to use standardized biological parts (BioBricks) to construct biological systems with actual functions. [https://competition.igem.org/](https://competition.igem.org/)
13. **Bio-based Innovation Student Challenge Europe.** The competition requires student teams to develop innovative products or processes based on Biomass. This usually involves the intersection of chemical engineering, materials science, and industrial biotechnology.
14. **Hello Tomorrow Global Challenge** [https://ufukavrupa.org.tr/sites/default/files/2025-11/2026%20Hello%20Tomorrow%20Challenge%20Brochure.pdf](https://ufukavrupa.org.tr/sites/default/files/2025-11/2026%20Hello%20Tomorrow%20Challenge%20Brochure.pdf) Involves multiple fields.