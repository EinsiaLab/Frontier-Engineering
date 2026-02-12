# 背景
AI4Research 领域目前的 bench 正确性评估大多为 0/1 形式，即使 Rubric 形式的评估方式，其结果也是在一个封闭的区间内，局限了 agent 在优化问题上能力的评估。近期类似于 ALE-Bench, MLE-Bench 和 FrontierCS 等 Bench 关注对开放性答案进行测试，从而最大程度评估 Agent 在开放问题上，以交互的方式进行迭代优化的能力。然而现存的 bench 大多关注 CS 领域的问题，领域单一且实际应用价值有限；或者将简单的实际问题高度抽象为数学问题，智能体无法充分利用自身和互联网的知识；现有 benchmark 的计算指标关注模型平均表现，但开放性优化问题应当对于模型在一个单一问题上的机制表现进行鼓励

# 解决方案
1. 针对实际应用价值的局限性，将关注的领域从CS转换到更广泛的 Eng，在实际场景中进行探索，并提升问题数量
2. 针对高度抽象的问题，为 agent 提供丰富上下文，agent 可以调用外部工具辅助解题
3. 针对评价指标的问题，在探索过程中提出一种新的 metric 进行评估

# 预期目标
1. 能提出一个规模较大（数百），涵盖工程学大部分领域的Bench，成为行业典范，问题有经济效益

# 示例样本
1. 航天动力学优化问题 容易验证，有实际背景，搜索空间大，需要参考的文档繁杂，难度大
2. 桁架的构造：可以参考很多经典的桥梁结构等
类似于游戏《Poly Bridge》
3. 非CUDA Kernel优化：参考资料少，但也重要的计算场景

# 样本要求
1. 和现实问题的差距较小，要考虑现实问题中可能存在的影响因素
2. 有一定经济效益的工程问题
3. 能够写出对应样本的验证程序，能在可接受的时间内进行评测
4. 可以优先考虑自己较为熟悉的领域的问题，尽可能保证自己清楚样本设置的合理性
5. 样本中要包含如下内容：
  1. 题目背景：要找到对应的现实例子依据）
  2. 题目描述：包含任务的流程，要求说明其中的细节信息，涉及到的数据尽可能参考现实案例给出，可以设计基础任务以及Bonus），同时规定好评分标准并进行详细说明
  3. 参考信息：解题所需要的一些基本信息（例如一些参数常数和方程等）或约束条件
  4. 数据格式：明确的输入输出数据格式，应当给出参考例子进行说明
  5. 验证程序：给出相应的验证代码，以及进行评测所需要的环境配置，如果可能可以给出docker
  6. （可选）给出问题的基础解法，并提供评测结果作为参考

# Related Work
1. FrontierCS: Evolving Challenges for Evolving Intelliges 
最优解未知，但解的质量可以评估的问题。但专注在TCS
2. CYBERGYM: EVALUATING AI AGENTS’ REAL-WORLD CYBERSECURITY CAPABILITIES AT SCALE 
https://arxiv.org/abs/2506.02548 很小的领域，不一定有经济价值
3. ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering
40个问题，问题少，规模小；经济性不一定有保证；没有实际意义，无法检验Agent使用工具的能力
4. PT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling
高度抽象的数学优化问题
https://arxiv.org/pdf/2601.19924
5. MLE-Bench：MLE-BENCH: EVALUATING MACHINE LEARNING AGENTS ON MACHINE LEARNING ENGINEERIN
75个ML问题，从Kaggle抓取，对agent能使用的资源进行限制，资源这部分的分析很有参考价值
6. LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics
      一个可供参考的评估方法

7. ICCAD CAD Contest（国际计算机辅助设计会议竞赛），利用现代HPC技术（如GPU加速、可微编程）在纳米级工艺节点下，对包含数百万个节点的离散组合系统进行多目标优化。https://research.nvidia.com/labs/electronic-design-automation/papers/yichen_iccad25_contest.pdf
8. 数据驱动的过程监控与故障诊断的综述 https://www.mdpi.com/2227-9717/12/2/251
9. ISCSO 2025 是一项国际学生结构优化竞赛。其主要目标是鼓励本科生和研究生解决工程优化问题https://www.brightoptimizer.com/
10. ISPD24 Contest: GPU/ML-Enhanced Large Scale Global Routing https://liangrj2014.github.io/ISPD24_contest/
11. The America’s cup of rocket science 行星际轨道设计 https://sophia.estec.esa.int/gtoc_portal/
12. 全球规模最大、影响力最深远的合成生物学赛事：iGEM 的核心理念在于“设计-构建-测试-学习”（Design-Build-Test-Learn）的工程循环，要求参赛者利用标准化的生物元件（BioBricks）构建具有实际功能的生物系统。https://competition.igem.org/
13. Bio-based Innovation Student Challenge Europe。比赛要求学生团队开发基于生物质（Biomass）的创新产品或工艺。这通常涉及化学工程、材料科学与工业生物技术的交叉 
14. Hello Tomorrow Global Challenge https://ufukavrupa.org.tr/sites/default/files/2025-11/2026%20Hello%20Tomorrow%20Challenge%20Brochure.pdf 涉及多个领域