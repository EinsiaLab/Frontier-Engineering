# 任务进度与规划

[English](TASK_PROGRESS.md) | 简体中文

此页面收录了原本嵌在根目录 `README_zh-CN.md` 中的 benchmark 覆盖表与规划说明。

下表列出了当前 Benchmark 各领域任务的覆盖情况。我们不仅欢迎代码贡献，也欢迎社区提出有挑战性的新工程问题构想。

说明：当前有效的 `v1` benchmark 任务池为 `47` 个题。为保留任务目录完整性，`MuonTomography` 仍保留在下表中，但目前暂不计入有效 `v1` 任务池，后续需先完成目标函数 / evaluator 对齐后的 benchmark 重构。

<table>
  <thead>
    <tr>
      <th>领域</th>
      <th>任务名称</th>
      <th>状态</th>
      <th>贡献者</th>
      <th>审查者</th>
      <th>备注</th>
      <th>版本</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Astrodynamics</b></td>
      <td><code>MannedLunarLanding</code></td>
      <td>已完成</td>
      <td>@jdp22</td>
      <td>@jdp22</td>
      <td>登月软着陆轨迹优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="2"><b>ParticlePhysics</b></td>
      <td><code>MuonTomography</code></td>
      <td>已完成</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>考虑缪子通量、预算与开挖约束的探测器布局优化；当前暂不计入有效 v1 任务池，后续待重构</td>
      <td></td>
    </tr>
    <tr>
      <td><code>ProtonTherapyPlanning</code></td>
      <td>已完成</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>在肿瘤覆盖、危及器官保护与束流成本约束下优化 IMPT 剂量权重</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Kernel Engineering</b></td>
      <td><code>MLA</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode MLA 解码内核</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>TriMul</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode 三角乘法</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>FlashAttention</code></td>
      <td>已完成</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>为 GPU 执行优化因果型 scaled dot-product attention 前向内核</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Single Cell Analysis</b></td>
      <td><code>denoising</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Open Problems 单细胞分析</td>
      <td></td>
    </tr>
    <tr>
      <td><code>perturbation_prediction</code></td>
      <td>已完成</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>OpenProblems 扰动响应预测（NeurIPS 2023 scPerturb）</td>
      <td></td>
    </tr>
    <tr>
      <td><code>predict_modality</code></td>
      <td>已完成</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>OpenProblems 模态预测（NeurIPS 2021，RNA→ADT）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>QuantumComputing</b></td>
      <td><code>routing qftentangled</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>路由导向优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>clifford t synthesis</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Clifford+T 综合优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>cross target qaoa</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>跨目标鲁棒优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Cryptographic</b></td>
      <td><code>AES-128 CTR</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Advanced Encryption Standard, 128-bit key, Counter mode</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>SHA-256</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 256-bit</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>SHA3-256</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 3 256-bit</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>CommunicationEngineering</b></td>
      <td><code>LDPCErrorFloor</code></td>
      <td>已完成</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>使用重要性采样估计LDPC码错误地板，针对trapping sets</td>
      <td></td>
    </tr>
    <tr>
      <td><code>PMDSimulation</code></td>
      <td>已完成</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>极化模色散（PMD）仿真</td>
      <td></td>
    </tr>
    <tr>
      <td><code>RayleighFadingBER</code></td>
      <td>已完成</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>瑞利衰落信道误码率分析，使用重要性采样模拟深衰落事件</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>EnergyStorage</b></td>
      <td><code>BatteryFastChargingProfile</code></td>
      <td>已完成</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>在电压、温升和退化约束下优化锂离子电池快充电流曲线</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>BatteryFastChargingSPMe</code></td>
      <td>已完成</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>在降阶 SPMe-T-Aging 风格的电化学、热、析锂和老化模型下优化分段快充策略</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><b>SustainableDataCenterControl</b></td>
      <td><code>hand_written_control</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>通过 unified 评测链路接入的 SustainDC 联合控制任务，覆盖负载迁移、冷却与电池调度</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="4"><b>ReactionOptimisation</b></td>
      <td><code>snar_multiobjective</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>连续流 SnAr 反应优化，在产能和废物之间做 Pareto 权衡</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>mit_case1_mixed</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>连续工艺变量与离散催化剂联合的混合变量收率最大化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>reizman_suzuki_pareto</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Reizman Suzuki 仿真器上的催化剂/工艺联合 Pareto 优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>dtlz2_pareto</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>通过 unified 评测链路接入的 DTLZ2 Pareto 前沿逼近任务</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><b>MolecularMechanics</b></td>
      <td><code>weighted_parameter_coverage</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>给定分子预算下的稀有力场参数覆盖优化</td>
      <td></td>
    </tr>
    <tr>
      <td><code>diverse_conformer_portfolio</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>低能量且高多样性的构象组合选择</td>
      <td></td>
    </tr>
    <tr>
      <td><code>torsion_profile_fitting</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>面向目标能量曲线的扭转参数缩放拟合</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="16"><b>Optics</b></td>
      <td><code>adaptive_constrained_dm_control</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>受约束可变形镜（DM）控制</td>
      <td></td>
    </tr>
    <tr>
      <td><code>adaptive_temporal_smooth_control</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>时序平滑与补偿质量折中</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>adaptive_energy_aware_control</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>能耗感知自适应光学控制</td>
      <td></td>
    </tr>
    <tr>
      <td><code>adaptive_fault_tolerant_fusion</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>容错多波前传感器融合</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_weighted_multispot_single_plane</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>单平面加权多焦点相位 DOE</td>
      <td></td>
    </tr>
    <tr>
      <td><code>phase_fourier_pattern_holography</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>傅里叶图案全息</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_dammann_uniform_orders</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dammann 光栅均匀级次优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_large_scale_weighted_spot_array</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>大规模加权焦点阵列合成</td>
      <td></td>
    </tr>
    <tr>
      <td><code>fiber_wdm_channel_power_allocation</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>WDM 信道与发射功率分配</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>fiber_mcs_power_scheduling</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>MCS 与功率联合调度</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>fiber_dsp_mode_scheduling</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>接收端 DSP 模式调度</td>
      <td></td>
    </tr>
    <tr>
      <td><code>fiber_guardband_spectrum_packing</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>带保护带约束的频谱打包</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multifocus_power_ratio</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>多焦点功率比控制</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multiplane_focusing</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>多平面全息聚焦</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multispectral_focusing</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>多光谱全息聚焦</td>
      <td></td>
    </tr>
    <tr>
      <td><code>holographic_polarization_multiplexing</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>偏振复用全息</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Computer Systems</b></td>
      <td><code>Malloc Lab</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>动态内存分配实验</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>DuckDBWorkloadOptimization</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>基于 DuckDB 官方 workload 的索引/物化视图选择与查询改写优化</td>
      <td></td>
    </tr>
    <tr>
      <td><b>EngDesign</b></td>
      <td><code>CY_03, WJ_01, XY_05, AM_02, AM_03, YJ_02, YJ_03</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td><a href="https://github.com/AGI4Engineering/EngDesign.git">EngDesign</a></td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="5"><b>InventoryOptimization</b></td>
      <td><code>tree_gsm_safety_stock</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>树形多级安全库存配置（GSM）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>general_meio</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>通用拓扑 MEIO（仿真驱动目标）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>joint_replenishment</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>多 SKU 共享订货成本下的联合补货优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>finite_horizon_dp</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>有限期随机库存控制（时变策略）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>disruption_eoqd</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>供应中断场景下的 EOQ 批量优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>PyPortfolioOpt</b></td>
      <td><code>robust_mvo_rebalance</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>含行业/因子/换手约束的鲁棒均值方差再平衡</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>cvar_stress_control</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>在收益与暴露约束下进行 CVaR 压力控制配置</td>
      <td></td>
    </tr>
    <tr>
      <td><code>discrete_rebalance_mip</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>带整数手数约束的离散再平衡混合整数优化</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="7"><b>JobShop</b></td>
      <td><code>abz</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 ABZ 家族（Adams, Balas, Zawack，1988）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ft</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 FT 家族（Fisher & Thompson，1963）</td>
      <td></td>
    </tr>
    <tr>
      <td><code>la</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 LA 家族（Lawrence，1984）</td>
      <td></td>
    </tr>
    <tr>
      <td><code>orb</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 ORB 家族（Applegate & Cook，1991）</td>
      <td></td>
    </tr>
    <tr>
      <td><code>swv</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 SWV 家族（Storer、Wu、Vaccari，1992）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ta</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 TA 家族（Taillard，1993）</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>yn</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>经典 JSSP 的 YN 家族（Yamada & Nakano，1992）</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="4"><b>StructuralOptimization</b></td>
      <td><code>ISCSO2015</code></td>
      <td>已完成</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>45 杆 2D 桁架尺寸+形状优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ISCSO2023</code></td>
      <td>已完成</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>284 杆 3D 桁架尺寸优化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>TopologyOptimization</code></td>
      <td>已完成</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>连续、体积约束的合规最小化, 连续、体积约束的合规最小化</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>PyMOTOSIMPCompliance</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>基于 pyMOTO 的二维梁拓扑优化（SIMP + OC/MMA），在体积分数约束下最小化柔度</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Robotics</b></td>
      <td><code>DynamicObstacleAvoidanceNavigation</code></td>
      <td>已完成</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>在二维环境中控制差分轮机器人从起点到终点</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>QuadrupedGaitOptimization</code></td>
      <td>已完成</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>通过优化 8 个步态参数，最大化宇树 A1 仿真四足机器人的前向运动速度</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>RobotArmCycleTimeOptimization</code></td>
      <td>已完成</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>使七自由度 KUKA LBR iiwa 机械臂从起始构型运动到目标构型的轨迹时间最短，同时保证全程无碰撞</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>PIDTuning</code></td>
      <td>已完成</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>二维四旋翼在多个飞行场景下调节级联 PID 控制器</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>UAVInspectionCoverageWithWind</code></td>
      <td>已完成</td>
      <td>@MichaelCaoo</td>
      <td>@ahydchh</td>
      <td>风场扰动下的无人机巡检</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>CoFlyersVasarhelyiTuning</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>调优 Vasarhelyi 群飞参数</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Aerodynamics</b></td>
      <td><code>CarAerodynamicsSensing</code></td>
      <td>已完成</td>
      <td>@LeiDQ, @llltttwww</td>
      <td>@llltttwww</td>
      <td>3D 汽车表面传感器布局优化，用于压力场重建</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>DawnAircraftDesignOptimization</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>在巡航高度、续航与载荷约束下，联合优化机翼/机身/动力参数以最小化飞机总重量</td>
      <td></td>
    </tr>
    <tr>
      <td><b>WirelessChannelSimulation</b></td>
      <td><code>HighReliableSimulation</code></td>
      <td>已完成</td>
      <td>@tonyhaohan</td>
      <td>@yks23, @ahydchh</td>
      <td>使用重要性采样估计 Hamming(127,120) 的误码率</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><b>PowerSystems</b></td>
      <td><code>EV2GymSmartCharging</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>电动车智能充电</td>
      <td></td>
    </tr>
    <tr>
      <td><b>AdditiveManufacturing</b></td>
      <td><code>DiffSimThermalControl</code></td>
      <td>已完成</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>研究增材制造中的 differentiable simulation 工艺优化</td>
      <td></td>
    </tr>
  </tbody>
</table>

> 💡 **有新的工程问题想法？**
> 即使你暂时无法提供完整的验证代码，我们也非常欢迎你分享好的**Task 构想**！
> 请创建一个 Issue 详细描述该问题的**现实背景**与**工程价值**。经讨论确认后，我们会将其加入上表，集结社区力量共同攻克。
