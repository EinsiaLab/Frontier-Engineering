# Task Progress & Planning

English | [简体中文](TASK_PROGRESS_zh-CN.md)

This page contains the benchmark coverage table and planning notes previously embedded in the root `README.md`.

The table below lists the current coverage of domain tasks in the Benchmark. We welcome not only code contributions but also ideas for challenging new engineering problems from the community.

Note: the current effective `v1` benchmark pool contains `47` tasks. `MuonTomography` remains listed below for completeness, but is temporarily excluded from the effective `v1` pool pending objective / evaluator redesign.

<table>
  <thead>
    <tr>
      <th>Domain</th>
      <th>Task Name</th>
      <th>Status</th>
      <th>Contributor</th>
      <th>Reviewer</th>
      <th>Remarks</th>
      <th>Version</th>
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
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="2"><b>ParticlePhysics</b></td>
      <td><code>MuonTomography</code></td>
      <td>Completed</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>Muon detector placement optimization under flux, budget, and excavation constraints; temporarily excluded from the effective v1 pool pending redesign</td>
      <td></td>
    </tr>
    <tr>
      <td><code>ProtonTherapyPlanning</code></td>
      <td>Completed</td>
      <td>@SeanDF333</td>
      <td>@ahydchh</td>
      <td>IMPT dose weight optimization under tumor coverage, OAR safety, and beam cost constraints</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Kernel Engineering</b></td>
      <td><code>MLA</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>TriMul</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>FlashAttention</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>Optimize a causal scaled dot-product attention forward kernel for GPU execution</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Single Cell Analysis</b></td>
      <td><code>denoising</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Open Problems in Single-Cell Analysis</td>
      <td></td>
    </tr>
    <tr>
      <td><code>perturbation_prediction</code></td>
      <td>Completed</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>NeurIPS 2023 scPerturb</td>
      <td></td>
    </tr>
    <tr>
      <td><code>predict_modality</code></td>
      <td>Completed</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>NeurIPS 2021, RNA→ADT</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>QuantumComputing</b></td>
      <td><code>routing qftentangled</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Routing-Oriented Optimization</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>clifford t synthesis</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Clifford+T Synthesis Optimization</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>cross target qaoa</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Cross-Target Robust Optimization</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Cryptographic</b></td>
      <td><code>AES-128 CTR</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Advanced Encryption Standard, 128-bit key, Counter mode</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>SHA-256</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 256-bit</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>SHA3-256</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 3 256-bit</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>CommunicationEngineering</b></td>
      <td><code>LDPCErrorFloor</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>LDPC code error floor estimation using importance sampling for trapping sets</td>
      <td></td>
    </tr>
    <tr>
      <td><code>PMDSimulation</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>Polarization Mode Dispersion simulation with importance sampling for rare outage events</td>
      <td></td>
    </tr>
    <tr>
      <td><code>RayleighFadingBER</code></td>
      <td>Completed</td>
      <td>@WayneJin0918</td>
      <td>@ahydchh</td>
      <td>BER analysis under Rayleigh fading with importance sampling for deep fade events</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>EnergyStorage</b></td>
      <td><code>BatteryFastChargingProfile</code></td>
      <td>Completed</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>Fast-charge current-profile optimization for a lithium-ion cell under voltage, thermal, and degradation constraints</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>BatteryFastChargingSPMe</code></td>
      <td>Completed</td>
      <td>@kunkun04</td>
      <td>@ahydchh</td>
      <td>Staged fast-charge optimization under a reduced SPMe-T-Aging style electrochemical, thermal, plating, and aging model</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><b>SustainableDataCenterControl</b></td>
      <td><code>hand_written_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>SustainDC joint control benchmark for load shifting, cooling, and battery dispatch through the unified evaluation pipeline</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="4"><b>ReactionOptimisation</b></td>
      <td><code>snar_multiobjective</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Continuous-flow SnAr reaction optimization with a Pareto trade-off between productivity and waste</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>mit_case1_mixed</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Mixed-variable reaction yield maximization with continuous process settings and a categorical catalyst</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>reizman_suzuki_pareto</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Reizman Suzuki emulator Pareto optimization over catalyst choice and operating conditions</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>dtlz2_pareto</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>DTLZ2 Pareto-front approximation task integrated through the unified evaluation pipeline</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><b>MolecularMechanics</b></td>
      <td><code>weighted_parameter_coverage</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Rare force-field parameter coverage under a molecule budget</td>
      <td></td>
    </tr>
    <tr>
      <td><code>diverse_conformer_portfolio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Low-energy, high-diversity conformer portfolio selection</td>
      <td></td>
    </tr>
    <tr>
      <td><code>torsion_profile_fitting</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Force-field torsion-scale fitting against target energy profiles</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="16"><b>Optics</b></td>
      <td><code>adaptive_constrained_dm_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Constrained deformable mirror control</td>
      <td></td>
    </tr>
    <tr>
      <td><code>adaptive_temporal_smooth_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Temporal smoothness versus correction quality</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>adaptive_energy_aware_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Energy-aware adaptive optics control</td>
      <td></td>
    </tr>
    <tr>
      <td><code>adaptive_fault_tolerant_fusion</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Fault-tolerant multi-WFS fusion</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_weighted_multispot_single_plane</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Single-plane weighted multispot phase DOE</td>
      <td></td>
    </tr>
    <tr>
      <td><code>phase_fourier_pattern_holography</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Fourier pattern holography</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_dammann_uniform_orders</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dammann grating uniform diffraction orders</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>phase_large_scale_weighted_spot_array</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Large-scale weighted spot array synthesis</td>
      <td></td>
    </tr>
    <tr>
      <td><code>fiber_wdm_channel_power_allocation</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>WDM channel and launch power allocation</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>fiber_mcs_power_scheduling</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Joint MCS and power scheduling</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>fiber_dsp_mode_scheduling</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Receiver DSP mode scheduling</td>
      <td></td>
    </tr>
    <tr>
      <td><code>fiber_guardband_spectrum_packing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Spectrum packing with guard-band constraints</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multifocus_power_ratio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-focus power ratio control</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multiplane_focusing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-plane holographic focusing</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>holographic_multispectral_focusing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multispectral holographic focusing</td>
      <td></td>
    </tr>
    <tr>
      <td><code>holographic_polarization_multiplexing</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Polarization-multiplexed holography</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Computer Systems</b></td>
      <td><code>Malloc Lab</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Dynamic memory allocation</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>DuckDBWorkloadOptimization</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Index/materialized-view selection and query rewriting optimization on official DuckDB workloads</td>
      <td></td>
    </tr>
    <tr>
      <td><b>EngDesign</b></td>
      <td><code>CY_03, WJ_01, XY_05, AM_02, AM_03, YJ_02, YJ_03</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td><a href="https://github.com/AGI4Engineering/EngDesign.git">EngDesign</a></td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="5"><b>InventoryOptimization</b></td>
      <td><code>tree_gsm_safety_stock</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Tree-structured multi-echelon safety-stock placement (GSM)</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>general_meio</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>General-topology MEIO with simulation-based objective</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>joint_replenishment</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Multi-SKU joint replenishment with shared setup cost</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>finite_horizon_dp</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Finite-horizon stochastic inventory control via time-varying policy</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>disruption_eoqd</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>EOQ lot-sizing optimization under supply disruptions</td>
      <td>v1</td>
    </tr>
    <tr>
      <td rowspan="3"><b>PyPortfolioOpt</b></td>
      <td><code>robust_mvo_rebalance</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Robust mean-variance rebalancing with sector/factor/turnover constraints</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>cvar_stress_control</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>CVaR stress-controlled portfolio allocation under return and exposure constraints</td>
      <td></td>
    </tr>
    <tr>
      <td><code>discrete_rebalance_mip</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Discrete lot-constrained rebalancing with mixed-integer optimization</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="7"><b>JobShop</b></td>
      <td><code>abz</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP ABZ family (Adams, Balas, Zawack 1988)</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ft</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP FT family (Fisher and Thompson 1963)</td>
      <td></td>
    </tr>
    <tr>
      <td><code>la</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP LA family (Lawrence 1984)</td>
      <td></td>
    </tr>
    <tr>
      <td><code>orb</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP ORB family (Applegate and Cook 1991)</td>
      <td></td>
    </tr>
    <tr>
      <td><code>swv</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP SWV family (Storer, Wu, Vaccari 1992)</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ta</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP TA family (Taillard 1993)</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>yn</code></td>
      <td>Completed</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Classical JSSP YN family (Yamada and Nakano 1992)</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="4"><b>StructuralOptimization</b></td>
      <td><code>ISCSO2015</code></td>
      <td>Completed</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>45-bar 2D truss size + shape</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>ISCSO2023</code></td>
      <td>Completed</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>284-member 3D truss sizing</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>TopologyOptimization</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>MBB beam 2D topology optimization (SIMP), Continuous, volume-constrained, compliance minimization</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>PyMOTOSIMPCompliance</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>pyMOTO-based 2D beam topology optimization (SIMP + OC/MMA) under a volume-fraction constraint</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Robotics</b></td>
      <td><code>DynamicObstacleAvoidanceNavigation</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Navigate a differential-drive robot from start to goal</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>QuadrupedGaitOptimization</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Maximize the forward locomotion speed of a quadruped robot by optimizing 8 gait parameters</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>RobotArmCycleTimeOptimization</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@yks23</td>
      <td>Minimize the motion time of a 7-DOF KUKA LBR iiwa arm moving from a start to a goal configuration, collision-free</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>PIDTuning</code></td>
      <td>Completed</td>
      <td>@Geniusyingmanji</td>
      <td>@ahydchh</td>
      <td>Tune a cascaded PID controller for a 2D quadrotor across multiple flight scenarios</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>UAVInspectionCoverageWithWind</code></td>
      <td>Completed</td>
      <td>@MichaelCaoo</td>
      <td>@ahydchh</td>
      <td>UAV inspection under wind field disturbance</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>CoFlyersVasarhelyiTuning</code></td>
      <td>In Progress</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Tune the original CoFlyers Vasarhelyi flocking parameters</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Aerodynamics</b></td>
      <td><code>CarAerodynamicsSensing</code></td>
      <td>Completed</td>
      <td>@LeiDQ, @llltttwww</td>
      <td>@llltttwww</td>
      <td>Sensor placement on 3D car surface for pressure field reconstruction</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><code>DawnAircraftDesignOptimization</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Jointly optimize wing, fuselage, and propulsion variables under cruise/endurance/payload constraints to minimize total aircraft mass</td>
      <td></td>
    </tr>
    <tr>
      <td><b>WirelessChannelSimulation</b></td>
      <td><code>HighReliableSimulation</code></td>
      <td>Completed</td>
      <td>@tonyhaohan</td>
      <td>@yks23, @ahydchh</td>
      <td>BER estimation with importance sampling for Hamming(127,120)</td>
      <td>v1</td>
    </tr>
    <tr>
      <td><b>PowerSystems</b></td>
      <td><code>EV2GymSmartCharging</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Upstream-aligned EV smart charging</td>
      <td></td>
    </tr>
    <tr>
      <td><b>AdditiveManufacturing</b></td>
      <td><code>DiffSimThermalControl</code></td>
      <td>Completed</td>
      <td>@DocZbs</td>
      <td>@DocZbs</td>
      <td>Study process optimization in additive manufacturing using differentiable simulation</td>
      <td></td>
    </tr>
  </tbody>
</table>

> 💡 **Have an idea for a new engineering problem?**
> Even if you cannot provide complete verification code for now, we highly welcome you to share good **Task concepts**!
> Please create an Issue detailing the **real-world background** and **engineering value** of the problem. After discussion and confirmation, we will add it to the table above to rally community power to solve it together.
