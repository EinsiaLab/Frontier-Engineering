# Engineering Bench Case 2

## Problem Background

2025 Integrated Circuit Computer-Aided Design (EDA) — Physical Design Optimization

## 1. Background

As semiconductor processes enter the nanometer scale (e.g., 7nm and below), the design complexity of Very Large Scale Integration (VLSI) circuits grows exponentially. In the **Late-stage Placement Optimization** phase of physical design, the trade-offs between **Timing**, **Power**, and **Area** become extremely rigorous. Traditional CPU-based heuristic algorithms often fall into local optima when handling discrete combinatorial systems containing millions of nodes and struggle to find global optimal solutions within a limited time.

To break through this bottleneck, utilizing modern High-Performance Computing (HPC) technologies (such as GPU acceleration and differentiable programming) for circuit optimization has become a research hotspot in both industry and academia. Based on "Incremental Placement Optimization," this task requires participants to repair timing violations and reduce static power consumption on **Million-gate** scale chip netlists using techniques such as logic gate sizing, buffer insertion, and cell movement under strict physical constraints.

## 2. Problem Description

To simulate a real industrial scenario, the task provides a chip design that has completed placement and routing (Place & Route) but still suffers from timing dissatisfaction or excessive power consumption. Participants need to develop an optimization program to perform "minimally invasive surgery" on the circuit.

### 2.1 Main Task Flow

* **Load Design:** Read circuit netlists and physical libraries based on industry-standard formats (Verilog, DEF, LEF, Liberty).
* **Incremental Optimization:** Optimize the circuit's PPA (Power, Performance, Area) metrics by adjusting gate sizes, inserting buffers, or moving cell positions while maintaining the logical function of the circuit.
* **Legalization & Output:** Ensure all modifications are physically legal (no overlaps, aligned to the grid) and output the final placement file (DEF) and Engineering Change Order (ECO).

### 2.2 Detailed Process Explanation

* **A. Gate Sizing:**
Replace instances in the netlist with cells from the library that have the same function but different driving capabilities (e.g., changing from X1 to X4).
*Constraint:* New and old cells must be **Footprint Compatible** and come from the same logic library.
* **B. Buffer Insertion:**
Insert buffers on interconnects to decouple the capacitive load of long wires or fix timing.
*Constraint:* Inserted buffers must have physical space for placement and must not cause severe local congestion.
* **C. Cell Relocation:**
Allow micro-tuning of cell coordinates  within a local range.
*Note:* Sizing often causes area changes, leading to overlaps with surrounding cells. These overlaps must be eliminated by moving surrounding cells (domino effect) to achieve **Legalization**.

## 3. Design Environment and Input Definitions

**Process Node:** Uses ASAP7 PDK (7nm predictive process library), including real Multi-Threshold Voltage (RVT, LVT, SLVT) cells.

**Input Files:**

* **Verilog Netlist (.v):** Logical connection relationships of the circuit (The Skeleton).
* **DEF (.def):** Physical layout information of the circuit, including coordinates, orientation, and placement status of all cells (The Body).
* **Liberty Library (.lib):** Timing and power models of standard cells (The Laws of Physics).
* **Parasitics (.spef):** Parasitic resistance and capacitance parameters of interconnects.

## 4. Physical Models and Calculation Formulas

This task involves Static Timing Analysis (STA) and power calculation, adhering to the following physical models.

### 4.1 Delay Model

Circuit delay consists of gate delay and interconnect delay.

**Interconnect Delay:** Uses the Elmore delay model. For a wire with resistance  and capacitance , driving a load , the delay is approximated as:

**Gate Delay (NLDM):** Based on the Non-Linear Delay Model (NLDM), calculated via two-dimensional lookup tables. Gate delay  and output slew  are functions of input slew  and output load capacitance :

### 4.2 Power Model

Focuses on **Static Leakage Power**, calculated as follows:

Where  has an exponential relationship with the transistor threshold voltage . The optimization goal is to replace Low  (high power, fast) gates with High  (low power, slow) gates.

### 4.3 Physical Displacement

The average displacement caused by incremental optimization is defined as:

## 5. Scoring Standards (Cost Function)

The benchmark score consists of a **Reward** term and a **Penalty** term.

### 5.1 Reward Calculation

The reward is based on the magnitude of optimization relative to the initial baseline.

* **TNS (Total Negative Slack):** The sum of all timing violations in the circuit (Primary optimization target).
* **Lkg (Leakage):** Total leakage power.
* **Area:** Total chip area.

### 5.2 Revised Scoring Formula (ICCAD 2025)

To limit excessive layout changes, the actual scoring introduces a displacement penalty:

## 6. Constraints

* **Timing Constraints:** **Setup Time:** . The optimized WNS (Worst Negative Slack) and TNS must improve or at least not degrade.
* **Physical Legalization:** All standard cells must be placed on predefined **Rows**. Cell coordinates must be integer multiples of the Site width. Overlapping between cells is strictly prohibited.
* **Design Rule Check (DRC):**
* **Max Capacitance:** The load capacitance of a node cannot exceed the library-defined upper limit.
* **Max Slew:** Signal transition time cannot exceed the upper limit.
* **Penalty:** 50 points are deducted for each DRC violation.


* **Runtime Limit:** The optimization process must be completed within the specified time budget (e.g., 1 hour for corresponding circuit scales). Timeouts will result in the inability to obtain a full score.

## 7. Result File Format

To verify the correctness of the calculation results, the participant's program must output the following standard format files.

### 7.1 Output Files

* **Solution DEF (.def):** A complete layout file containing the final positions and connection relationships of all cells after optimization.
* **ECO Changelist:** A text list recording all netlist changes (Resize, Insert Buffer, Move) for replay verification.

### 7.2 Verification Metrics Description

The scoring script will parse reports generated by OpenROAD/OpenSTA tools, focusing on the following metrics:

| Metric | Unit | Description | Target |
| --- | --- | --- | --- |
| **WNS** | ns | Worst Negative Slack |  ns |
| **TNS** | ns | Total Negative Slack | Eliminate > 98% |
| **Leakage Power** | mW | Static Leakage Power | Reduce 8-10% |
| **Avg Displacement** |  | Average Cell Displacement | As small as possible (< 5 ) |
| **DRC Violations** | Count | Design Rule Violations | 0 |

**Note:** If the output DEF file cannot be correctly parsed by OpenROAD, or if there are unfixed physical overlaps (Illegal Placement), the score for that test case will be directly recorded as **0 points**.