# ISCSO 2023 — 284-Member 3D Truss Sizing Optimization

## 1. Background

The **International Student Competition in Structural Optimization (ISCSO)** 2023 presents a large-scale 3D space truss sizing optimization problem. Unlike the 2015 problem which includes shape optimization, ISCSO 2023 is a **pure sizing optimization** problem with a much higher dimensionality (284 design variables).

The structure is a tapered 3D tower truss with 92 nodes and 284 members, subject to three distinct load cases representing wind and gravity loading. This problem tests an optimizer's ability to handle:

- High-dimensional continuous design spaces
- Multiple load cases with different constraint-critical members
- Coupling between member sizes through shared nodes
- A complex non-convex feasible region

## 2. Problem Description

The structure is a **3D tower truss** with a square cross-section that tapers from bottom to top:

- **92 nodes** arranged in 23 levels (4 nodes per level)
- **284 members**: vertical columns (88), horizontal perimeter bars (92), face diagonal bracing (88), and floor cross-bracing (16)
- **3 load cases**: lateral loads in X and Y directions, and vertical load in Z direction

The task is to minimize the total structural weight by optimizing the cross-sectional area of each member, subject to stress and displacement constraints under all three load cases.

### 2.1 Structural Layout

The tower has 23 levels (0 to 22) along the Z-axis. At each level, 4 nodes form a square cross-section. The cross-section width tapers linearly from 4000 mm at the base to 1000 mm at the top.

```
Level 22 (top):    o---o     Height: 40000 mm
                   |   |     Width: 1000 mm
                   o---o

     ...           tapers linearly

Level 0 (base):    O---O     Height: 0 mm
                   |   |     Width: 4000 mm
                   O---O
                   (pinned supports)
```

Member groups:
- **Vertical columns** (88): connect corresponding nodes between adjacent levels
- **Horizontal perimeter** (92): form the square frame at each level
- **Face diagonals** (88): one diagonal brace per face per bay
- **Floor cross-diagonals** (16): cross-bracing at selected levels (1, 4, 7, 10, 13, 16, 19, 22)

## 3. Mathematical Formulation

### 3.1 Design Variables

Total dimension: **284**

```
x = [A_0, A_1, ..., A_283]
```

- `A_i` (i = 0..283): cross-sectional area of member i, in mm²

### 3.2 Variable Bounds

| Variable | Lower Bound | Upper Bound | Unit |
| :--- | ---: | ---: | :--- |
| A_i (cross-sectional area) | 10.0 | 20000.0 | mm² |

### 3.3 Objective Function

Minimize total structural weight:

```
W(x) = sum_{i=0}^{283} rho * L_i * A_i
```

where:
- `rho = 7.86e-6 kg/mm³` (steel density)
- `L_i` is the length of member i (fixed, determined by geometry)
- `A_i` is the cross-sectional area of member i

**Note**: Unlike ISCSO 2015, all member lengths are fixed (no shape variables), making the objective linear in the design variables. However, the constraints are nonlinear.

### 3.4 Constraints

For **each** load case `k = 0, 1, 2`:

1. **Stress constraints** (all members):

```
|sigma_{i,k}(x)| <= sigma_allow = 248.2 MPa,  for i = 0..283
```

2. **Displacement constraints** (all free DOFs):

```
|delta_{j,k}(x)| <= delta_allow = 10.0 mm,  for all free DOFs j
```

3. **Variable bounds** (see Section 3.2)

A solution is **feasible** if and only if ALL constraints are satisfied across ALL load cases.

## 4. Physical Model

The structure is a **3D space truss** structure. The structural response (nodal displacements and member stresses) must be computed using appropriate structural analysis methods (e.g., finite element analysis) to evaluate constraint satisfaction.

Key physical relationships:
- Member stress depends on the applied loads, member cross-sectional areas, and structural geometry
- Nodal displacements depend on the structural stiffness, which is a function of member areas
- Both stress and displacement constraints must be satisfied simultaneously under all load cases

## 5. Problem Data

All problem data is stored in `references/problem_data.json`. The topology is generated parametrically from the tower description.

### 5.1 Material Properties

| Property | Value | Unit |
| :--- | ---: | :--- |
| Elastic modulus (E) | 200,000 | MPa |
| Density (rho) | 7.86e-6 | kg/mm³ |

### 5.2 Constraint Limits

| Constraint | Value | Unit |
| :--- | ---: | :--- |
| Allowable stress (sigma_allow) | 248.2 | MPa |
| Allowable displacement (delta_allow) | 10.0 | mm |

### 5.3 Load Cases

| Load Case | Description | Applied Loads |
| :--- | :--- | :--- |
| LC 0 | Lateral X-direction wind | 12 kN total in +X at top 4 nodes (3 kN each) |
| LC 1 | Lateral Y-direction wind | 12 kN total in +Y at top 4 nodes (3 kN each) |
| LC 2 | Vertical gravity load | 15 kN total in -Z at top 4 nodes (3.75 kN each) |

### 5.4 Support Conditions

| Nodes | Type | Fixed DOFs |
| :--- | :--- | :--- |
| 0, 1, 2, 3 (level 0) | Pinned | x, y, z (all) |

### 5.5 Tower Geometry Parameters

| Parameter | Value | Unit |
| :--- | ---: | :--- |
| Number of levels | 23 | - |
| Total height | 40,000 | mm |
| Bottom width | 4,000 | mm |
| Top width | 1,000 | mm |
| Cross-bracing levels | 1, 4, 7, 10, 13, 16, 19, 22 | - |

## 6. Input/Output Format

### 6.1 Submission Format

The solver must output a file named `submission.json` in the working directory:

```json
{
  "benchmark_id": "iscso_2023",
  "solution_vector": [A_0, A_1, ..., A_283],
  "algorithm": "<algorithm name>",
  "num_evaluations": <integer>
}
```

- `solution_vector`: array of 284 floating-point numbers (cross-sectional areas in mm²)

### 6.2 Evaluation Output

The evaluator returns:

```json
{
  "objective": <weight in kg>,
  "feasible": <true/false>,
  "max_stress_violation": <max (|sigma| - sigma_allow), 0 if feasible>,
  "max_displacement_violation": <max (|delta| - delta_allow), 0 if feasible>,
  "score": <weight if feasible, else Infinity>
}
```

## 7. Scoring Criteria

1. **Feasibility check**: All stress and displacement constraints must be satisfied (within tolerance `tol = 1e-6`) across all 3 load cases.
2. **If infeasible**: `score = +Infinity`
3. **If feasible**: `score = W(x)` (total structural weight in kg)
4. **Ranking**: Lower score is better.

## 8. References

- ISCSO Competition: [http://www.brightoptimizer.com/](http://www.brightoptimizer.com/)
- Problem data: `references/problem_data.json`
- Evaluation code: `verification/evaluator.py`

