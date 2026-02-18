# ISCSO 2015 — 45-Bar 2D Truss Size + Shape Optimization

## 1. Background

Structural optimization is a fundamental discipline in engineering that aims to design structures with minimum weight while satisfying safety constraints. The **International Student Competition in Structural Optimization (ISCSO)**, organized by Bright Optimizer, provides challenging real-world structural optimization problems annually.

The ISCSO 2015 problem is a **2D planar truss** optimization problem involving simultaneous **size optimization** (cross-sectional areas) and **shape optimization** (node positions). This combined optimization makes the problem significantly harder than pure sizing problems, as the geometry changes affect both member lengths and the structural stiffness matrix.

## 2. Problem Description

The structure is a **45-bar planar truss** spanning 20 meters, consisting of:

- **20 nodes**: 11 bottom-chord nodes (fixed geometry) and 9 top-chord nodes (variable y-coordinates)
- **45 members**: bottom chord (10), top chord (8), verticals (9), left diagonals (9), right diagonals (9)
- **2 load cases**: concentrated midspan load and distributed load

The task is to minimize the total structural weight by optimizing:
1. Cross-sectional areas of all 45 members
2. Vertical (y) coordinates of the 9 top-chord nodes

subject to stress and displacement constraints under all load cases.

### 2.1 Structural Layout

```
    11---12---13---14---15---16---17---18---19       (top chord, variable y)
   /|\  /|\  /|\  /|\  /|\  /|\  /|\  /|\  /|\
  / | \/ | \/ | \/ | \/ | \/ | \/ | \/ | \/ | \
 /  | /\ | /\ | /\ | /\ | /\ | /\ | /\ | /\ |  \
0---1----2----3----4----5----6----7----8----9---10    (bottom chord, y=0)
^                                              o
pin                                          roller
```

- Nodes 0 and 10 are supports (node 0: pinned; node 10: roller)
- Top-chord nodes 11–19 are positioned directly above bottom-chord nodes 1–9
- The y-coordinates of nodes 11–19 are design variables

## 3. Mathematical Formulation

### 3.1 Design Variables

Total dimension: **54**

```
x = [A_0, A_1, ..., A_44, y_11, y_12, ..., y_19]
```

- `A_i` (i = 0..44): cross-sectional area of member i, in mm²
- `y_j` (j = 11..19): y-coordinate of top-chord node j, in mm

### 3.2 Variable Bounds

| Variable | Lower Bound | Upper Bound | Unit |
| :--- | ---: | ---: | :--- |
| A_i (cross-sectional area) | 10.0 | 10000.0 | mm² |
| y_j (node y-coordinate) | 500.0 | 4000.0 | mm |

### 3.3 Objective Function

Minimize total structural weight:

```
W(x) = sum_{i=0}^{44} rho * L_i(x) * A_i
```

where:
- `rho = 7.86e-6 kg/mm³` (steel density)
- `L_i(x)` is the length of member i (depends on node coordinates, hence on shape variables)
- `A_i` is the cross-sectional area of member i

**Note**: Member lengths `L_i` depend on the shape variables `y_j`, making this a nonlinear objective.

### 3.4 Constraints

For **each** load case `k = 0, 1`:

1. **Stress constraints** (all members):

```
|sigma_{i,k}(x)| <= sigma_allow = 172.4 MPa,  for i = 0..44
```

2. **Displacement constraints** (all free DOFs):

```
|delta_{j,k}(x)| <= delta_allow = 50.0 mm,  for all free DOFs j
```

3. **Variable bounds** (see Section 3.2)

A solution is **feasible** if and only if ALL constraints are satisfied across ALL load cases.

## 4. Physical Model

The structure is a **2D planar truss** structure. The structural response (nodal displacements and member stresses) must be computed using appropriate structural analysis methods (e.g., finite element analysis) to evaluate constraint satisfaction.

Key physical relationships:
- Member stress depends on the applied loads, member cross-sectional areas, and structural geometry
- Nodal displacements depend on the structural stiffness, which is a function of member areas and geometry
- Both stress and displacement constraints must be satisfied simultaneously under all load cases

## 5. Problem Data

All problem data is stored in `references/problem_data.json`. Key parameters:

### 5.1 Material Properties

| Property | Value | Unit |
| :--- | ---: | :--- |
| Elastic modulus (E) | 200,000 | MPa |
| Density (rho) | 7.86e-6 | kg/mm³ |

### 5.2 Constraint Limits

| Constraint | Value | Unit |
| :--- | ---: | :--- |
| Allowable stress (sigma_allow) | 172.4 | MPa |
| Allowable displacement (delta_allow) | 50.0 | mm |

### 5.3 Load Cases

| Load Case | Description | Applied Loads |
| :--- | :--- | :--- |
| LC 0 | Concentrated midspan | 100 kN downward at node 5 |
| LC 1 | Distributed | 50 kN downward at nodes 3, 5, 7 |

### 5.4 Support Conditions

| Node | Type | Fixed DOFs |
| :--- | :--- | :--- |
| 0 | Pinned | x, y |
| 10 | Roller | y only |

## 6. Input/Output Format

### 6.1 Submission Format

The solver must output a file named `submission.json` in the working directory:

```json
{
  "benchmark_id": "iscso_2015",
  "solution_vector": [A_0, A_1, ..., A_44, y_11, y_12, ..., y_19],
  "algorithm": "<algorithm name>",
  "num_evaluations": <integer>
}
```

- `solution_vector`: array of 54 floating-point numbers
  - Indices 0–44: cross-sectional areas (mm²)
  - Indices 45–53: y-coordinates of nodes 11–19 (mm)

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

1. **Feasibility check**: All stress and displacement constraints must be satisfied (within tolerance `tol = 1e-6`) across all load cases.
2. **If infeasible**: `score = +Infinity`
3. **If feasible**: `score = W(x)` (total structural weight in kg)
4. **Ranking**: Lower score is better.

## 8. References

- ISCSO Competition: [http://www.brightoptimizer.com/](http://www.brightoptimizer.com/)
- Problem data: `references/problem_data.json`
- Evaluation code: `verification/evaluator.py`

