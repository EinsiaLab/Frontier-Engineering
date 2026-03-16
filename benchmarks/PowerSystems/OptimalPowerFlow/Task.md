# Task: DC Optimal Power Flow — IEEE 14-Bus

## Problem

Minimize total generation cost ($/h) for the IEEE 14-bus power system.

**Input** (`instance` dict):
- `base_mva` (float): 100 MVA system base
- `generators`: list of 5 dicts
  `{bus, pmin, pmax, cost_a, cost_b}` — bus number, MW limits, quadratic cost coefficients
- `buses`: list of 14 dicts
  `{id, pd}` — bus id and active load (MW)
- `branches`: list of 20 dicts
  `{from_bus, to_bus, x, rate_a}` — connectivity, per-unit reactance, thermal limit (MW)

**Output**: `list[float]` of length 5
Generator active power outputs `[Pg1, Pg2, Pg3, Pg4, Pg5]` in MW.

## Constraints

| Constraint | Formula | Tolerance |
|-----------|---------|-----------|
| Power balance | `\|sum(Pg) - 259\| ≤ 0.5 MW` | hard |
| Generation limits | `pmin_i ≤ Pg_i ≤ pmax_i` | hard |
| DC line flows | `\|P_ij\| ≤ rate_a_ij` via `B·θ = P_net/base_mva` | hard |

Any constraint violation → `valid = 0`, `score = 0`.

## Scoring

```
score = min(1.0, 7892.76 / cost)
```

where `cost = sum(cost_a[i] * Pg[i]^2 + cost_b[i] * Pg[i])` $/h.

- Human best (DC-OPF optimal): **7 892.76 $/h → score 1.000**
- Baseline (equal dispatch):   **9 154.77 $/h → score 0.862**

## Key Insight

Branch 2→3 has a **thermal limit of 36 MW** (binding at optimum). A naive
merit-order dispatch (send all cheap power from bus 1 or 2) will overload
this line and result in an infeasible solution. The optimal dispatch must
balance cost minimisation with the network constraint imposed by line 2→3.

## Hints

- The DC power flow equation is: `B · θ = P_net / base_mva`
  where `B[i,i] = sum(1/x_ij)`, `B[i,j] = -1/x_ij`
- Bus 1 (index 0) is the slack bus: `θ[0] = 0` (remove row/col when solving)
- Active power flow on branch i→j: `P_ij = (θ_i - θ_j) / x_ij * base_mva`
- Use `scipy.optimize.minimize` with SLSQP for the constrained QP
