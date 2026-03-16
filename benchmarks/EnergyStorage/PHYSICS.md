# EnergyStorage Physics Notes

This document explains the physical background, reduced-order modeling choices, and scoring formulas used by the `EnergyStorage` benchmarks.

## What "Reduced-Order" Means

Both tasks in this directory are more complex than a pure rule-based or black-box optimization toy problem, but they are still **reduced-order** models relative to full electrochemical PDE solvers.

In this repository, "reduced-order" means:

- the model keeps only the states that matter most for control and optimization,
- the model uses ODE-like updates instead of solving full spatial PDEs online,
- the model is fast enough to be evaluated hundreds or thousands of times inside AI-agent search loops.

So "reduced-order" does **not** mean "simple" or "unrealistic". It means "physics-informed but computationally tractable".

## Task 1: BatteryFastChargingProfile

### Physical Intent

`BatteryFastChargingProfile` is the lighter-weight task. It is designed to sit between:

- a pure equivalent-circuit model,
- and a more detailed electrochemical fast-charging model.

It keeps the following control-relevant effects:

- nonlinear OCV rise with SOC,
- SOC- and temperature-dependent ohmic resistance,
- polarization buildup,
- concentration-gradient proxy,
- lumped temperature rise,
- plating-risk proxy,
- aging-loss proxy.

### State Update Structure

The model evolves:

- `soc`
- `temp_c`
- `eta_pol_v`
- `eta_diff_v`
- `plating_loss_ah`
- `aging_loss_ah`

The terminal voltage is:

```math
V = U_{\mathrm{ocv}}(z) + I R_0(z, T) + \eta_{\mathrm{pol}} + \eta_{\mathrm{diff}}
```

where `z` is SOC.

The OCV function is a smooth nonlinear curve:

```math
U_{\mathrm{ocv}}(z)=
a_0 + a_1 z
+ a_2 \tanh(k_1(z-c_1))
+ a_3 \tanh(k_2(z-c_2))
```

The ohmic term is:

```math
R_0(z,T)=
r_0
+ r_{\mathrm{high}} \max(0, z-z_h)^2
+ r_{\mathrm{low}} \max(0, z_l-z)^2
+ r_T \max(0, T_{\mathrm{amb}}-T)
```

The polarization states use first-order relaxation:

```math
\dot{\eta}_{\mathrm{pol}} = \frac{\eta_{\mathrm{pol,target}}-\eta_{\mathrm{pol}}}{\tau_{\mathrm{pol}}}
```

```math
\dot{\eta}_{\mathrm{diff}} = \frac{\eta_{\mathrm{diff,target}}-\eta_{\mathrm{diff}}}{\tau_{\mathrm{diff}}}
```

### Plating Proxy

The task does not solve a full anode potential model. Instead it uses an anode safety margin:

```math
m_{\mathrm{anode}} =
m_0
+ k_z(1-z)
- k_I I_C
- k_d \eta_{\mathrm{diff}}
- k_T \max(0, T_{\mathrm{amb}}-T)
```

where `I_C` is the charging rate in C.

The plating drive is:

```math
p = \max(0, -m_{\mathrm{anode}})
```

and plating loss accumulates as:

```math
\Delta Q_{\mathrm{plating}} \propto I \, p \, \Delta t
```

### Thermal Model

The lumped thermal model is:

```math
C_{\mathrm{th}} \dot{T}
=
I^2 R_0 + k_h I |\eta_{\mathrm{pol}}+\eta_{\mathrm{diff}}|
- h(T-T_{\mathrm{amb}})
```

### Aging Proxy

The aging term is a stress surrogate:

```math
\dot{Q}_{\mathrm{aging}}
\propto
\left(1 + k_I I_C^\alpha\right)
\exp\left(k_T \max(0, T-T_{\mathrm{amb}})\right)
+ k_p Q_{\mathrm{plating}}
```

### Scoring

The final score is:

```math
\mathrm{time\_score}
=
\exp\left(
-\frac{t_{\mathrm{charge}}-t_{\mathrm{ref}}}{t_{\mathrm{scale}}}
\right)
```

```math
\mathrm{degradation\_score}
=
\exp\left(
-k_p Q_{\mathrm{plating}}
-k_a Q_{\mathrm{aging}}
\right)
```

```math
\mathrm{thermal\_score}
=
\exp\left(
-\frac{\max(0, T_{\max}-T_{\mathrm{ref}})}{T_{\mathrm{scale}}}
\right)
```

```math
\mathrm{voltage\_score}
=
\exp\left(
-\frac{\max(0, V_{\max}-V_{\mathrm{soft}})}{V_{\mathrm{scale}}}
\right)
```

```math
\mathrm{combined\_score}
=
S\cdot
\left(
w_t \,\mathrm{time\_score}
+ w_d \,\mathrm{degradation\_score}
+ w_h \,\mathrm{thermal\_score}
+ w_v \,\mathrm{voltage\_score}
\right)
```

All coefficients come from:

- `BatteryFastChargingProfile/references/battery_config.json`

## Task 2: BatteryFastChargingSPMe

### Physical Intent

`BatteryFastChargingSPMe` is the more physically structured task. It is still reduced-order, but it is closer to a control-oriented electrochemical model than the first task.

It keeps:

- negative-electrode average and surface stoichiometry,
- positive-electrode average and surface stoichiometry,
- electrolyte polarization,
- Butler-Volmer-style kinetic overpotential proxies,
- temperature-coupled electrochemical dynamics,
- plating margin,
- SEI-like aging loss.

### State Variables

The main dynamic states are:

- `theta_n_avg`
- `theta_p_avg`
- `delta_theta_n`
- `delta_theta_p`
- `electrolyte_state`
- `temp_c`
- `plating_loss_ah`
- `aging_loss_ah`

The average stoichiometries are mapped from SOC:

```math
\theta_n = \theta_{n,\min} + z(\theta_{n,\max}-\theta_{n,\min})
```

```math
\theta_p = \theta_{p,\max} - z(\theta_{p,\max}-\theta_{p,\min})
```

### Surface-State Dynamics

Surface deviations evolve as first-order reduced diffusion states:

```math
\dot{\delta \theta_n}
=
k_n I - \frac{\delta \theta_n}{\tau_n(T)}
```

```math
\dot{\delta \theta_p}
=
-k_p I - \frac{\delta \theta_p}{\tau_p(T)}
```

where the diffusion time constants are temperature-coupled by Arrhenius scaling:

```math
\Psi(T)=\Psi_{\mathrm{ref}}
\exp\left(
\frac{E_a}{R}
\left(
\frac{1}{T_{\mathrm{ref}}}-\frac{1}{T}
\right)
\right)
```

### Electrolyte Polarization

Electrolyte polarization is modeled as:

```math
\dot{\phi}_e
=
k_e I - \frac{\phi_e}{\tau_e(T)}
```

### OCV and Kinetic Overpotential

The positive and negative OCV curves are nonlinear functions of surface stoichiometry:

```math
U_p(\theta_p) =
b_0 + b_1 \theta_p
+ b_2 \tanh(g_1(\theta_p-c_1))
+ b_3 \tanh(g_2(\theta_p-c_2))
+ b_T (T-T_{\mathrm{ref}})
```

```math
U_n(\theta_n) =
d_0 + d_1 \theta_n
+ d_2 \tanh(h_1(\theta_n-s_1))
+ d_3 \tanh(h_2(\theta_n-s_2))
+ d_T (T-T_{\mathrm{ref}})
```

The exchange-current surrogates are:

```math
i_{0,n} \propto s_k(T)\sqrt{\theta_{n,\mathrm{surf}}(1-\theta_{n,\mathrm{surf}})} f_e
```

```math
i_{0,p} \propto s_k(T)\sqrt{\theta_{p,\mathrm{surf}}(1-\theta_{p,\mathrm{surf}})} f_e
```

with electrolyte factor:

```math
f_e = \max(0.2,\; 1-k_e |\phi_e|)
```

The kinetic overpotentials use a Butler-Volmer-inspired `asinh` form:

```math
\eta_n = \frac{2RT}{F}\operatorname{asinh}\left(\frac{I}{2i_{0,n}}\right)
```

```math
\eta_p = \frac{2RT}{F}\operatorname{asinh}\left(\frac{I}{2i_{0,p}}\right)
```

### Terminal Voltage

The terminal voltage is:

```math
V = (U_p-U_n) + \eta_p + \eta_n + I R_{\mathrm{ohm}} + \Phi_e
```

where:

```math
\Phi_e = k_{\phi} \phi_e
```

and the ohmic term is SOC- and temperature-dependent.

### Plating Margin

The reduced plating margin is:

```math
m_{\mathrm{plating}}
=
U_n
- \eta_n
- \frac{1}{2}\Phi_e
- k_T \max(0, 25-T)
```

This is not a full local anode-potential solver, but it plays the same optimization role:

- smaller margin means more plating risk,
- negative enough margin invalidates the candidate.

Plating loss is accumulated from soft-margin stress:

```math
s_{\mathrm{plating}}=\max(0, m_{\mathrm{soft}}-m_{\mathrm{plating}})
```

```math
\Delta Q_{\mathrm{plating}}
\propto
I \, s_{\mathrm{plating}}^{\alpha} \Delta t
```

### Thermal Model

The heat generation term combines irreversible and entropy-like heat:

```math
Q_{\mathrm{gen}}
=
|I(V-V_{\mathrm{oc}})|
+ \gamma |I T \Delta S|
```

and temperature evolves as:

```math
C_{\mathrm{th}} \dot{T}
=
Q_{\mathrm{gen}} - hA(T-T_{\mathrm{amb}})
```

### Aging Model

The SEI-like side-reaction proxy is:

```math
\dot{Q}_{\mathrm{aging}}
=
k_{\mathrm{sei,ref}} s_{\mathrm{sei}}(T)
\exp\left(
k_{\mathrm{stress}}
\max(0, m_{\mathrm{sei}}-m_{\mathrm{plating}})
\right)
```

This makes low plating margin and higher temperature more expensive even if the profile is still feasible.

### Scoring

The scores are:

```math
\mathrm{time\_score}
=
\exp\left(
-\frac{t_{\mathrm{charge}}-t_{\mathrm{ref}}}{t_{\mathrm{scale}}}
\right)
```

```math
\mathrm{aging\_score}
=
\exp(-k_a Q_{\mathrm{aging}})
```

```math
\mathrm{plating\_score}
=
\exp(-k_p Q_{\mathrm{plating}})
```

```math
\mathrm{thermal\_score}
=
\exp\left(
-\frac{\max(0, T_{\max}-T_{\mathrm{ref}})}{T_{\mathrm{scale}}}
\right)
```

```math
\mathrm{voltage\_score}
=
\exp\left(
-\frac{\max(0, V_{\max}-V_{\mathrm{soft}})}{V_{\mathrm{scale}}}
\right)
```

```math
\mathrm{combined\_score}
=
S\cdot
\left(
w_t \,\mathrm{time\_score}
+ w_a \,\mathrm{aging\_score}
+ w_p \,\mathrm{plating\_score}
+ w_h \,\mathrm{thermal\_score}
+ w_v \,\mathrm{voltage\_score}
\right)
```

All coefficients come from:

- `BatteryFastChargingSPMe/references/battery_config.json`

## Practical Interpretation

The difference between the two tasks is:

- `BatteryFastChargingProfile`: faster, lighter, closer to an enhanced ECM-plus-proxy benchmark.
- `BatteryFastChargingSPMe`: more structured, closer to a control-oriented electrochemical benchmark.

Both are meant for AI-agent optimization, not for chemistry identification or full PDE benchmarking.
