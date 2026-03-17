# EnergyStorage 物理说明

本文档统一说明 `EnergyStorage` 目录下两个任务的物理背景、降阶建模思路和评分公式。

## 任务 1：BatteryFastChargingProfile

### 物理定位

`BatteryFastChargingProfile` 是较轻量的快充任务，位于：

- 纯等效电路模型，
- 更详细的电化学快充模型

之间。

它保留了这些关键现象：

- 非线性 OCV 抬升，
- 与 SOC、温度相关的内阻，
- 极化累积，
- 浓差代理项，
- 集总温升，
- 析锂风险代理项，
- 老化损失代理项。

### 状态结构

模型内部更新的状态包括：

- `soc`
- `temp_c`
- `eta_pol_v`
- `eta_diff_v`
- `plating_loss_ah`
- `aging_loss_ah`

端电压公式为：

```math
V = U_{\mathrm{ocv}}(z) + I R_0(z, T) + \eta_{\mathrm{pol}} + \eta_{\mathrm{diff}}
```

其中 `z` 是 SOC。

OCV 用平滑非线性函数表示：

```math
U_{\mathrm{ocv}}(z)=
a_0 + a_1 z
+ a_2 \tanh(k_1(z-c_1))
+ a_3 \tanh(k_2(z-c_2))
```

内阻项为：

```math
R_0(z,T)=
r_0
+ r_{\mathrm{high}} \max(0, z-z_h)^2
+ r_{\mathrm{low}} \max(0, z_l-z)^2
+ r_T \max(0, T_{\mathrm{amb}}-T)
```

极化状态是一阶松弛过程：

```math
\dot{\eta}_{\mathrm{pol}} = \frac{\eta_{\mathrm{pol,target}}-\eta_{\mathrm{pol}}}{\tau_{\mathrm{pol}}}
```

```math
\dot{\eta}_{\mathrm{diff}} = \frac{\eta_{\mathrm{diff,target}}-\eta_{\mathrm{diff}}}{\tau_{\mathrm{diff}}}
```

### 析锂代理项

这个任务不直接解负极局部电势，而是构造一个负极安全裕度：

```math
m_{\mathrm{anode}} =
m_0
+ k_z(1-z)
- k_I I_C
- k_d \eta_{\mathrm{diff}}
- k_T \max(0, T_{\mathrm{amb}}-T)
```

其中 `I_C` 是倍率。

析锂驱动力定义为：

```math
p = \max(0, -m_{\mathrm{anode}})
```

析锂损失随之累积：

```math
\Delta Q_{\mathrm{plating}} \propto I \, p \, \Delta t
```

### 热模型

集总热模型为：

```math
C_{\mathrm{th}} \dot{T}
=
I^2 R_0 + k_h I |\eta_{\mathrm{pol}}+\eta_{\mathrm{diff}}|
- h(T-T_{\mathrm{amb}})
```

### 老化代理

老化项是一个应力代理：

```math
\dot{Q}_{\mathrm{aging}}
\propto
\left(1 + k_I I_C^\alpha\right)
\exp\left(k_T \max(0, T-T_{\mathrm{amb}})\right)
+ k_p Q_{\mathrm{plating}}
```

### 评分公式

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

所有系数来自：

- `BatteryFastChargingProfile/references/battery_config.json`

## 任务 2：BatteryFastChargingSPMe

### 物理定位

`BatteryFastChargingSPMe` 是更有物理结构的任务。它仍是降阶模型，但比前一个任务更接近面向控制的电化学模型。

它保留了：

- 负极平均与表面化学计量比，
- 正极平均与表面化学计量比，
- 电解液极化，
- Butler-Volmer 风格动力学过电势，
- 温度耦合电化学动态，
- 析锂裕度，
- SEI 风格老化损失。

### 状态变量

主要动态状态包括：

- `theta_n_avg`
- `theta_p_avg`
- `delta_theta_n`
- `delta_theta_p`
- `electrolyte_state`
- `temp_c`
- `plating_loss_ah`
- `aging_loss_ah`

平均化学计量比由 SOC 映射：

```math
\theta_n = \theta_{n,\min} + z(\theta_{n,\max}-\theta_{n,\min})
```

```math
\theta_p = \theta_{p,\max} - z(\theta_{p,\max}-\theta_{p,\min})
```

### 表面状态动力学

表面偏移量通过一阶降阶扩散状态更新：

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

扩散时间常数通过 Arrhenius 关系和温度耦合：

```math
\Psi(T)=\Psi_{\mathrm{ref}}
\exp\left(
\frac{E_a}{R}
\left(
\frac{1}{T_{\mathrm{ref}}}-\frac{1}{T}
\right)
\right)
```

### 电解液极化

电解液极化状态为：

```math
\dot{\phi}_e
=
k_e I - \frac{\phi_e}{\tau_e(T)}
```

### OCV 与动力学过电势

正负极 OCV 都是表面化学计量比的非线性函数：

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

交换电流代理为：

```math
i_{0,n} \propto s_k(T)\sqrt{\theta_{n,\mathrm{surf}}(1-\theta_{n,\mathrm{surf}})} f_e
```

```math
i_{0,p} \propto s_k(T)\sqrt{\theta_{p,\mathrm{surf}}(1-\theta_{p,\mathrm{surf}})} f_e
```

其中：

```math
f_e = \max(0.2,\; 1-k_e |\phi_e|)
```

动力学过电势采用 Butler-Volmer 风格的 `asinh` 形式：

```math
\eta_n = \frac{2RT}{F}\operatorname{asinh}\left(\frac{I}{2i_{0,n}}\right)
```

```math
\eta_p = \frac{2RT}{F}\operatorname{asinh}\left(\frac{I}{2i_{0,p}}\right)
```

### 端电压

端电压公式为：

```math
V = (U_p-U_n) + \eta_p + \eta_n + I R_{\mathrm{ohm}} + \Phi_e
```

其中：

```math
\Phi_e = k_{\phi} \phi_e
```

欧姆项还与 SOC、温度耦合。

### 析锂裕度

降阶析锂裕度定义为：

```math
m_{\mathrm{plating}}
=
U_n
- \eta_n
- \frac{1}{2}\Phi_e
- k_T \max(0, 25-T)
```

这不是完整的局部负极电势求解器，但在优化里起到相同作用：

- 裕度越小，析锂风险越高，
- 裕度足够低时直接判无效。

软边界下的析锂应力为：

```math
s_{\mathrm{plating}}=\max(0, m_{\mathrm{soft}}-m_{\mathrm{plating}})
```

析锂损失累积为：

```math
\Delta Q_{\mathrm{plating}}
\propto
I \, s_{\mathrm{plating}}^{\alpha} \Delta t
```

### 热模型

热源由不可逆热和熵项近似组成：

```math
Q_{\mathrm{gen}}
=
|I(V-V_{\mathrm{oc}})|
+ \gamma |I T \Delta S|
```

温度演化为：

```math
C_{\mathrm{th}} \dot{T}
=
Q_{\mathrm{gen}} - hA(T-T_{\mathrm{amb}})
```

### 老化模型

SEI 风格副反应代理为：

```math
\dot{Q}_{\mathrm{aging}}
=
k_{\mathrm{sei,ref}} s_{\mathrm{sei}}(T)
\exp\left(
k_{\mathrm{stress}}
\max(0, m_{\mathrm{sei}}-m_{\mathrm{plating}})
\right)
```

因此，即使策略还可行，只要析锂裕度过小、温度过高，也会被老化项惩罚。

### 评分公式

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

所有系数来自：

- `BatteryFastChargingSPMe/references/battery_config.json`

## 实际理解方式

两个任务的区别可以简单理解为：

- `BatteryFastChargingProfile`：更轻、更快，更接近“增强版 ECM + 代理项”。
- `BatteryFastChargingSPMe`：更有物理结构，更接近“面向控制的降阶电化学模型”。

两者都不是化学参数辨识任务，也不是完整 PDE 求解 benchmark，而是面向 AI agent 快充策略优化的工程任务。
