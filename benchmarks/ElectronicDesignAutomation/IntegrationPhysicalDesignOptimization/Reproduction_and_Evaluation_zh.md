# ICCAD 2025 Problem C 复现与评测指南（基于当前 data 目录）

## 1. 目标

本文聚焦目录：

- `benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data/ICCAD25_test_case`
- `benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data/ICCAD_problem_C_top-3_result`

给出两件事：

1. 如何复现（重放）已有结果。
2. 如何做可比的评测（baseline vs solution）。

## 2. data 目录结构分析

### 2.1 数据总览

- `ICCAD25_test_case`：约 `951M`
- `ICCAD_problem_C_top-3_result`：约 `846M`

### 2.2 testcase 组成

`ICCAD25_test_case` 下包含：

- `ASAP7/`：工艺库（`LIB/`, `LEF/`, `techlef/`, `setRC.tcl`）
- `ICCAD25_testcases/`：公开 testcase
- `openRoad_eval_script/eval_def.tcl`：OpenROAD 评测示例脚本（当前写死 `aes_cipher_top`）
- `ICCAD_ProbC_ENV/`：Docker + conda 环境示例

公开 testcase 设计名（6 个）：

- `ac97_top`
- `aes`
- `aes_cipher_top`
- `ariane`
- `des`
- `pci_bridge32`

每个 testcase 目录都包含 `.def/.v/.sdc/.spef/.ref.def` 等关键文件（可直接做 baseline 与 solution 对比）。

### 2.3 top-3 结果包组成

`ICCAD_problem_C_top-3_result/` 下有 3 个队伍目录：

- `cadc1001`
- `cadc1007`
- `cadc1051`

每个队伍都包含：

- `13` 个 `*.sol.def`
- `13` 个 `*.sol.changelist`

top-3 的 13 个设计名为：

- `NV_NVDLA_partition_c`, `ac97_top`, `aes`, `aes_cipher_top`, `ariane`, `des`, `des3`, `fpu`, `mc_top`, `mempool_tile_wrap`, `netcard_fast`, `pci_bridge32`, `tv80s`

与公开 testcase 的交集只有 6 个：

- `ac97_top`, `aes`, `aes_cipher_top`, `ariane`, `des`, `pci_bridge32`

结论：当前仓库仅能对这 6 个设计做“完整可复现评测”（因为其余 7 个缺少对应 baseline testcase）。

### 2.4 可复现 6 例的规模信息（来自 DEF 的 `COMPONENTS`）

| design | baseline components | cadc1001 delta | cadc1007 delta | cadc1051 delta |
| --- | ---: | ---: | ---: | ---: |
| ac97_top | 7750 | 0 | +64 | 0 |
| aes | 4393 | 0 | +32 | 0 |
| aes_cipher_top | 11630 | 0 | +64 | 0 |
| ariane | 105730 | +56 | 0 | 0 |
| des | 2317 | 0 | +32 | 0 |
| pci_bridge32 | 12091 | 0 | +64 | 0 |

说明：

- `delta = sol.def components - baseline components`。
- 可据此快速判断是否发生了 buffer 插入等新增实例行为。

### 2.5 changelist 操作类型统计（可复现 6 例）

| team | insert_buffer | size_cell |
| --- | ---: | ---: |
| cadc1001 | 56 | 120895 |
| cadc1007 | 256 | 261 |
| cadc1051 | 0 | 143911 |

## 3. 复现策略（建议）

### 3.1 推荐策略 A：直接评测 `sol.def`

这是最稳妥也最接近“结果复现”的方式：

1. 用 testcase 的库与约束（LIB/LEF/SDC）。
2. 替换 DEF 为 `*.sol.def`。
3. 运行 `report_tns/report_wns/report_power`。
4. 与 baseline DEF（`<design>.def`）对比。

优点：

- 不依赖 changelist 解释器差异。
- 直接验证最终布局结果。

### 3.2 策略 B：从 baseline 重放 `sol.changelist`

如果你要验证 ECO 操作本身，可尝试：

1. 读 baseline DEF。
2. `source <design>.sol.changelist`。
3. `detailed_placement`/`check_placement`（必要时）。
4. `write_def` 后再评测。

注意：不同 OpenROAD 版本对 `insert_buffer/size_cell` 细节可能有差异。若重放失败，以策略 A 为主。

## 4. 环境准备

### 4.1 OpenROAD 可执行路径

先确认你自己安装的 OpenROAD（你已安装）：

```bash
which openroad
openroad -version
```

如果 `which openroad` 为空，请用绝对路径，例如：

```bash
/your/openroad/path/openroad -version
```

### 4.2 当前仓库内二进制说明

仓库里有：

- `benchmarks/ElectronicDesignAutomation/install/OpenROAD/bin/openroad`

该二进制在本机可输出版本，但脚本执行时可能遇到 Tcl runfiles/版本问题。建议优先用你已正确安装并可跑 Tcl 的 OpenROAD。

## 5. 单个设计评测（baseline vs top3）

下面给出可复用 Tcl（不要用 data 里写死路径的 `eval_def.tcl`，改用参数化版本）。

### 5.1 参数化评测脚本

新建 `eval_any.tcl`（放哪都可以）：

```tcl
set base "/DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data/ICCAD25_test_case"
set design $::env(DESIGN)
set def_path $::env(DEF_PATH)
set tcdir "$base/ICCAD25_testcases/$design"

foreach libFile [glob "$base/ASAP7/LIB/*nldm*.lib"] {
  read_liberty $libFile
}
read_lef "$base/ASAP7/techlef/asap7_tech_1x_201209.lef"
foreach lef [glob "$base/ASAP7/LEF/*.lef"] {
  read_lef $lef
}

read_def $def_path
read_sdc "$tcdir/$design.sdc"
source "$base/ASAP7/setRC.tcl"
estimate_parasitics -placement

report_tns
report_wns
report_power
```

### 5.2 baseline 跑法

```bash
DESIGN=aes \
DEF_PATH=/DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data/ICCAD25_test_case/ICCAD25_testcases/aes/aes.def \
openroad eval_any.tcl | tee logs/aes.baseline.log
```

### 5.3 top3 结果跑法

```bash
DESIGN=aes \
DEF_PATH=/DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data/ICCAD_problem_C_top-3_result/cadc1007/aes.sol.def \
openroad eval_any.tcl | tee logs/aes.cadc1007.log
```

### 5.4 指标提取

```bash
rg -n "tns|wns|leakage|power" logs/aes.*.log -i
```

## 6. 批量评测（6 个可对齐设计）

### 6.1 可直接评测的设计集合

- `ac97_top aes aes_cipher_top ariane des pci_bridge32`

### 6.2 批量脚本示例

```bash
mkdir -p logs
DESIGNS="ac97_top aes aes_cipher_top ariane des pci_bridge32"
TEAMS="cadc1001 cadc1007 cadc1051"
BASE=/DATA_EDS2/haohan.chi.2311/Frontier-Engineering/benchmarks/ElectronicDesignAutomation/IntegrationPhysicalDesignOptimization/data

for d in $DESIGNS; do
  DESIGN=$d \
  DEF_PATH=$BASE/ICCAD25_test_case/ICCAD25_testcases/$d/$d.def \
  openroad eval_any.tcl > logs/${d}.baseline.log 2>&1

  for t in $TEAMS; do
    DESIGN=$d \
    DEF_PATH=$BASE/ICCAD_problem_C_top-3_result/$t/${d}.sol.def \
    openroad eval_any.tcl > logs/${d}.${t}.log 2>&1
  done
done
```

## 7. 评分计算建议（按题面公式）

你可以先从日志拿到每个设计的：

- `TNS`
- `Leakage`（或 `Total Leakage`）
- `Area`（若当前脚本没输出，可追加 `report_design_area`）
- `WNS`（用于硬约束检查）

再按 `Task_ch.md` 中公式计算 `Reward - Penalty`，并额外加上位移惩罚项（若你统计位移）。

实践上建议先做一个 CSV：

- 列：`design,team,tns,wns,leakage,area,score`
- 行：baseline + 每个 top3

## 8. 我对 top3 包的额外观察（便于你复现实验）

在“可对齐的 6 个设计”上：

- `cadc1051`：changelist 全是 `size_cell`，无 `insert_buffer`。
- `cadc1007`：`insert_buffer + size_cell`，ECO 条目较少。
- `cadc1001`：部分设计 changelist 很长，也有个别设计 changelist 为空（例如 `aes_cipher_top`, `des`）。

这意味着你做复现时可以分两类实验：

1. `sol.def` 直接评测（最稳定）。
2. changelist 重放一致性验证（更偏工具链兼容性测试）。

## 9. 常见问题

- `openroad: command not found`
  - 用绝对路径调用你已安装的 OpenROAD。
- `eval_def.tcl` 只能跑 `aes_cipher_top`
  - 原脚本写死路径，请改成本文的参数化版本。
- top3 某些设计无法评测
  - 当前 data 中缺少这 7 个设计的 testcase baseline（`tv80s/des3/fpu/...`）。
- `sol.def` 和 `sol.changelist` 不一致时怎么办
  - 先以 `sol.def` 的评测结果为准；changelist 主要用于重放/审计。
