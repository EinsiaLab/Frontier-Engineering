# Reliable Communication Simulation (ReliableSim)

ReliableSim是一个用于可靠通信系统模拟的Python工具包，主要用于评估误码率（BER）和测试不同采样方法的性能，特别适用于低信噪比（SNR）环境下的性能评估。项目专注于重要性采样（Importance Sampling）技术在通信系统性能分析中的应用。

## 项目背景

在通信系统中，当信噪比很低时，传统的Monte Carlo方法需要大量样本才能获得准确的误码率估计。本项目实现了多种高级采样技术，包括重要性采样（Importance Sampling）方法，大幅减少了获得可靠误码率估计所需的样本数量。

## 主要特性

### 编码与解码支持
- **汉明码（Hamming Code）**：支持 (2^r-1, 2^r-1-r) 形式的汉明码，r ∈ [3,8]
- **里德-穆勒码（Reed-Muller Code）**：RM(r,m) 形式（开发中）
- **多种解码器**：
  - 二进制硬判决解码
  - ORBGRAND (Ordered Reliability Bits GRAND)
  - SGRAND (Soft GRAND)
  - Chase 解码算法

### 高级采样技术
- **普通高斯采样（Naive Sampling）**：标准蒙特卡洛方法
- **平移高斯采样（Shift Sampling）**：基于最近邻码字的平移采样
- **贝塞尔采样（Bessel Sampling）**：使用贝塞尔分布的重要性采样
- **对称平移采样（SymShift Sampling）**：利用码字对称性的高效采样
- **方差控制采样**：自适应调整采样参数以优化性能

### 分析与可视化
- **性能对比**：误码率 vs 计算复杂度的综合分析
- **收敛性分析**：采样方法的收敛速度比较
- **参数敏感性**：不同噪声水平和码长下的性能分析
- **自动化报告**：JSON格式结果存储和图表自动生成

## 安装和依赖

### 系统要求
- Python 3.7+
- 建议内存：8GB+（用于大码长模拟）

### 依赖安装

```bash
pip install numpy scipy matplotlib scikit-commpy
```

可选依赖（用于高级功能）：
```bash
pip install pandas seaborn  # 增强数据分析和可视化
```

### 快速开始

```bash
git clone <repository-url>
cd reliable_sim

# 运行基础测试
python test_general.py

# 查看帮助信息
python test_general.py --help
```

## 使用方法

### 基础汉明码模拟

```python
from code_linear import HammingCode
from sampler import NaiveSampler

# 创建(7,4)汉明码
hamming = HammingCode(r=3, decoder='binary')

# 创建采样器
sampler = NaiveSampler(code=hamming)

# 运行模拟
err, weight, ratio = hamming.simulate(
    noise_std=0.5,    # 噪声标准差
    sampler=sampler,  # 采样器
    batch_size=10000, # 批处理大小
    num_samples=100000 # 总样本数
)

# 打印误码率
print(f'误码率: {ratio:.6e}')
```

### 命令行快速测试
```bash
# 运行完整的采样方法比较
python test_general.py

# 指定参数运行特定实验
python test_general.py --r 4 --sigma 0.3 --samples 100000 --samplers naive bessel sym
```

### 使用重要性采样

```python
from code_linear import HammingCode
from sampler import BesselSampler, SymShiftSampler

# 创建汉明码
hamming = HammingCode(r=4)  # (15,11)汉明码

# 创建重要性采样器
samplers = {
    'bessel': BesselSampler(code=hamming, scale_factor=1.0),
    'sym': SymShiftSampler(code=hamming, fix_tx=True, scale_factor=1.0)
}

# 使用对称平移采样器（通常效率最高）
err, weight, ratio = hamming.simulate(
    noise_std=0.3, 
    sampler=samplers['sym'],
    batch_size=10000,
    num_samples=100000
)
print(f"使用对称平移采样的误码率: {ratio:.2e}")
```

### 实验配置与批处理运行

```python
from test_general import SamplingExperimentRunner

# 创建实验运行器
runner = SamplingExperimentRunner()

# 配置实验参数
config = {
    'code_type': 'hamming',
    'r': 5,  # (31,26)汉明码
    'sigma': 0.25,
    'samples': 50000,
    'samplers': ['naive', 'bessel', 'sym'],
    'decoder': 'binary'
}

# 运行实验
results = runner.run_single_experiment(**config)
```

### 高级实验分析

#### 命令行实验运行
```bash
# 运行完整的采样器性能比较
python test_general.py --experiment convergence --r 4 --sigma 0.3

# 进行码长影响分析
python test_general.py --experiment code_length --r 3 4 5 6 --sigma 0.3 0.4

# 运行方差控制实验
python test_general.py --experiment variance_control --r 5 --samples 100000

# 并行运行多个配置
python test_general.py --parallel --config_file experiments.json
```

#### 程序化实验运行
```python
from test_general import run_experiments_from_config

# 定义实验配置
experiment_config = {
    "experiments": [
        {
            "type": "convergence",
            "r": 4,
            "sigma": 0.3,
            "samplers": ["naive", "bessel", "sym"],
            "samples": 100000
        },
        {
            "type": "code_length",
            "r": [3, 4, 5, 6],
            "sigma": 0.25,
            "samplers": ["sym"]
        }
    ]
}

# 运行实验
results = run_experiments_from_config(experiment_config)
```

## 代码结构与核心组件

### 核心模块

| 文件 | 功能描述 |
|------|----------|
| `code_linear.py` | 线性码基类和汉明码实现，包含编码、解码、模拟核心逻辑 |
| `sampler.py` | 采样器基类和所有重要性采样方法的实现 |
| `test_general.py` | 统一的实验运行框架和命令行接口 |
| `plot_general.py` | 结果可视化和图表生成工具 |

### 解码器模块
- `ORBGRAND.py`: Ordered Reliability Bits GRAND解码器
- `SGRAND.py`: Soft GRAND解码器实现
- `chase.py`: Chase解码算法
- `RM.py`: 里德-穆勒码的实现（开发中）

### 支持工具
- `distance_calculator.py`: 码字距离计算工具
- `decoder_analysis.py`: 解码器性能分析
- `plot_results.py`: 传统绘图工具（已废弃）
- `test_hamming.py`: 早期测试框架（已废弃）

### 数据目录
- `logs/`: 实验结果存储（按实验类型分子目录）
- `plots/`: 自动生成的图表
- `images/`: 参考图片和示例结果
- `libs/`: 预计算矩阵和噪声库

## 核心组件详解

### 线性码基类 (`code_linear.py`)

`LinearCodeBase` 是所有线性码的抽象基类，提供统一的接口：
- **编码功能**: 将信息比特映射到码字
- **解码功能**: 支持多种解码算法（硬判决/软判决）
- **模拟框架**: 集成采样器进行误码率评估
- **数学优化**: 使用对数域计算避免数值下溢

`HammingCode` 实现 (2^r-1, 2^r-1-r) 汉明码：
- 支持 r ∈ [3,8]，对应码长7到255
- 预计算所有码字和校验矩阵
- 提供最近邻码字查找优化

### 采样器体系 (`sampler.py`)

所有采样器继承自 `SamplerBase` 基类：

| 采样器类 | 原理 | 适用场景 | 性能特点 |
|----------|------|----------|----------|
| `NaiveSampler` | 标准高斯分布 | 基准测试 | 简单但效率低 |
| `BesselSampler` | 贝塞尔分布修正 | 中等SNR | 方差减小显著 |
| `SymShiftSampler` | 对称平移混合 | 低SNR | 效率最高 |
| `ShiftSampler` | 最近邻平移 | 特定场景 | 可调节性强 |

### 实验框架 (`test_general.py`)

`SamplingExperimentRunner` 提供完整的实验管理：
- **参数扫描**: 自动化的多维参数空间探索
- **结果缓存**: JSON格式结果持久化存储
- **并行计算**: 支持多实验并行执行
- **可视化**: 自动化图表生成

### 解码器支持

支持多种现代解码算法：
- **二进制解码**: 传统硬判决解码
- **ORBGRAND**: 基于可靠性的GRAND变种
- **SGRAND**: 软判决GRAND解码
- **Chase算法**: 基于可靠性的代数解码

## 实验结果与可视化

### 自动生成的图表类型

运行实验后，`plot_general.py` 会自动生成以下分析图表：

1. **收敛性分析**: 不同采样方法的误码率收敛速度对比
2. **码长影响**: 不同码长下的性能曲线
3. **计算复杂度**: 采样时间 vs 误码率精度权衡
4. **参数敏感性**: 缩放因子和噪声水平的影响分析

### 结果文件结构
```
logs/
├── convergence/          # 收敛性实验结果
├── code_length/         # 码长影响分析
├── variance_controlled/ # 方差控制实验
└── json/               # 结构化结果数据

plots/
├── convergence_*.png   # 收敛性图表
├── code_length_*.png   # 码长分析
└── variance_controlled_*.png # 方差分析
```

### 查看示例结果
```python
import json
import matplotlib.pyplot as plt

# 加载实验结果
with open('logs/convergence/convergence_test_r4_sigma0.3.json', 'r') as f:
    results = json.load(f)

# 提取关键指标
for sampler_name, data in results.items():
    print(f"{sampler_name}: BER={data['error_rate']:.2e}, "
          f"Time={data['execution_time']:.1f}s, "
          f"Samples={data['actual_samples']}")
```

## 性能优化建议

### 计算效率
- **批处理大小**: 建议 `batch_size=10000~50000` 平衡内存和速度
- **内存管理**: 大码长(r>6)建议使用较小批处理

### 采样器选择指南
| 场景 | 推荐采样器 | 优势 |
|------|------------|------|
| 高SNR(>3dB) | NaiveSampler | 简单可靠 |
| 中SNR(1-3dB) | BesselSampler | 方差减小显著 |
| 低SNR(<1dB) | SymShiftSampler | 效率最高 |
| 极端低SNR | 方差控制采样 | 自适应优化 |

### 调试与验证
```python
# 验证编码解码一致性
from code_linear import HammingCode
hamming = HammingCode(r=3)
is_consistent = hamming.test_encode_decode_consistency()
print(f"编码解码一致性: {is_consistent}")

# 检查码字距离分布
import numpy as np
distances = cdist(hamming.codewords, hamming.codewords, metric='hamming')
print(f"最小距离: {np.min(distances[distances>0]) * hamming.n}")
```

## 故障排除

### 常见问题
1. **内存不足**: 减小批处理大小或降低码长
2. **收敛缓慢**: 尝试使用重要性采样器(SymShiftSampler)
3. **精度不足**: 增加样本数量或使用方差控制

### 调试模式
```bash
# 启用详细输出
python test_general.py --debug --verbose

# 运行小规模测试
python test_general.py --r 3 --samples 1000 --quick-test
```

## 参与贡献

### 开发环境设置
```bash
git clone <repository-url>
cd reliable_sim

# 创建开发环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 贡献指南
1. **代码规范**: 遵循PEP8，使用中文注释
2. **测试要求**: 新功能需包含单元测试
3. **文档更新**: README和CLAUDE.md同步更新
4. **提交规范**: 使用清晰的commit信息，支持中英双语

### 路线图
- [ ] 实现并行重要性采样
- [ ] 支持更多GRAND变种
- [ ] 支持更多线性码

## 许可协议

本项目采用 [MIT License](LICENSE) 开源协议，允许商业使用和修改。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](https://github.com/yourusername/reliable_sim/issues)
- 发送邮件至: [your.email@domain.com](mailto:your.email@domain.com)
