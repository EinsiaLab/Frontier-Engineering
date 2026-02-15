# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
The code will be ran on a laptop with MacOS and Apple Silicon.

IMPORTANT: 此项目作者为中文母语者，请你在撰写文档和注释时使用中文。

## Project Overview

ReliableSim is a Python toolkit for reliable communication system simulation, focused on evaluating bit error rates (BER) and testing different sampling methods for performance evaluation in low SNR environments. It implements advanced sampling techniques including Importance Sampling methods to reduce sample requirements for reliable BER estimation.

## Key Commands

### Running Experiments
```bash
# Basic experiment
python test_general.py

# Parameter sweep
python test_general.py --r 3 4 5 --sigma 0.3 0.4 --samplers naive bessel sym

# Convergence analysis
python test_general.py --experiment convergence --r 4 --sigma 0.3

# Data generation and plotting
python plot_general.py

# Server synchronization
bash scp_tools.sh
```

### Dependencies
```bash
pip install numpy scipy matplotlib scikit-commpy pandas seaborn
```

## Core Architecture

### Module Relationships
```
test_general.py → code_linear.py → sampler.py → Decoder
        ↓
    Results Storage → logs/ → plot_general.py → plots/
```

### Main Components

**Linear Codes (`code_linear.py`)**
- `LinearCodeBase`: Base class for all linear codes
- `HammingCode`: Implements Hamming codes with configurable parameters
- Key methods: `encode()`, `decode()`, `simulate()`

**Sampling Methods (`sampler.py`)**
- `NaiveSampler`: Standard Gaussian sampling
- `BesselSampler`: Bessel distribution-based importance sampling
- `SymShiftSampler`: Symmetry-based efficient sampling (recommended for low SNR)
- `ShiftSampler`: Shifted Gaussian mixture sampling
- `CutoffSampler`: Cutoff distribution-based sampling

**Decoders**
- Binary decoder (built-in)
- ORBGRAND decoder (`ORBGRAND.py`)
- SGRAND decoder (`SGRAND.py`)
- Chase decoder (`chase.py`)

**Experiment Framework (`test_general.py`)**
- `SamplingExperimentRunner`: Unified experiment management with CLI
- Parameter sweeping and batch processing
- Parallel execution support

## Key Configuration Patterns

**Code Configuration**
```python
# Hamming Code (2^r-1, 2^r-1-r)
hamming = HammingCode(r=4, decoder='binary')  # (15,11) code

# Available decoders: 'binary', 'ORBGRAND', 'SGRAND', 'chase', 'chase2'
```

**Sampling Configuration**
```python
samplers = {
    'naive': NaiveSampler(code),
    'bessel': BesselSampler(code, scale_factor=1.0),  # 10-100x speedup
    'sym': SymShiftSampler(code, fix_tx=True, scale_factor=1.0)
}
```

**Experiment Configuration**
```python
config = {
    'code_type': 'hamming',
    'r': 4,
    'sigma': 0.3,
    'samples': 100000,
    'batch_size': 10000,  # Balance memory vs speed
    'samplers': ['naive', 'bessel', 'sym'],
    'decoder': 'binary'
}
```

## File Structure

```
├── code_linear.py      # Linear code implementations
├── sampler.py          # All sampling methods
├── test_general.py     # Experiment runner with CLI
├── plot_general.py     # Visualization utilities
├── ORBGRAND.py         # Ordered Reliability Bits GRAND decoder
├── SGRAND.py           # Soft GRAND decoder
├── chase.py            # Chase decoding algorithm
├── RM.py               # Reed-Muller codes (in development)
├── distance_calculator.py
├── decoder_analysis.py
├── logs/               # JSON result files
├── plots/              # Generated visualization
├── libs/               # Precomputed matrices and noise libraries
└── images/             # Reference plots
```

## Common Experiment Patterns

### Basic BER Testing
```python
from code_linear import HammingCode
from sampler import BesselSampler

hamming = HammingCode(r=4)  # (15,11) code
sampler = BesselSampler(code=hamming)
err, weight, ratio = hamming.simulate(
    noise_std=0.3,
    sampler=sampler,
    batch_size=1e5,
    num_samples=1e6
)
```

### Data Flow
1. **Simulation**: `code.simulate()` → `sampler.sample()` → `code.decode()` → BER calculation
2. **Storage**: JSON format results in logs/ with automatic caching
3. **Analysis**: Automated plot generation in plots/

## Performance Considerations

- **Memory**: Large r values (>7) require significant memory for Hamming codes
- **Batch Size**: Use 10k-50k for optimal memory/speed tradeoff
- **Sampling**: Importance sampling methods provide 10-100x speedup over naive sampling
- **Caching**: JSON format enables incremental analysis without recomputation

## Key Mathematical Concepts

- **Hamming Codes**: (2^r-1, 2^r-1-r) linear block codes
- **Importance Sampling**: Variance reduction technique for rare event simulation
- **BER Estimation**: Bit error rate calculation using log-space arithmetic to avoid underflow
- **GRAND Decoders**: Guessing Random Additive Noise Decoding algorithms

## Deprecated Components
- `test_hamming.py` → Use `test_general.py`
- `plot_results.py` → Use `plot_general.py`