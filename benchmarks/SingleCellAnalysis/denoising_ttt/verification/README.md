# OpenProblems Denoising Benchmark

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv venv .venv
source .venv/bin/activate

# Install requirements
uv pip install -r new_reqs.txt

# Git dependencies
uv pip install git+https://github.com/czbiohub/simscity.git
uv pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
uv pip install -e ./openproblems

# Minro API Change (if not already applied)
cd openproblems && git apply ../openproblems_api_fix.patch && cd .. 
```

## Known Issues

### 1. CZI cellxgene API changed
Tabula Muris loader fails. The API now uses:
- `dataset["dataset_id"]` instead of `dataset["id"]`
- Assets embedded in dataset: `dataset["assets"]`
- `asset["url"]` instead of `asset["presigned_url"]`

**Fix**: `openproblems_api_fix.patch`

### 2. NumPy 2.x compatibility
PyTorch pulls NumPy 2.x which breaks old syntax:
```python
# Old (breaks):
np.asarray(Y, dtype=np.float64, copy=False)

# New (works):
np.asarray(Y, dtype=np.float64)
```

