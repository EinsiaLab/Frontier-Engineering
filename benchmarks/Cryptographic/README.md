# Cryptographic Benchmarks

This domain contains algorithm-acceleration tasks for:

- `AES-128 CTR`
- `SHA-256`
- `SHA3-256`

Each task provides:

- baseline C++ implementation (`baseline/*.cpp`)
- correctness verification (`verification/validate.cpp`)
- throughput benchmark (`verification/evaluate.cpp`)
- reference PDF (`references/*.pdf`)

## Run with frontier_eval

```bash
# AES-128
python -m frontier_eval task=crypto_aes128 algorithm.iterations=10

# SHA-256
python -m frontier_eval task=crypto_sha256 algorithm.iterations=10

# SHA3-256
python -m frontier_eval task=crypto_sha3_256 algorithm.iterations=10
```

Optional reference injection for agents (default: disabled):

```bash
python -m frontier_eval task=crypto_sha256 task.include_pdf_reference=true
```

