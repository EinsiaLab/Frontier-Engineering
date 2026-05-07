# Cryptographic Benchmarks

This domain contains algorithm-acceleration tasks for:

- `AES-128 CTR`
- `SHA-256`
- `SHA3-256`

Each task provides:

- baseline C++ implementation (`baseline/*.cpp`)
- correctness verification (`verification/validate.cpp`)
- throughput benchmark (`verification/evaluate.cpp`)
- official reference URL note (`references/README.md`)

## Run with frontier_eval (unified)

```bash
# AES-128
python -m frontier_eval task=unified task.benchmark=Cryptographic/AES-128 algorithm.iterations=10

# SHA-256
python -m frontier_eval task=unified task.benchmark=Cryptographic/SHA-256 algorithm.iterations=10

# SHA3-256
python -m frontier_eval task=unified task.benchmark=Cryptographic/SHA3-256 algorithm.iterations=10
```

Backwards-compatible aliases (route to the same unified benchmark via config): `task=crypto_aes128`, `task=crypto_sha256`, `task=crypto_sha3_256`.

Optional reference injection for agents (default: disabled). The repository no
longer bundles local standard PDFs; this emits the official reference URL and
only extracts local PDF text if a user has supplied a permitted local copy:

```bash
python -m frontier_eval task=crypto_sha256 task.include_pdf_reference=true
```
