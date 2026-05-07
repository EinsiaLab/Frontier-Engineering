# SHA3-256 Algorithm Acceleration

We provide a basic C++ implementation `baseline/SHA3-256.cpp` that does not use third-party libraries. You need to modify this file to improve the algorithm's throughput while maintaining correctness.

The detailed algorithm reference is the official NIST FIPS 202 publication:
https://doi.org/10.6028/NIST.FIPS.202. A local PDF is not bundled; see
`references/README.md`.

Running `verification/valid.sh` will generate random data to verify the algorithm's correctness (using OpenSSL).

Running `verification/eval.sh` will perform multiple calculations using both 8Kbits and 8Mbits data streams to calculate the algorithm's efficiency.

## Run with frontier_eval (unified)

Unified benchmark: `task=unified task.benchmark=Cryptographic/SHA3-256`

```bash
python -m frontier_eval task=unified task.benchmark=Cryptographic/SHA3-256 algorithm.iterations=0
```

Backwards-compatible alias (routes to the same unified benchmark via config): `task=crypto_sha3_256`.
