# 密码学 Benchmark

该领域包含如下算法加速任务：

- `AES-128 CTR`
- `SHA-256`
- `SHA3-256`

每个任务都提供：

- 基线 C++ 实现（`baseline/*.cpp`）
- 正确性校验（`verification/validate.cpp`）
- 吞吐率评测（`verification/evaluate.cpp`）
- 算法参考 PDF（`references/*.pdf`）

## 在 frontier_eval 中运行（unified）

```bash
# AES-128
python -m frontier_eval task=unified task.benchmark=Cryptographic/AES-128 algorithm.iterations=10

# SHA-256
python -m frontier_eval task=unified task.benchmark=Cryptographic/SHA-256 algorithm.iterations=10

# SHA3-256
python -m frontier_eval task=unified task.benchmark=Cryptographic/SHA3-256 algorithm.iterations=10
```

兼容别名（通过配置路由到相同 unified benchmark）：`task=crypto_aes128`、`task=crypto_sha256`、`task=crypto_sha3_256`。

可选给 agent 注入 PDF references（默认关闭）：

```bash
python -m frontier_eval task=crypto_sha256 task.include_pdf_reference=true
```
