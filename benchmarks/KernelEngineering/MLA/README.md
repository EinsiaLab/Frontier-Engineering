# amd-mla-decode

This task originates from https://www.gpumode.com/leaderboard/463?tab=rankings. The file organization conforms to the standard format. Validation code is not yet implemented and will be added later. @ahydchh

The MLA reference implementation is located in `baseline/reference.py`, which is the basic implementation and also the standard for numerical correctness.

`baseline/mla_code_1/2/3.py` is the implementation provided by test-time training.

The agent can be modified based on `baseline/submission.py`, which is a template version to be optimized.

`baseline/util.py` provides common tools.

The evaluation entry point is located in `verification/eval.py`.

`verification/requirements-gpumode.txt` provides the required dependencies.

## Execution Method

``` cd benchmarks/KernelEngineering/MLA/verification

# Only check correctness
POPCORN_FD=1 python eval.py test mla_tests.txt

# Time each case, only perform an initial correctness check once, subsequent tests mainly focus on speed
POPCORN_FD=1 python eval.py benchmark mla_bench.txt

# Only run the last example, it will repeatedly recheck in a loop, for stricter testing
POPCORN_FD=1 python eval.py leaderboard mla_bench.txt

```

The above code will use `submission.custom_kernel` for evaluation. You can choose to replace `benchmarks/KernelEngineering/MLA/baseline/submission.py` with your own code, or replace all `from baseline.submission import custom_kernel` in `benchmarks/KernelEngineering/MLA/verification/eval.py` with importing from your specified code.