# Triangle Multiplication

This task originates from https://www.gpumode.com/leaderboard/496?tab=rankings. The file organization conforms to the standard format. Validation code is not yet implemented and will be added later. @ahydchh

The TriMul reference implementation is located in `baseline/reference.py`, which is the basic implementation and also the standard for numerical correctness.

`baseline/solution.py` is the implementation provided by `test-time-training`.

The agent can be modified based on `baseline/submission.py`; this is a template version to be optimized.

`baseline/util.py` provides common tools.

The evaluation entry point is located in `verification/eval.py`.

`verification/eval-profile.py` is a version with fine-grained timing diagnostics, used to locate where time is spent.

`verification/requirements-gpumode.txt` provides the required dependencies.