# amd-mla-decode

This task originates from https://www.gpumode.com/leaderboard/463?tab=rankings. The file organization conforms to the standard format. Validation code is not yet implemented and will be added later. @ahydchh

The MLA reference implementation is located in `baseline/reference.py`, which is the basic implementation and also the standard for numerical correctness.

`baseline/mla_code_1/2/3.py` is the implementation provided by test-time training.

The agent can be modified based on `baseline/submission.py`, which is a template version to be optimized.

`baseline/util.py` provides common tools.

The evaluation entry point is located in `verification/eval.py`.

`verification/requirements-gpumode.txt` provides the required dependencies.