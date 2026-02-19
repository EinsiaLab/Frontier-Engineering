# Noise Removal
Removing noise from sparse single-cell RNA sequencing count data

This task originates from https://openproblems.bio/benchmarks/denoising?version=v1.0.0. The file organization conforms to the standard format. Validation code has not yet been implemented and will be added later. @ahydchh

`baseline/benchmark_denoising.ipynb` is the script in the test-time-training repository that compares the basic MAGIC algorithm with the TTT implementation.

`baseline/run_magic/ttt_denoising.py` is the runtime file for the two methods after our decomposition.

`verification/evaluate_denoising_results.py` is the decomposition evaluation script, which evaluates the results of the Magic and TTT methods.

For required dependencies, please refer to `verification/README.md` and `requirements-denoising.txt`.