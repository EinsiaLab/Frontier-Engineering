# Denoising
Removing noise from sparse single-cell RNA-sequencing count data

This task originates from https://openproblems.bio/benchmarks/denoising?version=v1.0.0

## How to Run
```bash
cd benchmarks/SingleCellAnalysis/denoising

git clone [https://github.com/openproblems-bio/task_denoising.git](https://github.com/openproblems-bio/task_denoising.git)
cd task_denoising
git submodule update --init --recursive # Initialize common

# Build components/containers
bash scripts/project/build_all_components.sh
bash scripts/project/build_all_docker_containers.sh

# Sync resources
bash scripts/sync_resources.sh

# Run tests
bash scripts/run_benchmark/run_test_local.sh
# Datasets: 2
# cxg_immune_cell_atlas_subsample, cxg_mouse_pancreas_atlas_subsample (see temp/results/testrun_xxx/dataset_uns.yaml).
# Methods: 7
# no_denoising, perfect_denoising, alra, cellmapper, dca, knn_smoothing, magic (scprint is excluded to avoid slow or unstable testing caused by GPU/network dependencies, see temp/results/testrun_xxx/run_test_local.sh).
# Metrics: 2
# mse and poisson (see temp/results/testrun_xxx/metric_configs.yaml).

```

### Integrating Your Method into the Benchmark

How to implement and integrate a new method into the test

Generate the method skeleton (Recommended)

```bash
common/scripts/create_component --type method --language python --name my_method
```

Code implementation: Edit `src/methods/my_method/config.vsh.yaml` + `src/methods/my_method/script.py`
The input and output interfaces must satisfy the following:

* Input train: layers["counts"] + uns["dataset_id"] + uns["dataset_organism"] (`src/api/file_train.yaml`)
* Output prediction: layers["denoised"] + uns["dataset_id"] + uns["method_id"] (`src/api/file_prediction.yaml`)

> You can refer to `src/methods/magic/script.py`.

#### Component-level Testing

```bash
viash test src/methods/my_method/config.vsh.yaml
```

The above command will perform:

* Configuration specification checks (metadata, links, references, resource tags) `common/component_tests/check_config.py`
* Execution and output format checks `common/component_tests/run_and_check_output.py`

#### Integrating into the Benchmark Workflow

Add my_method to the method list: `src/workflows/run_benchmark/main.nf` (line 17)
Add methods/my_method to the workflow dependencies: `src/workflows/run_benchmark/config.vsh.yaml` (line 69)
Rebuild and run integration tests

```bash
viash ns build --parallel --setup cachedbuild --query '^(methods/my_method|workflows/run_benchmark)$'

bash scripts/run_benchmark/run_test_local.sh
```
Verify successful integration
Check if `method_id: my_method` appears in `temp/results/testrun_*/score_uns.yaml`.

## Template Implementation

Here is a template implementation without denoising. You can also modify the code here.

Apply the modifications using the following commands:

``` 
cd benchmarks/SingleCellAnalysis/denoising
mkdir -p task_denoising/src/methods/submission
cp submission_template/method_submission/config.vsh.yaml task_denoising/src/methods/submission/config.vsh.yaml
cp submission_template/method_submission/script.py task_denoising/src/methods/submission/script.py

git -C task_denoising apply ../submission_template/patches/run_benchmark_main.nf.patch
git -C task_denoising apply ../submission_template/patches/run_benchmark_config.vsh.yaml.patch

cd task_denoising
viash test src/methods/submission/config.vsh.yaml
viash ns build --parallel --setup cachedbuild --query '^(methods/submission|workflows/run_benchmark)$'
```