# Denoising
Removing noise from sparse single-cell RNA-sequencing count data

This task originates from https://openproblems.bio/benchmarks/denoising?version=v1.0.0

## Repo-local bootstrap

The recommended setup path in this repository is:

```bash
bash scripts/bootstrap/setup_denoising_task.sh
source benchmarks/SingleCellAnalysis/denoising/env.sh
```

This installs repo-local tooling under `benchmarks/SingleCellAnalysis/denoising/.tools/`,
clones the external `task_denoising` repository into
`benchmarks/SingleCellAnalysis/denoising/task_denoising/`, and prepares the
`submission` method template expected by `frontier_eval`.

Optional heavier steps:

```bash
bash scripts/bootstrap/setup_denoising_task.sh --sync-resources
bash scripts/bootstrap/setup_denoising_task.sh --build-components --build-containers
```

### Docker requirement

Building containers requires Docker. The Docker daemon must be accessible to the
current user (i.e. the user must be in the `docker` group):

```bash
sudo usermod -aG docker $USER
newgrp docker        # apply without re-login
```

If Docker Hub is not reachable (common in mainland China), configure an HTTP proxy
for the Docker daemon before building containers. Create
`/etc/systemd/system/docker.service.d/http-proxy.conf`:

```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1"
```

Then reload and restart Docker:

```bash
sudo systemctl daemon-reload && sudo systemctl restart docker
```

### Known compatibility fix applied to `task_denoising`

`openproblems/base_python:1` ships Python 3.12. The `scprep` package requires
`pandas < 2.1`, which has no Python 3.12 binary wheels and cannot be built from
source on Python 3.12 due to a `pkg_resources` / setuptools incompatibility.

The three affected components (`methods/magic`, `metrics/mse`, `metrics/poisson`)
have been patched inside `task_denoising` to use `python:3.10` as their base image
and to list `anndata`/`scanpy` explicitly (previously inherited from the base image).
These patches are applied automatically by `setup_denoising_task.sh`.

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
The evaluation results will be located in `benchmarks/SingleCellAnalysis/denoising/task_denoising/temp/results/xxx/score_uns.yaml`. 
You can extract the scores from it by running `benchmarks/SingleCellAnalysis/denoising/verification/rank_scores.py`.


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
