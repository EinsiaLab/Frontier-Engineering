# 去噪
去除稀疏单细胞RNA测序计数数据中的噪声

此任务源自 https://openproblems.bio/benchmarks/denoising?version=v1.0.0

## 仓库内推荐初始化方式

本仓库里推荐先执行：

```bash
bash scripts/bootstrap/setup_denoising_task.sh
source benchmarks/SingleCellAnalysis/denoising/env.sh
```

这会把本地工具安装到 `benchmarks/SingleCellAnalysis/denoising/.tools/`，
把外部 `task_denoising` 仓库 clone 到
`benchmarks/SingleCellAnalysis/denoising/task_denoising/`，并准备好
`frontier_eval` 期望的 `submission` 模板方法。

需要更重的前置步骤时，可额外执行：

```bash
bash scripts/bootstrap/setup_denoising_task.sh --sync-resources
bash scripts/bootstrap/setup_denoising_task.sh --build-components --build-containers
```

### Docker 前提条件

构建容器需要 Docker，且当前用户需在 `docker` 组中：

```bash
sudo usermod -aG docker $USER
newgrp docker   # 无需重新登录即可生效
```

若 Docker Hub 无法访问（国内常见），需为 Docker daemon 配置 HTTP 代理。
创建 `/etc/systemd/system/docker.service.d/http-proxy.conf`：

```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1"
```

然后重载并重启 Docker：

```bash
sudo systemctl daemon-reload && sudo systemctl restart docker
```

### 已知兼容性修复（已应用到 `task_denoising`）

`openproblems/base_python:1` 内置 Python 3.12。`scprep` 依赖 `pandas < 2.1`，
而该版本 pandas 没有 Python 3.12 的二进制 wheel，且在 Python 3.12 上从源码构建会因
`pkg_resources` / setuptools 不兼容而失败。

受影响的三个组件（`methods/magic`、`metrics/mse`、`metrics/poisson`）已在
`task_denoising` 内部打补丁，将基础镜像改为 `python:3.10`，并显式声明
`anndata`/`scanpy` 依赖（原先从基础镜像继承）。
`setup_denoising_task.sh` 会自动应用这些补丁。

## 运行方式
```
cd benchmarks/SingleCellAnalysis/denoising

git clone https://github.com/openproblems-bio/task_denoising.git
cd task_denoising
git submodule update --init --recursive # 初始化 common

# 构建组件/容器
bash scripts/project/build_all_components.sh
bash scripts/project/build_all_docker_containers.sh

# 同步资源
bash scripts/sync_resources.sh

# 运行测试
bash scripts/run_benchmark/run_test_local.sh
# 数据集：2 个
# cxg_immune_cell_atlas_subsample、cxg_mouse_pancreas_atlas_subsample（见 temp/results/testrun_xxx/dataset_uns.yaml）。
# 方法：7 个
# no_denoising, perfect_denoising, alra, cellmapper, dca, knn_smoothing, magic（scprint 被排除，避免 GPU/网络依赖导致测试慢或不稳定，见 temp/results/testrun_xxx/run_test_local.sh）。
# 指标：2 个
# mse 和 poisson（见 temp/results/testrun_xxx/metric_configs.yaml）。
```

评测结果将位于 benchmarks/SingleCellAnalysis/denoising/task_denoising/temp/results/xxx/score_uns.yaml
您可以通过运行 benchmarks/SingleCellAnalysis/denoising/verification/rank_scores.py 来提取其中的分数


### 将您的方法接入测试

新增一个方法，怎么实现并接入测试
生成方法骨架（推荐）
```
common/scripts/create_component --type method --language python --name my_method
```
代码实现：编辑 `src/methods/my_method/config.vsh.yaml` + `src/methods/my_method/script.py`
输入输出接口必须满足：
- 输入 train：layers["counts"] + uns["dataset_id"] + uns["dataset_organism"]（src/api/file_train.yaml）
- 输出 prediction：layers["denoised"] + uns["dataset_id"] + uns["method_id"]（src/api/file_prediction.yaml）

> 可参考 src/methods/magic/script.py。

#### 组件级测试
```
viash test src/methods/my_method/config.vsh.yaml
```
上述命令会进行：
- 配置规范检查（metadata、links、references、资源标签）`common/component_tests/check_config.py`
- 执行并检查输出格式 `common/component_tests/run_and_check_output.py`

#### 接入 benchmark 工作流
在方法列表里加上 my_method：`src/workflows/run_benchmark/main.nf` (line 17)
在 workflow 依赖里加上 methods/my_method：`src/workflows/run_benchmark/config.vsh.yaml` (line 69)
重新 build 并跑集成测试
```
viash ns build --parallel --setup cachedbuild --query '^(methods/my_method|workflows/run_benchmark)$'

bash scripts/run_benchmark/run_test_local.sh
```
验证接入成功
看 `temp/results/testrun_*/score_uns.yaml` 里是否出现 method_id: my_method。

## 模版实现
这里提供一个不进行去噪的模板实现，您也可以在这里的代码基础上修改
通过如下命令应用修改
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
