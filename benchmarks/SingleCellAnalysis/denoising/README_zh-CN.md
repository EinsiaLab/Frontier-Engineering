# 去噪
去除稀疏单细胞RNA测序计数数据中的噪声

此任务源自 https://openproblems.bio/benchmarks/denoising?version=v1.0.0

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