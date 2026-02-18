# 扰动响应预测（OpenProblems / NeurIPS 2023 scPerturb）

## 1. 背景
单细胞扰动筛选用于测量小分子化合物在不同细胞类型中引发的转录组响应。实验成本高、周期长；如果能够准确预测扰动后
的表达变化，将有助于加速药物研发与生物学假设验证。

本任务参考并改编自 OpenProblems Bio 的 perturbation prediction 任务仓库 `openproblems-bio/task_perturbation_prediction`，
其数据与设定来自 NeurIPS 2023 Open Problems 竞赛（Kaggle：*Open Problems – Single-Cell Perturbations*）。

## 2. 数据集（来源）
使用 OpenProblems 在公共 `openproblems-data`（S3）上发布的处理后数据：

- dataset id：`neurips-2023-data`
- 文件：`de_train.h5ad`、`de_test.h5ad`、`id_map.csv`

其中 `de_train.h5ad` / `de_test.h5ad` 为按 *(compound, cell_type)* 聚合的差异表达（DE）特征，包含多个 layer。
默认评测使用的真实标签 layer 为：

- `clipped_sign_log10_pval`

## 3. 任务定义
给定训练集 DE profile，预测测试集中每个 *(compound, cell_type)* 的 DE profile。

### 3.1 输入
方法可使用：

- `de_train.h5ad`（训练 DE）
- `id_map.csv`（需要预测的测试对及其顺序）

### 3.2 输出
写出 `prediction.h5ad`（AnnData）：

- `layers["prediction"]`：形状 `(n_test, n_genes)` 的预测矩阵
- `obs_names`：必须与 `id_map.csv` 的 `id` 列一致（字符串），且顺序一致
- `var_names`：与 `de_test.var_names` 对齐的基因列表
- `uns["dataset_id"]`：数据集标识
- `uns["method_id"]`：方法标识

## 4. 评测
评测对比 `de_test.layers["clipped_sign_log10_pval"]`（真实）与 `prediction.layers["prediction"]`（预测），并在基因维度上
逐行计算指标。

输出指标（与 OpenProblems 任务一致）：

- `mean_rowwise_rmse`
- `mean_rowwise_mae`
- `mean_rowwise_pearson`
- `mean_rowwise_spearman`
- `mean_rowwise_cosine`

### 4.1 综合得分
将相关性与误差合成为单一可最大化得分：

- `corr_score = (mean_rowwise_pearson + 1) / 2`
- `err_score = 1 / (1 + mean_rowwise_rmse)`
- `combined_score = (corr_score + err_score) / 2`

得分越高越好；不合法输出得分为 0。

## 5. 参考
- OpenProblems 任务仓库：https://github.com/openproblems-bio/task_perturbation_prediction
- Kaggle 比赛：https://www.kaggle.com/competitions/open-problems-single-cell-perturbations

