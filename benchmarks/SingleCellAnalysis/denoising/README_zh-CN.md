# 去噪
去除稀疏单细胞RNA测序计数数据中的噪声

此任务源自 https://openproblems.bio/benchmarks/denoising?version=v1.0.0

baseline/benchmark_denoising.ipynb 是 test-time-training 仓库中对比基础 MAGIC 算法与 ttt 实现的脚本
baseline/run_magic/ttt_denoising.py 是经我们拆解后的两种方法的运行文件
verification/evaluate_denoising_results.py 是拆解后的评测脚本，会对 Magic 和 ttt 两种方法的结果进行评测
所需依赖请参照 verification/README.md, requirements-denoising.txt