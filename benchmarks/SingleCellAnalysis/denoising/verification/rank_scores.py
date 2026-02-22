import argparse
import yaml
import pandas as pd
from pathlib import Path

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="解析单细胞去噪基准测试得分，计算标准化得分并排名。")
    parser.add_argument("input_file", type=str, help="输入的 score_uns.yaml 文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="保存结果的文件夹路径。默认为输入文件所在的文件夹。")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 找不到输入文件 '{input_path}'。")
        return

    # 确定输出目录
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ranked_normalized_scores.csv"
    out_txt = out_dir / "ranked_scores_report.txt"

    # 从文件读取 YAML 数据
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"解析 YAML 文件时发生错误: {e}")
            return

    # 转换为 DataFrame 方便处理
    df = pd.DataFrame(data)

    # 提取需要的列
    cols_to_keep = ['dataset_id', 'metric_ids', 'method_id', 'metric_values']
    try:
        df = df[cols_to_keep]
    except KeyError as e:
        print(f"错误: YAML 数据中缺少必要的字段 {e}。")
        return

    results = []
    report_lines = []

    # 按照数据集和指标进行分组
    grouped = df.groupby(['dataset_id', 'metric_ids'])
    
    for (dataset, metric), group in grouped:
        report_lines.append(f"=== Dataset: {dataset} | Metric: {metric} ===")

        # 获取控制组（基线）的得分
        try:
            score_0 = group.loc[group['method_id'] == 'no_denoising', 'metric_values'].values[0]
            score_1 = group.loc[group['method_id'] == 'perfect_denoising', 'metric_values'].values[0]
        except IndexError:
            report_lines.append("  [!] 缺少基线数据 (no_denoising 或 perfect_denoising)，跳过该组数据的标准化计算。\n")
            continue

        # 计算标准化得分
        # 规避分母为0的情况（虽然理论上完美去噪和不处理不应该得分相同）
        if score_1 == score_0:
            group['normalized_score'] = 0.0
        else:
            group['normalized_score'] = (group['metric_values'] - score_0) / (score_1 - score_0)

        # 排名逻辑：标准化得分越高越好（1 为完美，0 为基线，负数为劣化数据）
        group_sorted = group.sort_values(by='normalized_score', ascending=False).copy()
        group_sorted['rank'] = range(1, len(group_sorted) + 1)

        results.append(group_sorted)

        # 格式化输出表头
        report_lines.append(f"{'Rank':<6} | {'Method':<25} | {'Raw Score':<15} | {'Normalized Score':<15}")
        report_lines.append("-" * 70)
        
        # 格式化输出每一行数据
        for _, row in group_sorted.iterrows():
            report_lines.append(f"{row['rank']:<6} | {row['method_id']:<25} | {row['metric_values']:<15.4f} | {row['normalized_score']:<15.4f}")
        report_lines.append("\n")

    # 聚合结果并保存
    if results:
        final_df = pd.concat(results, ignore_index=True)
        # 保存为便于程序后续读取的 CSV
        final_df.to_csv(out_csv, index=False)

        # 保存为便于人类阅读的 TXT 报告
        report_text = "\n".join(report_lines)
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 在终端打印结果
        print(report_text)
        print("=" * 70)
        print(f"数据已成功处理并保存至：\n  - CSV 数据表: {out_csv}\n  - 文本报告:   {out_txt}")
    else:
        print("没有找到有效的包含基线的数据组，无法生成排名。")

if __name__ == "__main__":
    main()