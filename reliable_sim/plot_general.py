import matplotlib.pyplot as plt
import numpy as np
from test_general import *
from scipy.special import logsumexp
import json
import os
import pandas as pd

def logmeanexp(a, axis=-1):
    if a.ndim == 1:
        return logsumexp(a) - np.log(len(a))
    return logsumexp(a, axis=axis) - np.log(a.shape[axis])

def logstdexp(a, axis=-1):
    xmean = logmeanexp(a, axis=axis)
    if np.isscalar(xmean):
        # 处理1维数组的情况
        e = np.exp(a - xmean)
        return np.std(e, ddof=1)
    else:
        # 处理多维数组的情况
        e = np.exp(a - xmean)
        return np.std(e, axis=axis, ddof=1)

def read_and_filter_logs(log_files, select_params=None):
    """
    读取多个日志文件，筛选符合条件的数据条目，并合并相同实验设置的结果
    自动检测实验类型（普通实验 vs 方差控制实验）
    
    参数:
        log_files: 日志文件路径列表
        select_params: 选择参数的字典，用于筛选数据，格式为 {参数名: 参数值}
                      如果参数值是列表，则匹配列表中的任意一个值
                      
    返回:
        pandas DataFrame，包含筛选和处理后的数据
    """
    if select_params is None:
        select_params = {}
    
    all_entries = []
    
    # 读取所有日志文件中的条目
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"警告: 日志文件 '{log_file}' 不存在，将被跳过")
            continue
            
        print(f"正在读取文件: {log_file}")
        with open(log_file, 'r') as f:
            # 读取头部信息
            header_line = f.readline().strip()
            try:
                header = json.loads(header_line)
                print(f"文件头信息: {header['test_type']} - {header['timestamp']}")
            except json.JSONDecodeError as e:
                print(f"警告: 无法解析头部信息: {header_line}. 错误: {e}")
            
            # 读取每行实验结果
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析日志条目: {line}. 错误: {e}")
            
            print(f"从文件 {log_file} 读取了 {line_count} 条数据记录")
    
    # 如果没有条目，返回空DataFrame
    if not all_entries:
        print("警告: 没有找到有效的实验条目")
        return pd.DataFrame()
    
    # 将条目转换为DataFrame
    df = pd.DataFrame(all_entries)
    print(f"读取的总数据条目: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 自动检测实验类型
    is_variance_controlled = 'target_std' in df.columns
    if is_variance_controlled:
        print("检测到方差控制实验数据")
    else:
        print("检测到普通实验数据")
    
    # 应用选择过滤条件
    if select_params:
        print(f"应用选择过滤条件: {select_params}")
        
    for param, values in select_params.items():
        if param not in df.columns:
            print(f"警告: 选择参数 '{param}' 在数据中不存在，将被忽略")
            continue
            
        if isinstance(values, (list, tuple)) and len(values) == 2 and isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
            # 如果是长度为2的列表/元组，且元素为数字，则视为区间判断 [min, max]
            min_val, max_val = values
            df = df[(df[param] >= min_val) & (df[param] <= max_val)]
        elif isinstance(values, list):
            # 如果是列表，匹配列表中的任意一个值
            df = df[df[param].isin(values)]
        else:
            # 否则精确匹配单个值
            df = df[df[param] == values]
    
    print(f"过滤后的数据条目: {len(df)}")
    
    if df.empty:
        print("警告: 应用选择条件后没有符合条件的数据")
        return df
    
    # 确定分组的键 - 根据实验类型调整
    if is_variance_controlled:
        # 方差控制实验：按target_std分组，排除actual_samples, actual_std, converged等结果字段
        result_columns = ['err_rate', 'err_ratio', 'exec_time', 'actual_std', 'actual_samples', 'converged']
        exclude_columns = result_columns + ['repetition', 'sample_count', 'experiment_id', 'max_samples', 'min_errors']
    else:
        # 普通实验：按num_samples分组
        result_columns = ['err_rate', 'err_ratio', 'exec_time']
        exclude_columns = result_columns + ['repetition', 'sample_count', 'experiment_id']
    
    group_columns = [col for col in df.columns if col not in exclude_columns]
    
    # 检查分组列是否都存在
    for col in group_columns:
        if col not in df.columns:
            print(f"警告: 分组列 '{col}' 不在数据中")
    
    print(f"分组列: {group_columns}")
    
    # 处理分组列中的NaN和None值
    for col in group_columns:
        # 如果列中存在NaN或None，用一个特殊标记替代
        if df[col].isnull().any():
            print(f"列 '{col}' 包含NaN或None值，将被替换为特殊标记")
            df[col] = df[col].fillna("NA_VALUE")
    
    # 转换为字符串类型以确保分组键可以哈希
    for col in group_columns:
        if df[col].dtype == 'object' or isinstance(df[col].iloc[0], (list, dict)):
            print(f"列 '{col}' 包含对象类型，将转换为字符串")
            df[col] = df[col].astype(str)
    
    # 诊断信息：检查每列的唯一值
    for col in group_columns:
        unique_values = df[col].unique()
        print(f"列 '{col}' 的唯一值: {unique_values[:5]}{'...' if len(unique_values) > 5 else ''}")
    
    # 对数据进行分组和聚合
    grouped_results = {}
    
    # 确保所有列都是合适的数据类型，避免分组问题
    for col in group_columns:
        if df[col].dtype == 'object':
            try:
                # 尝试转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass  # 如果转换失败，保持原样
    
    # 使用try-except来捕获可能的分组错误
    try:
        group_count = 0
        for key, group_df in df.groupby(group_columns):
            group_count += 1
            # 对于每个分组，计算平均值
            if isinstance(key, tuple):
                pass  # key已经是tuple了
            else:
                # 如果只有一个分组列，key会是标量，需要转为元组
                key = (key,)
            
            # 使用logmeanexp计算err_rate的平均值和logstdexp计算标准差
            if 'err_rate' in group_df.columns:
                err_rates = group_df['err_rate'].values
                avg_err_rate = logmeanexp(err_rates)
                std_err_rate = logstdexp(err_rates) if len(err_rates) > 1 else np.nan
            else:
                avg_err_rate = np.nan
                std_err_rate = np.nan
            
            # 计算其他指标的普通平均值和标准差
            avg_err_ratio = group_df['err_ratio'].mean() if 'err_ratio' in group_df.columns else np.nan
            std_err_ratio = group_df['err_ratio'].std() if 'err_ratio' in group_df.columns and len(group_df) > 1 else np.nan
            
            avg_exec_time = group_df['exec_time'].mean() if 'exec_time' in group_df.columns else np.nan
            std_exec_time = group_df['exec_time'].std() if 'exec_time' in group_df.columns and len(group_df) > 1 else np.nan
            
            # 为方差控制实验添加额外字段的处理
            extra_fields = {}
            if is_variance_controlled:
                # 计算收敛率（成功收敛的比例）
                if 'converged' in group_df.columns:
                    convergence_rate = group_df['converged'].mean()
                    extra_fields['convergence_rate'] = convergence_rate
                
                # 计算平均实际样本数和标准差
                if 'actual_samples' in group_df.columns:
                    avg_actual_samples = group_df['actual_samples'].mean()
                    std_actual_samples = group_df['actual_samples'].std() if len(group_df) > 1 else np.nan
                    extra_fields['actual_samples'] = avg_actual_samples
                    extra_fields['actual_samples_std'] = std_actual_samples
                
                # 计算平均实际标准差
                if 'actual_std' in group_df.columns:
                    avg_actual_std = group_df['actual_std'].mean()
                    std_actual_std = group_df['actual_std'].std() if len(group_df) > 1 else np.nan
                    extra_fields['avg_actual_std'] = avg_actual_std
                    extra_fields['std_actual_std'] = std_actual_std
            
            # 添加到结果字典
            if key not in grouped_results:
                grouped_results[key] = {
                    'err_rate': [avg_err_rate],
                    'err_rate_std': [std_err_rate],
                    'err_ratio': [avg_err_ratio],
                    'err_ratio_std': [std_err_ratio],
                    'exec_time': [avg_exec_time],
                    'exec_time_std': [std_exec_time],
                    'count': 1,
                    **{k: [v] for k, v in extra_fields.items()}
                }
            else:
                grouped_results[key]['err_rate'].append(avg_err_rate)
                grouped_results[key]['err_rate_std'].append(std_err_rate)
                grouped_results[key]['err_ratio'].append(avg_err_ratio)
                grouped_results[key]['err_ratio_std'].append(std_err_ratio)
                grouped_results[key]['exec_time'].append(avg_exec_time)
                grouped_results[key]['exec_time_std'].append(std_exec_time)
                grouped_results[key]['count'] += 1
                for k, v in extra_fields.items():
                    if k in grouped_results[key]:
                        grouped_results[key][k].append(v)
                    else:
                        grouped_results[key][k] = [v]
        
        print(f"成功创建 {group_count} 个分组")
    except Exception as e:
        print(f"分组过程中出错: {str(e)}")
        # 如果分组失败，我们可以尝试一种备用方法:
        print("尝试使用备用分组方法...")
        grouped_results = {}
        
        # 转换为字典列表以便手动分组
        records = df.to_dict('records')
        
        for record in records:
            # 创建分组键
            key_values = [record.get(col) for col in group_columns]
            key = tuple(key_values)
            
            # 提取结果值
            err_rate = record.get('err_rate', np.nan)
            err_ratio = record.get('err_ratio', np.nan)
            exec_time = record.get('exec_time', np.nan)
            
            # 添加到结果字典
            if key not in grouped_results:
                grouped_results[key] = {
                    'err_rate': [err_rate],
                    'err_rate_std': [np.nan],
                    'err_ratio': [err_ratio],
                    'err_ratio_std': [np.nan],
                    'exec_time': [exec_time],
                    'exec_time_std': [np.nan],
                    'count': 1
                }
            else:
                grouped_results[key]['err_rate'].append(err_rate)
                grouped_results[key]['err_ratio'].append(err_ratio)
                grouped_results[key]['exec_time'].append(exec_time)
                grouped_results[key]['count'] += 1
    
    # 将分组后的结果重新转换为DataFrame
    result_rows = []
    
    for key, values in grouped_results.items():
        # 再次应用logmeanexp和logstdexp来合并多个文件中的相同实验结果
        if len(values['err_rate']) > 0:
            valid_rates = [r for r in values['err_rate'] if not np.isnan(r)]
            if valid_rates:
                avg_err_rate = logmeanexp(np.array(valid_rates))
                std_err_rate = logstdexp(np.array(valid_rates)) if len(valid_rates) > 1 else np.nan
            else:
                avg_err_rate = np.nan
                std_err_rate = np.nan
        else:
            avg_err_rate = np.nan
            std_err_rate = np.nan
        
        valid_ratios = [r for r in values['err_ratio'] if not np.isnan(r)]
        avg_err_ratio = np.mean(valid_ratios) if valid_ratios else np.nan
        std_err_ratio = np.std(valid_ratios) if len(valid_ratios) > 1 else np.nan
        
        valid_times = [t for t in values['exec_time'] if not np.isnan(t)]
        avg_exec_time = np.mean(valid_times) if valid_times else np.nan
        std_exec_time = np.std(valid_times) if len(valid_times) > 1 else np.nan
        
        # 检查键的长度是否与列名匹配
        if len(key) != len(group_columns):
            print(f"警告: 键长度 ({len(key)}) 与分组列数 ({len(group_columns)}) 不匹配")
            print(f"键: {key}")
            print(f"分组列: {group_columns}")
            # 调整键长度以匹配分组列
            if len(key) < len(group_columns):
                key = tuple(list(key) + [None] * (len(group_columns) - len(key)))
            else:
                key = key[:len(group_columns)]
        
        row = dict(zip(group_columns, key))
        row['err_rate'] = avg_err_rate
        row['err_rate_std'] = std_err_rate
        row['err_ratio'] = avg_err_ratio
        row['err_ratio_std'] = std_err_ratio
        row['exec_time'] = avg_exec_time
        row['exec_time_std'] = std_exec_time
        row['sample_count'] = values['count']
        
        # 为方差控制实验添加额外字段
        if is_variance_controlled:
            for field_name in ['convergence_rate', 'actual_samples', 'actual_samples_std', 'avg_actual_std', 'std_actual_std']:
                if field_name in values:
                    field_values = [v for v in values[field_name] if not (np.isscalar(v) and np.isnan(v))]
                    if field_values:
                        row[field_name] = np.mean(field_values)
                    else:
                        row[field_name] = np.nan
        
        result_rows.append(row)
    
    # 创建最终的DataFrame
    result_df = pd.DataFrame(result_rows)
    print(f"最终结果包含 {len(result_df)} 条数据")
    
    return result_df

def plot_general(data_df, x_param, y_param, row_param=None, col_param=None, curve_param=None, 
                 figsize=(15, 10), dpi=100, x_log_scale=True, y_log_scale=False, grid=True, legend=True,
                 title=None, xlabel=None, ylabel=None, markers=None, colors=None, 
                 linestyles=None, filename=None, error_bar=False, share_x_axis=True, share_y_axis=True):
    """
    根据提供的参数绘制实验结果图表
    
    参数:
        data_df: 包含实验数据的DataFrame
        x_param: 横坐标参数名
        y_param: 纵坐标参数名
        row_param: 纵向子图参数名，默认None
        col_param: 横向子图参数名，默认None
        curve_param: 曲线参数名，默认None
        figsize: 图形大小，默认(15, 10)
        dpi: 图像分辨率，默认100
        x_log_scale: 是否使用对数尺度，默认True
        y_log_scale: 是否使用对数尺度，默认False
        grid: 是否显示网格，默认True
        legend: 是否显示图例，默认True
        title: 图表标题，默认None
        xlabel: x轴标签，默认None（使用x_param）
        ylabel: y轴标签，默认None（使用y_param）
        markers: 标记样式映射，默认None
        colors: 颜色映射，默认None
        linestyles: 线型映射，默认None
        filename: 保存图像的文件名，默认None（不保存）
        error_bar: 是否显示误差条，默认False
        share_x_axis: 是否共享X轴，默认True
        share_y_axis: 是否共享Y轴，默认True
        
    返回:
        matplotlib Figure对象
    """
    # 检查数据与参数
    if data_df.empty:
        print("警告：数据为空，无法绘图")
        return None
        
    for param in [x_param, y_param, row_param, col_param, curve_param]:
        if param is not None and param not in data_df.columns:
            print(f"警告：参数 '{param}' 不在数据中，无法绘图")
            return None
            
    # 获取子图划分的唯一值
    row_values = [None] if row_param is None else sorted(data_df[row_param].unique())
    col_values = [None] if col_param is None else sorted(data_df[col_param].unique())
    
    # 创建子图网格
    n_rows = len(row_values)
    n_cols = len(col_values)
    
    # 根据share_x_axis和share_y_axis参数设置sharex和sharey
    sharex = 'col' if share_x_axis else False
    sharey = 'row' if share_y_axis else False
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi,
                            squeeze=False, sharex=sharex, sharey=sharey)
    
    # 设置全局标题
    if title:
        fig.suptitle(title, fontsize=16)
    
    # 为不同曲线设置默认样式
    default_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    default_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # 如果有曲线参数，则获取其唯一值
    curve_values = [None] if curve_param is None else sorted(data_df[curve_param].unique())
    
    # 创建样式映射
    if markers is None:
        markers = {val: default_markers[i % len(default_markers)] for i, val in enumerate(curve_values)}
    if colors is None:
        colors = {val: default_colors[i % len(default_colors)] for i, val in enumerate(curve_values)}
    if linestyles is None:
        linestyles = {val: default_linestyles[i % len(default_linestyles)] for i, val in enumerate(curve_values)}
    
    # 检查是否有对应的标准差列
    y_std_param = f"{y_param}_std"
    has_std_data = y_std_param in data_df.columns
    
    if error_bar and not has_std_data:
        print(f"警告：未找到'{y_std_param}'列，无法显示误差条")
        error_bar = False
    
    # 获取x轴参数的最小和最大值，用于设置轴范围
    x_min = data_df[x_param].min()
    x_max = data_df[x_param].max()
    
    # 如果是对数刻度，可以稍微扩展范围，但不要太远
    if x_log_scale:
        x_min = x_min * 0.9  # 稍微向下扩展
        x_max = x_max * 1.1  # 稍微向上扩展
    
    # 遍历每个子图
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]
            
            # 过滤当前子图的数据
            subplot_data = data_df.copy()
            if row_param is not None:
                subplot_data = subplot_data[subplot_data[row_param] == row_val]
            if col_param is not None:
                subplot_data = subplot_data[subplot_data[col_param] == col_val]
                
            if subplot_data.empty:
                ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center')
                continue
                
            # 为每个曲线值绘制一条曲线
            for curve_val in curve_values:
                curve_data = subplot_data
                if curve_param is not None:
                    curve_data = curve_data[curve_data[curve_param] == curve_val]
                
                if curve_data.empty:
                    continue
                
                # 按照x参数对数据进行分组和排序
                if error_bar and has_std_data:
                    grouped_curve_data = curve_data.groupby(x_param, as_index=False)[[y_param, y_std_param]].mean()
                else:
                    grouped_curve_data = curve_data.groupby(x_param, as_index=False)[y_param].mean()
                grouped_curve_data = grouped_curve_data.sort_values(by=x_param)
                
                # 绘制曲线
                marker = markers.get(curve_val, 'o')
                color = colors.get(curve_val, 'blue')
                linestyle = linestyles.get(curve_val, '-')
                
                label = f"{curve_param}={curve_val}" if curve_param else None
                
                # 如果启用误差条且有标准差数据，则使用errorbar绘图
                if error_bar and has_std_data and not grouped_curve_data[y_std_param].isnull().all():
                    # 对数据进行处理，避免标准差为负数
                    yerr = grouped_curve_data[y_std_param].values
                    # 对于err_rate这样的对数值，标准差需要特殊处理
                    if y_param == 'err_rate':
                        # 将标准差转换为误差条的上下界
                        y_val = grouped_curve_data[y_param].values
                        # yerr_upper = np.exp(np.log(y_val) + yerr) - y_val
                        # yerr_lower = y_val - np.exp(np.log(y_val) - yerr)
                        # yerr = [yerr_lower, yerr_upper]
                    
                    ax.errorbar(grouped_curve_data[x_param], grouped_curve_data[y_param],
                            yerr=yerr, marker=marker, color=color, linestyle=linestyle,
                            linewidth=2, markersize=8, label=label, capsize=5)
                else:
                    ax.plot(grouped_curve_data[x_param], grouped_curve_data[y_param],
                            marker=marker, color=color, linestyle=linestyle,
                            linewidth=2, markersize=8, label=label)
            
            # 设置轴标签 - 为每个子图都设置标签，如果不共享坐标轴
            if (i == n_rows - 1) or not share_x_axis:  # 如果是最下面一行或不共享X轴
                ax.set_xlabel(xlabel if xlabel else x_param)
            if (j == 0) or not share_y_axis:  # 如果是最左边一列或不共享Y轴
                ax.set_ylabel(ylabel if ylabel else y_param)
                
            # 设置子图标题
            subplot_title = ""
            if row_param is not None:
                subplot_title += f"{row_param}={row_val}"
            if col_param is not None:
                if subplot_title:
                    subplot_title += ", "
                subplot_title += f"{col_param}={col_val}"
            if subplot_title:
                ax.set_title(subplot_title)
                
            # 设置x轴范围
            ax.set_xlim(x_min, x_max)
            
            # 设置对数刻度
            if x_log_scale:
                ax.set_xscale('log')
            if y_log_scale:
                ax.set_yscale('log')
                    
            # 添加网格
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
                
            # 添加图例
            if legend and curve_param is not None:
                ax.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # 为标题腾出空间
    
    # 保存图形
    if filename:
        plt.savefig(filename, dpi=dpi)# , bbox_inches='tight'
        print(f"figure saved at: {filename}")
    
    return fig

def filename_filter(tag, log_dir='logs'):
    """
    Search for files in the logs directory that start with the given tag.
    """
    log_dir = os.path.join(log_dir, tag)
    if not os.path.exists(log_dir):
        print(f"Warning: Directory '{log_dir}' does not exist")
        return []

    all_files = os.listdir(log_dir)
    matching_files = [os.path.join(log_dir, file) for file in all_files if file.startswith(tag)]
        
    print(f"Found {len(matching_files)} files matching tag '{tag}'")
    
    return matching_files

if __name__ == "__main__":
    run_general = True
    general_experiment = 'general_experiment_small'
    run_vc = True
    vc_experiment = 'vc_small'
    
    if run_general:
        # name_list = filename_filter('chase_diff_sampler_conv')
        name_list = filename_filter(general_experiment)
        print(name_list)
        result_df = read_and_filter_logs(name_list, select_params={
            # 'code_type': 'hamming', 
            # 'sigma': 0.4,
            'r': [3, 5, 7],
            # 'sampler': ['naive', 'bessel', 'sym_direct', 'cutoff']
            })
        print(result_df)
        fig0 = plot_general(
            data_df=result_df, 
            x_param='snr', 
            y_param='err_rate',
            col_param='r',
            row_param='decoder_type',
            curve_param='sampler',
            title='Error Rate vs SNR',
            xlabel='Sample Size',
            ylabel='error rate (log10)',
            x_log_scale=False,
            y_log_scale=False,
            # error_bar=True,  # 启用误差条
            share_y_axis=False,
            filename='plots/general_full_err_rate_vs_snr.png'
        )

        fig1 = plot_general(
            data_df=result_df, 
            x_param='snr', 
            y_param='exec_time',
            col_param='r',
            row_param='decoder_type',
            curve_param='sampler',
            title='Execution Time vs SNR',
            xlabel='Sample Size',
            ylabel='Execution Time (s)',
            x_log_scale=False,
            y_log_scale=True,
            # error_bar=True,  # 启用误差条
            share_y_axis=False,
            filename='plots/general_full_time_vs_snr.png'
        )

        # fig2 = plot_general(
        #     data_df=result_df, 
        #     x_param='snr', 
        #     y_param='err_rate',
        #     col_param='r',
        #     row_param='decoder_type',
        #     curve_param='sampler',
        #     title='Error Rate vs SNR with Error Bars',
        #     xlabel='Sample Size',
        #     ylabel='error rate (log10)',
        #     x_log_scale=False,
        #     y_log_scale=False,
        #     error_bar=True,  # 启用误差条
        #     share_y_axis=False,
        #     filename='plots/general_full_errorbar_vs_snr.png'
        # )

    if run_vc:
        # 方差控制实验示例
        name_list = filename_filter(vc_experiment)
        if name_list:
            result_df = read_and_filter_logs(name_list, select_params={
                'r': [3, 5, 7],
                # 'code_type': 'hamming',
                # 'target_std': [0.05],  # 筛选特定的目标标准差
                # 'sigma': [0.2, 0.3, 0.4],  # 筛选特定噪声水平
                # 'max_samples': 10000000,  # 限制最大样本数
                # 'sampler': ['naive', 'sym_direct'],  # 筛选特定采样器
                # 'decoder_type': ['chase', 'chase2', 'binary'],  # 筛选特定的解码器类型
            })
            print("方差控制实验数据:")
            print(result_df)
            
            # 检查是否有actual_samples数据
            if 'actual_samples' in result_df.columns:
                print("找到actual_samples数据，可以绘制样本数图表")

                fig0 = plot_general(
                    data_df=result_df, 
                    x_param='snr', 
                    y_param='err_rate',
                    col_param='r',  # 不同码长作为列
                    # col_param='sigma',  # 不同噪声水平作为列
                    row_param='decoder_type',  # 不同的解码器作为行
                    curve_param='sampler',  # 不同采样器作为曲线
                    title='Error Rate vs SNR (Variance Controlled)',
                    xlabel='SNR',
                    ylabel='error rate (log10)',
                    x_log_scale=False,
                    # error_bar=True,  # 启用误差条
                    share_y_axis=False,
                    filename='plots/sigmas_vc_full_err_rate_vs_snr.png'
                )
                
                # 绘制实际样本数随r变化的图
                fig1 = plot_general(
                    data_df=result_df, 
                    x_param='snr', 
                    y_param='actual_samples',  # 使用实际样本数作为y轴
                    col_param='r',  # 不同码长作为列
                    # col_param='sigma',  # 不同噪声水平作为列
                    row_param='decoder_type',  # 不同的解码器作为行
                    curve_param='sampler',  # 不同采样器作为曲线
                    title='Actual Samples vs SNR (Variance Controlled)',
                    xlabel='SNR',
                    ylabel='Actual Samples',
                    x_log_scale=False,
                    y_log_scale=True,
                    error_bar=False,  # 启用误差条显示标准差
                    share_y_axis=True,
                    filename='plots/sigmas_vc_full_actual_samples_vs_snr.png'
                )
                
                # 绘制执行时间随r变化的图（作为对比）
                fig2 = plot_general(
                    data_df=result_df,
                    x_param='snr',
                    y_param='exec_time',  # 使用执行时间作为y轴
                    col_param='r',  # 不同码长作为列
                    # col_param='sigma',  # 不同噪声水平作为列
                    row_param='decoder_type',  # 不同的解码器作为行
                    curve_param='sampler',  # 不同采样器作为曲线
                    title='Execution Time vs SNR (Variance Controlled)',
                    xlabel='SNR',
                    ylabel='Time (s)',
                    x_log_scale=False,
                    y_log_scale=True,
                    share_y_axis=False,
                    error_bar=True,
                    filename='plots/sigmas_vc_full_time_vs_snr.png'
                )
                
            else:
                print("未找到actual_samples数据，请检查实验数据")
                print("可用的列:", result_df.columns.tolist())
            
        else:
            print("未找到方差控制实验文件")

    plt.show()