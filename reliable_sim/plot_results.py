#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

def plot_error_rates(code_lengths, info_lengths, err_rates, err_ratios, sigma_values, samplers):
    """绘制误码率图像"""
    # 创建图形和子图
    fig, axes = plt.subplots(1, len(sigma_values), figsize=(5*len(sigma_values), 6), sharey=False)
    
    # 为每个sigma创建一个子图
    for i, sigma in enumerate(sigma_values):
        ax = axes[i] if len(sigma_values) > 1 else axes
        
        # 绘制每个采样器的结果
        for sampler_info in samplers:
            sampler_name = sampler_info["name"]
            color = sampler_info["color"]
            markers = sampler_info.get("markers", [sampler_info.get("marker", 'o')])
            
            # 绘制对数误码率
            ax.plot(code_lengths, np.array(err_rates[sampler_info["name"]][str(sigma)]) / np.log(10),
                    marker=markers[0],
                    linestyle='-',
                    color=color,
                    label=f'{sampler_name} sampling (rate)')
            
            # 对于非Naive采样器，绘制实际误码率
            if sampler_info["name"] != "Naive" and len(markers) > 1:
                ax.plot(code_lengths, np.log10(err_ratios[sampler_info["name"]][str(sigma)]),
                        marker=markers[1],
                        linestyle='--',
                        color=color,
                        label=f'{sampler_info["name"]} sampling (ratio)')
        
        # 美化子图
        ax.set_title(f'sigma = {sigma:.2f}', fontsize=14)
        ax.set_xlabel('code length (n)', fontsize=12)
        ax.set_ylabel('Log10(error rate)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 设置x轴为对数尺度
        ax.set_xscale('log')
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)
    
    # 图例放在图形下方，给予更多空间
    fig.subplots_adjust(bottom=0.2)
    handles, labels = axes[-1].get_legend_handles_labels() if len(sigma_values) > 1 else axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    return fig

def plot_execution_times(code_lengths, info_lengths, exec_times, sigma_values, samplers):
    """绘制执行时间图像"""
    # 创建图形和子图
    fig, axes = plt.subplots(1, len(sigma_values), figsize=(5*len(sigma_values), 6), sharey=False)
    
    # 为每个sigma创建一个子图
    for i, sigma in enumerate(sigma_values):
        ax = axes[i] if len(sigma_values) > 1 else axes
        
        # 绘制每个采样器的执行时间
        for sampler_info in samplers:
            sampler_name = sampler_info["name"]
            color = sampler_info["color"]
            markers = sampler_info.get("markers", [sampler_info.get("marker", 'o')])
            
            ax.plot(code_lengths, exec_times[sampler_name][str(sigma)],
                    marker=markers[0],
                    linestyle='-',
                    color=color,
                    label=f'{sampler_name} sampling')
        
        # 美化子图
        ax.set_title(f'sigma = {sigma:.2f}', fontsize=14)
        ax.set_xlabel('code length (n)', fontsize=12)
        ax.set_ylabel('Execution time (seconds)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_yscale('log')
        
        # 设置x轴为对数尺度
        ax.set_xscale('log')
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)
    
    # 图例放在图形下方，给予更多空间
    fig.subplots_adjust(bottom=0.2)
    handles, labels = axes[-1].get_legend_handles_labels() if len(sigma_values) > 1 else axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    return fig

def plot_convergence_results(results, samplers, rs, sigmas):
    """绘制误码率收敛性图表
    
    参数:
        results: run_convergence_test返回的结果字典
        samplers: 采样器配置列表
        rs: r值列表
        sigmas: sigma值列表
    """
    # 计算子图布局
    num_plots = len(results)
    cols = len(sigmas)
    rows = len(rs)
    
    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    
    # 处理每一个r-sigma组合
    for i, (key, data) in enumerate(results.items()):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        sample_sizes = data['sample_sizes']
        
        for sampler_info in samplers:
            sampler_name = sampler_info["name"]
            color = sampler_info["color"]
            marker = sampler_info.get("marker", 'o')
            
            if sampler_name in data['samplers']:
                sampler_data = data['samplers'][sampler_name]
                
                # 计算各采样大小下的平均误码率和标准差
                err_rates_mean = np.mean(sampler_data['err_rates'], axis=1)
                err_rates_std = np.std(sampler_data['err_rates'], axis=1)
                
                # 绘制误码率曲线和误差条
                ax.errorbar(sample_sizes, 
                        err_rates_mean / np.log(10),  # 转换为log10
                        yerr=err_rates_std / np.log(10),  # 标准差也要转换
                        marker=marker,
                        color=color,
                        label=f'{sampler_name}',
                        capsize=3)
        
        # 美化子图
        n = data.get('hamming_n', 0)
        k = data.get('hamming_k', 0)
        ax.set_title(f"{key}, ({n},{k}) Hamming code", fontsize=12)
        ax.set_xlabel('Number of samples', fontsize=10)
        ax.set_ylabel('Log10(error rate)', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 如果是空子图（当总数不能被列数整除时）
        if i >= len(results):
            ax.set_visible(False)
    
    # 图例放在图形下方，给予更多空间
    fig.subplots_adjust(bottom=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), 
            bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    return fig

def load_json_files(file_list):
    """读取指定的JSON文件并返回数据列表"""
    data_list = []
    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 保存文件路径以便跟踪来源
                data['file_path'] = file_path
                data_list.append(data)
                print(f"成功加载文件: {file_path}")
        except Exception as e:
            print(f"无法加载文件 {file_path}: {e}")
    
    return data_list

def merge_convergence_results(data_list):
    """合并具有相同设置的convergence_test实验结果"""
    # 过滤出convergence_test类型的文件
    convergence_data = []
    for data in data_list:
        if data.get('test_type') == 'convergence_test':
            convergence_data.append(data)
        else:
            print(f"警告: 文件 {data.get('file_path')} 不是convergence_test类型，将被忽略")
    
    if not convergence_data:
        print("错误: 没有有效的convergence_test数据文件")
        return None
    
    # 提取所有实验参数
    all_r_values = set()
    all_sigma_values = set()
    all_samplers = set()
    all_sample_sizes = set()
    
    for data in convergence_data:
        params = data.get('parameters', {})
        
        # 收集实验参数
        r_values = [float(r) for r in params.get('r_values', [])]
        sigma_values = [float(s) for s in params.get('sigma_values', [])]
        samplers = params.get('samplers', [])
        
        all_r_values.update(r_values)
        all_sigma_values.update(sigma_values)
        all_samplers.update(samplers)
        
        # 从结果中收集采样大小
        for key, result in data.get('results', {}).items():
            for sampler_name, sampler_data in result.get('samplers', {}).items():
                for size_result in sampler_data.get('sample_results', []):
                    all_sample_sizes.add(size_result.get('num_samples'))
    
    # 转换为有序列表
    r_values = sorted(list(all_r_values))
    sigma_values = sorted(list(all_sigma_values))
    samplers = sorted(list(all_samplers))
    sample_sizes = sorted(list(all_sample_sizes))
    
    print(f"发现的参数组合:")
    print(f"  r值: {r_values}")
    print(f"  sigma值: {sigma_values}")
    print(f"  采样器: {samplers}")
    print(f"  采样大小: {sample_sizes}")
    
    # 创建合并结果的数据结构
    merged_results = {}
    
    # 对每个r-sigma组合创建结果条目
    for r in r_values:
        for sigma in sigma_values:
            key = f"r={r:.0f}_sigma={sigma:.2f}"
            merged_results[key] = {
                'sample_sizes': sample_sizes,
                'samplers': {},
                'hamming_n': 0,
                'hamming_k': 0,
            }
    
    # 合并数据
    for data in convergence_data:
        results = data.get('results', {})
        
        for key, result in results.items():
            if key in merged_results:
                # 更新码长信息
                if result.get('hamming_n') > 0:
                    merged_results[key]['hamming_n'] = result.get('hamming_n')
                    merged_results[key]['hamming_k'] = result.get('hamming_k')
                
                # 处理每个采样器的结果
                for sampler_name, sampler_data in result.get('samplers', {}).items():
                    if sampler_name not in merged_results[key]['samplers']:
                        merged_results[key]['samplers'][sampler_name] = {
                            'all_repetitions': [[] for _ in sample_sizes],
                            'size_indices': {}
                        }
                    
                    # 记录每个采样大小的索引
                    for size_result in sampler_data.get('sample_results', []):
                        num_samples = size_result.get('num_samples')
                        if num_samples in sample_sizes:
                            size_index = sample_sizes.index(num_samples)
                            merged_results[key]['samplers'][sampler_name]['size_indices'][num_samples] = size_index
                            
                            # 收集所有重复实验的结果
                            for rep in size_result.get('repetitions', []):
                                rep_result = {
                                    'err_rate': rep.get('err_rate'),
                                    'err_ratio': rep.get('err_ratio'),
                                    'exec_time': rep.get('exec_time')
                                }
                                merged_results[key]['samplers'][sampler_name]['all_repetitions'][size_index].append(rep_result)
    
    # 转换为plot_convergence_results兼容的格式
    for key, result in merged_results.items():
        for sampler_name, sampler_data in result['samplers'].items():
            all_reps = sampler_data['all_repetitions']
            
            # 计算每个采样大小的重复次数
            max_reps = max([len(reps) for reps in all_reps]) if all_reps else 0
            if max_reps == 0:
                continue
            
            # 创建numpy数组存储结果
            err_rates = np.zeros((len(sample_sizes), max_reps))
            err_ratios = np.zeros((len(sample_sizes), max_reps))
            exec_times = np.zeros((len(sample_sizes), max_reps))
            
            # 填充数组
            for i, reps in enumerate(all_reps):
                for j, rep in enumerate(reps):
                    if j < max_reps:  # 确保不超出数组边界
                        err_rates[i, j] = rep['err_rate']
                        err_ratios[i, j] = rep['err_ratio']
                        exec_times[i, j] = rep['exec_time']
            
            # 更新采样器数据
            result['samplers'][sampler_name] = {
                'err_rates': err_rates,
                'err_ratios': err_ratios,
                'exec_times': exec_times
            }
    
    return merged_results, r_values, sigma_values, samplers

def prepare_sampler_configs(sampler_names):
    """准备绘图使用的采样器配置"""
    sampler_configs = {
        'naive': {"name": "Naive", "class": None, "color": "blue", "marker": 'o', 'index': 'naive'},
        'bessel': {"name": "Bessel", "class": None, "color": "red", "marker": 'D', 'index': 'bessel'},
        'sym': {"name": "Symmetrical", "class": None, "color": "green", "marker": 's', 'index': 'sym'},
        'sym_direct': {"name": "Sym_Direct", "class": None, "color": "purple", "marker": 'x', 'index': 'sym_direct'},
    }
    
    sampler_list = []
    for name in sampler_names:
        if name.lower() in sampler_configs:
            sampler_list.append(sampler_configs[name.lower()])
    
    return sampler_list

def visualize_convergence_results(merged_results, r_values, sigma_values, samplers, output_prefix=None):
    """使用合并的结果绘制收敛性图表"""
    # 准备采样器配置
    sampler_list = prepare_sampler_configs(samplers)
    
    # 调用绘图函数
    fig = plot_convergence_results(merged_results, sampler_list, r_values, sigma_values)
    
    # 如果指定了输出前缀，保存图像
    if output_prefix:
        timestamp = datetime.datetime.now().strftime("%m%d")
        output_path = f"images/{output_prefix}_convergence_{timestamp}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_path}")
    
    plt.show()
    return fig

def merge_code_length_results(data_list):
    """合并code_length_test实验结果"""
    # 过滤出code_length_test类型的文件
    code_length_data = []
    for data in data_list:
        if data.get('test_type') == 'code_length_test':
            code_length_data.append(data)
    
    if not code_length_data:
        print("错误: 没有有效的code_length_test数据文件")
        return None, None, None, None
    
    # 提取所有实验参数
    all_r_values = set()
    all_sigma_values = set()
    all_samplers = set()
    code_lengths = []
    info_lengths = []
    
    # 收集所有参数和代码长度信息
    for data in code_length_data:
        # 从参数中收集r值和sigma值
        params = data.get('parameters', {})
        r_values_param = [float(r) for r in params.get('r_values', [])]
        sigma_values_param = [float(s) for s in params.get('sigma_values', [])]
        
        all_r_values.update(r_values_param)
        all_sigma_values.update(sigma_values_param)
        
        # 从结果收集采样器名称和代码长度
        results = data.get('results', {})
        for r_key, r_data in results.items():
            # 提取n和k值
            n = r_data.get('n', 0)
            k = r_data.get('k', 0)
            
            # 检查是否已存在该代码长度
            if n not in code_lengths and n > 0:
                code_lengths.append(n)
                info_lengths.append(k)
            
            # 从sigma结果中收集采样器
            sigmas_data = r_data.get('sigmas', {})
            for sigma_key, sigma_data in sigmas_data.items():
                samplers_data = sigma_data.get('samplers', {})
                for sampler_name in samplers_data.keys():
                    all_samplers.add(sampler_name)
    
    # 确保代码长度和信息长度一一对应并排序
    if len(code_lengths) > 0:
        # 创建元组对并排序
        pairs = sorted(zip(code_lengths, info_lengths))
        code_lengths = [p[0] for p in pairs]
        info_lengths = [p[1] for p in pairs]
    
    print(f"发现的参数组合:")
    print(f"  r值: {sorted(list(all_r_values))}")
    print(f"  sigma值: {sorted(list(all_sigma_values))}")
    print(f"  采样器: {sorted(list(all_samplers))}")
    print(f"  代码长度: {code_lengths}")
    print(f"  信息长度: {info_lengths}")
    
    # 转换为有序列表
    r_values = sorted(list(all_r_values))
    sigma_values = sorted(list(all_sigma_values))
    samplers = sorted(list(all_samplers))
    
    # 创建结果存储结构
    err_rates = {}
    err_ratios = {}
    exec_times = {}
    
    # 初始化结果字典 - 使用采样器名称作为键
    for sampler in samplers:
        err_rates[sampler] = {}
        err_ratios[sampler] = {}
        exec_times[sampler] = {}
        
        for sigma in sigma_values:
            sigma_str = str(sigma)
            err_rates[sampler][sigma_str] = [0] * len(code_lengths)  # 初始化为0
            err_ratios[sampler][sigma_str] = [0] * len(code_lengths)
            exec_times[sampler][sigma_str] = [0] * len(code_lengths)
    
    # 从新格式中提取结果
    for data in code_length_data:
        results = data.get('results', {})
        
        for r_key, r_data in results.items():
            n = r_data.get('n', 0)
            if n in code_lengths:
                code_index = code_lengths.index(n)
                
                # 处理每个sigma的结果
                sigmas_data = r_data.get('sigmas', {})
                for sigma_key, sigma_data in sigmas_data.items():
                    # 从键名提取sigma值，格式为"sigma=X.XX"
                    sigma_value = float(sigma_key.split('=')[1])
                    sigma_str = str(sigma_value)
                    
                    # 处理每个采样器
                    samplers_data = sigma_data.get('samplers', {})
                    for sampler_name, sampler_results in samplers_data.items():
                        if sampler_name in samplers:
                            # 保存误码率、误码比例和执行时间
                            if 'err_rate' in sampler_results:
                                err_rates[sampler_name][sigma_str][code_index] = sampler_results['err_rate']
                            
                            if 'err_ratio' in sampler_results:
                                err_ratios[sampler_name][sigma_str][code_index] = sampler_results['err_ratio']
                            
                            if 'exec_time' in sampler_results:
                                exec_times[sampler_name][sigma_str][code_index] = sampler_results['exec_time']
    
    # 返回合并的结果和参数
    merged_results = {
        'code_lengths': code_lengths,
        'info_lengths': info_lengths,
        'err_rates': err_rates,
        'err_ratios': err_ratios,
        'exec_times': exec_times
    }
    
    return merged_results, r_values, sigma_values, samplers

def visualize_code_length_results(merged_results, samplers, sigma_values, output_prefix=None):
    """使用合并的结果绘制码长测试图表"""
    # 准备采样器配置
    sampler_list = prepare_sampler_configs(samplers)
    
    # 逆序排列sigma值（从大到小）
    sorted_sigma_values = sorted(sigma_values, reverse=True)
    
    # 调用误码率绘图函数
    fig = plot_error_rates(merged_results['code_lengths'], 
                    merged_results['info_lengths'],
                    merged_results['err_rates'], 
                    merged_results['err_ratios'], 
                    sorted_sigma_values, 
                    sampler_list)
    
    # 如果指定了输出前缀，保存图像
    if output_prefix:
        timestamp = datetime.datetime.now().strftime("%m%d")
        plt.savefig(f"images/{output_prefix}_code_length_error_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure()  # 创建新图形
    
    # 调用执行时间绘图函数
    fig = plot_execution_times(merged_results['code_lengths'], 
                        merged_results['info_lengths'],
                        merged_results['exec_times'], 
                        sorted_sigma_values, 
                        sampler_list)
    
    # 如果指定了输出前缀，保存图像
    if output_prefix:
        timestamp = datetime.datetime.now().strftime("%m%d")
        plt.savefig(f"images/{output_prefix}_code_length_time_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """主函数：解析命令行参数并处理文件"""
    parser = argparse.ArgumentParser(description='合并和可视化汉明码实验结果')
    parser.add_argument('files', nargs='+', help='要处理的JSON结果文件')
    parser.add_argument('--type', choices=['convergence', 'code_length'], default='convergence',
                      help='实验类型 (默认: convergence)')
    parser.add_argument('--output', '-o', help='输出图像的前缀')
    parser.add_argument('--output-dir', '-d', default='images', help='图像输出目录 (默认: images)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 如果指定了输出前缀，构建完整路径
    output_prefix = None
    if args.output:
        output_prefix = os.path.join(args.output_dir, args.output)
    
    # 加载指定的JSON文件
    data_list = load_json_files(args.files)
    
    if not data_list:
        print("错误: 未能加载任何有效数据文件")
        return
    
    # 处理数据并绘图
    if args.type == 'convergence':
        merged_results, r_values, sigma_values, samplers = merge_convergence_results(data_list)
        if merged_results:
            visualize_convergence_results(merged_results, r_values, sigma_values, samplers, output_prefix)
    elif args.type == 'code_length':
        merged_results, r_values, sigma_values, samplers = merge_code_length_results(data_list)
        if merged_results:
            visualize_code_length_results(merged_results, samplers, sigma_values, output_prefix)

if __name__ == "__main__":
    # main()
    filename=["logs/convergence_test_ORBGRAND_20250427_100828.json"]
    data_list = load_json_files(filename)
    # merged_results, r_values, sigma_values, samplers = merge_code_length_results(data_list)
    # visualize_code_length_results(merged_results, samplers, sigma_values, "ORBGRAND")
    merged_results, r_values, sigma_values, samplers = merge_convergence_results(data_list)
    visualize_convergence_results(merged_results, r_values, sigma_values, samplers, "ORBGRAND")
