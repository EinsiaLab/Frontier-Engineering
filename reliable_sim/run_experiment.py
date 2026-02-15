#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReliableSim实验运行脚本
提供简化的命令行接口，用于运行各种通信系统误码率实验

使用方法示例：
    # 基础汉明码实验
    python run_experiment.py --r 3 --sigma 0.3 --samples 10000
    
    # 采样器性能比较
    python run_experiment.py --experiment convergence --r 4 5 --samplers naive bessel sym
    
    # 方差控制实验
    python run_experiment.py --experiment variance_control --r 5 --target-std 0.05
    
    # 快速测试
    python run_experiment.py --quick-test
"""

import sys
import os
import json
from datetime import datetime

# 将当前目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_general import run_from_args, parse_args

def print_help_examples():
    """打印使用示例"""
    print("""
ReliableSim实验运行器 - 使用示例

1. 基础实验
   python run_experiment.py --r 3 --sigma 0.3 --samples 10000
   
2. 采样器性能比较
   python run_experiment.py --experiment convergence --r 4 5 6 --samplers naive bessel sym
   
3. 码长影响分析
   python run_experiment.py --experiment code_length --r 3 4 5 6 7 --sigma 0.3 --samples 100000
   
4. 方差控制实验
   python run_experiment.py --experiment variance_control --r 4 --sigma 0.35 --target-std 0.1
   
5. 快速测试（小参数）
   python run_experiment.py --quick-test
   
6. 详细输出模式
   python run_experiment.py --experiment convergence --r 4 --sigma 0.3 --verbose
   
7. 指定解码器
   python run_experiment.py --decoder binary ORBGRAND --r 4 --sigma 0.3
   
8. RM码实验
   python run_experiment.py --code rm --rm "1,3" "2,4" --sigma 0.4

参数说明：
  --experiment: 实验类型 [general|convergence|code_length|variance_control]
  --r: 汉明码参数r (3-8)
  --sigma: 噪声标准差 (0.1-1.0)
  --samples: 样本数量 (1e3-1e7)
  --samplers: 采样器类型 [naive|bessel|sym|sym_direct|...]
  --decoder: 解码器类型 [binary|ORBGRAND|SGRAND|chase|...]
""")

def create_config_file():
    """创建示例配置文件"""
    config = {
        "experiments": [
            {
                "name": "basic_hamming_test",
                "description": "基础汉明码测试",
                "parameters": {
                    "experiment": "general",
                    "code": ["hamming"],
                    "r": [3, 4, 5],
                    "decoder": ["binary"],
                    "samplers": ["naive", "bessel", "sym"],
                    "sigma": [0.5, 0.4, 0.3],
                    "samples": [1e4],
                    "repetitions": 1
                }
            },
            {
                "name": "convergence_analysis",
                "description": "采样器收敛性分析",
                "parameters": {
                    "experiment": "convergence",
                    "code": ["hamming"],
                    "r": [4],
                    "decoder": ["binary"],
                    "samplers": ["naive", "bessel", "sym"],
                    "sigma": [0.3],
                    "samples": [1e3, 1e4, 1e5, 1e6],
                    "repetitions": 3
                }
            },
            {
                "name": "variance_control_test",
                "description": "方差控制实验",
                "parameters": {
                    "experiment": "variance_control",
                    "code": ["hamming"],
                    "r": [5],
                    "decoder": ["binary"],
                    "samplers": ["naive", "sym"],
                    "sigma": [0.35, 0.3, 0.25],
                    "target_std": [0.1, 0.05],
                    "max_samples": 1e6,
                    "repetitions": 1
                }
            }
        ]
    }
    
    with open('experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("已创建配置文件: experiment_config.json")

def run_from_config(config_file):
    """从配置文件运行实验"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for exp in config["experiments"]:
            print(f"\n运行实验: {exp['name']} - {exp['description']}")
            
            # 将参数字典转换为命令行参数
            params = exp["parameters"]
            sys.argv = ['run_experiment.py']
            
            for key, value in params.items():
                if isinstance(value, list):
                    sys.argv.extend([f'--{key.replace("_", "-")}'] + [str(v) for v in value])
                else:
                    sys.argv.extend([f'--{key.replace("_", "-")}', str(value)])
            
            # 运行实验
            args = parse_args()
            results = run_from_args(args)
            
            print(f"实验 {exp['name']} 完成，获得 {len(results)} 条结果")
            
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_file} 不存在")
        create_config_file()
    except Exception as e:
        print(f"错误: 读取配置文件失败 - {e}")

def main():
    """主函数"""
    
    # 检查是否需要帮助
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help_examples()
        return
    
    # 检查是否需要创建配置文件
    if '--create-config' in sys.argv:
        create_config_file()
        return
    
    # 检查是否使用配置文件
    if '--config' in sys.argv:
        try:
            config_index = sys.argv.index('--config')
            config_file = sys.argv[config_index + 1]
            run_from_config(config_file)
            return
        except (IndexError, ValueError):
            print("错误: 请指定配置文件路径")
            print("用法: python run_experiment.py --config experiment_config.json")
            return
    
    # 正常运行实验
    try:
        args = parse_args()
        
        # 记录开始时间
        start_time = datetime.now()
        print(f"实验开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行实验
        results = run_from_args(args)
        
        # 记录结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n实验完成！")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {duration.total_seconds():.1f} 秒")
        print(f"获得结果数: {len(results)}")
        
        # 显示最新结果
        if results:
            latest_result = results[-1]
            print(f"\n最新结果示例:")
            print(f"  码类型: {latest_result.get('code_type', 'N/A')}")
            print(f"  解码器: {latest_result.get('decoder_type', 'N/A')}")
            print(f"  采样器: {latest_result.get('sampler', 'N/A')}")
            print(f"  噪声水平: {latest_result.get('sigma', 'N/A')}")
            print(f"  误码率: {latest_result.get('err_ratio', 'N/A'):.2e}")
            print(f"  执行时间: {latest_result.get('exec_time', 'N/A'):.2f}s")
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        print("使用 --help 查看使用示例")

if __name__ == "__main__":
    main()