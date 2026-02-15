import numpy as np
import itertools
import json
import os
# from heapq import heappush, heappop
from code_linear import *
from RM import RMCode

class ORBGRANDDecoder:
    """
    ORBGRAND decoder, a soft decoder suitable for all linear codes.
    """
    def __init__(self, code, max_log_weight=25, judge_bs=20):
        self.code = code
        self.decoder_type = "ORBGRAND"
        if isinstance(code, RMCode):
            self.code_type = "RM"
        elif isinstance(code, HammingCode):
            self.code_type = "Hamming"
        else:
            raise ValueError("Unsupported code type. Please use RMCode or HammingCode.")
        self.n = code.dim
        
        self.max_log_weight = max_log_weight
        self.judge_bs = judge_bs  # 批处理大小参数
        self.noise_lib_file = f"libs/ORB_noise_lib_n{self.n}_max{max_log_weight}_vec.json"
        
        # 生成位置权重向量：[1,2,3,...,n]
        self.pos_weights = 1 + np.arange(self.n)
        
        # 预生成噪声模式库（如果不存在）
        if not os.path.exists(self.noise_lib_file):
            print(f"噪声模式库不存在，正在生成： n={self.n}, max_log_weight={max_log_weight}...")
            self._precompute_noise_library()
        
        # 加载噪声库 {weighted_sum: [[0,1,0,...],...]}
        with open(self.noise_lib_file, 'r') as f:
            self.noise_library = json.load(f)
        
        # 将噪声库转换为numpy数组，按加权和排序
        self.noise_patterns_array = np.zeros((0, self.n), dtype=int)
        
        # 顺序连接所有噪声模式
        for weight in sorted(map(int, self.noise_library.keys())):
            patterns = np.array(self.noise_library[str(weight)])
            if len(patterns) > 0:
                self.noise_patterns_array = np.vstack([self.noise_patterns_array, patterns])

    def get_r(self, tx_bin=None):
        """
        获取此算法的最小错误距离
        """
        if self.code_type == "Hamming":
            return np.sqrt(8/3)
        elif self.code_type == "RM":
            logd = self.code.m - self.code.r
            if logd == 0:
                return 1
            elif logd == 1:
                return np.sqrt(2)
            elif logd == 2:
                return 2
            elif logd == 3:
                return np.sqrt(8-1/4)
            else:
                raise ValueError("Not yet studied.")
        else:
            raise ValueError("Unsupported code type. Please use HammingCode or RMCode.")
    
    def _precompute_noise_library(self):
        """预生成所有加权和<=max_log_weight的噪声模式"""
        noise_lib = {}
        
        # 遍历所有可能的汉明权重（翻转位数）
        max_choose_numbers = min(self.n, int(np.sqrt(2*self.max_log_weight)+1))
        for choose_numbers in range(max_choose_numbers + 1):
            # 生成所有weight位的组合
            for bits in itertools.combinations(range(min(self.n, self.max_log_weight)), choose_numbers):
                # 创建0-1噪声向量
                noise_vec = np.zeros(self.n, dtype=int)
                noise_vec[list(bits)] = 1
                
                # 计算加权和：(位置索引+1) * 噪声值 的和
                weighted_sum = np.sum(self.pos_weights * noise_vec)
                
                if weighted_sum > self.max_log_weight:
                    continue
                    
                # 以字符串形式存储加权和作为键
                if str(weighted_sum) not in noise_lib:
                    noise_lib[str(weighted_sum)] = []
                    
                noise_lib[str(weighted_sum)].append(noise_vec.tolist())
        
        # 每个array中按照1的个数从大到小重排
        for key, patterns in noise_lib.items():
            patterns.sort(key=lambda x: sum(x), reverse=True)
            noise_lib[key] = patterns

        # 按加权和排序
        sorted_weights = sorted(map(int, noise_lib.keys()))
        sorted_lib = {str(w): noise_lib[str(w)] for w in sorted_weights}
        sorted_lib['0'] = [[0] * self.n]  # 添加全0模式
        
        with open(self.noise_lib_file, 'w') as f:
            json.dump(sorted_lib, f)
    
    def decode(self, received_signals):
        # 初始化解码结果数组，只包含数据位
        batch_size = received_signals.shape[0]
        
        # 并行硬判决（-1->0, 1->1）
        hard_decisions = (received_signals > 0).astype(int)
        
        rel_orders = np.argsort(np.abs(received_signals), axis=1)# (batch_size, n)
        sorted_hard = np.take_along_axis(hard_decisions, rel_orders, axis=1)
        rel_orders_back = np.zeros_like(rel_orders, dtype=int)
        rel_orders_back[np.arange(batch_size)[:, np.newaxis], rel_orders] = np.arange(self.n)
        # hard_back_back = np.take_along_axis(sorted_hard, rel_orders_back, axis=1)
        # assert np.all(hard_back_back == hard_decisions)

        n_noise = len(self.noise_patterns_array)
        decoded = hard_decisions.copy()
        
        max_queries = min(10000, n_noise)
        found = np.zeros(batch_size, dtype=bool)
        for start_idx in range(0, max_queries, self.judge_bs):
            end_idx = min(start_idx + self.judge_bs, max_queries)

            # 如果所有样本都已找到解，提前结束
            if np.all(found):
                break

            noised_sorted = (sorted_hard[~found, None, :] + self.noise_patterns_array[start_idx:end_idx, :]) % 2 # (batch_size, judge_bs, n)
            noised_codes = np.take_along_axis(noised_sorted, rel_orders_back[~found][:, None, :], axis=2) # (batch_size, judge_bs, n)
            valid = self.code.is_codeword(noised_codes) # (batch_size, judge_bs)

            # new_found = np.logical_and(np.any(valid, axis=1), ~found) # (batch_size,)
            # found = np.logical_or(found, new_found) # 更新找到有效码字的样本
            new_found = np.any(valid, axis=1) # (batch_size,)
            if not np.any(new_found):
                continue
            found_old = found.copy()
            found[~found] = new_found # 更新找到有效码字的样本
            new_found_long = np.logical_and(found, ~found_old) # 只处理新找到的样本
            # 只处理找到有效码字的样本

            first_valid = np.argmax(valid[new_found] * (self.judge_bs - np.arange(end_idx - start_idx)), axis=1) # (batch_size,)
            decoded[new_found_long] = noised_codes[new_found, first_valid, :] # (batch_size, n)
            
        returned = self.code.decode_binary(decoded)
        # 如果输入是一维数组，返回一维数组
        return returned

# ---------------------------
# 使用示例
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    r=3
    code = HammingCode(r)
    # code = RMCode(2, 5)
    decoder = ORBGRANDDecoder(code=code, max_log_weight=25, judge_bs=5**2)
    code.set_decoder(decoder)
    
    n = decoder.n
    # bs = 1e4
    # sigma = 0.5
    # # 生成随机接收信号
    # received = -1 * np.ones((int(bs), n)) + sigma * np.random.randn(int(bs), n)

    received = -1 * np.ones((1, n))
    received[0, [0,1,3]] = [0.1, 1.01, 1.1]

    import time
    start_time = time.time()
    decoded1 = code.decode(received)
    end_time = time.time()
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print(np.mean(np.all(decoded1 == 0, axis=1)))  # 检查是否为0码字