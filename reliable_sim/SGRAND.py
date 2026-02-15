import numpy as np
import scipy
import itertools
import json
import os
from heapq import heappush, heappop
from code_linear import *
from RM import RMCode

class GRANDDecoder:
    """
    GRAND解码器
    """
    def __init__(self, code, max_errors=7, max_queries=1000):
        self.code = code
        self.decoder_type = "GRAND"
        if isinstance(code, RMCode):
            self.code_type = "RM"
        elif isinstance(code, HammingCode):
            self.code_type = "Hamming"
        else:
            raise ValueError("Unsupported code type. Please use RMCode or HammingCode.")
        self.n = code.dim

        self.max_errors = max_errors
        self.max_queries = max_queries

    def get_r(self, tx_bin=None):
        """
        获取此算法的最小错误距离
        """
        if self.code_type == "Hamming":
            return np.sqrt(3)
        elif self.code_type == "RM":
            return np.sqrt(2**(self.code.m - self.code.r - 1))
        else:
            raise ValueError("Unsupported code type. Please use HammingCode or RMCode.")

    def generate_noise_candidates(self, weight):
        """GRAND：按汉明权重升序生成噪声序列（最多max_errors位错误）"""
        n = self.n
        candidates = np.zeros((int(scipy.special.comb(n, weight)), n), dtype=int)
        indices = np.arange(n)
        for i, bits in enumerate(itertools.combinations(indices, weight)):
            noise = np.zeros(n, dtype=int)
            noise[list(bits)] = 1
            candidates[i] = noise
        return candidates

    def decode_vec(self, y):
        """GRAND解码主函数"""
        y_hard = y > 0
        for i in range(self.max_errors+1):
            candidates = self.generate_noise_candidates(i)  # Pass the current weight
            for j, noise in enumerate(candidates):
                if j >= self.max_queries:
                    return y_hard  # 擦除
                candidate = (y_hard - noise) % 2
                if self.code.is_codeword(candidate):
                    return candidate
        return y_hard

    def decode(self, received):
        """解码函数"""
        decoded = np.zeros_like(received)
        for i, y in enumerate(received):
            decoded[i] = self.decode_vec(y)
        return self.code.decode_binary(decoded)

class SGRANDDecoder:
    """
    SGRAND解码器
    """
    def __init__(self, code, max_errors=7, max_queries=1000):
        self.code = code
        self.decoder_type = "SGRAND"
        if isinstance(code, RMCode):
            self.code_type = "RM"
        elif isinstance(code, HammingCode):
            self.code_type = "Hamming"
        else:
            raise ValueError("Unsupported code type. Please use RMCode or HammingCode.")
        self.n = code.dim

        self.max_errors = max_errors
        self.max_queries = max_queries

    def get_r(self, tx_bin=None):
        """
        获取此算法的最小错误距离
        """
        if self.code_type == "Hamming":
            return np.sqrt(3)
        elif self.code_type == "RM":
            return np.sqrt(2**(self.code.m - self.code.r))
        else:
            raise ValueError("Unsupported code type. Please use HammingCode or RMCode.")
    
    def hard_decision(self, y):
        """硬判决函数"""
        return (y > 0).astype(int)

    def decode_vec(self, y):
        """SGRAND解码主函数（带软信息优化）"""
        # 首先检查硬判决结果是否为有效码字（全0噪声情况）
        hard_decision = self.hard_decision(y)
        if self.code.is_codeword(hard_decision):
            return hard_decision
        
        n = self.n
        # 计算可靠性（绝对值越小越不可靠）
        reliability = np.abs(y)
        sorted_indices = np.argsort(reliability)  # 按可靠性升序排序
        
        # 初始化堆 - 直接使用索引而非位置
        heap = []
        
        # 直接添加第一个噪声模式（翻转索引0对应的位置）
        first_idx_tuple = (0,)  # 使用元组存储索引，更高效
        first_pos = sorted_indices[0]
        first_log_prob = np.sum(reliability) + 2 * (1 - reliability[first_pos])
        heappush(heap, (first_log_prob, first_idx_tuple))
        
        for _ in range(self.max_queries):
            if not heap:
                return hard_decision
            
            log_prob, idx_tuple = heappop(heap)
            
            # 快速生成候选码字
            candidate = hard_decision.copy()
            pos = sorted_indices[list(idx_tuple)]
            candidate[pos] = 1 - candidate[pos]

            # 检验码字
            if self.code.is_codeword(candidate):
                return candidate
            
            # 直接获取最大索引和下一个索引 - O(1)操作
            last_idx = idx_tuple[-1]
            next_idx = last_idx + 1
            
            # 生成子节点 - 更高效的方式
            if next_idx < n:
                # 子节点1：添加下一个最不可靠位的索引
                new_idx_tuple1 = idx_tuple + (next_idx,)
                next_pos = sorted_indices[next_idx]
                
                # 增量计算对数概率 - 只需考虑新添加的位置
                # 对数概率应为负值，值越小表示概率越大
                new_log_prob1 = log_prob + 2 * reliability[next_pos]
                heappush(heap, (new_log_prob1, new_idx_tuple1))
                
                if len(idx_tuple) == 0:
                    continue
                new_idx_tuple2 = idx_tuple[:-1] + (next_idx,)
                prev_pos = sorted_indices[last_idx]
                
                # 增量计算对数概率 - 考虑替换的影响
                # 需要加回被替换位置的影响
                new_log_prob2 = new_log_prob1 - 2 * reliability[prev_pos]
                heappush(heap, (new_log_prob2, new_idx_tuple2))
        
        return hard_decision  # 超出查询次数，返回硬判决结果

    def decode(self, received):
        """解码函数"""
        decoded = np.zeros_like(received)
        for i, y in enumerate(received):
            decoded[i] = self.decode_vec(y)  # 无需传入额外参数
        return self.code.decode_binary(decoded)

# ---------------------------
# 使用示例
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    # r=3
    # code = HammingCode(r=r)
    code = RMCode(2, 4)
    grand_decoder = GRANDDecoder(code=code, max_errors=2, max_queries=1000)
    sgrand_decoder = SGRANDDecoder(code=code, max_errors=2, max_queries=1000)

    from ORBGRAND import ORBGRANDDecoder
    orbgrand_decoder = ORBGRANDDecoder(code=code, max_log_weight=25, judge_bs=5)
    
    bs = 1e3
    n = grand_decoder.n
    sigma = 0.6
    # 生成随机接收信号
    # noise_base = np.array([4/3, 2/3, 0, 2/3, 0, 0, 0])
    # noise_base = np.array([1.1, 1.1, 0, 0.8, 0, 0, 0])
    noise_base = 0
    received = -1 * np.ones((int(bs), n)) + sigma * np.random.randn(int(bs), n) + noise_base
    # received = np.array([[0.05,   -0.04,   -1.21,  -0.02, -1.21, -1.16, -1.21]])
    # print("Received signals:")
    # print(received)

    import time
    start_time = time.time()
    decoded1 = code.decode(received)
    end_time = time.time()
    correct = np.all(decoded1 == 0, axis=1)  # 检查是否为0码字
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print(np.mean(correct))  # 检查是否为0码字
    mindist, minvec = find_error_dist(received, correct)
    print(f"min dist: {mindist}")
    print(minvec)

    start_time = time.time()
    code.set_decoder(grand_decoder)
    decoded2 = grand_decoder.decode(received)
    end_time = time.time()
    correct = np.all(decoded2 == 0, axis=1)  # 检查是否为0码字
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print(np.mean(correct))  # 检查是否为0码字
    mindist, minvec = find_error_dist(received, correct)
    print(f"min dist: {mindist}")
    print(minvec)
    
    start_time = time.time()
    code.set_decoder(sgrand_decoder)
    decoded3 = sgrand_decoder.decode(received)
    end_time = time.time()
    correct = np.all(decoded3 == 0, axis=1)  # 检查是否为0码字
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print(np.mean(correct))  # 检查是否为0码字
    if np.any(~correct):
        mindist, minvec = find_error_dist(received, correct)
        print(f"min dist: {mindist}")
        print(minvec)

    start_time = time.time()
    decoded4 = orbgrand_decoder.decode(received)
    end_time = time.time()
    correct = np.all(decoded4 == 0, axis=1)  # 检查是否为0码字
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print(np.mean(correct))  # 检查是否为0码字
    if np.any(~correct):
        mindist, minvec = find_error_dist(received, correct)
        print(f"min dist: {mindist}")
        print(minvec)