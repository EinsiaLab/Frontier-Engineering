import numpy as np
from code_linear import *
from RM import RMCode
import itertools

class ChaseDecoder:
    """
    Chase decoder for linear codes.
    """
    def __init__(self, code, t=None):
        self.code = code
        if t is None:
            if isinstance(code, HammingCode):
                self.code_type = "Hamming"
                t = 3
            elif isinstance(code, RMCode):
                self.code_type = "RM"
                if t is None:
                    t = 2**(code.m - code.r)
            else:
                raise ValueError("Unsupported code type. Please use HammingCode or RMCode.")
        if t >= 10:
            print("Warning: Chase decoder is not suitable for t >= 10.")
        self.errs = t
        self.decoder_type = "chase"
        self.n = code.dim
        
        # 预生成所有可能的错误模式
        self.error_patterns = []
        # 生成所有可能的t位错误模式组合（在t个不可靠位置上尝试各种组合）
        for i in range(self.errs+1):
            patterns = list(itertools.combinations(range(self.errs), i))
            for pattern in patterns:
                # 创建错误模式向量
                error = np.zeros(self.n, dtype=int)
                # error = np.zeros(self.errs, dtype=int)
                error[list(pattern)] = 1
                self.error_patterns.append(error)
        
        # 将错误模式列表转换为numpy数组，便于并行处理
        self.error_patterns = np.array(self.error_patterns)

    def get_r(self, tx_bin=None):
        """
        Get the minimum error distance of this algorithm.
        """
        if self.errs == 2:
            return np.sqrt(8/3)
        elif self.errs >= 3:
            return np.sqrt(2 * self.errs)
        else:
            return 0 # not supported
    
    def decode_vector(self, received_signal):
        """
        Decode a single received signal using the Chase algorithm.
        """
        # 初始化解码结果数组，只包含数据位
        decoded = np.zeros(self.n, dtype=int)
        
        # 硬判决（-1->0, 1->1）
        hard_decision = (received_signal > 0).astype(int)
        if self.code.is_codeword(hard_decision):
            return hard_decision[self.code.r:]
        
        rel = np.abs(received_signal)
        rel_orders = np.argsort(rel)
        err_pos = rel_orders[:self.errs] # 选择最不可靠的t个位置

        cand_rel = rel * np.ones((self.error_patterns.shape[0], self.n)) # (n_noise, n)
        cand_rel[:,err_pos] *= (1-2*self.error_patterns)
        
        candidates = hard_decision * np.ones((self.error_patterns.shape[0], self.n)) # (n_noise, n)
        candidates[:, err_pos] = (hard_decision[err_pos] + self.error_patterns) % 2 # (n_noise, n)

        valid = self.code.is_codeword(candidates) # (n_noise,)
        rel_noised = (np.sum(cand_rel, axis=1) + 2*self.n) * valid # (n_noise,)
        rel_choose = np.argmax(rel_noised)
        decoded = candidates[rel_choose, :] # (n,)
        return decoded[self.code.r:]

    def decode(self, received_signals, batch=True):
        if not batch:
            decoded = np.zeros((len(received_signals), self.code.k), dtype=int)
            for i, signal in enumerate(received_signals):
                decoded[i] = self.decode_vector(signal)
            return decoded
        # 初始化解码结果数组，只包含数据位
        batch_size = received_signals.shape[0]
        
        # 并行硬判决（-1->0, 1->1）
        hard_decisions = (received_signals > 0).astype(int)
        
        found = self.code.is_codeword(hard_decisions) # (batch_size,)
        if np.all(found):
            return self.code.decode_binary(hard_decisions) # (batch_size, k)

        n_nf = np.sum(~found) # 还未找到解的样本数
        decoded = hard_decisions.copy()
        
        rel = np.abs(received_signals[~found]) # (batch_size, n)
        rel_orders = np.argsort(rel, axis=1)# (batch_size, n)
        sorted_hard = np.take_along_axis(hard_decisions[~found], rel_orders, axis=1)
        rel_orders_back = np.zeros_like(rel_orders, dtype=int)
        rel_orders_back[np.arange(n_nf)[:, np.newaxis], rel_orders] = np.arange(self.n)
        sorted_rel = np.take_along_axis(rel, rel_orders, axis=1) # (batch_size, n)

        noised_sorted = (sorted_hard[:, None, :] + self.error_patterns) % 2 # (batch_size, n_noise, n)
        noised_codes = np.take_along_axis(noised_sorted, rel_orders_back[:, None, :], axis=2) # (batch_size, n_noise, n)
        valid = self.code.is_codeword(noised_codes) # (batch_size, n_noise)

        rel_noised = np.sum(sorted_rel[:, None, :] * (1-2*self.error_patterns), axis=2) # (batch_size, n_noise)
        rel_choose = np.argmax((rel_noised + 2*self.n) * valid, axis=1) # (batch_size,)
        sorted_decoded = noised_codes[np.arange(n_nf), rel_choose, :] # (batch_size, n)
        # unsorted_decoded = np.take_along_axis(sorted_decoded, rel_orders_back, axis=1) # (batch_size, n)
        decoded[~found] = sorted_decoded # (batch_size, n)
            
        # returned = self.code.decode_binary(decoded)
        returned = self.code.decode_binary(decoded) # (batch_size, k)
        return returned

# 使用示例
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    
    # 测试Hamming码的Chase解码
    r = 3
    code = HammingCode(r)
    # t=1表示Chase-1算法，考虑最不可靠的1个位置
    decoder = ChaseDecoder(code=code, t=3)
    code.set_decoder(decoder)
    # print(code.get_nearest_neighbors())
    
    n = code.dim
    # 生成测试接收信号
    received = np.array([1,-1,1,-1,-1,-1,1], dtype=float)*np.ones((2,1))
    # 第一个信号有一个不太可靠的位（可能出错的位）
    received[0, [0, 1, 5]] *= [0.3, 0.25, -0.4]
    # 第二个信号有两个不太可靠的位
    # received[1, [0, 1]] *= [0.3, -0.35]
    
    import time
    start_time = time.time()
    decoded = code.decode(received)
    end_time = time.time()
    
    print(f"解码时间: {end_time - start_time:.4f}秒")
    print("解码结果:")
    print(decoded)

