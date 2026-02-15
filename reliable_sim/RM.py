import numpy as np
import scipy
from scipy.special import comb
from code_linear import *

class RMCode(LinearCodeBase):
    """
    Reed-Muller Code
    """
    def __init__(self, r, m, decoder="binary"):
        self.r = r
        self.m = m
        n = 2 ** m
        k = self.compute_k(r, m)
        super().__init__(dim=n, bin_dim=k)
        self.decoder = "binary"

    def get_r(self, tx_bin=None):
        if self.decoder == "binary":
            return np.sqrt(2**(self.m-self.r-1))
        else:
            return self.decoder.get_r(tx_bin=tx_bin)

    def compute_k(self, r, m):
        """计算信息位长度k"""
        k = np.sum([int(comb(m, i)) for i in range(r + 1)])
        return k

    def rm_encoder_recursive(self, info_bits, r=None, m=None):
        """
        Reed-Muller码递归编码器
        参数:
            info_bits (0-1 ndarray): 信息比特序列, (batch_size, k)
            r (int): RM码阶数
            m (int): 变量数，码长n=2^m
        返回:
            ndarray: 编码后的码字, (batch_size, n)
        """
        if r is None or m is None:
            r, m = self.r, self.m
        
        N = info_bits.shape[0] # batch_size
        if r == 0:  # 重复码
            result = np.tile(info_bits, (1, 2**m))
            return result
            
        elif r == m:  # 全阶码，使用生成矩阵
            G = self.build_full_rank_generator(m)
            result = (info_bits @ G) % 2
            return result
            
        elif r > 0 and r < m:
            # 计算u部分信息位长度
            u_num = self.compute_k(r, m-1)
            
            # 分割信息位
            info_u = info_bits[:, :u_num]
            info_v = info_bits[:, u_num:]
            
            # 递归编码子码
            subcode_u = self.rm_encoder_recursive(info_u, r, m-1)
            subcode_v = self.rm_encoder_recursive(info_v, r-1, m-1)
            
            # Plotkin构造：(u, u+v)
            result = np.concatenate((subcode_u, (subcode_u + subcode_v) % 2), axis=-1)
            return result
        
        else:
            raise ValueError("无效参数")

    def build_full_rank_generator(self, m):
        """构建全阶生成矩阵（克罗内克积）"""
        G = np.array([[1, 1], [0, 1]], dtype=int)
        for _ in range(m-1):
            G = np.kron(G, np.array([[1, 1], [0, 1]]))
        return G
    
    def choose_k(self, r, m):
        """ 构建生成矩阵 by choosing the rows of G """
        # G0 = self.build_full_rank_generator(m)
        if r == 0:
            k = np.array([0])
        elif r == m:
            k = np.arange(2**m)
        elif r == 1:
            k = np.hstack((0, 2**np.arange(m)))
        else:
            k1 = self.choose_k(r, m-1)
            k2 = self.choose_k(r-1, m-1)
            k = np.hstack((k1, k2 + 2**(m-1)))
        return k
    
    def encode(self, info_bits):
        """标准编码接口，使用生成矩阵"""
        is_1d = info_bits.ndim == 1
        if is_1d:
            info_bits = info_bits[None, :]
        ans = self.rm_encoder_recursive(info_bits, self.r, self.m)
        ans = ans*2 - 1

        if is_1d:
            return ans[0]
        return ans
    
    def rm_decoder_recursive(self, codeword, r, m):
        """
        Reed-Muller码递归解码器
        
        参数:
            codeword (ndarray): 接收到的码字
            r (int): RM码阶数
            m (int): 变量数，码长n=2^m
            
        返回:
            tuple: 更新后的info_bits
        """
        if r > 1 and r < m:
            n = codeword.shape[-1]
            # 分解码字为u和v部分 (Plotkin构造)
            codeword_u = codeword[:, :, :n//2]
            codeword_uv = codeword[:, :, n//2:]
            codeword_v = (codeword_uv - codeword_u) % 2
            
            info_v, decoded_v = self.rm_decoder_recursive(codeword_v, r-1, m-1)
            u_from_v = (codeword_uv - decoded_v) % 2
            codeword_u2 = np.concatenate((codeword_u, u_from_v), axis=0)
            info_u, decoded_u = self.rm_decoder_recursive(codeword_u2, r, m-1)
            info_bits = np.concatenate((info_u, info_v), axis=-1)
            decoded = np.concatenate((decoded_u, (decoded_u + decoded_v) % 2), axis=-1)
            
        elif r == m:  # 全阶码解码
            G = self.build_full_rank_generator(m)
            decoded = self.majority_vote(codeword, axis=0)
            info_bits = decoded @ G % 2

        elif r == 1:  # RM(1,m)解码
            info_bits, decoded = self.rm_decoder_r1(codeword, m)
            
        elif r == 0:  # 重复码解码 (多数表决)
            n = codeword.shape[0] * codeword.shape[-1]
            x = 2 * np.sum(codeword, axis=(0,-1)) >= n
            info_bits = x.astype(int)[:, None]
            decoded = info_bits * np.ones(codeword.shape[-1], dtype=int)
            
        else:
            raise ValueError("Invalid parameters")
        
        return info_bits, decoded
    
    def rm_decoder_r1(self, codeword, m):
        """ decoder for RM(1,m) with majority vote """
        N = codeword.shape[1] # batch_size
        info_bits = np.zeros((N, 1+m), dtype=int)
        decoded = np.zeros((N, codeword.shape[-1]), dtype=int)
        for i in range(1, m+1):
            pos1 = np.hstack((np.zeros(2**(i-1), dtype=bool), np.ones(2**(i-1), dtype=bool)))
            pos = np.tile(pos1, 2**(m-i))
            bits_with, bits_without = codeword[:, :, pos], codeword[:, :, ~pos]
            # difference = bits_with != bits_without
            # info_bits[i] = 1 if 2*difference >= 2**(m-1) else 0
            info_bits[:, i] = self.majority_vote(bits_with - bits_without, (0, -1))
            decoded[:, pos] += info_bits[:, [i]]
        # info_bits[0] = 1 if np.sum((decoded - codeword) % 2) >= 2**(m-1) else 0
        info_bits[:, 0] = self.majority_vote(decoded - codeword, (0, -1))
        decoded = (decoded + info_bits[:, [0]]) % 2
        return info_bits, decoded
        
    def decode(self, received):
        """标准解码接口"""
        is_1d = received.ndim == 1
        if is_1d:
            received = received[None, :]

        if self.decoder == "binary":
            hard_decision = np.where(received > 0, 1, 0)
            info_bits, _ = self.rm_decoder_recursive(hard_decision[None, :], self.r, self.m)
        else:
            info_bits = self.decoder.decode(received)

        if is_1d:
            return info_bits[0]
        return info_bits
    
    def decode_binary(self, hard_decision):
        """标准解码接口"""
        is_1d = hard_decision.ndim == 1
        if is_1d:
            hard_decision = hard_decision[None, :]

        hard_decision = np.where(hard_decision > 0, 1, 0)
        info_bits, _ = self.rm_decoder_recursive(hard_decision[None, :], self.r, self.m)

        if is_1d:
            return info_bits[0]
        return info_bits
    
    def majority_vote(self, codeword, axis=-1):
        """多数表决解码"""
        if isinstance(axis, tuple):
            n = np.prod([codeword.shape[a] for a in axis], dtype=int)
        else:
            n = codeword.shape[axis]
        sums = np.sum(codeword % 2, axis=axis)
        return np.where(2 * sums >= n, 1, 0)
    
    def is_codeword(self, codeword, r=None, m=None):
        """判断是否为码字"""
        is_1d = codeword.ndim == 1
        is_3d = codeword.ndim == 3
        if is_1d:
            codeword = codeword[None, :]
        elif is_3d:
            # reshape to 2D, keep the -1 dimension
            d0, d1 = codeword.shape[0], codeword.shape[1]
            codeword = codeword.reshape(-1, codeword.shape[-1])

        if r is None or m is None:
            r, m = self.r, self.m

        if r == 0:
            code_first = codeword[:, 0][:, None]
            ans = np.all(codeword == code_first, axis=-1)
        elif r == m:
            ans = np.ones((codeword.shape[0],), dtype=bool)
        else:
            u = codeword[:, :2**(m-1)]
            uv = codeword[:, 2**(m-1):]
            v = (uv - u) % 2
            ans = self.is_codeword(u, r, m-1) & self.is_codeword(v, r-1, m-1)

        if is_1d:
            return ans[0]
        elif is_3d:
            # ans is 1d, reshape back to 2d
            ans = ans.reshape(d0, d1)
        return ans

if __name__ == "__main__":
    # 示例
    r = 3  # RM阶数
    m = 7  # 变量数
    rm_code = RMCode(r, m)
    k = rm_code.bin_dim  # 信息位长度
    n = rm_code.dim  # 码字长度
    print(f"RM({r},{m})码: 信息位长度={k}, 码字长度={n}")

    batch_size = 10
    signal_bin = np.random.randint(0, 2, (batch_size, k))  # 随机生成信息比特
    encoded = rm_code.encode(signal_bin)
    error_idx = np.random.randint(0, n, 7)  # 随机生成错误位索引
    encoded[:, error_idx] = 1 - encoded[:, error_idx]  # 人为制造错误
    decoded = rm_code.decode(encoded)
    print(f"解码成功: {np.array_equal(signal_bin, decoded)}")

    print(rm_code.is_codeword(encoded))
    
    # G = rm_code.generate_standard_G(r, m)
    # G0 = rm_code.build_full_rank_generator(m)
    # print(f"生成矩阵G0:\n{G0}")
    # print(f"信息比特: {signal_bin}")
    # print(f"矩阵生成码字: {signal_bin @ G[:-1, :] % 2}")
    # print(f"编码后的码字: {encoded}")
    # print(G @ G % 2)
    # print(f"解码比特: {decoded}")

    # ks = rm_code.choose_k(r, m)
    # print(f"选择的行索引: {ks}")
    # G = G0[ks, :]
    # H = G0[:, ks]
    # print(f"矩阵生成码字: {signal_bin @ G % 2}")
    # print(f"矩阵解码码字: {encoded @ H % 2}")
    # print(f"解码成功: {np.array_equal(encoded @ H % 2, signal_bin)}")