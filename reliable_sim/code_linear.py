import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import time
from numpy.random import Generator, PCG64, Philox

def log_adder(a, b):
    return np.logaddexp(a, b)

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

class LinearCodeBase:
    def __init__(self, dim, bin_dim):
        """
        初始化线性码基类
        
        参数:
        dim: 空间维度
        num: 码字数量
        """
        self.dim = dim
        self.bin_dim = bin_dim  # 码字数量
        # self.codewords = self._generate_codewords()
        self.rng = Generator(Philox())  # 随机数生成器
        # self.codeword_map = {}  # 存储二进制向量到索引的映射
        self.decoder = None  # 解码器

    def set_decoder(self, decoder):
        self.decoder = decoder
        
    def bin_to_idx(self, binary_vec):
        """
        将二进制向量转换为索引
        binary_vec: 二进制向量，可以是1D数组(单个向量)或2D数组(多个向量)
        返回: 整数索引，如果输入是2D数组则返回包含多个索引的1D数组
        """
        if binary_vec.ndim == 1:
            # 将二进制向量转换为字符串，然后转换为整数
            return int(''.join(map(str, binary_vec)), 2)
        elif binary_vec.ndim == 2:
            # 对每个向量应用相同的转换
            return np.array([int(''.join(map(str, vec)), 2) for vec in binary_vec])
        else:
            raise ValueError("输入必须是1D或2D数组")
    
    def idx_to_bin(self, idx):
        """
        将索引转换为二进制向量
        idx: 整数索引或索引数组
        返回: 二进制向量（1D数组）或二进制向量数组（2D数组）
        """
        # 处理单个整数索引
        if np.isscalar(idx):
            return np.array(list(map(int, np.binary_repr(idx, width=self.bin_dim))), dtype=int)
        else:
            return np.array([list(map(int, np.binary_repr(i, width=self.bin_dim))) for i in idx], dtype=int)

    def vec_to_idx(self, vec):
        """Find the nonzero indices of a vector"""
        return np.where(vec != 0)[0]
    
    def idx_to_vec(self, idx):
        """Convert indices to a 0-1 vector"""
        idx = np.asarray(idx)
        
        if idx.ndim == 1:
            # 单个向量的情况
            vec = np.zeros(self.dim, dtype=int)
            vec[idx] = 1
            return vec
        else:
            # 多维情况：支持2D, 3D等
            # 获取除最后一维外的所有维度形状
            batch_shape = idx.shape[:-1]
            d = idx.shape[-1]  # 最后一维是索引个数
            
            # 创建输出向量
            vec = np.zeros(batch_shape + (self.dim,), dtype=int)
            
            # 使用展平的方法进行向量化赋值
            # 展平除最后一维外的所有维度
            idx_flat = idx.reshape(-1, d)
            vec_flat = vec.reshape(-1, self.dim)
            
            # 使用numpy的put_along_axis的替代方法
            row_indices = np.arange(len(idx_flat))[:, np.newaxis]
            vec_flat[row_indices, idx_flat] = 1
            
            return vec

    def get_r(self):
        """
        计算码的最小汉明距离
        对于一般的线性码，需在子类中实现
        """
        raise NotImplementedError
    
    def encode(self, message):
        """编码函数"""
        raise NotImplementedError
    
    def decode(self, rx_signals):
        """解码函数，接收float array，自带并行化功能，可接受1d、2d array"""
        raise NotImplementedError
    
    def decode_binary(self, rx_signals):
        """二进制解码函数，接受0-1数组"""
        raise NotImplementedError
    
    def random_bits(self, num=1):
        """生成随机码"""
        return self.rng.integers(0, 2, size=(num, self.bin_dim), dtype=int)
    
    def simulate(self, noise_std, sampler=None, batch_size=1e5, num_samples=1e8, scale_factor=1.0, fix_tx=True, **kwargs):
        """仿真过程，使用不同采样器"""
        batch_size, num_samples = int(batch_size), int(num_samples)
        # rounds = int(num_samples/batch_size/len(self.codewords))+1
        # batch_size_use = int(num_samples/rounds/len(self.codewords))
        rounds = int(num_samples/batch_size)
        batch_size_use = int(num_samples/rounds)

        errors = -np.inf
        weights = -np.inf
        err_nums = 0
        for i in range(rounds):
            # for j, codeword in enumerate(self.codewords):
            if fix_tx:
                # 使用固定的码字进行仿真
                code_bits = np.zeros(self.bin_dim, dtype=int)
            else:   
                # 随机生成码字
                code_bits = self.random_bits()[0]
            if sampler is None:
                error, weight, err_num = self._simulate_batch(noise_std, batch_size=batch_size_use, tx_bin=code_bits)
            else:
                error, weight, err_num = self._simulate_importance_batch(sampler, noise_std, tx_bin=code_bits, scale_factor=scale_factor, batch_size=batch_size_use, **kwargs)
            errors = log_adder(errors, error)
            weights = log_adder(weights, weight)
            err_nums += err_num
            if i % 10 == 0:
                print(f'Sample {(i+1)*batch_size_use / 10000}/{num_samples / 10000}, log_errors: {errors:.2f}-{weights:.2f}, error_ratio: {err_num}/{batch_size_use}')
        return errors, weights, err_nums / (rounds * batch_size_use)
    
    def _simulate_batch(self, noise_std, batch_size=1e7, tx_bin=None):
        """使用普通采样进行一个批次的仿真"""
        batch_size = int(batch_size)
        if tx_bin is None:
            tx_bin = self.random_bits(batch_size)
        tx_signals = self.encode(tx_bin)

        # 添加高斯噪声
        noise = self.rng.normal(0, noise_std, (batch_size, self.dim))
        rx_signals = tx_signals + noise

        # 解码
        rx_bin = self.decode(rx_signals)
        # 对于每一行，有一个不同就是不同的码字        
        # rx_bin 和 tx_bin 都是(batch_size, self.dim)维度的array
        errors = np.sum(rx_bin != tx_bin, axis=1)
        sum_error = np.sum(errors > 0)
        if sum_error == 0:
            return -np.inf, np.log(batch_size), sum_error
        else:
            return np.log(sum_error), np.log(batch_size), sum_error
    
    def _simulate_importance_batch(self, sampler, noise_std, tx_bin=None, batch_size=1e5, scale_factor=1.0, **kwargs):
        """使用重要性采样进行一个批次的仿真"""
        batch_size = int(batch_size)
        if tx_bin is None:
            tx_bin = self.random_bits(1)
                
        # 使用采样器获取噪声样本和PDF值
        noise_samples, pdf_values = sampler.sample(noise_std*np.sqrt(scale_factor), tx_bin, batch_size, **kwargs)
        log_pdf_original = -(np.sum(noise_samples**2, axis=1))/(2*noise_std**2) - self.dim/2 * np.log(2*np.pi*noise_std**2)
        
        # 生成接收信号
        rx_signals = self.encode(tx_bin) + noise_samples
        
        # 解码
        rx_bin = self.decode(rx_signals)
        
        # 计算错误权重
        # errors_if = np.sum(rx_bin != tx_bin, axis=-1)
        # error_indices = np.where(errors_if > 0)[0]
        error_mask = np.any(rx_bin != tx_bin, axis=-1)
        log_weights = log_pdf_original - pdf_values
        log_weights_errors = log_weights[error_mask]

        # imax_weight = np.argmax(log_weights_errors)
        # print(f"max weight: {log_weights_errors[imax_weight]:.2f}, idx: {imax_weight}")
        # print(noise_samples[imax_weight])

        # total_weight = np.log(batch_size)
        total_weight = logsumexp(log_weights)
        if len(log_weights_errors) == 0:
            total_weighted_errors = -np.inf
        else:
            total_weighted_errors = logsumexp(log_weights_errors)
        # print(f"sum of weight: {total_weight} vs {np.log(batch_size)}")

        return total_weighted_errors, total_weight, len(log_weights_errors)
    
    def _simulate_importance_batch_rx(self, rx_target, sampler, noise_std, tx_bin=None, batch_size=1e5, scale_factor=1.0, **kwargs):
        """使用重要性采样进行一个批次的仿真"""
        batch_size = int(batch_size)
        if tx_bin is None:
            tx_bin = self.random_bits(1)
                
        # 使用采样器获取噪声样本和PDF值
        noise_samples, pdf_values = sampler.sample(noise_std*np.sqrt(scale_factor), tx_bin, batch_size, **kwargs)
        log_pdf_original = -(np.sum(noise_samples**2, axis=1))/(2*noise_std**2) - self.dim/2 * np.log(2*np.pi*noise_std**2)
        # print(np.mean(noise_samples, axis=0))

        # 生成接收信号
        rx_signals = self.encode(tx_bin) + noise_samples
        
        # 解码
        rx_bin = self.decode(rx_signals)
        
        # 计算错误权重
        errors_if = np.sum(rx_bin != tx_bin, axis=-1)
        error_indices = np.where(errors_if > 0)[0]
        log_weights = log_pdf_original - pdf_values
        log_weights_errors = log_weights[error_indices]

        # 计算等于目标信号的权重
        errors_if_rx = np.all(rx_bin == rx_target, axis=-1)
        error_indices_rx = np.where(errors_if_rx)[0]
        log_weights_rx = log_pdf_original - pdf_values
        log_weights_errors_rx = log_weights_rx[error_indices_rx]
        
        total_weight = np.log(batch_size)
        if len(log_weights_errors) == 0:
            total_weighted_errors = -np.inf
        else:
            total_weighted_errors = logsumexp(log_weights_errors)
        
        if len(log_weights_errors_rx) == 0:
            total_weighted_errors_rx = -np.inf
        else:
            total_weighted_errors_rx = logsumexp(log_weights_errors_rx)

        return total_weighted_errors, total_weight, len(log_weights_errors), total_weighted_errors_rx, len(log_weights_errors_rx)
    
    def simulate_variance_controlled(self, noise_std, target_std, max_samples, 
                                   sampler=None, batch_size=1e4, scale_factor=1.0, 
                                   fix_tx=True, min_errors=10, min_batches=10, **kwargs):
        """
        基于方差控制的仿真过程
        
        参数:
        noise_std: 噪声标准差
        target_std: 目标方差阈值（对数域）
        max_samples: 最大样本数
        sampler: 采样器，None表示使用普通采样
        batch_size: 批处理大小
        scale_factor: 缩放因子
        fix_tx: 是否固定发送码字
        min_errors: 最小错误数量，低于此数量不能因方差过小而停止
        **kwargs: 其他参数
        
        返回:
        errors: 错误对数
        weights: 权重对数
        err_ratio: 错误率
        total_samples: 实际使用的样本数
        converged: 是否收敛
        """
        batch_size = int(batch_size)
        max_samples = int(max_samples)
        
        # 存储每批次的结果
        batch_errors = []
        batch_weights = []
        total_err_nums = 0
        total_samples = 0
        current_std = 1
        
        print(f"开始方差控制仿真：目标方差={target_std:.4f}, 最大样本数={max_samples}")
        
        while total_samples < max_samples:
            # 生成码字
            if fix_tx:
                code_bits = np.zeros(self.bin_dim, dtype=int)
            else:   
                code_bits = self.random_bits()[0]
            
            # 执行一个批次的仿真
            if sampler is None:
                error, weight, err_num = self._simulate_batch(
                    noise_std, batch_size=batch_size, tx_bin=code_bits)
            else:
                error, weight, err_num = self._simulate_importance_batch(
                    sampler, noise_std, tx_bin=code_bits, 
                    scale_factor=scale_factor, batch_size=batch_size, **kwargs)
            
            # 更新累计统计
            batch_errors.append(error)
            batch_weights.append(weight)
            total_err_nums += err_num
            total_samples += batch_size
            
            # 检查是否有足够的有效错误数据
            if len(batch_errors) >= min_batches and total_err_nums >= min_errors:
                # 计算当前的方差估计
                
                # 计算对数域的标准差作为方差指标
                current_std = logstdexp(np.array(batch_errors)) / np.sqrt(len(batch_errors))
                
                if total_samples % 10000 == 0:
                    print(f"错误数: {total_err_nums/batch_size}/{total_samples/batch_size}, "
                        f"相对标准差: {current_std/target_std:.2f}, "
                        f"对数错误率: {batch_errors[-1]-batch_weights[-1]:.2f}, ")
                
                # 检查是否满足收敛条件
                if current_std < target_std:
                    print(f"方差收敛！在 {total_samples} 样本后停止")
                    break
        
        # 计算最终结果
        final_errors = logsumexp(np.array(batch_errors))
        final_weights = logsumexp(np.array(batch_weights))
        err_ratio = total_err_nums / total_samples
        converged = total_samples < max_samples
        
        print(f"仿真完成：样本数={total_samples}, 错误数={total_err_nums}, "
              f"相对标准差: {current_std/target_std:.2f}, "
              f"对数错误率: {batch_errors[-1]-batch_weights[-1]:.2f}, "
              f"收敛={'是' if converged else '否'}")
        
        return final_errors, final_weights, err_ratio, total_samples, current_std, converged

class HammingCode(LinearCodeBase):
    def __init__(self, r=3, decoder='binary'):
        """
        初始化汉明码
        
        参数:
        r: 校验位数量，默认为3对应(7,4)汉明码
        decoder: 解码方法，默认为'binary'
        """
        # 汉明码参数: n=2^r-1, k=2^r-1-r
        self.n = 2**r - 1
        self.k = 2**r - 1 - r
        self.r = r
        super().__init__(dim=self.n, bin_dim=self.k)
        # self.G, self.H = self._generate_matrix()
        # self._nearest_neighbors = self._find_nearest_neighbors()
        self.decoder = decoder
        # self.G, self.H = self._generate_matrix_custom()
        self.G, self.H = self._generate_matrix()
        
    def get_r(self, tx_bin=None):
        """汉明码的最小汉明距离总是3"""
        if self.decoder == 'nearest':
            return np.sqrt(3)
        elif self.decoder == 'binary':
            return np.sqrt(2)
        else:
            return self.decoder.get_r(tx_bin)
        
    def _generate_matrix(self):
        """生成汉明码的生成矩阵G和校验矩阵H"""
        try:
            name = f'libs/hamming_matrix_{self.r}.csv'
            M = np.loadtxt(name, delimiter=',', dtype=int)
            # delta = M[:self.k, :self.r]
            # G = np.hstack((np.eye(self.k, dtype=int), delta))
            # H = np.hstack((delta.T, np.eye(self.r, dtype=int)))
            G = M[:self.k, :]
            H = M[self.k:, :]

            # build hash table for H
            sums = 2**np.arange(self.r) @ H
            self.H_hash = np.zeros(2**self.r, dtype=int)
            for i, s in enumerate(sums):
                self.H_hash[s] = i
            return G, H
        except ImportError:
            print("警告: 未预备汉明码生成和校验矩阵，使用自定义实现")
            # 使用我们自己的实现作为后备
            # return self._generate_matrix_custom()
            return self._generate_hamming_matrices()
    
    def _generate_hamming_matrices(self):
        """
        生成汉明码的生成矩阵G和校验矩阵H
        
        参数:
        r -- 校验位数
        
        返回:
        G -- 生成矩阵
        H -- 校验矩阵
        """
        # 计算总位数n和信息位数k
        r = self.n - self.k
        n = self.n
        k = self.k
        
        # 构造H矩阵的左侧部分 - 所有非零的r位二进制列向量
        # 列向量从1到2^r-1
        H_left = np.zeros((r, n-r), dtype=int)
        
        col = 0
        # 从1到2^r-1生成所有非零的r位二进制数
        for i in range(1, 2**r):
            # 将整数i转换为二进制并去掉'0b'前缀
            bin_str = bin(i)[2:].zfill(r)
            # 将二进制字符串转换为列向量
            bin_vec = np.array([int(bit) for bit in bin_str])
            
            # 跳过与单位矩阵列相同的向量，以确保H的秩为r
            if np.sum(bin_vec) > 1:  # 至少有两个1的列
                H_left[:, col] = bin_vec
                col += 1
                if col >= n-r:
                    break
        
        # 构造H矩阵 - 左侧部分和r×r单位矩阵
        H_right = np.eye(r, dtype=int)
        H = np.hstack((H_left, H_right))
        
        # 构造G矩阵 - k×k单位矩阵和H左侧部分的转置
        G_left = np.eye(k, dtype=int)
        G_right = H_left.T  # 在二进制域中，负转置等于转置
        G = np.hstack((G_left, G_right))
        
        # 在二进制域中进行运算（模2）
        return G % 2, H % 2

    def _generate_codewords(self):
        """生成所有2^k个合法码字"""
        print("生成所有合法码字, k=", self.k)
        info_bits = self.idx_to_bin(np.arange(2**self.k))
        # print(info_bits.shape, self.G.shape)
        codewords_binary = (info_bits @ self.G) % 2
        return 2 * codewords_binary.astype(np.float32) - 1  # 转换为±1形式
    
    def encode(self, tx_bin):
        # print(tx_bin.shape, self.G.shape)
        tx_code = (tx_bin @ self.G) % 2
        tx_code = tx_code.astype(np.float32)
        return 2 * (tx_code) - 1  # 转换为±1形式
    
    def is_codeword(self, vecs):
        """
        校验是否为合法码字
        支持1维、2维或3维数组：
        - 1维: 单个向量 (n,)
        - 2维: 多个向量 (batch, n)
        - 3维: 多个样本的多个向量 (batch, candidates, n)
        """
        if vecs.ndim == 3:
            syndromes = np.einsum('ijk,lk->ijl', vecs, self.H) % 2
            return np.all(syndromes == 0, axis=-1)  # (batch, candidates)
        else:
            return np.all((self.H @ vecs.T) % 2 == 0, axis=0)
        
    def decode(self, rx_signals):
        """
        最近邻解码
        """
        is_1d = (rx_signals.ndim == 1)
        if is_1d:
            rx_signals = rx_signals.reshape(1, -1)
        if self.decoder == 'binary':
            # 转换为二进制格式
            rx_binary = np.where(rx_signals > 0, 1, 0).astype(int)
            ans = self.decode_binary(rx_binary)
        elif self.decoder == 'nearest':
            raise NotImplementedError("最近邻解码复杂度高，被禁用")
            # 最近邻解码
            if not hasattr(self, 'codewords'):
                self.codewords = self._generate_codewords()
            distances = cdist(rx_signals, self.codewords, 'euclidean')
            idx = np.argmin(distances, axis=-1)
            return self.idx_to_bin(idx)
        else:
            ans = self.decoder.decode(rx_signals)
            
        if is_1d:
            return ans[0]
        return ans
            
    def decode_binary(self, rx_binary):
        """
        高效的汉明码解码算法 - 基于矩阵运算，支持批量处理
        takes 0-1 arrays instead of float arrays
        """
        
        # 计算syndrome (错误模式)
        syndrome = (rx_binary @ self.H.T) % 2
        correct = np.all(syndrome == 0, axis=1)
        error_codes = np.where(~correct)[0]
        error_syn = syndrome[error_codes]

        if len(error_codes) == 0:
            # 没有错误
            return rx_binary[:, self.r:]
        # error_match = np.all(error_syn[:, None, :] == (self.H.T), axis=2) # (errors, n)
        # rx_binary[error_codes] = (rx_binary[error_codes] + error_match) % 2
        int_syn = 2**np.arange(self.r, dtype=int) @ error_syn.T # (errors, 1)
        err_pos = self.H_hash[int_syn] # (errors, 1)
        # change the bit at the error position
        rx_binary[error_codes] = (rx_binary[error_codes] + np.eye(self.n)[err_pos]) % 2
        
        # 提取信息位部分
        result = rx_binary[:, self.r:]
        return result

    def _find_nearest_neighbors(self):
        """
        找到全0码字的最近邻
        对于汉明码，最小汉明距离是3，所以寻找汉明距离为3的码字
        """
        import itertools
        # 将校验矩阵H转换为NumPy数组
        H_np = np.array(self.H, dtype=int)
        
        # 获取H的列向量
        cols = H_np.T  # 转置获取列向量
        n = cols.shape[0]  # 列数
        
        # 预先生成所有可能的列对组合
        combinations = list(itertools.combinations(range(n), 2))
        
        valid_triples = []
        
        # 批量处理组合，提高效率
        batch_size = 10000  # 可根据内存大小调整
        for start in range(0, len(combinations), batch_size):
            end = min(start + batch_size, len(combinations))
            batch_combos = combinations[start:end]
            
            # 提取当前批次的i和j索引
            i_indices = [combo[0] for combo in batch_combos]
            j_indices = [combo[1] for combo in batch_combos]
            
            # 批量计算所有s = cols[i] + cols[j]
            batch_s = (cols[i_indices] + cols[j_indices]) % 2
            
            # 对每个s，检查它是否等于cols中的某一列
            for idx, (i, j, s) in enumerate(zip(i_indices, j_indices, batch_s)):
                # 使用向量化操作找出与s匹配的所有列
                # (cols == s).all(axis=1)计算每列是否与s完全匹配
                matches = np.where((cols == s).all(axis=1))[0]
                
                # 过滤符合条件的k
                for k in matches:
                    if k > j and k != i and k != j:
                        valid_triples.append((i, j, k))
        
        # # 生成码字向量
        # codewords = []
        # for triple in valid_triples:
        #     c = np.zeros(n, dtype=int)
        #     c[list(triple)] = 1
        #     codewords.append(c)
        
        self.neighbors_idx = np.array(valid_triples)
        self.neighbors = self.idx_to_vec(self.neighbors_idx)
        return self.neighbors
    
    def get_nearest_neighbors(self, tx_bin=None):
        """
        获取指定码字的最近邻列表
        """
        if not hasattr(self, 'neighbors'):
            self.neighbors = self._find_nearest_neighbors()
        if tx_bin is None:
            return self.neighbors
        vector = ((self.encode(tx_bin) + 1 ) / 2).astype(int)
        return (self.neighbors + vector) % 2
    
    def get_nearest_neighbors_idx(self):
        """
        获取指定码字的最近邻列表, 返回索引
        """
        if not hasattr(self, 'neighbors_idx'):
            self._find_nearest_neighbors()
        return self.neighbors_idx
    
    def find_third_bit(self, error_pairs):
        """
        对于给定的两位错误组合，找到对应的唯一第三位错误位置
        error_pairs: n×2的数组，每行表示为1的两位的编号
        third_bits: n维数组，表示每一组两位所对应的唯一的第三位的编号
        """
        error_pairs = np.asarray(error_pairs, dtype=int)
        if error_pairs.ndim ==1:
            error_pairs = error_pairs.reshape(1, -1)
        
        # 预计算所有H列的整数表示（一次性完成）
        if not hasattr(self, '_H_col_ints'):
            H_cols = self.H.T
            self._H_col_ints = np.packbits(H_cols, axis=1, bitorder='little').flatten()
            
            # 创建查找表（一次性完成）
            self._lookup_table = np.zeros(2**self.r, dtype=int)
            for k, col_int in enumerate(self._H_col_ints):
                self._lookup_table[col_int] = k
        
        # 向量化计算syndrome对应的整数
        i_indices = error_pairs[:, 0]
        j_indices = error_pairs[:, 1]
        
        # 计算syndrome的整数表示
        syndrome_ints = (self._H_col_ints[i_indices] ^ self._H_col_ints[j_indices])
        
        # 使用查找表找到对应的第三位
        third_bits = self._lookup_table[syndrome_ints]
        
        return third_bits

def find_error_dist(rx_signals, correct):
    """
    对于错误的码字，寻找其最近距离
    """
    if len(correct) == 0:
        return 0, None
    errors = rx_signals[~correct]
    distances = np.sum((errors+1)**2, axis=1)
    return np.min(distances), errors[np.argmin(distances)]

if __name__ == "__main__":
    # from test_hamming import run_simulation_test
    # run_simulation_test(num_samples=1e6, hamming_r=3)

    # # 测试代码
    # r = 8
    # hamming_code = HammingCode(r=r, decoder='binary')
    # random_bits = hamming_code.random_bits(10)
    # encoded = hamming_code.encode(random_bits)
    # error_num = 1
    # # 添加错误
    # noised = encoded.copy()
    # for vec in noised:
    #     for i in range(error_num):
    #         idx = np.random.randint(0, hamming_code.n)
    #         vec[idx] *= -1
    # # 解码
    # decoded = hamming_code.decode(noised)
    # correct = np.all(decoded == random_bits, axis=1)
    # print(1*correct)

    # hamming_code = HammingCode(r=3, decoder='binary')
    # neighbors = hamming_code.get_nearest_neighbors()
    # print(neighbors)

    code = HammingCode(r=3, decoder='binary')
    print(code.H)
    # code.simulate_variance_controlled(
    #     noise_std=0.4, target_std=0.01, max_samples=1e6, 
    #     batch_size=1e3, min_errors=10
    # )