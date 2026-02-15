import numpy as np
from scipy.special import comb, logsumexp
from numpy.random import Generator, PCG64, Philox
from scipy.special import gamma, iv, ive
import itertools
from itertools import combinations
from code_linear import *
from ORBGRAND import ORBGRANDDecoder
from SGRAND import SGRANDDecoder, GRANDDecoder
from distance_calculator import *

def log_adder(a, b):
    return np.logaddexp(a, b)

class SamplerBase:
    """采样器基类"""
    def __init__(self):
        self.rng = Generator(Philox())  # 随机数生成器
        
    def sample(self, noise_std, tx_bin, batch_size):
        """
        生成噪声样本及其PDF值
        
        参数:
        noise_std: 噪声标准差
        tx_bin: 发送码字二进制表示
        batch_size: 批次大小
        
        返回:
        noise_samples: 噪声样本，形状为(batch_size, n)
        log_pdf_values: 对数PDF值，形状为(batch_size,)
        """
        raise NotImplementedError


class NaiveSampler(SamplerBase):
    """普通高斯采样器"""
    def __init__(self, code):
        """
        参数:
        n: 码长
        """
        super().__init__()
        self.n = code.dim  # 码长
        
    def sample(self, noise_std, tx_bin, batch_size):
        """生成标准高斯噪声及其PDF"""
        batch_size = int(batch_size)
        noise = self.rng.normal(0, noise_std, (batch_size, self.n))
        log_pdf = -(np.sum(noise**2, axis=1))/(2*noise_std**2) - self.n/2 * np.log(2*np.pi*noise_std**2)
        return noise, log_pdf


class ShiftSampler(SamplerBase):
    """平移高斯混合采样器"""
    def __init__(self, code):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        """
        super().__init__()
        self.code = code
        
    def sample(self, noise_std, tx_bin, batch_size):
        """生成向最近邻码字平移的高斯噪声及其PDF"""
        batch_size = int(batch_size)
            
        # 获取编码后的信号
        tx_signal = self.code.encode(tx_bin)
        
        # 获取当前码字的最近邻
        neighbor_bins = self.code.get_nearest_neighbors(tx_bin)
        chosen_neighbors = self.rng.choice(neighbor_bins, size=batch_size)
        
        # 计算均值偏移量
        chosen_neighbor_vecs = self.code.encode(chosen_neighbors)
        mean_shifts = (chosen_neighbor_vecs - tx_signal) / 2.0
        
        # 生成噪声样本
        noise = self.rng.normal(0, noise_std, (batch_size, self.code.dim)) + mean_shifts
        
        # 计算混合分布的PDF值
        log_pdfs = -np.inf
        neighbor_vecs = self.code.encode(np.array(neighbor_bins))
        for neighbor_vec in neighbor_vecs:
            shift = (neighbor_vec - tx_signal) / 2.0
            log_prob = -np.sum((noise - shift)**2, axis=1)/(2*noise_std**2) - self.code.dim/2 * np.log(2*np.pi*noise_std**2) - np.log(len(neighbor_bins))
            log_pdfs = log_adder(log_pdfs, log_prob)
        
        return noise, log_pdfs

class SingleShiftSampler(SamplerBase):
    """平移高斯混合采样器，只平移到一个点附近"""
    def __init__(self, code, shift_vec=0):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        """
        super().__init__()
        self.code = code
        self.shift_vec = shift_vec
        
    def sample(self, noise_std, tx_bin, batch_size, shift_vec=None):
        """生成向最近邻码字平移的高斯噪声及其PDF"""
        batch_size = int(batch_size)
        if shift_vec is None:
            shift_vec = self.shift_vec
        else:
            shift_vec = np.array(shift_vec)
        
        # 生成噪声样本
        noise_raw = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
        noise = noise_raw + shift_vec

        # 计算分布的PDF值
        log_pdfs = - np.sum((noise_raw)**2, axis=1)/(2*noise_std**2) - self.code.dim/2 * np.log(2*np.pi*noise_std**2)
        
        return noise, log_pdfs


class BesselSampler(SamplerBase):
    """贝塞尔高斯采样器"""
    def __init__(self, code):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        """
        super().__init__()
        self.code = code
        self.dim = code.dim
        self.r = self.code.get_r()

    def log_pdf(self, x, noise_std):
        """计算贝塞尔高斯分布的对数PDF"""
        d21 = self.dim / 2 - 1
        r = self.r
        s2nr = r / noise_std # sqrt(2nr)
        ynorm = np.linalg.norm(x, axis=1) / noise_std
        logpdf_gaussian = - np.sum(x**2, axis=-1) / (2 * noise_std**2) - self.dim / 2 * np.log(2 * np.pi * noise_std**2)
        t2 = - s2nr**2 / 2
        t3 = np.log(gamma(self.dim / 2)) + d21 * (np.log(2) - np.log(s2nr))
        # t4 = np.log(iv(d21, s2nr * ynorm)) - d21 * np.log(ynorm)
        t4 = np.log(ive(d21, s2nr * ynorm)) + np.abs(s2nr*ynorm) - d21 * np.log(ynorm)  # use scaled bessel function to avoid overflow
        logpdf_bessel = t2 + t3 + t4
        return logpdf_bessel + logpdf_gaussian

    def sample(self, noise_std, tx_bin, batch_size):
        """生成贝塞尔高斯噪声及其PDF"""
        batch_size = int(batch_size)
        # r = self.code.get_r(tx_bin)
        r = self.r
        # print('minimum distance r:', r)
        dim = self.dim
        # 1. 生成高斯噪声
        G = noise_std * self.rng.normal(0, 1, (batch_size, dim))
        
        # 2. 生成单位球面均匀分布U (半径=1)
        U = np.random.randn(batch_size, dim)  # 生成高斯随机向量
        U /= np.linalg.norm(U, axis=1, keepdims=True)  # 归一化到单位球面

        noise = G + r * U
        log_pdfs = self.log_pdf(noise, noise_std)
        
        return noise, log_pdfs

def max_distances_logsumexp(x, r, factor=1, const=0, threshold=0.01):
    """
    计算x到所有n维{-1,1}向量(恰好r个1)的欧氏距离平方的logsumexp
    
    参数:
    x: 浮点向量
    r: 目标向量中1的个数
    factor: 影响距离计算的因子
    const: 常数项，用于调整计算
    threshold: 早停阈值
    
    返回:
    所有距离平方的logsumexp值
    """
    n = len(x)
    if n <= 2*r:
        raise ValueError("n必须大于2r")
    
    # 计算每个维度的贡献差异: (x[i]-1)^2 - (x[i]-0)^2 = -2*x[i] +1
    contributions = (-2 * x + 1) * factor
    
    # 按贡献从大到小排序
    sorted_indices = np.argsort(-contributions)
    sorted_contribs = contributions[sorted_indices]
    
    # 基础距离(所有维度均为0时的距离)
    base_distance = np.sum((x - 0)**2) * factor + const
    
    # 使用动态规划计算各层的组合距离
    # dp[i][j]是一个列表，表示从前i个元素中选j个的所有可能组合的距离增量
    dp = [[[] for _ in range(r+1)] for _ in range(n+1)]
    logsums = np.ones(n+1) * -np.inf  # 初始化logsumexp数组

    # 初始化：空集的距离增量为0
    dp[0][0] = [0]  # 修改：应该是[0]而不是空列表

    for i in range(1, n+1):
        # 不选任何元素的情况
        dp[i][0] = [0]  # 修改：保持一致性
        
        for j in range(1, min(i, r)+1):
            dp[i][j] = []  # 显式初始化为空列表
            
            # 不选第i-1个元素的情况
            if dp[i-1][j]:  # 确保列表非空
                dp[i][j].extend(dp[i-1][j])
            
            # 选第i-1个元素的情况
            if dp[i-1][j-1]:  # 确保列表非空
                dp[i][j].extend([val + sorted_contribs[i-1] for val in dp[i-1][j-1]])
            
        # 早停条件
        if i >= r:
            term = logsumexp(np.array(dp[i][r]) + base_distance)
            # logsums[i] = log_adder(logsums[i-1], term)
            logsums[i] = term
            if logsums[i] - logsums[i-1] < threshold:
                # print(f"Early stopping at i={i}/{n}, added term={term}/{logsums[i]}")
                break
    
    return logsums[i]
    
def max_distances_logsumexp_batch(x, r, factor=1, const=0, threshold=0.001):
    """
    批量计算x中每个向量到所有n维{-1,1}向量(恰好r个1)的欧氏距离平方的logsumexp
    
    参数:
    x: 形状为(batch_size, n)的numpy数组，每行是一个需要处理的向量
    r: 目标向量中1的个数
    factor: 影响距离计算的因子
    const: 常数项，用于调整计算
    threshold: 早停阈值
    
    返回:
    所有距离平方的logsumexp值数组，形状为(batch_size,)
    """
    # 确保x是2D数组
    if x.ndim == 1:
        x = x.reshape(1, -1)
    batch_size, n = x.shape
    if n <= 2*r:
        raise ValueError("n必须大于2r")
    
    # 计算每个维度的贡献差异: (x[i]-1)^2 - (x[i]-0)^2 = -2*x[i] +1
    contributions = (-2 * x + 1) * factor  # 形状为(batch_size, n)
    sorted_contribs = -np.sort(-contributions, axis=1)  # 形状为(batch_size, n)
    base_distances = np.sum((x - 0)**2, axis=1) * factor + const
    
    # 为整个批次创建DP表
    # dp_values[i][j]是一个数组，形状为(batch_size, combinations)
    # 表示每个向量从前i个元素中选j个的所有可能组合的距离增量
    dp = {}
    
    # 初始化：对所有向量，空集的距离增量为0
    dp[(0, 0)] = np.zeros((batch_size, 1))
    
    # 创建活跃向量掩码和结果数组
    active = np.ones(batch_size, dtype=bool)
    results = np.zeros(batch_size)
    prev_results = np.zeros(batch_size)
    
    # 主循环
    for i in range(1, n+1):
        # j=0的情况：不选任何元素
        dp[(i, 0)] = np.zeros((batch_size, 1))
        
        for j in range(1, min(i, r)+1):
            # 合并两种情况的所有组合：不选第i-1个和选第i-1个
            # 1. 不选第i-1个元素：沿用dp[i-1][j]的所有组合
            not_include = dp.get((i-1, j), np.zeros((batch_size, 0)))
            
            # 2. 选第i-1个元素：dp[i-1][j-1]的所有组合加上第i-1个元素的贡献
            include_base = dp.get((i-1, j-1), np.zeros((batch_size, 0)))
            
            # 计算新的组合数量
            not_include_count = not_include.shape[1]
            include_count = include_base.shape[1]
            total_count = not_include_count + include_count
            
            if total_count > 0:
                # 创建新的组合数组
                dp[(i, j)] = np.zeros((batch_size, total_count))
                dp[(i, j)][:, :not_include_count] = not_include
                dp[(i, j)][:, not_include_count:] = include_base + sorted_contribs[:, i-1].reshape(-1, 1)
            else:
                dp[(i, j)] = np.zeros((batch_size, 0))
        
        # 早停检查
        if i >= r and dp[(i, r)].shape[1] > 0:
            # 计算当前的logsumexp值
            if i > r:
                prev_results = results.copy()
            
            # 批量计算logsumexp - 并行版本
            # 为所有向量计算基础距离加上每个组合的距离增量
            all_distances = dp[(i, r)] + base_distances[:, np.newaxis]
            
            # 仅为活跃向量计算logsumexp
            # logsumexp可以直接应用于2D数组，axis=1表示沿行方向计算
            results[active] = logsumexp(all_distances[active], axis=1)
            
            # 检查是否所有向量都可以早停
            if i > r:
                active = (results - prev_results) > threshold
                if not np.any(active):
                    break
            
            # 清理不再需要的状态以节省内存
            if i > 1:
                for k in range(r+1):
                    dp.pop((i-2, k), None)
    
    return results

def compute_distances_to_binary_vectors(X, r, batch_size=10000):
    """
    计算X中每个向量到所有n维{0,1}向量（恰好有r个1）的欧氏距离平方
    
    参数:
    X: 形状为(m, n)的numpy数组，m是向量数量，n是维度
    r: 目标向量中1的个数
    batch_size: 每批处理的组合数，用于控制内存使用
    
    返回:
    distances: 形状为(m, comb(n,r))的numpy数组，表示距离平方
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)  # 确保X是2D数组
    
    m, n = X.shape  # 向量数量和维度
    num_combinations = int(comb(n, r))  # 总组合数
    
    # 预先计算常数部分：每个向量的平方和
    x_squared_sum = np.sum(X**2, axis=1)  # 形状为(m,)
    
    # 初始化距离矩阵
    distances = np.zeros((m, num_combinations))
    
    # 批处理生成所有组合
    combo_iter = combinations(range(n), r)
    for batch_start in range(0, num_combinations, batch_size):
        batch_end = min(batch_start + batch_size, num_combinations)
        batch_size_actual = batch_end - batch_start
        
        # 为当前批次创建二进制向量矩阵
        batch_vectors = np.zeros((batch_size_actual, n))
        
        # 填充批次向量
        for i, combo in enumerate(itertools.islice(combo_iter, batch_size_actual)):
            batch_vectors[i, list(combo)] = 1
        
        # 计算点积：X @ batch_vectors.T 形状为(m, batch_size_actual)
        dot_product = X @ batch_vectors.T
        
        # 计算欧氏距离平方：||x-v||^2 = ||x||^2 + ||v||^2 - 2x·v
        # 其中||v||^2 = r (因为v有r个1)
        distances[:, batch_start:batch_end] = x_squared_sum[:, np.newaxis] - 2 * dot_product + r
    
    return distances

def generate_01_array(b, n, r, rng=None):
    """
    生成一个b*n的数组，其中每一行恰好有r个1，其余为0
    
    参数:
    b (int): 行数
    n (int): 列数
    r (int): 每行中1的个数
    rng (numpy.random.Generator, optional): 随机数生成器
    
    返回:
    numpy.ndarray: 形状为(b,n)的二维数组，每行恰好有r个1
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 创建一个b×n的零数组
    result = np.zeros((b, n), dtype=int)
    
    # 生成随机值并获取每行最大的r个值的索引
    rand_vals = rng.random((b, n))
    indices = np.argpartition(-rand_vals, r, axis=1)[:, :r]
    
    # 使用高级索引设置1
    row_indices = np.arange(b)[:, np.newaxis]
    result[row_indices, indices] = 1
    
    return result

class SymShiftSampler(SamplerBase):
    def __init__(self, code, pdf_type='direct', decoder_type=None):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        pdf_type: 概率密度函数类型，默认是'direct'
        decoder_type: 解码器类型，默认按照decoder的类型
        """
        super().__init__()
        self.code = code
        self.dim = code.dim
        self.pdf_type = pdf_type
        self.decoder = code.decoder
        
        if decoder_type is not None:
            self.decoder_type = decoder_type
        elif self.decoder == 'binary':
            self.decoder_type = 'binary'
        else:
            self.decoder_type = self.decoder.decoder_type

        if self.decoder_type == 'chase':
            if self.decoder.errs == 2:
                self.decoder_type = 'ORBGRAND'
            elif self.decoder.errs >= 3:
                self.decoder_type = 'SGRAND'

        
        if self.pdf_type == 'direct':
            if self.decoder_type == 'ORBGRAND':
                self.centers = self.generate_ORBGRAND_centers()
            elif self.decoder_type == 'SGRAND':
                self.centers = self.generate_SGRAND_centers()

    def sample(self, noise_std, tx_bin, batch_size, pdf_type='direct', threshold=0.001, decoder_type=None):
        """利用对称性，只选取某一个点附近的噪声，并分析其它点的pdf对此点的贡献"""
        if decoder_type is not None:
            self.decoder_type = decoder_type
        if self.decoder_type == 'ORBGRAND':
            # return self.sample_centered(noise_std, tx_bin, batch_size, pdf_type, threshold)
            return self.sample_ORBGRAND(noise_std, tx_bin, batch_size)
        elif self.decoder_type == 'SGRAND':
            return self.sample_SGRAND(noise_std, tx_bin, batch_size)
        batch_size = int(batch_size)
        
        tx_signal = self.code.encode(tx_bin)
        # assert that tx_signal is a negative vector
        if np.any(tx_signal > 0):
            raise ValueError("tx_signal must all -1")
        err_num = int(self.code.get_r(tx_bin) ** 2)
        # tx_01 = ((1 + tx_signal) / 2).astype(int)

        # shift_01 = np.zeros(self.dim)
        # shift_01[0:err_num] = 1
        # shift = shift_01
        shift = generate_01_array(batch_size, self.dim, err_num, self.rng)

        # 生成噪声样本
        noise_raw = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
        noise = noise_raw + shift

        log_pdfs = -np.inf * np.ones(batch_size)
        num_neighbors = comb(self.dim, err_num)
        logpdf_const = - self.dim / 2 * np.log(2 * np.pi * noise_std ** 2) - np.log(num_neighbors)

        # direct way to compute logpdf
        if pdf_type == 'direct':
            pairs = np.array(list(combinations(range(self.code.dim), 2)))
            # dists = compute_distances_to_binary_vectors(noise, err_num, batch_size)
            dists = sparse_distance_calculator(noise, pairs)  # (batch_size, n_centers)
            terms = logsumexp(- dists / (2 * noise_std ** 2), axis=-1)
            log_pdfs = logpdf_const + terms
            # dists_sorted = np.sort(dists, axis=1)
            # print(dists_sorted[0:5, 0:15])
        elif pdf_type == 'layered':
            # selective way
            log_pdfs = max_distances_logsumexp_batch(noise, err_num, factor=-1/(2 * noise_std ** 2), const=logpdf_const, threshold=threshold)
        
        return noise, log_pdfs

    def generate_ORBGRAND_centers(self):
        """
        从neighbors生成修改后的中心点数组
        neighbors: 2D数组，每行有3个值为1的元素，其余为0
        centers: 2D数组，行数是neighbors行数的3倍
        """
        if not hasattr(self.code, 'neighbors'):
            neighbors = self.code._find_nearest_neighbors()
        else:
            neighbors = self.code.neighbors
        n_rows, n_cols = neighbors.shape
        # 确保每行有且仅有3个1
        assert np.all(np.sum(neighbors, axis=1) == 3)
        
        centers = np.zeros((n_rows * 3, n_cols))
        for i in range(n_rows):
            # 找到第i行中值为1的元素的索引
            one_indices = np.where(neighbors[i] == 1)[0]
            
            # 创建基本行模板，所有非零位置设置为2/3
            base_row = np.zeros(n_cols)
            base_row[one_indices] = 2/3
            
            # 为每个非零元素位置创建一个变体
            for j, idx in enumerate(one_indices):
                row = base_row.copy()
                row[idx] = 4/3  # 将当前位置的值改为4/3
                centers[i * 3 + j] = row
                
        self.centers = centers
        self.centers_idx = self.code.get_nearest_neighbors_idx()
        return centers

    def generate_SGRAND_centers(self):
        """Just copy the neighbors from the code"""
        if not hasattr(self.code, 'neighbors'):
            neighbors = self.code._find_nearest_neighbors()
        else:
            neighbors = self.code.neighbors
        if not hasattr(self.code, 'neighbors_idx'):
            neighbors_idx = self.code.get_nearest_neighbors_idx()
        else:
            neighbors_idx = self.code.neighbors_idx
        # 确保每行有且仅有3个1
        assert np.all(np.sum(neighbors, axis=1) == 3)
        self.centers = neighbors
        self.centers_idx = neighbors_idx
        return neighbors

    def sample_SGRAND(self, noise_std, tx_bin, batch_size):
        """使用SGRAND解码器进行采样"""
        if not hasattr(self, 'centers_idx') or not hasattr(self, 'centers'):
            self.generate_SGRAND_centers()
        centers_idx = self.centers_idx
        centers = self.centers

        # 选择中心点
        centers_use = centers[self.rng.integers(0, len(centers), size=batch_size)]
        noise_raw = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
        noise = noise_raw + centers_use

        # 计算PDF
        dists = sparse_distance_calculator(noise, centers_idx)# (batch_size, n_centers)
        dist_item = logsumexp(-dists / (2 * noise_std ** 2), axis=-1)
        log_pdfs = dist_item - self.code.dim / 2 * np.log(2 * np.pi * noise_std**2) - np.log(dists.shape[-1])
        return noise, log_pdfs
    
    def sample_ORBGRAND(self, noise_std, tx_bin, batch_size):
        """使用ORBGRAND解码器进行采样"""
        if not hasattr(self, 'centers_idx') or not hasattr(self, 'centers'):
            self.generate_ORBGRAND_centers()
        centers_idx = self.centers_idx
        centers = self.centers

        # 选择中心点
        centers_use = centers[self.rng.integers(0, len(centers), size=batch_size)]
        noise_raw = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
        noise = noise_raw + centers_use

        # 计算PDF
        # d_vec = np.array([4/3, 2/3, 2/3])
        # dists = sparse_distance_calculator_permuted(noise, centers_idx, d_vec=d_vec)# (batch_size, n_centers)
        dists = sparse_distance_calculator_4323(noise, centers_idx)  # (batch_size, n_centers)
        dist_item = logsumexp(-dists / (2 * noise_std ** 2), axis=-1)
        log_pdfs = dist_item - self.code.dim / 2 * np.log(2 * np.pi * noise_std**2) - np.log(dists.shape[-1])
        return noise, log_pdfs

    def sample_centered(self, noise_std, tx_bin, batch_size, pdf_type='direct', threshold=0.001):
        """针对ORBGRAND, SGRAND解码器进行采样
        slow, so dumped it
        """
        if pdf_type == 'direct':
            if not hasattr(self, 'centers'):
                if self.decoder_type == 'ORBGRAND':
                    self.centers = self.generate_ORBGRAND_centers()
                elif self.decoder_type == 'SGRAND':
                    self.centers = self.generate_SGRAND_centers()
            centers = self.centers
            n_centers = len(centers)

            # sampling
            centers_use = centers[self.rng.integers(0, n_centers, size=batch_size)]
            noise_raw = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
            noise = noise_raw + centers_use

            # compute logpdf
            noise_norms = np.sum(noise**2, axis=1, keepdims=True)  # (batch_size, 1)
            centers_norms = np.sum(centers**2, axis=1)  # (n_centers,)
            dot_products = noise @ centers.T  # (batch_size, n_centers)
            squared_distances = noise_norms + centers_norms - 2 * dot_products

            log_pdfs = logsumexp(-squared_distances / (2 * noise_std ** 2), axis=1) - self.code.dim / 2 * np.log(2 * np.pi * noise_std ** 2) - np.log(n_centers)
        else:
            raise ValueError("Unsupported pdf_type: {}".format(pdf_type))

        return noise, log_pdfs
    
from scipy.stats import norm, chi2, truncnorm

def gaussian_cutoff_sampling(d0, batch_size):
    """
    标准高斯分布的右截断采样，从x > d0的区域采样
    
    参数:
    d0: 截断阈值，x > d0的区域
    batch_size: 采样数量
    
    返回:
    samples: 采样点（均为正值，x > d0）
    log_pdfs: 每个采样点的对数概率密度（在原始高斯中的对数密度）
    """
    batch_size = int(batch_size)
    
    # 使用逆变换采样避免数值问题
    rng = np.random.default_rng()
    
    # 计算截断区域的生存函数值（右尾概率）
    sf_d0 = norm.sf(d0)  # P(X > d0)
    
    # 生成均匀随机数并映射到截断区域
    u = rng.uniform(0, 1, size=batch_size)
    
    # 使用逆生存函数生成样本，直接从x > d0区域采样
    samples = norm.isf(u * sf_d0)
    
    # 计算原始高斯中的对数PDF
    log_pdf_original = norm.logpdf(samples)
    
    return samples, log_pdf_original

def chi2_cutoff_sampling(df, d0, batch_size):
    """
    卡方分布的截断采样，从x > d0的区域采样
    """
    batch_size = int(batch_size)
    # 计算对数尾概率
    log_tail_prob = chi2.logsf(d0, df)
    
    # 检查是否需要使用正态近似
    use_normal_approx = (d0 > df + 50 * np.sqrt(2 * df)) or (log_tail_prob < -700)
    
    if not use_normal_approx:
        # 原始方法
        u = np.random.uniform(0, 1, batch_size)
        u_safe = np.clip(u, 1e-300, 1)
        log_u = np.log(u_safe)
        log_s = log_tail_prob + log_u
        samples = chi2.isf(np.exp(log_s), df)
    else:
        # Wilson-Hilferty 变换参数
        t0 = (d0 / df) ** (1/3)
        mu = 1 - 2/(9*df)
        sigma = np.sqrt(2/(9*df))
        z0 = (t0 - mu) / sigma
        
        # 生成均匀随机数 (避免log(0))
        u = np.random.uniform(0, 1, batch_size)
        u_safe = np.clip(u, 1e-300, 1)
        
        # 手动实现正态尾部采样 (基于Mills ratio渐近展开)
        log_u = np.log(u_safe)
        
        # 主要项：sqrt(z0^2 - 2*log(u))
        z_sq = z0**2 - 2 * log_u
        
        # 修正项：考虑log(z)项的影响
        # 使用迭代法：z ≈ sqrt(z_sq - log(z_sq) + log(z0))
        z1 = np.sqrt(np.maximum(z_sq, 0))  # 防止负值
        z2 = np.sqrt(np.maximum(z_sq - np.log(z1) + np.log(z0), 1e-100))
        
        # 最终采样值 (右尾取正)
        z = np.where(z_sq > 0, z2, z0)  # 处理z_sq<0的情况
        
        # 转换回卡方分布
        t = mu + sigma * z
        samples = df * t**3
    
    # # 计算截断PDF (始终在对数空间操作)
    # log_pdf_original = chi2.logpdf(samples, df)
    # log_pdf_trunc = log_pdf_original - log_tail_prob
    # pdf_trunc = np.exp(log_pdf_trunc)
    
    return samples

class CutoffSampler(SamplerBase):
    """截断高斯采样器"""
    def __init__(self, code, dp_find=True, decoder_type=None, gaussian_ratio=0.8, use_bessel=True):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        pdf_type: 概率密度函数类型，默认是'direct'
        decoder_type: 解码器类型，默认按照decoder的类型
        """
        super().__init__()
        self.code = code
        self.dim = code.dim
        self.decoder = code.decoder
        self.dp_find = dp_find
        self.gaussian_ratio = gaussian_ratio

        self.n_neighbors = self.dim*(self.dim-1)//6

        if dp_find == False:
            self.neighbors_idx = self.code.get_nearest_neighbors_idx()
        
        if decoder_type is not None:
            self.decoder_type = decoder_type
        elif self.decoder == 'binary':
            self.decoder_type = 'binary'
        else:
            self.decoder_type = self.decoder.decoder_type

        if self.decoder_type == 'chase':
            if self.decoder.errs == 2:
                self.decoder_type = 'ORBGRAND'
            elif self.decoder.errs >= 3:
                self.decoder_type = 'SGRAND'

        if self.decoder_type =='binary':
            self.v = [1, 1, 0]
        elif self.decoder_type == 'ORBGRAND':
            self.v = [4/3, 2/3, 2/3]
        elif self.decoder_type == 'SGRAND':
            self.v = [1, 1, 1]
        else:
            raise ValueError("Unsupported decoder_type: {}".format(self.decoder_type))
        self.v = np.array(self.v, dtype=float)
        # self.v = np.sort(self.v)[::-1]

        if np.allclose(self.v, self.v[0]):
            self.sym=3
            self.n_planes = self.n_neighbors
        elif np.isclose(self.v[1], self.v[2]) or np.isclose(self.v[0], self.v[1]):
            self.sym=2
            self.n_planes = self.n_neighbors * 3
        else:
            self.sym=1
            self.n_planes = self.n_neighbors * 6

        self.use_bessel = use_bessel
        if use_bessel:
            self.bessel_sampler = BesselSampler(code)

    def judge_all_cutoff_dp(self, noise):
        """
        真正的批量三重判断并行架构
        完全基于排序后的第0,1,2,3,4号元素进行判断
        """
        if noise.ndim == 1:
            noise = noise.reshape(1, -1)
        batch_size, dim = noise.shape
        sum_v = np.sum(self.v**2)
        # v0, v1, v2 = self.v[0], self.v[1], self.v[2]
        nums = np.zeros(batch_size, dtype=int)  # 记录每个噪声向量满足条件的计数
        
        # 对每个噪声向量，找到最大的5个元素及其索引
        top_5_indices = np.argpartition(-noise, 5, axis=1)[:, :5]  # 找到最大5个元素的索引
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        top_5_values = noise[batch_indices, top_5_indices]  # 获取对应的值

        # 对每行的5个值进行降序排序
        sort_indices = np.argsort(-top_5_values, axis=1)  # 在5个值中的排序索引
        sorted_top_5_indices = top_5_indices[batch_indices, sort_indices]  
        sorted_top_5_values = top_5_values[batch_indices, sort_indices]

        # sums = v0 * sorted_top_5_values[:, 0] + v1 * sorted_top_5_values[:, 3] + v2 * sorted_top_5_values[:, 4]
        sums = np.dot(self.v, sorted_top_5_values[:, [0, 3, 4]].T)
        edge_case_mask = sums > sum_v

        # tell case 1 from case 2
        expected_3pos = self.code.find_third_bit(sorted_top_5_indices[:, :2])
        case1_mask = (expected_3pos == sorted_top_5_indices[:, 2]) & ~edge_case_mask
        case2_mask = (~case1_mask) & (~edge_case_mask)

        # case1: 前三大元素点积判断
        if np.any(case1_mask):
            nums[case1_mask] = self._judge_cutoff_reshuffle(sorted_top_5_values[case1_mask, :3])

        # case2: 前三大元素两两找第三位
        if np.any(case2_mask):
            noise2 = noise[case2_mask]
            top_indices2 = sorted_top_5_indices[case2_mask, :3]

            # 生成所有两两组合 (i,j) 其中 i,j ∈ {0,1,2}
            all_pairs = [(0,1), (0,2), (1,2)]
            total_counts = np.zeros(len(noise2), dtype=int)

            for i, j in all_pairs:
                pairs = top_indices2[:, [i, j]]
                third_bits = self.code.find_third_bit(pairs)
                triplets_idx = np.column_stack([pairs, third_bits])

                # 获取对应的噪声值
                batch_idx = np.arange(len(noise2))[:, np.newaxis]
                noise_values = noise2[batch_idx, triplets_idx]

                # 使用现有函数判断
                counts = self._judge_cutoff_reshuffle(noise_values)
                total_counts += counts

            nums[case2_mask] = total_counts

        # edge case: 逐条处理
        i_case3 = np.where(edge_case_mask)[0]
        for i in i_case3:
            nums[i] = self._judge_edge_case_vec(noise[i, :])

        return nums
    
    def _judge_cutoff_reshuffle(self, triplets):
        """Input: (n_i, n_j, n_k), where (i, j, k) is a valid tuple
        Output: number of exceeded planes"""
        if self.sym == 3:
            combos = [(0,1,2)]
        elif self.sym == 2:
            combos = [(0,1,2), (0,2,1), (1,2,0)]
        else:
            combos = [(0,1,2), (0,2,1), (1,2,0), (1,0,2), (2,0,1), (2,1,0)]

        nums = np.zeros(len(triplets), dtype=int)
        sum_v = np.sum(self.v**2)
        for (i,j,k) in combos:
            noise = triplets[:, [i, j, k]]
            sums = np.dot(self.v, noise.T)
            nums += (sums > sum_v).astype(int)
        return nums

    def _find_valid_pairs(self, noise_vec):
        """
        通用双指针算法找到所有满足条件的(i,j)对
        """
        dim = len(noise_vec)
        v0, v1, v2 = self.v[0], self.v[1], self.v[2]
        sum_v = np.sum(self.v**2)
        
        # 使用numpy的argsort进行高效排序（降序）
        indices = np.argsort(noise_vec)[::-1]
        values = noise_vec[indices]
        
        valid_pairs = []
        coeff = v1 + v2
        
        # 双指针算法：对于每个i，j从i+1开始向右移动
        # 由于数组是降序排列，一旦找到不满足条件的元素就停止
        
        for i_pos in range(dim):
            i = indices[i_pos]
            val_i = values[i_pos]
            
            # 计算j需要满足的最小值
            min_val_j = (sum_v - v0 * val_i) / coeff
                
            if values[i_pos + 1] < min_val_j:
                break
                
            for j_pos in range(i_pos + 1, dim):
                if values[j_pos] < min_val_j:
                    break
                j = indices[j_pos]
                valid_pairs.append((i, j))
        
        return valid_pairs
    
    def _judge_edge_case_vec(self, noise_vec):
        """
        情况3：逐条处理漏网之鱼
        只处理一条向量，返回整数计数
        """
        v0, v1, v2 = self.v[0], self.v[1], self.v[2]
        sum_v = np.sum(self.v**2)
        dim = len(noise_vec)
        
        # 使用_find_valid_pairs找到所有有效二元组
        valid_pairs = self._find_valid_pairs(noise_vec)
        
        if not valid_pairs:
            return 0
        
        # 使用find_third_bit找到对应的第三位
        pairs_array = np.array(valid_pairs)
        third_bits = self.code.find_third_bit(pairs_array)
        
        # 验证完整条件并去重
        valid_triplets = set()
        for (i, j), k in zip(valid_pairs, third_bits):
            n0, n1, n2 = noise_vec[i], noise_vec[j], noise_vec[k]
            if v0 * n0 + v1 * n1 + v2 * n2 > sum_v:
                triplet = tuple(sorted([i, j, k]))
                valid_triplets.add(triplet)
        
        return len(valid_triplets)

    def judge_all_cutoff_3sym(self, noise):
        """3sym版本：每个最近邻居对应一个半空间 x^T n >= sum(v)"""
        if noise.ndim == 1:
            noise = noise.reshape(1, -1)
        
        sum_v = np.sum(self.v**2)  # 计算v的平方和
        
        noise_values = noise[:, self.neighbors_idx]  # (batch_size, M, 3)
        dot_products = self.v[0] * np.sum(noise_values, axis=2)  # (batch_size, M)
        
        return dot_products >= sum_v
        
    def judge_all_cutoff_2sym(self, noise):
        """2sym版本：每个最近邻居对应三个半空间，通过重排v的三个元素得到"""
        noise = np.asarray(noise, dtype=float)
        if noise.ndim == 1:
            noise = noise.reshape(1, -1)
        
        noise_values = noise[:, self.neighbors_idx]  # (batch_size, M, 3)
        
        # 三个重排：v的循环移位
        v_perms = [
            np.array([self.v[0], self.v[1], self.v[2]]),
            np.array([self.v[1], self.v[2], self.v[0]]),
            np.array([self.v[2], self.v[0], self.v[1]])
        ]
        sum_v = np.sum(self.v**2)  # 计算v的平方和
        
        # 对每个邻居计算三个半空间的结果
        results = []
        for v_perm in v_perms:
            dot_products = np.sum(noise_values * v_perm, axis=2)  # (batch_size, M)
            results.append(dot_products >= sum_v)
        
        # 合并结果：按最近邻居分组，每个邻居的三个半空间挨在一起
        # 使用np.stack然后reshape来得到正确的顺序
        result_stack = np.stack(results, axis=2)  # (batch_size, M, 3)
        return result_stack.reshape(result_stack.shape[0], -1)  # (batch_size, 3M)
    
    def sample_gaussian_cutoff(self, batch_size, noise_std):
        """
        实现半平面截断高斯采样：x^T n >= ||v||^2
        使用稀疏操作优化，将复杂度从O(n)降低到O(1)
        """
        batch_size = int(batch_size)
        
        # 获取半平面参数
        if not hasattr(self, 'neighbors_idx'):
            self.neighbors_idx = self.code.get_nearest_neighbors_idx()
            
        neighbors_idx = self.neighbors_idx  # (M, 3) 每行是三个非零位的索引
        v_norm = np.linalg.norm(self.v)  # v的范数
        
        # 随机选择邻居（半平面）
        num_neighbors = len(neighbors_idx)
        chosen_indices = neighbors_idx[self.rng.integers(0, num_neighbors, size=batch_size)]  # (batch_size, 3)
        
        # 批量随机重排self.v（完全向量化实现）
        # 生成随机索引矩阵
        random_indices = self.rng.permuted(
            np.tile(np.arange(3), (batch_size, 1)), axis=1
        )
        v_perms = self.v[random_indices] / v_norm  # 归一化
        
        # 生成批量高斯噪声
        orthogonal_noise = self.rng.normal(0, noise_std, (batch_size, self.code.dim))
        
        # 使用稀疏操作计算投影（点积）
        # 只计算非零位置的点积：sum(orthogonal_noise[indices] * v_perms)
        projections = np.sum(orthogonal_noise[np.arange(batch_size)[:, None], chosen_indices] * v_perms, axis=1)
        
        # 计算截断阈值
        cutoff_scaled = v_norm / noise_std
        
        # 批量从截断正态分布采样平行分量
        parallel_components = truncnorm.rvs(
            cutoff_scaled, np.inf, loc=0, scale=1, size=batch_size, random_state=self.rng
        )
        parallel_components = parallel_components * noise_std

        # 使用稀疏操作构建最终噪声向量
        # 只需要更新3个非零位置，而不是整个向量
        noise = orthogonal_noise.copy()
        scale_factors = (parallel_components - projections)  # (batch_size,)
        
        # 批量更新非零位置
        batch_indices = np.arange(batch_size)[:, None]  # (batch_size, 1)
        noise[batch_indices, chosen_indices] += scale_factors[:, None] * v_perms
        
        return noise

    def sample_chi2_cutoff(self, batch_size, noise_std):
        """
        实现球面截断高斯采样：从半径为norm(v)的球外采样
        正确实现：||n|| >= ||v||
        """
        batch_size = int(batch_size)
        radius = np.linalg.norm(self.v)
        
        # 正常情况：使用chi-square截断采样
        chi2_threshold = (radius / noise_std) ** 2
        df = self.code.dim
            
        directions = self.rng.normal(0, 1, (batch_size, self.code.dim))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        # 使用更稳定的截断采样
        distances_squared = chi2_cutoff_sampling(df, chi2_threshold, batch_size)
        distances = np.sqrt(distances_squared) * noise_std
        
        noise = directions * distances[:, np.newaxis]
            
        return noise

    def sample(self, noise_std, tx_bin, batch_size):
        """
        主采样函数：混合半平面截断和球面截断高斯采样
        使用二项分布决定两种采样方法的数量
        """
        batch_size = int(batch_size)
        
        # 获取编码信号（确保是-1向量）
        tx_signal = self.code.encode(tx_bin)
        if np.any(tx_signal > 0):
            raise ValueError("tx_signal must all -1")
        
        # 使用二项分布决定两种采样方法的数量
        gaussian_samples = self.rng.binomial(batch_size, self.gaussian_ratio)
        chi2_samples = batch_size - gaussian_samples
        
        # 生成噪声样本
        noise = np.empty((batch_size, self.code.dim))
        log_pdfs = np.empty(batch_size)
        
        # 半平面截断高斯采样
        if gaussian_samples > 0:
            gaussian_noise = self.sample_gaussian_cutoff(gaussian_samples, noise_std)
            noise[:gaussian_samples] = gaussian_noise
        
        # 球面截断高斯采样
        if chi2_samples > 0:
            if not self.use_bessel:
                chi2_noise = self.sample_chi2_cutoff(chi2_samples, noise_std)
            else:
                chi2_noise, _ = self.bessel_sampler.sample(noise_std, tx_bin, chi2_samples)
            noise[gaussian_samples:] = chi2_noise

        log_pdf_gaussian = self._log_pdf_gaussian_cutoff(noise, noise_std)

        if self.use_bessel:
            log_pdf_chi2 = self.bessel_sampler.log_pdf(noise, noise_std)
        else:
            log_pdf_chi2 = self._log_pdf_chi2_cutoff(noise, noise_std)

        log_pdfs = np.logaddexp(
            log_pdf_gaussian + np.log(self.gaussian_ratio),
            log_pdf_chi2 + np.log(1 - self.gaussian_ratio)
        )

        return noise, log_pdfs

    def _log_pdf_gaussian_cutoff(self, noise, noise_std):
        """
        计算半平面截断高斯采样的对数PDF
        直接调用judge_all_cutoff方法：统计在多少个半平面之外
        """
        # batch_size = noise.shape[0]
        
        # 获取标准高斯PDF（未归一化）
        log_pdf_standard = -self.code.dim/2 * np.log(2*np.pi*noise_std**2) - np.sum(noise**2, axis=1) / (2*noise_std**2)
        
        # 获取半平面参数
        v_squared_sum = np.sum(self.v ** 2)
        v_norm = np.sqrt(v_squared_sum)
        
        # 根据v的情况选择使用3sym还是2sym
        if self.dp_find:
            num_valid_planes = self.judge_all_cutoff_dp(noise)
        else:
            if self.sym == 3:
                valid_planes = self.judge_all_cutoff_3sym(noise)
            else:
                valid_planes = self.judge_all_cutoff_2sym(noise)
            num_valid_planes = np.sum(valid_planes, axis=1)  # (batch_size,)
        
        n_planes = self.n_planes
        # 计算CDF修正因子（每个半平面的生存概率）
        log_cdf_correction = norm.logsf(v_norm / noise_std)  # P(Z >= v_norm/sigma)
        
        # 最终PDF = 标准高斯PDF × (CDF修正因子)
        # 避免log(0)的情况
        log_num = np.where(num_valid_planes > 0, np.log(num_valid_planes), -np.inf)  # 防止除以0
        log_pdf = log_pdf_standard - log_cdf_correction + log_num - np.log(n_planes)

        return log_pdf

    def _log_pdf_chi2_cutoff(self, noise, noise_std):
        """
        计算球面截断高斯采样的对数PDF
        基于球面截断：||n|| >= ||v||
        """
        # 计算球面半径
        radius = np.linalg.norm(self.v)
        
        # 计算噪声向量的范数
        noise_norm2 = np.sum(noise**2, axis=1)  # (batch_size,)
        if_out = noise_norm2 < radius**2

        # 标准高斯PDF
        log_pdf_standard = -self.code.dim/2 * np.log(2*np.pi*noise_std**2) - noise_norm2 / (2*noise_std**2)
        
        # 球面截断的CDF修正因子
        # P(||n|| >= radius) = P(χ²(dim) >= (radius/σ)²)
        chi2_threshold = (radius / noise_std) ** 2
        log_cdf_correction = chi2.logsf(chi2_threshold, self.code.dim)
        # log_cdf_correction = np.log(cdf_correction)
        
        # 最终PDF = 标准高斯PDF × (CDF修正因子)
        log_pdf_final = log_pdf_standard - log_cdf_correction
        log_pdf_final = np.where(if_out, log_pdf_final, -np.inf)  # 如果在球内，则PDF为0
        
        return log_pdf_final

class FixedShiftSampler(SamplerBase):
    """固定平移高斯采样器"""
    def __init__(self, code):
        """
        参数:
        code: 编码对象，需要提供码字和最近邻信息
        """
        super().__init__()
        self.code = code
        
    def sample(self, noise_std, tx_bin, batch_size):
        """生成向固定最近邻码字平移的高斯噪声及其PDF"""
        batch_size = int(batch_size)
        
        neighbor_indices = self.code.get_nearest_neighbors(tx_bin)
        _ = neighbor_indices[0]  # 选择第一个最近邻（但不使用）
        
        # 计算均值偏移量（固定为0，即不偏移）
        mean_shift = 0
        
        # 生成噪声样本
        noise = self.rng.normal(mean_shift, noise_std, (batch_size, self.code.dim))
        
        # 计算PDF值
        log_pdfs = -np.sum((noise - mean_shift)**2, axis=1)/(2*noise_std**2) - self.code.dim/2 * np.log(2*np.pi*noise_std**2)
        
        return noise, log_pdfs
    
import time
if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    from test_hamming import *
    from chase import ChaseDecoder
    code = HammingCode(7)
    # decoder = SGRANDDecoder(code)
    # decoder = 'binary'
    # decoder = ORBGRANDDecoder(code)
    decoder = ChaseDecoder(code,t=3)
    code.set_decoder(decoder)
    # sampler = SymShiftSampler(code, pdf_type='direct', decoder_type='ORBGRAND')
    # noise, log_pdf = sampler.sample(noise_std=0.5, batch_size=10000, threshold=0.001, tx_bin=np.zeros(code.bin_dim))
    SNR = 13
    sigma = np.sqrt(1 / SNR)
    sampler_bessel = BesselSampler(code)
    sampler_true = CutoffSampler(code,gaussian_ratio=0.85, dp_find=True)
    # sampler_false = CutoffSampler(code,gaussian_ratio=0.85, dp_find=False)
    sampler_false = SymShiftSampler(code, pdf_type='direct', decoder_type='SGRAND')

    result_0 = code.simulate(sigma, num_samples=1e6, sampler=sampler_bessel, batch_size=1e5)[0] - np.log(10)
    start_time = time.time()
    result_true = code.simulate(sigma, num_samples=1e5, sampler=sampler_true, batch_size=1e4)[0]
    true_time = time.time() - start_time
    start_time = time.time()
    result_false = code.simulate(sigma, num_samples=1e5, sampler=sampler_false, batch_size=1e4)[0]
    false_time = time.time() - start_time
    print(f"Time taken true vs false: {true_time:.3f} vs {false_time:.3f} seconds")
    print(f"Result true vs false:{result_0:.3f} vs {result_true:.3f} vs {result_false:.3f}")
    # run_simulation_test(num_samples=1e6, hamming_r=6, batch_size=1e4, sigma_values=[0.3, 0.2, 0.1])
    # run_code_length_test(decoder='ORBGRAND', samplers=['naive', 'bessel', 'sym_direct'], num_samples=1e5, sigma_values=[0.5, 0.3], scale_factor=1, r_values=[4, 5], batch_size=1e4, naive_extra=1)