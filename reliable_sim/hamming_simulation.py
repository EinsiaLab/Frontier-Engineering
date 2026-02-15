import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import time
from numpy.random import Generator, PCG64, Philox

def log_adder(a, b):
    return np.logaddexp(a, b)

class HammingCode:
    def __init__(self, n=7, k=4):
        self.n = n
        self.k = k
        self.G = self._generate_matrix()
        self.codewords = self._generate_codewords()
        self._nearest_neighbors = self._find_nearest_neighbors()
        self.rng = Generator(Philox())

    def _generate_matrix(self):
        # 标准(7,4)汉明码生成矩阵
        return np.array([
            [1,0,0,0,1,1,0],
            [0,1,0,0,1,0,1],
            [0,0,1,0,0,1,1],
            [0,0,0,1,1,1,1]
        ], dtype=np.uint8)

    def _generate_codewords(self):
        # 生成所有2^k个合法码字
        info_bits = np.array([list(np.binary_repr(i, width=self.k)) 
                            for i in range(2**self.k)], dtype=np.uint8)
        codewords_binary = (info_bits @ self.G) % 2
        return 2 * codewords_binary.astype(np.float32) - 1  # 转换为±1形式

    def simulate(self, noise_std, sampler=None, batch_size=1e7, num_samples=1e8, scale_factor=1.0):
        batch_size, num_samples = int(batch_size), int(num_samples)
        rounds = int(num_samples/batch_size/len(self.codewords))+1
        batch_size_use = int(num_samples/rounds/len(self.codewords))

        errors = -np.inf
        weights = -np.inf
        for i in range(rounds):
            for j, codeword in enumerate(self.codewords):
                if sampler == None:
                    error, weight = self._simulate_batch(noise_std, batch_size=batch_size_use, tx_idx=j)
                else:
                    error, weight = self._simulate_importance_batch(sampler, noise_std, tx_idx=j, scale_factor=scale_factor, batch_size=batch_size_use)
                errors = log_adder(errors, error)
                weights = log_adder(weights, weight)
                print(f'Sample {(i*len(self.codewords)+j)*batch_size_use}/{num_samples}, log_errors: {errors:.2f}/{weights:.2f}')
        return errors, weights

    def _simulate_batch(self, noise_std, batch_size=1e7, tx_idx=None):
        if tx_idx is None:
            tx_idx = np.random.randint(0, len(self.codewords), batch_size)
            tx_signals = self.codewords[tx_idx]
        else:
            # tx_signals = np.tile(self.codewords[tx_idx], (batch_size, 1))
            tx_signals = self.codewords[tx_idx]

        # 添加高斯噪声
        # noise = np.random.normal(0, noise_std, (batch_size, self.n))
        noise = self.rng.normal(0, noise_std, (batch_size, self.n))
        rx_signals = tx_signals + noise

        # 最近邻解码
        distances = cdist(rx_signals, self.codewords, 'euclidean')
        rx_indices = np.argmin(distances, axis=1)

        # 计算误码率
        bit_error = np.sum(tx_idx != rx_indices)
        # print('Bit error:', bit_error, 'Batch size:', batch_size)
        return np.log(bit_error), np.log(batch_size)

    def _find_nearest_neighbors(self):
        # 找到每个码字的最近邻（汉明距离3）
        neighbors = []
        for i, cw in enumerate(self.codewords):
            distances = np.sum(self.codewords != cw, axis=1)
            neighbors.append(np.where(distances == 3)[0].tolist())
        return neighbors

    def shift_sampler(self, noise_std, tx_indice, batch_size):
        # 平移采样器实现
        if tx_indice is None:
            raise ValueError("必须指定发送码字索引")

        neighbor_indices = self._nearest_neighbors[tx_indice]
        chosen_neighbor = np.random.choice(neighbor_indices, size=batch_size)
        
        # 计算均值偏移量（合法码字差异）
        mean_shift = (self.codewords[chosen_neighbor] - self.codewords[tx_indice]) / 2.0
        
        # 生成噪声样本
        # noise = np.random.normal(mean_shift, noise_std, (batch_size, self.n))
        noise = self.rng.normal(mean_shift, noise_std, (batch_size, self.n))
        
        # 计算混合分布的PDF值（平均7个高斯分布的密度）
        log_pdfs = -np.inf
        for neighbor in neighbor_indices:
            shift = (self.codewords[neighbor] - self.codewords[tx_indice]) / 2.0
            # print('Neighbor ', neighbor,' shift:', shift)
            log_prob = -np.sum((noise - shift)**2, axis=1)/(2*noise_std**2) - self.n/2 * np.log(2*np.pi*noise_std**2) - np.log(len(neighbor_indices))
            log_pdfs = log_adder(log_pdfs, log_prob)
        
        return noise, log_pdfs

    def shift_sampler_fixed(self, noise_std, tx_indice, batch_size):
        # 平移采样器实现
        if tx_indice is None:
            raise ValueError("必须指定发送码字索引")

        neighbor_indices = self._nearest_neighbors[tx_indice]
        chosen_neighbor = neighbor_indices[0]
        
        # 计算均值偏移量（合法码字差异）
        # mean_shift = (self.codewords[chosen_neighbor] - self.codewords[tx_indice]) / 2.0
        mean_shift = 0
        
        # 生成噪声样本
        # noise = np.random.normal(mean_shift, noise_std, (batch_size, self.n))
        noise = self.rng.normal(mean_shift, noise_std, (batch_size, self.n))
        
        # 计算混合分布的PDF值（平均7个高斯分布的密度）
        log_pdfs = -np.sum((noise - mean_shift)**2, axis=1)/(2*noise_std**2) - self.n/2 * np.log(2*np.pi*noise_std**2)
        
        return noise, log_pdfs

    def _simulate_importance_batch(self, sampler, noise_std, tx_idx=0, batch_size=1e6, scale_factor=1.0):
        batch_size = int(batch_size)
        # total_weighted_errors = 0.0
        # total_weight = 0.0

        # 使用采样器获取噪声样本和PDF值
        noise_samples, pdf_values = sampler(noise_std*np.sqrt(scale_factor), tx_idx, batch_size)
        log_pdf_original = -(np.sum(noise_samples**2, axis=1))/(2*noise_std**2) - self.n/2 * np.log(2*np.pi*noise_std**2)
        
        # 生成接收信号
        rx_signals = self.codewords[tx_idx] + noise_samples
        
        # 解码
        distances = cdist(rx_signals, self.codewords, 'euclidean')
        rx_indices = np.argmin(distances, axis=1)
        
        # 计算错误权重
        errors = np.array(rx_indices != tx_idx, dtype=bool)
        log_weights = log_pdf_original - pdf_values
        log_weights_errors = log_weights[errors==True]
        
        total_weight = np.log(batch_size) #logsumexp(log_weights)
        if len(log_weights_errors) == 0:
            total_weighted_errors = -np.inf
        else:
            total_weighted_errors = logsumexp(log_weights_errors)
        # print('num of errors:', np.sum(errors), len(errors))
        # print('total weighted errors:', total_weighted_errors, total_weight)

        return total_weighted_errors, total_weight

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # print(np.logaddexp(1,1),np.log(2))

    hamming = HammingCode()
    print(hamming._nearest_neighbors)
    sigma_values = np.exp(np.linspace(0, np.log(1e-1),7))
    scale_factors = [1,10]
    batch_size = 1e7
    num_samples = 1e7

    err_rate_naive = []
    err_rate_importance = []

    for sigma in sigma_values:
        start_time = time.time()
        err_naive, weight_naive = hamming.simulate(sigma, batch_size=batch_size, num_samples=num_samples*10)
        print(f'Naive Sampling Time: {time.time()-start_time:.3f} s')

        err_rate_step = []
        for scale_factor in scale_factors:
            start_time = time.time()
            err_importance, weight_importance = hamming.simulate(sigma, sampler=hamming.shift_sampler, batch_size=batch_size, num_samples=num_samples, scale_factor=scale_factor)
            print(f'Scale Factor: {scale_factor}')
            print(f'Importance Sampling Time: {time.time()-start_time:.3f} s')
            err_rate_step.append(err_importance - weight_importance)

        err_rate_importance.append(err_rate_step)
        err_rate_naive.append(err_naive - weight_naive)
        print(f'SNR: {np.log10(1/sigma**2):.2f} dB, log BER: {(err_naive - weight_naive):.4f}')

    err_rate_importance = np.array(err_rate_importance).T
    print(err_rate_naive)
    print(err_rate_importance)
    for i in range(len(scale_factors)):
        plt.plot(sigma_values, err_rate_importance[i] / np.log(10), '--', label=f'IS, n={scale_factors[i]}')
    plt.plot(sigma_values, err_rate_naive / np.log(10), '-o', label='Naive')
    plt.xlabel('sigma')
    plt.ylabel('Loged Bit Error Rate')
    plt.xscale('log')
    # plt.yscale('log')
    plt.title('Hamming Code (7,4) Performance')
    plt.grid(True)
    plt.legend()
    # plt.savefig('ber_curve.png')
    plt.show()


    # # 新增实验配置
    # fixed_sigma = 0.5
    # num_samples_list = [int(1e4), int(1e5), int(1e6), int(1e7), int(1e8), int(1e9)]
    # err_rate_naive_conv = []
    # err_rate_importance_conv = []
    
    # for num_samples in num_samples_list:
    #     # 普通采样
    #     err_naive, weight_naive = hamming.simulate(fixed_sigma, batch_size=min(num_samples,batch_size), num_samples=num_samples)
    #     # 重要性采样（固定scale factor=7）
    #     err_importance, weight_importance = hamming.simulate(fixed_sigma, sampler=hamming.shift_sampler, 
    #             batch_size=min(num_samples,batch_size), num_samples=num_samples, scale_factor=7)
        
    #     err_rate_naive_conv.append(err_naive - weight_naive)
    #     err_rate_importance_conv.append(err_importance - weight_importance)

    # # 新增绘图逻辑
    # plt.figure(figsize=(10,6))
    # plt.plot(num_samples_list, np.array(err_rate_naive_conv)/np.log(10), 's-', label='Naive Sampling')
    # plt.plot(num_samples_list, np.array(err_rate_importance_conv)/np.log(10), 'o-', label='Importance Sampling')
    # plt.xscale('log')
    # plt.xlabel('Number of Samples')
    # plt.ylabel('Log BER')
    # plt.title(f'Convergence Comparison (sigma={fixed_sigma})')
    # plt.grid(True)
    # plt.legend()
    # plt.show()