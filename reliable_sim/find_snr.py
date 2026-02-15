
import sys
import os
import numpy as np
sys.path.append('/Users/haohan/workspace/Frontier-Engineering/reliable_sim')

from code_linear import HammingCode
from sampler import NaiveSampler

def find_target_snr():
    # Hamming code with r=7 (n=127, k=120)
    r = 7
    hamming = HammingCode(r=r)
    sampler = NaiveSampler(code=hamming)
    
    # Range of SNRs to test (linear scale)
    # SNR (dB) = 10 log10(SNR_linear)
    # We want BER ~ 1e-7. 
    # For Uncoded BPSK, BER = Q(sqrt(2*SNR)). 1e-7 corresponds to roughly SNR_dB ~ 11.3 dB -> SNR_linear ~ 13.5
    # Coding gain might change this.
    
    snr_values = [6, 7, 8, 9, 10, 11, 12] # Linear SNR values to try around the expected range
    
    print(f"Testing Hamming({hamming.n}, {hamming.k})")
    
    for snr in snr_values:
        sigma = 1.0 / np.sqrt(snr)
        
        # Run a quick simulation. 
        # Naive sampling is slow for low BER, so we might need a lot of samples or use the property of error correction to guess.
        # Alternatively, use BesselSampler for faster estimation if we trust it.
        # But let's try Naive first with limited samples to get a ballpark.
        
        # Actually, for 1e-7, NaiveSampler needs ~1e8 samples which is too slow for interactive.
        # I will use BesselSampler from reliable_sim to estimate it quickly.
        from sampler import BesselSampler
        fast_sampler = BesselSampler(code=hamming, scale_factor=1.0)
        
        try:
            err, weight, ratio = hamming.simulate(
                noise_std=sigma, 
                sampler=fast_sampler,
                batch_size=10000, 
                num_samples=100000 
            )
            print(f"SNR: {snr:.2f} (sigma={sigma:.4f}) -> Estimated BER: {ratio:.2e}")
        except Exception as e:
            print(f"SNR: {snr:.2f} -> Error: {e}")

if __name__ == "__main__":
    find_target_snr()
