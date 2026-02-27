"""Base sampler class for LDPC error floor estimation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, Philox


class SamplerBase:
    """Base class for importance sampling in LDPC error floor estimation."""
    
    def __init__(self, code=None, *, seed: int = 0):
        """Initialize sampler.
        
        Args:
            code: LDPC code object
            seed: Random seed
        """
        self.rng = Generator(Philox(int(seed)))
        self.code = code
    
    def sample(self, noise_std, tx_bits, batch_size, **kwargs):
        """Sample noise vectors from biasing distribution.
        
        Args:
            noise_std: Standard deviation of true noise
            tx_bits: Transmitted bits (for reference)
            batch_size: Number of samples to generate
        
        Returns:
            tuple: (noise_vectors, log_pdf_values)
                - noise_vectors: shape (batch_size, n)
                - log_pdf_values: shape (batch_size,)
        """
        raise NotImplementedError
    
    def simulate_variance_controlled(
        self,
        *,
        code,
        sigma: float,
        target_std: float,
        max_samples: int,
        batch_size: int,
        fix_tx: bool = True,
        min_errors: int = 10,
    ):
        """Run variance-controlled importance sampling simulation.
        
        Args:
            code: LDPC code object
            sigma: Noise standard deviation
            target_std: Target standard deviation for estimate
            max_samples: Maximum number of samples
            batch_size: Samples per batch
            fix_tx: Whether to fix transmitted codeword
            min_errors: Minimum number of errors to observe
        
        Returns:
            tuple or dict with simulation results
        """
        raise NotImplementedError


class NaiveSampler(SamplerBase):
    """Naive Monte Carlo sampler (no biasing)."""
    
    def __init__(self, code, *, seed: int = 0):
        super().__init__(code, seed=seed)
        self.n = code.n if code is not None else 1008
    
    def sample(self, noise_std, tx_bits, batch_size, **kwargs):
        """Sample from true Gaussian distribution."""
        batch_size = int(batch_size)
        noise = self.rng.normal(0, noise_std, (batch_size, self.n))
        log_pdf = (
            -np.sum(noise**2, axis=1) / (2 * noise_std**2)
            - self.n / 2 * np.log(2 * np.pi * noise_std**2)
        )
        return noise, log_pdf
    
    def simulate_variance_controlled(
        self,
        *,
        code,
        sigma: float,
        target_std: float,
        max_samples: int,
        batch_size: int,
        fix_tx: bool = True,
        min_errors: int = 10,
    ):
        """Run naive Monte Carlo simulation."""
        return code.simulate_variance_controlled(
            noise_std=sigma,
            target_std=target_std,
            max_samples=max_samples,
            sampler=self,
            batch_size=batch_size,
            fix_tx=fix_tx,
            min_errors=min_errors,
        )

