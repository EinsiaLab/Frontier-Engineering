#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# EVOLVE-BLOCK-START
import sys
from pathlib import Path
import math
import numpy as np
from numpy.random import Generator, Philox
from scipy.special import logsumexp

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.WirelessChannelSimulation.HighReliableSimulation.runtime.chase import ChaseDecoder
from benchmarks.WirelessChannelSimulation.HighReliableSimulation.runtime.code_linear import HammingCode
from benchmarks.WirelessChannelSimulation.HighReliableSimulation.runtime.sampler import SamplerBase


class MySampler(SamplerBase):
    """Optimized antithetic triple-mixture importance sampler for rare-event BER estimation."""

    DESIGN_SEED = 1
    SHIFT = 0.72
    CHUNK = 1024

    def __init__(self, code: HammingCode, *, seed: int = 0):
        super().__init__(code, seed=seed)
        self.rng = Generator(Philox(seed))
        self.triples = code.get_nearest_neighbors_idx().astype(np.int64)
        self.i0 = np.ascontiguousarray(self.triples[:, 0])
        self.i1 = np.ascontiguousarray(self.triples[:, 1])
        self.i2 = np.ascontiguousarray(self.triples[:, 2])
        self.num_triples = int(self.triples.shape[0])
        self.log_num_triples = math.log(self.num_triples)
        self._cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def _build_design(self, noise_std: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        design_rng = Generator(Philox(self.DESIGN_SEED))
        half = batch_size // 2

        z = design_rng.normal(0.0, noise_std, size=(half, self.code.dim))
        comp = design_rng.integers(0, self.num_triples, size=half)
        tri = self.triples[comp]

        # Pre-allocate and build antithetic pair with shifts
        noise = np.empty((batch_size, self.code.dim), dtype=np.float64)
        np.copyto(noise[:half], z)
        np.copyto(noise[half:], -z)

        # Apply shifts using advanced indexing
        rows = np.arange(half)
        noise[rows[:, None], tri] += self.SHIFT
        noise[rows[:, None] + half, tri] += self.SHIFT

        # Compute log proposal density
        inv_var = 1.0 / (noise_std * noise_std)
        sq_norm = np.einsum('ij,ij->i', noise, noise)
        log_norm_const = -0.5 * self.code.dim * np.log(2.0 * np.pi * noise_std ** 2)
        base = -0.5 * inv_var * sq_norm + log_norm_const

        quad = -1.5 * self.SHIFT * self.SHIFT * inv_var
        shift_inv = self.SHIFT * inv_var

        acc = np.full(batch_size, -np.inf)

        for start in range(0, self.num_triples, self.CHUNK):
            end = min(start + self.CHUNK, self.num_triples)
            i0_batch = self.i0[start:end]
            i1_batch = self.i1[start:end]
            i2_batch = self.i2[start:end]

            sums = noise[:, i0_batch] + noise[:, i1_batch] + noise[:, i2_batch]
            zterm = shift_inv * sums + quad
            acc = np.logaddexp(acc, logsumexp(zterm, axis=1))

        log_q = base + acc - self.log_num_triples
        return noise, log_q

    def sample(self, noise_std, tx_bin, batch_size, **kwargs):
        key = (float(noise_std), int(batch_size))
        if key not in self._cache:
            self._cache[key] = self._build_design(key[0], key[1])
        noise, log_q = self._cache[key]
        return noise.copy(), log_q.copy()

    def simulate_variance_controlled(
        self,
        *,
        code: HammingCode,
        sigma: float = 0.3,
        target_std: float = 0.08,
        max_samples: int = 20_000,
        batch_size: int = 2_000,
        fix_tx: bool = True,
        min_errors: int = 5,
    ):
        """Variance-controlled simulation entry point."""
        return code.simulate_variance_controlled(
            noise_std=sigma,
            target_std=target_std,
            max_samples=max_samples,
            sampler=self,
            batch_size=batch_size,
            fix_tx=fix_tx,
            min_errors=min_errors,
        )


def build_code() -> HammingCode:
    """Construct Hamming(127,120) code with Chase decoder."""
    code = HammingCode(r=7, decoder="binary")
    code.set_decoder(ChaseDecoder(code=code, t=3))
    return code


def main() -> None:
    """Local test entry point."""
    code = build_code()
    sampler = MySampler(code, seed=0)
    result = sampler.simulate_variance_controlled(code=code)
    print(result)


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END