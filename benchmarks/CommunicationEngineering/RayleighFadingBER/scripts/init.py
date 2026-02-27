#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initial starter code for Rayleigh Fading BER analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from numpy.random import Generator, Philox

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.sampler import NaiveSampler


class DeepFadeSampler(NaiveSampler):
    """Initial starter: Naive Monte Carlo sampler."""
    
    def __init__(self, channel_model=None, *, seed: int = 0):
        super().__init__(channel_model, seed=seed)
        self.rng = Generator(Philox(seed))
    
    def simulate_variance_controlled(
        self,
        *,
        channel_model,
        diversity_type: str = "MRC",
        modulation: str = "BPSK",
        snr_db: float = 10.0,
        target_std: float = 0.1,
        max_samples: int = 50000,
        batch_size: int = 5000,
        min_errors: int = 20,
    ):
        """Run variance-controlled simulation."""
        return channel_model.simulate_variance_controlled(
            diversity_type=diversity_type,
            modulation=modulation,
            snr_db=snr_db,
            target_std=target_std,
            max_samples=max_samples,
            sampler=self,
            batch_size=batch_size,
            min_errors=min_errors,
        )


if __name__ == "__main__":
    from benchmarks.CommunicationEngineering.RayleighFadingBER.runtime.channel_model import RayleighFadingChannel
    
    channel = RayleighFadingChannel(num_branches=4, sigma_h=1.0)
    sampler = DeepFadeSampler(channel, seed=0)
    result = sampler.simulate_variance_controlled(
        channel_model=channel,
        diversity_type="MRC",
        modulation="BPSK",
        snr_db=10.0,
    )
    print(result)

