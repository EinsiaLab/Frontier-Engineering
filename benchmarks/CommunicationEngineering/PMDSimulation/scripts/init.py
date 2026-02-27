#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initial starter code for PMD simulation."""

from __future__ import annotations

import sys
from pathlib import Path
from numpy.random import Generator, Philox

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.CommunicationEngineering.PMDSimulation.runtime.sampler import NaiveSampler


class PMDSampler(NaiveSampler):
    """Initial starter: Naive Monte Carlo sampler."""
    
    def __init__(self, fiber_model=None, *, seed: int = 0):
        super().__init__(fiber_model, seed=seed)
        self.rng = Generator(Philox(seed))
    
    def simulate_variance_controlled(
        self,
        *,
        fiber_model,
        dgd_threshold: float = 30.0,
        target_std: float = 0.1,
        max_samples: int = 50000,
        batch_size: int = 5000,
        min_outages: int = 20,
    ):
        """Run variance-controlled simulation."""
        return fiber_model.simulate_variance_controlled(
            dgd_threshold=dgd_threshold,
            target_std=target_std,
            max_samples=max_samples,
            sampler=self,
            batch_size=batch_size,
            min_outages=min_outages,
        )


if __name__ == "__main__":
    from benchmarks.CommunicationEngineering.PMDSimulation.runtime.fiber_model import PMDFiberModel
    
    fiber = PMDFiberModel(length_km=100.0, pmd_coefficient=0.5, num_segments=100)
    sampler = PMDSampler(fiber, seed=0)
    result = sampler.simulate_variance_controlled(
        fiber_model=fiber,
        dgd_threshold=30.0,
    )
    print(result)

