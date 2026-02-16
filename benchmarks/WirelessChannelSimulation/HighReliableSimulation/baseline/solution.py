#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from numpy.random import Generator, Philox

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))

from chase import ChaseDecoder
from code_linear import HammingCode
from sampler import BesselSampler


class MySampler(BesselSampler):
    """示例解：使用 Bessel 重要性采样。"""

    def __init__(self, code: HammingCode, *, seed: int = 0):
        super().__init__(code)
        self.rng = Generator(Philox(seed))

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
    code = HammingCode(r=7, decoder="binary")
    code.set_decoder(ChaseDecoder(code=code, t=3))
    return code


if __name__ == "__main__":
    code = build_code()
    sampler = MySampler(code, seed=0)
    print(sampler.simulate_variance_controlled(code=code))
