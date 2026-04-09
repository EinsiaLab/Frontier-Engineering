# EVOLVE-BLOCK-START
import math
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's fused scaled_dot_product_attention."""
    config, Q, K, V = data

    return F.scaled_dot_product_attention(
        Q,
        K,
        V,
        scale=config.scale,
        is_causal=config.causal,
    )
# EVOLVE-BLOCK-END
