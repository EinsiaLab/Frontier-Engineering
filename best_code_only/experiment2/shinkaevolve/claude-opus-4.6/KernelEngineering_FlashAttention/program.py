# EVOLVE-BLOCK-START
import math
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's SDPA with FlashAttention backend."""
    config, Q, K, V = data

    # Use PyTorch's scaled_dot_product_attention which dispatches to
    # FlashAttention2 or memory-efficient attention on CUDA
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=config.causal,
        scale=config.scale,
    )
    return output
# EVOLVE-BLOCK-END