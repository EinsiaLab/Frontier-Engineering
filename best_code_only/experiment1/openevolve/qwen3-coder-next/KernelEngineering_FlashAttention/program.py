# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's fused kernel with explicit scale."""
    config, Q, K, V = data

    # Ensure inputs are contiguous for optimal memory access
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()

    # Use PyTorch's optimized scaled dot-product attention
    # Explicitly pass scale parameter to match the task's scaling factor
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=config.causal,
        scale=config.scale
    )
    
    return output
# EVOLVE-BLOCK-END
