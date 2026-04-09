# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's built-in scaled_dot_product_attention."""
    config, Q, K, V = data
    # Make tensors contiguous for optimal memory access
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Use PyTorch's optimized attention function
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=config.causal,
        scale=config.scale
    )
    return output
# EVOLVE-BLOCK-END
