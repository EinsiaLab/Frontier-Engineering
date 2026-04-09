# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Optimized kernel using PyTorch's fused scaled‑dot‑product attention."""
    config, Q, K, V = data

    # The fused kernel handles scaling, optional causal masking,
    # softmax, and the final matmul in a single fast GPU operation.
    # It works on CPUs as well (fall‑back is handled internally by PyTorch).
    output = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=config.causal,
        scale=config.scale,
    )
    return output
# EVOLVE-BLOCK-END
