# EVOLVE-BLOCK-START
import math
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t

# Enable the FlashAttention kernel and turn off slower fallbacks.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's native FlashAttention."""
    config, Q, K, V = data

    # Direct call to the highly‑optimized CUDA kernel with explicit scale.
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