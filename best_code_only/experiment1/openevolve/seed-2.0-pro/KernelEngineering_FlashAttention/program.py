# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t

# Force exclusive use of fastest FlashAttention backend, eliminate runtime backend selection overhead
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)


def custom_kernel(data: input_t) -> output_t:
    """Naive attention implementation. ALLOWED TO MODIFY."""
    config, Q, K, V = data

    # Direct call to pre-configured fast FlashAttention backend
    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=config.causal,
        scale=config.scale
    )
# EVOLVE-BLOCK-END
