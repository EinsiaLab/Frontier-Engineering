# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t

# Cache the function lookup once to minimize Python-side overhead on each call.
_SDPA = F.scaled_dot_product_attention


def custom_kernel(data: input_t) -> output_t:
    config, Q, K, V = data
    return _SDPA(Q, K, V, is_causal=config.causal, scale=config.scale)
# EVOLVE-BLOCK-END
