# EVOLVE-BLOCK-START
import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa
from .task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    config, Q, K, V = data
    return _sdpa(Q, K, V, dropout_p=0.0,
                 is_causal=config.causal, scale=config.scale)
# EVOLVE-BLOCK-END
