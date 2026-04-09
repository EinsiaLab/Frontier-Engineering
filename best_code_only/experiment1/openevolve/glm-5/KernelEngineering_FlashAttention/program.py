# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import Config, input_t, output_t

# Precompute version check at module load time to avoid per-call overhead
_HAS_SCALE_PARAM = tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 1)


def custom_kernel(data: input_t) -> output_t:
    """Optimized attention using PyTorch's fused SDPA."""
    config, Q, K, V = data
    
    if _HAS_SCALE_PARAM:
        return F.scaled_dot_product_attention(
            Q, K, V, is_causal=config.causal, scale=config.scale
        )
    Q_scaled = Q * config.scale if config.scale != 1.0 else Q
    return F.scaled_dot_product_attention(Q_scaled, K, V, is_causal=config.causal)
# EVOLVE-BLOCK-END
