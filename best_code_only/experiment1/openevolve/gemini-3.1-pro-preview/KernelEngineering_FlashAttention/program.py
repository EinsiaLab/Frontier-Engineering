# EVOLVE-BLOCK-START
from torch.nn.functional import scaled_dot_product_attention as s

def custom_kernel(d):
    return s(d[1], d[2], d[3], is_causal=d[0].causal)
# EVOLVE-BLOCK-END
