# EVOLVE-BLOCK-START
import torch
from dataclasses import dataclass
from .task import input_t, output_t

class RoPE:
    def __init__(self, d_model: int):
        self.theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))

    def __call__(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        t = self.theta.to(x.device)
        p = torch.arange(start_pos, start_pos + x.size(-2), device=x.device)
        a = torch.outer(p, t)
        c = a.cos().to(torch.bfloat16)
        s = a.sin().to(torch.bfloat16)
        x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1)


class KVCache(torch.nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda'))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()
    
    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        n = c_kv.size(1)
        assert self.seq_len + n <= self.data.size(1), "KV Cache Exceeded"
        self.data[:, self.seq_len:self.seq_len + n, :] = c_kv
        self.seq_len += n
        return self.data[:, :self.seq_len], self.seq_len
    
@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class MLA:
    def __init__(self, config: Config):
        self.n_heads = config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scale = (self.rope_head_dim + self.nope_head_dim) ** -0.5
        self.wqd = config.Q_proj_down_weight
        self.wqu = config.Q_proj_up_weight
        self.wkd = config.KV_proj_down_weight
        self.wku = config.KV_proj_up_weight
        self.wo = config.wo_weight
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

    def __call__(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        b = x.size(0)
        ql = x @ self.wqd.t()
        kl, n = kv_cache(x @ self.wkd.t())
        q = (ql @ self.wqu.t()).view(b, 1, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        qn, qr = q.split((self.nope_head_dim, self.rope_head_dim), -1)
        km, kr = kl.split((self.kv_lora_rank, self.rope_head_dim), -1)
        kv = (km @ self.wku.t()).view(b, n, self.n_heads, self.nope_head_dim + self.v_head_dim)
        kn, v = kv.split((self.nope_head_dim, self.v_head_dim), -1)
        q = torch.cat((qn.transpose(1, 2), self.q_rope(qr.transpose(1, 2), n - 1)), -1)
        k = torch.cat((kn.permute(0, 2, 1, 3), self.k_rope(kr[:, None]).expand(-1, self.n_heads, -1, -1)), -1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v.permute(0, 2, 1, 3), scale=self.scale
        ).reshape(b, 1, -1)
        return y @ self.wo.t(), kv_cache.get_data()

def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    with torch.inference_mode():
        return MLA(config)(x, kv_cache)
# EVOLVE-BLOCK-END
