# EVOLVE-BLOCK-START
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .task import input_t, output_t

class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin


class KVCache(nn.Module):
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
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

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

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # Down-projection matrices
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False, dtype=torch.bfloat16)

        # Up-projection and rope projection matrices
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)

        # RoPE on half embeddings
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

        # Output projection
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6
   
    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        # seq_len = 1 always here
        batch_size, seq_len, model_dim = x.size()

        ################################################################################
        #                 Step 1: Handle down-projection + KV cache                    #
        ################################################################################
        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        ################################################################################
        #                  Step 2: Up-project and prepare NoPE + RoPE                  #
        ################################################################################

        # Handle queries Q first
        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Handle keys and values K/V. V does not need RoPE
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

        ################################################################################
        #                    Step 3: Handle RoPE Stream                                #
        ################################################################################

        # Compute RoPE for queries and combine with no-RoPE part
        q_rope = q_rope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q = torch.concat([q_nope, q_rope], dim=-1)


        # Compute RoPE for keys and combine with no-RoPE part
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1,self.n_heads,-1,-1)
        k_nope = k_nope.permute(0, 2, 1, 3) # bs x kv_len x n_heads x rope_head_dim
        k = torch.concat([k_nope, k_rope], dim=-1)
                
        ################################################################################
        #                        Compute Multi-head Attention                          #
        ################################################################################
        v = v.permute(0, 2, 1, 3) # bs x n_heads x kv_len x v_head_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)

        return y, kv_cache.get_data()

def _rope_apply(x: torch.Tensor, start_pos: int, theta_base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to tensor x of shape (..., seq_len, d). Matches reference RoPE exactly."""
    seq_len = x.size(-2)
    d = x.size(-1)
    half_d = d // 2
    theta = theta_base ** (-torch.arange(0, half_d, dtype=torch.bfloat16, device=x.device) / half_d)
    seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
    idx_theta = torch.einsum('s,d->sd', seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
    cos_a = idx_theta2.cos().to(torch.bfloat16)
    sin_a = idx_theta2.sin().to(torch.bfloat16)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos_a + rotated * sin_a

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    batch_size = x.size(0)
    n_heads = config.n_heads
    nope_dim = config.qk_nope_head_dim
    rope_dim = config.qk_rope_head_dim
    v_dim = config.v_head_dim
    kv_lora_rank = config.kv_lora_rank

    # KV_proj_up_weight: [(nope_dim + v_dim) * n_heads, kv_lora_rank]
    # Reshape to [n_heads, nope_dim + v_dim, kv_lora_rank]
    W_uv = config.KV_proj_up_weight.view(n_heads, nope_dim + v_dim, kv_lora_rank)
    W_uk = W_uv[:, :nope_dim, :]  # [n_heads, nope_dim, kv_lora_rank]
    W_v = W_uv[:, nope_dim:, :]   # [n_heads, v_dim, kv_lora_rank]

    # Q path: down-project then up-project
    q_lora = F.linear(x, config.Q_proj_down_weight)  # (bs, 1, q_lora_rank)
    q_full = F.linear(q_lora, config.Q_proj_up_weight).view(batch_size, n_heads, nope_dim + rope_dim)
    q_nope, q_rope_part = q_full.split([nope_dim, rope_dim], dim=-1)
    # q_nope: [bs, n_heads, nope_dim]

    # Absorb W_uk into q_nope: q_nope_compressed = q_nope @ W_uk
    # q_nope: [bs, n_heads, nope_dim], W_uk: [n_heads, nope_dim, kv_lora_rank]
    # Result: [bs, n_heads, kv_lora_rank]
    q_nope_compressed = torch.einsum('bhn,hnd->bhd', q_nope, W_uk)

    # KV path: down-project and cache
    kv_down = F.linear(x, config.KV_proj_down_weight)  # (bs, 1, kv_lora_rank + rope_dim)
    kv_lora_all, kv_len = kv_cache(kv_down)
    query_pos = kv_len - 1

    # Split cached KV into nope-part and rope-part
    kv_nope_part, k_rope_part = kv_lora_all.split([kv_lora_rank, rope_dim], dim=-1)
    # kv_nope_part: [bs, kv_len, kv_lora_rank]

    # Compute nope attention scores in compressed space
    # q_nope_compressed: [bs, n_heads, kv_lora_rank], kv_nope_part: [bs, kv_len, kv_lora_rank]
    # scores_nope: [bs, n_heads, kv_len]
    scores_nope = torch.einsum('bhd,bsd->bhs', q_nope_compressed, kv_nope_part)

    # Apply RoPE for rope scores
    q_rope_part = q_rope_part.unsqueeze(2)  # [bs, n_heads, 1, rope_dim]
    q_rope_part = _rope_apply(q_rope_part, start_pos=query_pos)

    k_rope_part = k_rope_part.unsqueeze(1)  # [bs, 1, kv_len, rope_dim]
    k_rope_part = _rope_apply(k_rope_part, start_pos=0)

    # scores_rope: [bs, n_heads, 1, rope_dim] x [bs, 1, kv_len, rope_dim] -> [bs, n_heads, kv_len]
    scores_rope = torch.einsum('bhsd,bcsd->bhs', q_rope_part, k_rope_part)

    # Combined scores
    scale = 1.0 / math.sqrt(rope_dim + nope_dim)
    scores = (scores_nope + scores_rope) * scale  # [bs, n_heads, kv_len]

    # Softmax
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16)  # [bs, n_heads, kv_len]

    # Compute value output in compressed space
    # attn: [bs, n_heads, kv_len], kv_nope_part: [bs, kv_len, kv_lora_rank]
    # attn_kv: [bs, n_heads, kv_lora_rank]
    attn_kv = torch.einsum('bhs,bsd->bhd', attn, kv_nope_part)

    # Up-project to get value output: attn_kv @ W_v^T
    # attn_kv: [bs, n_heads, kv_lora_rank], W_v: [n_heads, v_dim, kv_lora_rank]
    # y: [bs, n_heads, v_dim]
    y = torch.einsum('bhd,hnd->bhn', attn_kv, W_v)

    y = y.reshape(batch_size, 1, n_heads * v_dim)
    y = F.linear(y, config.wo_weight)

    return y, kv_cache.get_data()
# EVOLVE-BLOCK-END
