# EVOLVE-BLOCK-START
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .task import input_t, output_t


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


# Global cache for precomputed absorbed weights
_absorbed_cache = {}


def _get_absorbed_weights(config):
    """Precompute and cache absorbed weights for MLA decode."""
    key = id(config.KV_proj_up_weight)
    if key in _absorbed_cache:
        return _absorbed_cache[key]

    n_heads = config.n_heads
    kv_lora_rank = config.kv_lora_rank
    nope_head_dim = config.qk_nope_head_dim
    rope_head_dim = config.qk_rope_head_dim
    v_head_dim = config.v_head_dim
    q_lora_rank = config.q_lora_rank

    # KV_proj_up_weight: [(nope+v)*n_heads, kv_lora_rank]
    KV_proj_up_w = config.KV_proj_up_weight
    KV_proj_up_reshaped = KV_proj_up_w.view(n_heads, nope_head_dim + v_head_dim, kv_lora_rank)
    W_k_nope = KV_proj_up_reshaped[:, :nope_head_dim, :]  # [n_heads, nope, kv_lora_rank]
    W_v = KV_proj_up_reshaped[:, nope_head_dim:, :]       # [n_heads, v, kv_lora_rank]

    # Precompute absorbed query projection for nope part:
    # Q_proj_up: [(nope+rope)*n_heads, q_lora_rank]
    Q_proj_up_w = config.Q_proj_up_weight
    Q_proj_up_reshaped = Q_proj_up_w.view(n_heads, nope_head_dim + rope_head_dim, q_lora_rank)
    W_q_nope = Q_proj_up_reshaped[:, :nope_head_dim, :]   # [n_heads, nope, q_lora_rank]
    W_q_rope = Q_proj_up_reshaped[:, nope_head_dim:, :]   # [n_heads, rope, q_lora_rank]

    # Absorbed: W_q_absorbed = W_k_nope^T @ W_q_nope -> [n_heads, kv_lora_rank, q_lora_rank]
    # q_absorbed = x @ Q_proj_down^T @ W_q_absorbed^T = q_lora @ W_q_absorbed^T
    # This gives [bs, n_heads, kv_lora_rank] directly
    W_q_absorbed = torch.bmm(W_k_nope.transpose(1, 2), W_q_nope)  # [n_heads, kv_lora_rank, q_lora_rank]

    # Precompute RoPE theta
    theta = (10000.0 ** (-torch.arange(0, rope_head_dim // 2, dtype=torch.float32, device=W_v.device) / (rope_head_dim // 2)))
    theta = theta.to(torch.bfloat16)

    _absorbed_cache[key] = (W_k_nope, W_v, W_q_absorbed, W_q_rope, theta)
    return _absorbed_cache[key]


def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data

    n_heads = config.n_heads
    kv_lora_rank = config.kv_lora_rank
    nope_head_dim = config.qk_nope_head_dim
    rope_head_dim = config.qk_rope_head_dim
    v_head_dim = config.v_head_dim

    Q_proj_down_w = config.Q_proj_down_weight    # [q_lora_rank, dim]
    KV_proj_down_w = config.KV_proj_down_weight  # [kv_lora_rank+rope, dim]
    wo_w = config.wo_weight                      # [dim, v*n_heads]

    W_k_nope, W_v, W_q_absorbed, W_q_rope, theta = _get_absorbed_weights(config)

    batch_size = x.size(0)

    ############################################################################
    # Step 1: Down-projections + KV cache update
    ############################################################################
    # Squeeze seq dim for decode (seq_len=1)
    x_sq = x.squeeze(1)  # [bs, dim]
    q_lora = F.linear(x_sq, Q_proj_down_w)      # [bs, q_lora_rank]
    kv_new = F.linear(x_sq, KV_proj_down_w)     # [bs, kv_lora_rank + rope_head_dim]

    # Update KV cache
    kv_lora_all, kv_len = kv_cache(kv_new.unsqueeze(1))  # [bs, kv_len, kv_lora_rank + rope_head_dim]
    query_pos = kv_len - 1

    ############################################################################
    # Step 2: Compute absorbed nope query: q_absorbed = q_lora @ W_q_absorbed^T
    # W_q_absorbed: [n_heads, kv_lora_rank, q_lora_rank]
    # q_lora: [bs, q_lora_rank]
    # Result: [bs, n_heads, kv_lora_rank]
    ############################################################################
    # Use einsum for this: q_absorbed[b,h,d] = sum_r q_lora[b,r] * W_q_absorbed[h,d,r]
    q_absorbed = torch.einsum('br,hdr->bhd', q_lora, W_q_absorbed)  # [bs, n_heads, kv_lora_rank]

    ############################################################################
    # Step 3: Compute rope query
    # W_q_rope: [n_heads, rope, q_lora_rank]
    ############################################################################
    q_rope_raw = torch.einsum('br,hdr->bhd', q_lora, W_q_rope)  # [bs, n_heads, rope]

    ############################################################################
    # Step 4: Split KV cache
    ############################################################################
    kv_latent = kv_lora_all[:, :, :kv_lora_rank]   # [bs, kv_len, kv_lora_rank]
    k_rope_all = kv_lora_all[:, :, kv_lora_rank:]  # [bs, kv_len, rope_head_dim]

    ############################################################################
    # Step 5: Nope attention scores via bmm
    # q_absorbed: [bs, n_heads, kv_lora_rank]
    # kv_latent: [bs, kv_len, kv_lora_rank]
    # scores_nope: [bs, n_heads, kv_len]
    ############################################################################
    scores_nope = torch.einsum('bhd,bsd->bhs', q_absorbed, kv_latent)

    ############################################################################
    # Step 6: RoPE attention scores
    ############################################################################
    half_rope = rope_head_dim // 2

    # Query RoPE - compute angles
    q_angles = float(query_pos) * theta  # [rope/2]
    q_cos = torch.cos(q_angles).to(torch.bfloat16)  # [rope/2]
    q_sin = torch.sin(q_angles).to(torch.bfloat16)  # [rope/2]

    # Apply RoPE to query: q_rope_raw [bs, n_heads, rope]
    q1 = q_rope_raw[:, :, :half_rope]
    q2 = q_rope_raw[:, :, half_rope:]
    q_rope_applied = torch.cat([
        q1 * q_cos - q2 * q_sin,
        q2 * q_cos + q1 * q_sin,
    ], dim=-1)  # [bs, n_heads, rope]

    # Key RoPE - compute angles for all positions
    k_pos_idx = torch.arange(0, kv_len, dtype=torch.bfloat16, device=x.device)
    k_angles = k_pos_idx.unsqueeze(-1) * theta.unsqueeze(0)  # [kv_len, rope/2]
    k_cos = torch.cos(k_angles).to(torch.bfloat16)  # [kv_len, rope/2]
    k_sin = torch.sin(k_angles).to(torch.bfloat16)  # [kv_len, rope/2]

    # Apply RoPE to keys: k_rope_all [bs, kv_len, rope]
    k1 = k_rope_all[:, :, :half_rope]
    k2 = k_rope_all[:, :, half_rope:]
    k_rope_applied = torch.cat([
        k1 * k_cos.unsqueeze(0) - k2 * k_sin.unsqueeze(0),
        k2 * k_cos.unsqueeze(0) + k1 * k_sin.unsqueeze(0),
    ], dim=-1)  # [bs, kv_len, rope]

    # Rope scores
    scores_rope = torch.einsum('bhr,bsr->bhs', q_rope_applied, k_rope_applied)

    ############################################################################
    # Step 7: Combine scores, softmax, compute output
    ############################################################################
    scale = 1.0 / math.sqrt(rope_head_dim + nope_head_dim)
    scores = (scores_nope + scores_rope) * scale
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16)  # [bs, n_heads, kv_len]

    # Absorbed V: context_latent = attn @ kv_latent, then y = context_latent @ W_v^T
    context_latent = torch.einsum('bhs,bsd->bhd', attn, kv_latent)  # [bs, n_heads, kv_lora_rank]

    # W_v: [n_heads, v, kv_lora_rank]
    y = torch.einsum('bhd,hvd->bhv', context_latent, W_v)  # [bs, n_heads, v_head_dim]
    y = y.reshape(batch_size, 1, n_heads * v_head_dim)

    ############################################################################
    # Step 8: Output projection
    ############################################################################
    output = F.linear(y, wo_w)  # [bs, 1, dim]

    return output, kv_cache.get_data()
# EVOLVE-BLOCK-END