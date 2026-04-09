# EVOLVE-BLOCK-START
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .task import input_t, output_t

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


def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache_obj = data
    
    batch_size = config.batch_size
    dim = config.dim
    n_heads = config.n_heads
    q_lora_rank = config.q_lora_rank
    kv_lora_rank = config.kv_lora_rank
    nope_head_dim = config.qk_nope_head_dim
    rope_head_dim = config.qk_rope_head_dim
    v_head_dim = config.v_head_dim
    
    Q_proj_down_w = config.Q_proj_down_weight
    Q_proj_up_w = config.Q_proj_up_weight
    KV_proj_down_w = config.KV_proj_down_weight
    KV_proj_up_w = config.KV_proj_up_weight
    wo_w = config.wo_weight
    
    device = x.device
    
    # x: [bs, 1, dim]
    x_sq = x.squeeze(1)  # [bs, dim]
    
    # Down projections - fused into one matmul by concatenating weight matrices
    # Q down: [q_lora_rank, dim], KV down: [kv_lora_rank + rope, dim]
    # Combined: [q_lora_rank + kv_lora_rank + rope, dim]
    combined_down_w = torch.cat([Q_proj_down_w, KV_proj_down_w], dim=0)
    combined_down = x_sq @ combined_down_w.t()  # [bs, q_lora_rank + kv_lora_rank + rope]
    q_lora = combined_down[:, :q_lora_rank]
    kv_down = combined_down[:, q_lora_rank:]
    
    # Update KV cache
    kv_cache_data = kv_cache_obj.data
    seq_len_before = kv_cache_obj.seq_len
    kv_cache_data[:, seq_len_before, :] = kv_down.to(kv_cache_data.dtype)
    kv_cache_obj.seq_len = seq_len_before + 1
    kv_len = seq_len_before + 1
    query_pos = kv_len - 1
    
    # Split KV_proj_up_w into nope and v parts to avoid materializing full kv_up
    # KV_proj_up_w: [(nope+v)*n_heads, kv_lora_rank]
    # Reshape to [n_heads, nope+v, kv_lora_rank]
    KV_proj_up_reshaped = KV_proj_up_w.view(n_heads, nope_head_dim + v_head_dim, kv_lora_rank)
    W_k_nope = KV_proj_up_reshaped[:, :nope_head_dim, :]  # [n_heads, nope, kv_lora_rank]
    W_v = KV_proj_up_reshaped[:, nope_head_dim:, :]  # [n_heads, v, kv_lora_rank]
    
    # Q up-projection
    q_up = q_lora @ Q_proj_up_w.t()  # [bs, (nope+rope)*n_heads]
    q_up = q_up.view(batch_size, n_heads, nope_head_dim + rope_head_dim)
    q_nope = q_up[:, :, :nope_head_dim]  # [bs, n_heads, nope]
    q_rope_raw = q_up[:, :, nope_head_dim:]  # [bs, n_heads, rope]
    
    # Absorbed attention: instead of projecting all KV up then doing attention,
    # compute q_nope @ W_k_nope to get q_absorbed: [bs, n_heads, kv_lora_rank]
    # Then attention score nope part = q_absorbed @ kv_nope_data^T
    # q_nope: [bs, n_heads, nope], W_k_nope: [n_heads, nope, kv_lora_rank]
    q_absorbed = torch.einsum('bhn,hnd->bhd', q_nope, W_k_nope)  # [bs, n_heads, kv_lora_rank]
    
    kv_lora_all = kv_cache_data[:, :kv_len, :]  # [bs, kv_len, kv_lora_rank + rope]
    kv_nope_data = kv_lora_all[:, :, :kv_lora_rank]  # [bs, kv_len, kv_lora_rank]
    k_rope_raw = kv_lora_all[:, :, kv_lora_rank:]  # [bs, kv_len, rope]
    
    # Nope attention scores: [bs, n_heads, kv_len]
    scores_nope = torch.einsum('bhd,bkd->bhk', q_absorbed, kv_nope_data)
    
    # RoPE
    half_rope = rope_head_dim // 2
    theta = 10000.0 ** (-torch.arange(0, half_rope, dtype=torch.float32, device=device) / half_rope)
    theta = theta.to(torch.bfloat16)
    
    # Query RoPE
    angles_q = float(query_pos) * theta  # [rope/2]
    cos_q = angles_q.cos()
    sin_q = angles_q.sin()
    q_r1, q_r2 = q_rope_raw[..., :half_rope], q_rope_raw[..., half_rope:]
    q_rope = torch.cat([q_r1 * cos_q - q_r2 * sin_q, q_r2 * cos_q + q_r1 * sin_q], dim=-1)
    
    # Key RoPE
    seq_idx = torch.arange(0, kv_len, device=device, dtype=torch.bfloat16)
    angles_k = seq_idx[:, None] * theta[None, :]  # [kv_len, rope/2]
    cos_k = angles_k.cos()
    sin_k = angles_k.sin()
    k_r1, k_r2 = k_rope_raw[..., :half_rope], k_rope_raw[..., half_rope:]
    k_rope = torch.cat([k_r1 * cos_k - k_r2 * sin_k, k_r2 * cos_k + k_r1 * sin_k], dim=-1)
    
    # Rope attention scores
    scores_rope = torch.einsum('bhr,bkr->bhk', q_rope, k_rope)
    
    scale = math.sqrt(rope_head_dim + nope_head_dim)
    scores = (scores_nope + scores_rope) / scale
    
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16)  # [bs, n_heads, kv_len]
    
    # Absorbed value attention: attn @ kv_nope_data -> [bs, n_heads, kv_lora_rank]
    # Then project through W_v
    attn_kv = torch.einsum('bhk,bkd->bhd', attn, kv_nope_data)  # [bs, n_heads, kv_lora_rank]
    
    # y = attn_kv @ W_v^T: [bs, n_heads, v_head_dim]
    y = torch.einsum('bhd,hvd->bhv', attn_kv, W_v)  # [bs, n_heads, v_head_dim]
    
    y = y.reshape(batch_size, n_heads * v_head_dim)
    
    output = y @ wo_w.t()  # [bs, dim]
    output = output.unsqueeze(1)  # [bs, 1, dim]
    
    return output, kv_cache_obj.get_data()
# EVOLVE-BLOCK-END
