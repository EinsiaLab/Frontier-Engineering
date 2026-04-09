# EVOLVE-BLOCK-START
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .task import input_t, output_t

_ROPE_CACHE = {}

def apply_rope(t: torch.Tensor, pos: int) -> torch.Tensor:
    dim = t.size(-1)
    seq_len = t.size(-2)
    device = t.device
    dtype = t.dtype
    
    key = (dim, device, dtype)
    if key not in _ROPE_CACHE:
        max_len = 16384
        # Match the reference implementation's CPU bfloat16 initialization precision
        theta = (10000 ** (-torch.arange(0, dim // 2, dtype=torch.bfloat16) / (dim // 2))).to(device)
        seq_idx = torch.arange(0, max_len, device=device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(dtype)
        sin = idx_theta2.sin().to(dtype)
        _ROPE_CACHE[key] = (cos, sin)
        
    cos, sin = _ROPE_CACHE[key]
    cos = cos[pos:pos+seq_len]
    sin = sin[pos:pos+seq_len]
    
    t1, t2 = t.chunk(2, dim=-1)
    t_out = t * cos
    half = dim // 2
    t_out[..., :half].addcmul_(t2, sin[..., :half], value=-1.0)
    t_out[..., half:].addcmul_(t1, sin[..., half:])
    return t_out


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

_WEIGHT_CACHE = {"wid": None, "tensors": None}

def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    batch_size, seq_len, _ = x.shape

    wid = id(config.KV_proj_up_weight)
    if _WEIGHT_CACHE.get("wid") != wid:
        weight_up = config.KV_proj_up_weight.view(config.n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
        W_UK = weight_up[:, :config.qk_nope_head_dim, :].contiguous()
        W_UV_t = weight_up[:, config.qk_nope_head_dim:, :].transpose(1, 2).contiguous()
        W_DOWN = torch.cat([config.Q_proj_down_weight, config.KV_proj_down_weight], dim=0)
        _WEIGHT_CACHE["wid"] = wid
        _WEIGHT_CACHE["tensors"] = (W_UK, W_UV_t, W_DOWN)
    
    W_UK, W_UV_t, W_DOWN = _WEIGHT_CACHE["tensors"]

    q_kv_lora = F.linear(x.squeeze(1), W_DOWN)
    q_lora = q_kv_lora[:, :config.q_lora_rank]
    kv_lora = q_kv_lora[:, config.q_lora_rank:].unsqueeze(1)
    
    kv_lora, kv_len = kv_cache(kv_lora)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora, config.Q_proj_up_weight).view(
        batch_size, config.n_heads, config.qk_nope_head_dim + config.qk_rope_head_dim)
    
    q_nope, q_rope = torch.split(q_up, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)

    q_rope = apply_rope(q_rope.unsqueeze(2), query_pos).squeeze(2)

    kv_nope, k_rope = torch.split(kv_lora, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)
    k_rope = apply_rope(k_rope, 0)

    q_nope_proj = torch.bmm(q_nope.transpose(0, 1), W_UK).transpose(0, 1)

    scores = torch.bmm(q_rope, k_rope.transpose(1, 2))
    inv_norm = 1.0 / math.sqrt(config.qk_rope_head_dim + config.qk_nope_head_dim)
    scores.baddbmm_(q_nope_proj, kv_nope.transpose(1, 2), beta=inv_norm, alpha=inv_norm)
    
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16)

    attn_kv = torch.bmm(attn, kv_nope)

    y = torch.bmm(attn_kv.transpose(0, 1), W_UV_t).transpose(0, 1)
    y = y.reshape(batch_size, seq_len, config.n_heads * config.v_head_dim)

    output = F.linear(y, config.wo_weight)

    return output, kv_cache.get_data()
# EVOLVE-BLOCK-END
