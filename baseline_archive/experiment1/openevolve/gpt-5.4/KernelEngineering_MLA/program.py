# EVOLVE-BLOCK-START
import math

import torch
import torch.nn.functional as F

from .task import input_t, output_t


_THETA_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}
_ROPE_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_theta(device: torch.device, rope_dim: int) -> torch.Tensor:
    key = (device, rope_dim)
    cached = _THETA_CACHE.get(key)
    if cached is None:
        cached = 10000 ** (
            -torch.arange(0, rope_dim // 2, dtype=torch.bfloat16, device=device)
            / (rope_dim // 2)
        )
        _THETA_CACHE[key] = cached
    return cached


def _apply_rope_base(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _rope_tables(
    device: torch.device, rope_dim: int, end_pos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, rope_dim)
    cached = _ROPE_CACHE.get(key)
    if cached is None or cached[0].size(0) < end_pos:
        size = max(end_pos, 128 if cached is None else cached[0].size(0) * 2)
        theta = _get_theta(device, rope_dim)
        angles = torch.outer(torch.arange(size, dtype=theta.dtype, device=device), theta)
        cached = (angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16))
        _ROPE_CACHE[key] = cached
    return cached


def _apply_rope(x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    end_pos = start_pos + x.size(-2)
    cos, sin = _rope_tables(x.device, x.size(-1), end_pos)
    return _apply_rope_base(x, cos[start_pos:end_pos], sin[start_pos:end_pos])


def _apply_rope_one(x: torch.Tensor, pos: int) -> torch.Tensor:
    cos, sin = _rope_tables(x.device, x.size(-1), pos + 1)
    return _apply_rope_base(x, cos[pos], sin[pos])


def _split_kv_up_weights(config) -> tuple[torch.Tensor, torch.Tensor]:
    cached = getattr(config, "_frontier_kv_up_split", None)
    if cached is not None:
        return cached

    kv_up = config.KV_proj_up_weight.view(
        config.n_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    cached = (
        kv_up[:, : config.qk_nope_head_dim, :].contiguous(),
        kv_up[:, config.qk_nope_head_dim :, :].contiguous(),
    )
    setattr(config, "_frontier_kv_up_split", cached)
    return cached


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    batch_size = x.size(0)
    scale = getattr(config, "_frontier_scale", None)
    if scale is None:
        scale = 1.0 / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim)
        setattr(config, "_frontier_scale", scale)

    q_lora = F.linear(x, config.Q_proj_down_weight)
    kv_update = F.linear(x, config.KV_proj_down_weight)
    kv_lora, kv_len = kv_cache(kv_update)
    kv_len = int(kv_len)
    query_pos = kv_len - 1

    q_full = F.linear(q_lora, config.Q_proj_up_weight).view(
        batch_size,
        config.n_heads,
        config.qk_nope_head_dim + config.qk_rope_head_dim,
    )
    q_nope = q_full[..., : config.qk_nope_head_dim]
    q_rope = _apply_rope_one(q_full[..., config.qk_nope_head_dim :], query_pos)

    kv_nope, k_rope = torch.split(
        kv_lora,
        [config.kv_lora_rank, config.qk_rope_head_dim],
        dim=-1,
    )
    k_rope = _apply_rope(k_rope)
    kv_nope_t = kv_nope.transpose(1, 2)
    k_rope_t = k_rope.transpose(1, 2)

    w_k_nope, w_v = _split_kv_up_weights(config)
    head_outputs = torch.empty(
        (batch_size, config.n_heads, config.v_head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )

    head_chunk = 128 if kv_len <= 1024 else 64 if kv_len <= 4096 else 32
    for head_start in range(0, config.n_heads, head_chunk):
        head_stop = min(head_start + head_chunk, config.n_heads)

        q_nope_chunk = q_nope[:, head_start:head_stop]
        q_rope_chunk = q_rope[:, head_start:head_stop]
        w_k_chunk = w_k_nope[head_start:head_stop]
        w_v_chunk = w_v[head_start:head_stop]

        q_latent = torch.einsum("bhd,hdc->bhc", q_nope_chunk, w_k_chunk)
        scores = torch.matmul(q_latent, kv_nope_t)
        scores.add_(torch.matmul(q_rope_chunk, k_rope_t))
        attn = scores.mul_(scale).softmax(dim=-1).to(torch.bfloat16)

        context_latent = torch.matmul(attn, kv_nope)
        head_outputs[:, head_start:head_stop] = torch.einsum(
            "bhc,hdc->bhd", context_latent, w_v_chunk
        )

    output = F.linear(head_outputs.reshape(batch_size, 1, -1), config.wo_weight)
    return output, kv_cache.get_data()
# EVOLVE-BLOCK-END
