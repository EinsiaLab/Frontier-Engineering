# EVOLVE-BLOCK-START
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from .task import Config, input_t, output_t


@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len_q, seq_len_kv, scale,
    n_heads,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads

    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        end_n = tl.minimum(seq_len_kv, (pid_m + 1) * BLOCK_M + (seq_len_kv - seq_len_q))
    else:
        end_n = seq_len_kv

    num_steps = tl.cdiv(end_n, BLOCK_N)

    for step in range(0, num_steps):
        offs_n = step * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] + (seq_len_kv - seq_len_q) >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        qk = tl.where(offs_n[None, :] < seq_len_kv, qk, float('-inf'))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)

        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (offs_n[:, None] < seq_len_kv) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        p = tl.exp(qk - m_new[:, None])
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=o_mask)


def custom_kernel(data: input_t) -> output_t:
    config, Q, K, V = data

    B, H, S_q, D = Q.shape
    _, _, S_kv, _ = K.shape

    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=config.causal,
        scale=config.scale,
    )
    return output
# EVOLVE-BLOCK-END
