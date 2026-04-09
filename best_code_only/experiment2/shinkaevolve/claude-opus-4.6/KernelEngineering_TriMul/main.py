# EVOLVE-BLOCK-START
import torch
from torch import nn
import torch.nn.functional as F
from .task import input_t, output_t

# Cache for compiled function
_compiled_fn = None


def _trimul_core(
    x: torch.Tensor,
    mask_bf: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    fused_weight_bf: torch.Tensor,
    to_out_norm_weight: torch.Tensor,
    to_out_norm_bias: torch.Tensor,
    to_out_weight_bf: torch.Tensor,
    hidden_dim: int,
) -> torch.Tensor:
    batch_size, seq_len, _, dim = x.shape

    # LayerNorm in float32 for numerical stability
    x = F.layer_norm(x, (dim,), norm_weight, norm_bias)

    # Fused projection in bfloat16
    x_bf = x.to(torch.bfloat16)
    all_proj = F.linear(x_bf, fused_weight_bf)  # bfloat16

    left, right, left_gate, right_gate, out_gate = all_proj.split(hidden_dim, dim=-1)

    # Apply masks and gates entirely in bfloat16
    mask_expanded = mask_bf.unsqueeze(-1)
    left = left * mask_expanded * left_gate.sigmoid()
    right = right * mask_expanded * right_gate.sigmoid()
    out_gate_f32 = out_gate.float().sigmoid()

    # Triangle multiplication via bmm in bfloat16
    # out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
    left_p = left.permute(0, 3, 1, 2).contiguous()
    right_p = right.permute(0, 3, 2, 1).contiguous()

    # Reshape to [bs*hidden_dim, seq_len, seq_len]
    left_r = left_p.reshape(batch_size * hidden_dim, seq_len, seq_len)
    right_r = right_p.reshape(batch_size * hidden_dim, seq_len, seq_len)

    out = torch.bmm(left_r, right_r)

    # Reshape back: [bs, hidden_dim, seq_len, seq_len] -> [bs, seq_len, seq_len, hidden_dim]
    out = out.reshape(batch_size, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1)

    # Output LayerNorm needs float32 for precision
    out = out.float()
    out = F.layer_norm(out, (hidden_dim,), to_out_norm_weight, to_out_norm_bias)
    out = out * out_gate_f32

    # Output projection in bfloat16 for speed
    out = F.linear(out.to(torch.bfloat16), to_out_weight_bf).float()

    return out


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _compiled_fn
    input_tensor, mask, weights, config = data
    hidden_dim = config["hidden_dim"]

    norm_weight = weights['norm.weight'].float()
    norm_bias = weights['norm.bias'].float()

    # Pre-fuse and convert to bfloat16 once
    fused_weight_bf = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).to(torch.bfloat16).contiguous()

    to_out_norm_weight = weights['to_out_norm.weight'].float()
    to_out_norm_bias = weights['to_out_norm.bias'].float()
    to_out_weight_bf = weights['to_out.weight'].to(torch.bfloat16).contiguous()

    # Pre-convert mask to bfloat16
    mask_bf = mask.to(torch.bfloat16)

    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _trimul_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
        )

    output = _compiled_fn(
        input_tensor, mask_bf,
        norm_weight, norm_bias,
        fused_weight_bf,
        to_out_norm_weight, to_out_norm_bias,
        to_out_weight_bf,
        hidden_dim,
    )

    return output.to(torch.float32)
# EVOLVE-BLOCK-END