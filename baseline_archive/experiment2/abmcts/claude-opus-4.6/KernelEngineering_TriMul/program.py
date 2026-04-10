# EVOLVE-BLOCK-START
import torch
from torch import nn
from .task import input_t, output_t

@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    batch_size, seq_len, _, _ = input_tensor.shape

    # Pre-extract weights
    norm_weight = weights['norm.weight']
    norm_bias = weights['norm.bias']
    to_out_norm_weight = weights['to_out_norm.weight']
    to_out_norm_bias = weights['to_out_norm.bias']

    # Use float16 for compute on GPU where possible
    compute_dtype = torch.float16

    left_proj_w = weights['left_proj.weight'].to(compute_dtype)
    right_proj_w = weights['right_proj.weight'].to(compute_dtype)
    left_gate_w = weights['left_gate.weight'].to(compute_dtype)
    right_gate_w = weights['right_gate.weight'].to(compute_dtype)
    out_gate_w = weights['out_gate.weight'].to(compute_dtype)
    to_out_w = weights['to_out.weight'].to(compute_dtype)

    # LayerNorm in float32
    x = input_tensor.to(torch.float32)
    x = torch.nn.functional.layer_norm(x, (dim,), norm_weight.float(), norm_bias.float())

    # Convert to compute dtype for projections
    x_half = x.to(compute_dtype)
    del x

    # Reshape for batched linear: [B*N*N, dim]
    x_flat = x_half.reshape(-1, dim)
    del x_half

    # Fuse all 5 projections into one matmul
    combined_w = torch.cat([left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w], dim=0)
    combined_out = x_flat @ combined_w.t()
    del x_flat, combined_w

    left = combined_out[:, :hidden_dim].reshape(batch_size, seq_len, seq_len, hidden_dim)
    right = combined_out[:, hidden_dim:2*hidden_dim].reshape(batch_size, seq_len, seq_len, hidden_dim)
    left_gate = combined_out[:, 2*hidden_dim:3*hidden_dim].reshape(batch_size, seq_len, seq_len, hidden_dim)
    right_gate = combined_out[:, 3*hidden_dim:4*hidden_dim].reshape(batch_size, seq_len, seq_len, hidden_dim)
    out_gate = combined_out[:, 4*hidden_dim:].reshape(batch_size, seq_len, seq_len, hidden_dim)
    del combined_out

    # Apply mask
    mask_expanded = mask.unsqueeze(-1).to(compute_dtype)
    left = left * mask_expanded
    right = right * mask_expanded
    del mask_expanded

    # Apply gates with sigmoid
    torch.sigmoid_(left_gate)
    torch.sigmoid_(right_gate)
    torch.sigmoid_(out_gate)

    left = left * left_gate
    right = right * right_gate
    del left_gate, right_gate

    # Einsum: ... i k d, ... j k d -> ... i j d
    # left: [B, i, k, d] -> [B*d, i, k]
    # right: [B, j, k, d] -> [B*d, k, j]
    left_perm = left.permute(0, 3, 1, 2).reshape(batch_size * hidden_dim, seq_len, seq_len)
    right_perm = right.permute(0, 3, 2, 1).reshape(batch_size * hidden_dim, seq_len, seq_len)
    del left, right

    out = torch.bmm(left_perm, right_perm)
    del left_perm, right_perm

    out = out.reshape(batch_size, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1)

    # LayerNorm on hidden_dim in float32
    out = out.to(torch.float32)
    out = torch.nn.functional.layer_norm(out, (hidden_dim,), to_out_norm_weight.float(), to_out_norm_bias.float())

    # Apply out_gate (convert out_gate to float32 for multiply)
    out = out * out_gate.float()
    del out_gate

    # Final linear projection in compute_dtype
    out = out.to(compute_dtype)
    out = out.reshape(-1, hidden_dim) @ to_out_w.t()
    out = out.reshape(batch_size, seq_len, seq_len, dim)

    return out.to(torch.float32)
# EVOLVE-BLOCK-END
