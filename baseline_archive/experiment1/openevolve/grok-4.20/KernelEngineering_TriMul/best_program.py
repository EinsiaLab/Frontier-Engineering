# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import input_t, output_t

torch.set_float32_matmul_precision('high')

_compiled_funcs = {}


def _tri_mul_forward(x, mask, norm_w, norm_b, left_proj_w, right_proj_w,
                     left_gate_w, right_gate_w, out_gate_w,
                     to_out_norm_w, to_out_norm_b, to_out_w,
                     hidden_dim):
    x = F.layer_norm(x, (x.shape[-1],), weight=norm_w, bias=norm_b)

    left = F.linear(x, left_proj_w)
    right = F.linear(x, right_proj_w)

    mask = mask.unsqueeze(-1)
    mask = mask.to(left.dtype)
    left = left * mask
    right = right * mask

    left_gate = F.linear(x, left_gate_w).sigmoid()
    right_gate = F.linear(x, right_gate_w).sigmoid()
    out_gate = F.linear(x, out_gate_w).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # Explicit matmul for better GEMM fusion on tensor cores (avoids einsum overhead)
    # Using TF32 on float32 (enabled globally) avoids bf16 cast overhead
    left_f = left.permute(0, 3, 1, 2).contiguous()
    right_f = right.permute(0, 3, 2, 1).contiguous()
    out_f = torch.matmul(left_f, right_f)
    out = out_f.permute(0, 2, 3, 1).contiguous()

    out = F.layer_norm(out, (hidden_dim,), weight=to_out_norm_w, bias=to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out


def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device
    key = (dim, hidden_dim, device.index if device.type == "cuda" else str(device))

    if key not in _compiled_funcs:
        fn = torch.compile(_tri_mul_forward, mode="max-autotune", fullgraph=True)
        _compiled_funcs[key] = fn
    else:
        fn = _compiled_funcs[key]

    # All weights are expected in float32
    norm_w = weights['norm.weight'].to(torch.float32)
    norm_b = weights['norm.bias'].to(torch.float32)
    left_proj_w = weights['left_proj.weight'].to(torch.float32)
    right_proj_w = weights['right_proj.weight'].to(torch.float32)
    left_gate_w = weights['left_gate.weight'].to(torch.float32)
    right_gate_w = weights['right_gate.weight'].to(torch.float32)
    out_gate_w = weights['out_gate.weight'].to(torch.float32)
    to_out_norm_w = weights['to_out_norm.weight'].to(torch.float32)
    to_out_norm_b = weights['to_out_norm.bias'].to(torch.float32)
    to_out_w = weights['to_out.weight'].to(torch.float32)

    x = input_tensor.to(torch.float32) if input_tensor.dtype != torch.float32 else input_tensor
    # Ensure mask is on correct device (defensive)
    if mask.device != x.device:
        mask = mask.to(x.device)

    output = fn(x, mask, norm_w, norm_b, left_proj_w, right_proj_w,
                left_gate_w, right_gate_w, out_gate_w,
                to_out_norm_w, to_out_norm_b, to_out_w,
                hidden_dim)

    return output
# EVOLVE-BLOCK-END
