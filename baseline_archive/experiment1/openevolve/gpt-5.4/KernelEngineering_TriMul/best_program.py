# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from torch import einsum
from .task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    dev = input_tensor.device
    x = F.layer_norm(
        input_tensor.float(),
        (config["dim"],),
        weights["norm.weight"].to(dev, torch.float32),
        weights["norm.bias"].to(dev, torch.float32),
    )
    m = mask.unsqueeze(-1)
    left = F.linear(x, weights["left_proj.weight"].to(dev, torch.float32))
    right = F.linear(x, weights["right_proj.weight"].to(dev, torch.float32))
    left = left * m * F.linear(x, weights["left_gate.weight"].to(dev, torch.float32)).sigmoid()
    right = right * m * F.linear(x, weights["right_gate.weight"].to(dev, torch.float32)).sigmoid()
    out = einsum("...ikd,...jkd->...ijd", left.bfloat16(), right.bfloat16()).float()
    out = F.layer_norm(
        out,
        (config["hidden_dim"],),
        weights["to_out_norm.weight"].to(dev, torch.float32),
        weights["to_out_norm.bias"].to(dev, torch.float32),
    )
    out = out * F.linear(x, weights["out_gate.weight"].to(dev, torch.float32)).sigmoid()
    return F.linear(out, weights["to_out.weight"].to(dev, torch.float32)).float()
# EVOLVE-BLOCK-END
