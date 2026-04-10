# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import input_t, output_t


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data

    w = weights
    dim = config["dim"]
    h = config["hidden_dim"]

    # LayerNorm on input
    x = F.layer_norm(input_tensor.float(), (dim,), w['norm.weight'].float(), w['norm.bias'].float())

    # Fused projection in bf16 for tensor core acceleration on the linear layers
    combined_w = torch.cat([w['left_proj.weight'], w['right_proj.weight'],
                            w['left_gate.weight'], w['right_gate.weight'],
                            w['out_gate.weight']], dim=0).to(torch.bfloat16)
    all_proj = F.linear(x.to(torch.bfloat16), combined_w).float()
    del combined_w, x

    left, right, lg, rg, og = all_proj.split(h, dim=-1)
    del all_proj

    # Fuse mask + gate application
    m = mask.unsqueeze(-1)
    left = left * (m * lg.sigmoid())
    right = right * (m * rg.sigmoid())
    out_gate = og.sigmoid()
    del lg, rg, og, m

    # Triangular multiplication via bmm in bfloat16
    bs, sl, _, d = left.shape
    left_bf = left.to(torch.bfloat16).permute(0, 3, 1, 2).contiguous().view(bs * d, sl, sl)
    right_bf = right.to(torch.bfloat16).permute(0, 3, 2, 1).contiguous().view(bs * d, sl, sl)
    del left, right
    out = torch.bmm(left_bf, right_bf)
    del left_bf, right_bf
    out = out.float().view(bs, d, sl, sl).permute(0, 2, 3, 1).contiguous()

    # Output LayerNorm + gate + projection
    out = F.layer_norm(out, (h,), w['to_out_norm.weight'].float(), w['to_out_norm.bias'].float())
    out = out * out_gate
    return F.linear(out, w['to_out.weight'].float())
# EVOLVE-BLOCK-END
