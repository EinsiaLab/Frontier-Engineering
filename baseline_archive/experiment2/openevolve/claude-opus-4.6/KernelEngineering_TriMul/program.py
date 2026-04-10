# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import input_t, output_t


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    bs, sl, _, dim = input_tensor.shape
    hd = config["hidden_dim"]

    nw = weights['norm.weight'].float()
    nb = weights['norm.bias'].float()
    x = F.layer_norm(input_tensor.float(), [dim], nw, nb)

    left = F.linear(x, weights['left_proj.weight'].float())
    right = F.linear(x, weights['right_proj.weight'].float())

    m = mask.unsqueeze(-1)
    left.mul_(m)
    right.mul_(m)
    del m

    left.mul_(F.linear(x, weights['left_gate.weight'].float()).sigmoid_())
    right.mul_(F.linear(x, weights['right_gate.weight'].float()).sigmoid_())
    out_gate = F.linear(x, weights['out_gate.weight'].float()).sigmoid_()
    del x

    n = bs * hd
    left_r = left.permute(0, 3, 1, 2).contiguous().view(n, sl, sl).to(torch.bfloat16)
    del left
    right_r = right.permute(0, 3, 2, 1).contiguous().view(n, sl, sl).to(torch.bfloat16)
    del right

    out = torch.bmm(left_r, right_r)
    del left_r, right_r

    out = out.view(bs, hd, sl, sl).permute(0, 2, 3, 1).contiguous().float()

    out = F.layer_norm(out, [hd], weights['to_out_norm.weight'].float(), weights['to_out_norm.bias'].float())
    out.mul_(out_gate)
    del out_gate

    return F.linear(out, weights['to_out.weight'].float())
# EVOLVE-BLOCK-END
