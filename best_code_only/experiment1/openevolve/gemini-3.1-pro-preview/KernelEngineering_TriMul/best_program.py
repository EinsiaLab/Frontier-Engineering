# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F
from .task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    x, m, w, c = data
    f32, bf16 = torch.float32, torch.bfloat16
    
    x = F.layer_norm(x, (c["dim"],), w['norm.weight'].to(f32), w['norm.bias'].to(f32)).to(f32)
    
    w_cat = torch.cat([
        w['left_proj.weight'], w['right_proj.weight'],
        w['left_gate.weight'], w['right_gate.weight'], w['out_gate.weight']
    ], dim=0).to(f32)
    
    l, r, lg, rg, og = F.linear(x, w_cat).chunk(5, dim=-1)
    
    m = m.unsqueeze(-1)
    l = (l * m * lg.sigmoid()).to(bf16).permute(0, 3, 1, 2)
    r = (r * m * rg.sigmoid()).to(bf16).permute(0, 3, 2, 1)
    
    out = torch.matmul(l, r).permute(0, 2, 3, 1).to(f32)
    
    out = F.layer_norm(out, (c["hidden_dim"],), w['to_out_norm.weight'].to(f32), w['to_out_norm.bias'].to(f32))
    return F.linear(out * og.sigmoid(), w['to_out.weight'].to(f32))
# EVOLVE-BLOCK-END
