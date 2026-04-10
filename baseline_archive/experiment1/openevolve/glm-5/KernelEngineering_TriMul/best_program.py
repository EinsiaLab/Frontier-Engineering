# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
from .task import input_t, output_t

torch.set_float32_matmul_precision('high')

class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).to(torch.float32)
        
        # Project once and reuse
        left = self.left_proj(x)
        right = self.right_proj(x)
        
        # Compute all gates in parallel
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        
        # Fuse mask and gate operations to reduce memory bandwidth
        mask = mask.unsqueeze(-1)
        left = left * mask * left_gate
        right = right * mask * right_gate
        
        # Core einsum operation - bfloat16 for tensor core efficiency
        out = einsum('... i k d, ... j k d -> ... i j d', 
                     left.to(torch.bfloat16), right.to(torch.bfloat16))
        
        # Fuse final operations
        out = self.to_out_norm(out.to(torch.float32)) * out_gate
        return self.to_out(out)

# Cache for compiled models to avoid recreation overhead
_trimul_cache = {}

def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dim, hidden_dim = config["dim"], config["hidden_dim"]
    cache_key = (device, dim, hidden_dim)
    
    if cache_key not in _trimul_cache:
        trimul = TriMul(dim, hidden_dim).to(device)
        trimul = torch.compile(trimul, mode="reduce-overhead")
        _trimul_cache[cache_key] = trimul
    else:
        trimul = _trimul_cache[cache_key]

    # Update weights efficiently using copy_
    with torch.no_grad():
        trimul.norm.weight.copy_(weights['norm.weight'].to(torch.float32))
        trimul.norm.bias.copy_(weights['norm.bias'].to(torch.float32))
        trimul.left_proj.weight.copy_(weights['left_proj.weight'].to(torch.float32))
        trimul.right_proj.weight.copy_(weights['right_proj.weight'].to(torch.float32))
        trimul.left_gate.weight.copy_(weights['left_gate.weight'].to(torch.float32))
        trimul.right_gate.weight.copy_(weights['right_gate.weight'].to(torch.float32))
        trimul.out_gate.weight.copy_(weights['out_gate.weight'].to(torch.float32))
        trimul.to_out_norm.weight.copy_(weights['to_out_norm.weight'].to(torch.float32))
        trimul.to_out_norm.bias.copy_(weights['to_out_norm.bias'].to(torch.float32))
        trimul.to_out.weight.copy_(weights['to_out.weight'].to(torch.float32))

    with torch.no_grad():
        return trimul(input_tensor, mask)
# EVOLVE-BLOCK-END
