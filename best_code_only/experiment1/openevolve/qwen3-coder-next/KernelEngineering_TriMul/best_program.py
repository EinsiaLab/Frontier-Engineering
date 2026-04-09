# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
from .task import input_t, output_t

class TriMul(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply normalization first
        x = self.norm(x)
        
        # Project to left and right paths
        left = self.left_proj(x)
        right = self.right_proj(x)
        
        # Apply mask early to reduce computation
        mask_expanded = mask.unsqueeze(-1)
        left = left * mask_expanded
        right = right * mask_expanded
        
        # Compute gates
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        
        # Apply gates after masking
        left = left * left_gate
        right = right * right_gate
        
        # Triangle multiplication with optimized type handling
        # Use bfloat16 for the heavy computation
        out = einsum('... i k d, ... j k d -> ... i j d', 
                    left.to(torch.bfloat16), right.to(torch.bfloat16))
        
        # Convert back to float32 for normalization
        out = out.to(torch.float32)
        
        # Apply final transformations
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)
    
    # Load weights more efficiently by directly assigning matching weights
    for name, param in trimul.named_parameters():
        if name in weights:
            param.data = weights[name].data
    
    return trimul(input_tensor, mask)
# EVOLVE-BLOCK-END
