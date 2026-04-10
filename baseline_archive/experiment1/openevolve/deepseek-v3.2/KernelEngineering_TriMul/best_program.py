# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
from .task import input_t, output_t

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
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x).to(torch.float32)
        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        # Convert to bfloat16 for faster computation and tensor core utilization
        left_bf16 = left.to(torch.bfloat16)
        right_bf16 = right.to(torch.bfloat16)
        
        # Permute dimensions to [batch, hidden_dim, seq_len, seq_len] for efficient matmul
        left_permuted = left_bf16.permute(0, 3, 1, 2).contiguous()  # [b, d, i, k]
        right_permuted = right_bf16.permute(0, 3, 1, 2).contiguous()  # [b, d, j, k]
        # Compute matmul: left_permuted @ right_permuted.transpose(-1, -2)
        # Result shape: [b, d, i, j]
        out_permuted = torch.matmul(left_permuted, right_permuted.transpose(-1, -2))
        # Permute back to [batch, seq_len, seq_len, hidden_dim]
        out = out_permuted.permute(0, 2, 3, 1).to(torch.float32)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    # Enable TensorFloat32 for better performance on float32 matmul
    torch.set_float32_matmul_precision('high')
    
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight.data = weights['norm.weight'].to(torch.float32)
    trimul.left_proj.weight.data = weights['left_proj.weight'].to(torch.float32)
    trimul.right_proj.weight.data = weights['right_proj.weight'].to(torch.float32)
    trimul.left_gate.weight.data = weights['left_gate.weight'].to(torch.float32)
    trimul.right_gate.weight.data = weights['right_gate.weight'].to(torch.float32)
    trimul.out_gate.weight.data = weights['out_gate.weight'].to(torch.float32)
    trimul.to_out_norm.weight.data = weights['to_out_norm.weight'].to(torch.float32)
    trimul.to_out.weight.data = weights['to_out.weight'].to(torch.float32)
    trimul.norm.bias.data = weights['norm.bias'].to(torch.float32)
    trimul.to_out_norm.bias.data = weights['to_out_norm.bias'].to(torch.float32)

    trimul = torch.compile(trimul, mode="max-autotune")
    output = trimul(input_tensor, mask).to(torch.float32)

    return output
# EVOLVE-BLOCK-END
