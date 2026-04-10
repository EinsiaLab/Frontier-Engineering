# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
from .task import input_t, output_t

# --------------------------------------------------------------
# Small runtime‑hints for CUDA:
# – Enable TF32 matrix multiplies (often faster with negligible error)
# – Turn on cuDNN benchmark mode to select the best kernels
# These are safe defaults for the forward‑only workload.
# --------------------------------------------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True          # allow TensorFloat‑32
    torch.backends.cudnn.benchmark = True               # auto‑tune kernels

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
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        # Normalise once – the LayerNorm already returns a float tensor.
        x = self.norm(x)

        # Linear projections.
        left = self.left_proj(x)
        right = self.right_proj(x)

        # Apply mask (broadcasted over the hidden dimension)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        # Gating signals.
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        # Apply gates and cast to bfloat16 for faster Tensor‑Core einsum.
        left = (left * left_gate).to(torch.bfloat16)
        right = (right * right_gate).to(torch.bfloat16)

        # Core tri‑multiplication – compute in bfloat16 then cast back.
        out = einsum('... i k d, ... j k d -> ... i j d',
                     left, right).to(torch.float32)

        # Normalise, apply output gate and final linear projection.
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    # Fill in the given weights of the model
    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output
# EVOLVE-BLOCK-END
