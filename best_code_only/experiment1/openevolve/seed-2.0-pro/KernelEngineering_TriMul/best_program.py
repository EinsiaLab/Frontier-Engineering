# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Optimized TriMul implementation with zero module overhead, fused operations, and disabled autograd.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]
    device = input_tensor.device

    # Disable autograd entirely since we only need forward pass (no gradients required)
    with torch.no_grad():
        # Temporarily set optimal GPU performance settings for tensor core utilization
        original_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('medium')
        original_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True

        # Load all weights to device in single batch with non-blocking to overlap data movement with computation
        w_norm = weights['norm.weight'].to(device, torch.float32, non_blocking=True)
        b_norm = weights['norm.bias'].to(device, torch.float32, non_blocking=True)
        w_left_proj = weights['left_proj.weight'].to(device, torch.float32, non_blocking=True)
        w_right_proj = weights['right_proj.weight'].to(device, torch.float32, non_blocking=True)
        w_left_gate = weights['left_gate.weight'].to(device, torch.float32, non_blocking=True)
        w_right_gate = weights['right_gate.weight'].to(device, torch.float32, non_blocking=True)
        w_out_gate = weights['out_gate.weight'].to(device, torch.float32, non_blocking=True)
        w_out_norm = weights['to_out_norm.weight'].to(device, torch.float32, non_blocking=True)
        b_out_norm = weights['to_out_norm.bias'].to(device, torch.float32, non_blocking=True)
        w_to_out = weights['to_out.weight'].to(device, torch.float32, non_blocking=True)

        # Forward pass with in-place operations and fusions to reduce kernel launches
        x = F.layer_norm(input_tensor, (dim,), weight=w_norm, bias=b_norm).to(torch.float32, non_blocking=True)
        
        left = F.linear(x, w_left_proj)
        right = F.linear(x, w_right_proj)

        mask = mask.unsqueeze(-1).to(torch.float32, non_blocking=True)
        left_gate = F.linear(x, w_left_gate).sigmoid()
        right_gate = F.linear(x, w_right_gate).sigmoid()
        out_gate = F.linear(x, w_out_gate).sigmoid()

        # In-place fused mask + gate operation to eliminate memory copies and reduce kernel launches
        left *= mask * left_gate
        right *= mask * right_gate

        # Use tensor-core optimized batched matmul (better hardware utilization than einsum)
        left_bf16 = left.to(torch.bfloat16, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        right_bf16 = right.to(torch.bfloat16, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        out = torch.matmul(left_bf16, right_bf16.transpose(-1, -2)).permute(0, 2, 3, 1).contiguous()
        
        out = out.to(torch.float32, non_blocking=True)
        out = F.layer_norm(out, (hidden_dim,), weight=w_out_norm, bias=b_out_norm)
        out *= out_gate
        out = F.linear(out, w_to_out)

        # Restore original global GPU settings to avoid side effects
        torch.set_float32_matmul_precision(original_matmul_precision)
        torch.backends.cudnn.benchmark = original_cudnn_benchmark

    return out.to(torch.float32, non_blocking=True)
# EVOLVE-BLOCK-END
