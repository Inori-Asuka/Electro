import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 冻结原始线性层
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA 参数
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.linear(x)
        
        x_dropout = self.dropout(x)
        
        original_shape = x_dropout.shape
        if x_dropout.dim() > 2:
            x_flat = x_dropout.view(-1, original_shape[-1])
        else:
            x_flat = x_dropout
        
        # LoRA 计算: x @ A^T @ B^T
        # x_flat: [N, in_features]
        # lora_A: [r, in_features] -> A^T: [in_features, r]
        # lora_B: [out_features, r] -> B^T: [r, out_features]
        lora_output = x_flat @ self.lora_A.T  # [N, r]
        lora_output = lora_output @ self.lora_B.T  # [N, out_features]
        lora_output = lora_output * self.scaling
    
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [lora_output.shape[-1]]
            lora_output = lora_output.view(*output_shape)
        
        return original_output + lora_output
    
    def extra_repr(self) -> str:
        return f'r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}'