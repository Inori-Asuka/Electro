import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearRegressionHead(RegressionHead):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5):
        super().__init__(input_dim, output_dim, dropout)
        self.head = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MLPRegressionHead(RegressionHead):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__(input_dim, output_dim, dropout)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AttentionRegressionHead(RegressionHead):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__(input_dim, output_dim, dropout)
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        return self.output(x)


class GatedRegressionHead(RegressionHead):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 hidden_dim: Optional[int] = None, use_simple: bool = False):
        super().__init__(input_dim, output_dim, dropout)
        if hidden_dim is None:
            hidden_dim = input_dim
        
        if use_simple:
            self.main = nn.Linear(input_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
        else:
            self.main_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            self.gate_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Sigmoid()
            )
            self.output = nn.Linear(hidden_dim // 2, output_dim)
        self.use_simple = use_simple
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_simple:
            main_out = self.main(x)
            gate_weights = self.gate(x)
            return main_out * gate_weights
        else:
            main_features = self.main_branch(x)
            gate_weights = self.gate_branch(x)
            gated_features = main_features * gate_weights
            return self.output(gated_features)


class ResidualRegressionHead(RegressionHead):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 hidden_dim: int = 512, num_blocks: int = 3):
        super().__init__(input_dim, output_dim, dropout)
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)
            prev_dim = hidden_dim
        
        self.output = nn.Linear(prev_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, prev_dim) if input_dim != prev_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = x
        for block in self.blocks:
            out = block(out)
        out = out + residual
        return self.output(out)