import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class AttentionPoolingHead(nn.Module):
    """Attention Pooling Head for spatial features"""
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 num_heads: int = 8, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Learnable query token for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) - Batch, Num_patches, Channels
        Returns:
            output: (B, output_dim)
        """
        B, N, C = x.shape
        
        # Project input features
        x_proj = self.input_proj(x)  # (B, N, hidden_dim)
        
        # Expand query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        
        # Attention pooling: query attends to all patches
        attn_out, attn_weights = self.attention(query, x_proj, x_proj)  # (B, 1, hidden_dim)
        
        # Squeeze and project to output
        pooled = attn_out.squeeze(1)  # (B, hidden_dim)
        output = self.output_proj(pooled)  # (B, output_dim)
        
        return output


class MultiScaleFusionHead(nn.Module):
    """Multi-scale feature fusion head"""
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.5,
                 num_scales: int = 4, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.num_scales = num_scales
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        # Feature fusion layers
        self.fusion_layers = nn.ModuleList()
        prev_dim = input_dim * num_scales
        
        for hidden_dim in hidden_dims:
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: List of (B, N, C) tensors from different layers
        Returns:
            output: (B, output_dim)
        """
        # Global average pooling for each scale
        pooled_features = []
        for feat in features_list:
            # feat: (B, N, C)
            pooled = feat.mean(dim=1)  # (B, C)
            pooled_features.append(pooled)
        
        # Concatenate all scales
        fused = torch.cat(pooled_features, dim=1)  # (B, num_scales * C)
        
        # Pass through fusion layers
        x = fused
        for layer in self.fusion_layers:
            x = layer(x)
        
        # Output
        output = self.output(x)
        return output
