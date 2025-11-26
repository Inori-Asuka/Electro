import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import timm
from safetensors.torch import load_file

# Removed PEFT import - using manual LoRA implementation instead



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


def create_regression_head(head_type: str, input_dim: int, output_dim: int = 1,
                          dropout: float = 0.5, **kwargs) -> RegressionHead:

    head_type = head_type.lower()
    if head_type == 'linear':
        return LinearRegressionHead(input_dim, output_dim, dropout)
    elif head_type == 'mlp':
        hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        return MLPRegressionHead(input_dim, output_dim, dropout, hidden_dims)
    elif head_type == 'attention':
        num_heads = kwargs.get('num_heads', 8)
        num_layers = kwargs.get('num_layers', 2)
        return AttentionRegressionHead(input_dim, output_dim, dropout, num_heads, num_layers)
    elif head_type == 'gated':
        hidden_dim = kwargs.get('hidden_dim', None)
        use_simple = kwargs.get('use_simple', False)
        return GatedRegressionHead(input_dim, output_dim, dropout, hidden_dim, use_simple)
    elif head_type == 'residual':
        hidden_dim = kwargs.get('hidden_dim', 512)
        num_blocks = kwargs.get('num_blocks', 3)
        return ResidualRegressionHead(input_dim, output_dim, dropout, hidden_dim, num_blocks)
    else:
        raise ValueError(f"不支持的回归头类型: {head_type}")


class DINOv3RegressionModel(nn.Module):
    """基于DINOv3 ConvNeXt-Large的回归模型
    支持特征融合（多层CLS token + Patch token平均池化）
    """
    
    def __init__(
        self,
        model_path: str,
        num_outputs: int = 1,
        head_type: str = 'mlp',
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        
        self.model_path = model_path
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        
        # 创建ConvNeXt-Large结构
        self.backbone = timm.create_model(
            "convnext_large.dinov3_lvd1689m",
            pretrained=False,
            num_classes=0
        )
        
        # 加载DINOv3权重
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[DINOv3] Loading Weights: {model_path}")
            print(f"[DINOv3] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            print("Error")
        
        # 获取特征维度
        embed_dim = self.backbone.num_features
        
        # 计算回归头的输入维度（使用最后一层的特征）
        linear_in_dim = embed_dim
        
        # 设置微调策略（顺序很重要：先处理LoRA，再处理freeze）
        if use_lora:
            modules_to_replace = {}
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear) and (name.endswith('.fc1') or name.endswith('.fc2')):
                    modules_to_replace[name] = module
            
            sorted_names = sorted(modules_to_replace.keys(), key=lambda x: x.count('.'), reverse=True)
            
            for name in sorted_names:
                module = modules_to_replace[name]
                parts = name.split('.')
                module_name = parts[-1]
                parent_path = '.'.join(parts[:-1])
                
                parent = self.backbone
                for part in parent_path.split('.'):
                    parent = getattr(parent, part)
                
                lora_linear = LoRALinear(module, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
                setattr(parent, module_name, lora_linear)
            
            print(f"[DINOv3] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层")
                
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[DINOv3] 完全冻结backbone，仅训练回归头")
            
        elif freeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
            print(f"[DINOv3] 冻结层: {freeze_layers}")
        else:
            print("[DINOv3] 全参数微调")
        
        head_kwargs = head_kwargs or {}
        self.head = create_regression_head(
            head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
        )
        print(f"[DINOv3] 回归头类型: {head_type}")
        print(f"[DINOv3] 特征维度: {linear_in_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (without passing through regression head)
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features: Extracted features [B, embed_dim]
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            # If features are spatial, perform global average pooling
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
        return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

