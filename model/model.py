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
    
    注意：ConvNeXt 使用 GAP (Global Average Pooling) 作为默认的 pooling 方式
    """
    
    def __init__(
        self,
        model_path: str,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 为了统一接口，但 ConvNeXt 只支持 'gap'
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
        self.pooling = pooling  # 统一接口，但 ConvNeXt 只支持 'gap'
        
        if pooling != 'gap':
            print(f"[DINOv3 ConvNeXt] Warning: pooling='{pooling}' 不支持，将使用 'gap'")
            self.pooling = 'gap'
        
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


class DINOv3ViTRegressionModel(nn.Module):
    """基于DINOv3 ViT Large的回归模型
    支持三种方案：
    - 方案A: GAP + MLP (pooling='gap')
    - 方案B: 保留空间特征 + Attention Pooling (pooling='attention')
    - 方案C: 多尺度特征融合 (pooling='multiscale')
    """
    
    def __init__(
        self,
        model_path: str,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'gap', 'attention', 'multiscale'
        head_type: str = 'mlp',
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        head_kwargs: Optional[dict] = None,
        multiscale_layers: Optional[List[int]] = None,  # 用于方案C，指定要提取的层索引
    ):
        super().__init__()
        
        self.model_path = model_path
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.pooling = pooling
        self.multiscale_layers = multiscale_layers or [6, 12, 18, 23]  # 默认提取4个不同深度的层
        
        # 创建ViT Large结构
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=False,
            num_classes=0,  # 移除分类头
            dynamic_img_size=True  # 支持动态输入尺寸
        )
        
        # 加载DINOv3权重
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            # 清理state_dict键名（移除backbone.前缀等）
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("backbone.", "").replace("model.", "")
                cleaned_state_dict[new_k] = v
            
            missing, unexpected = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
            print(f"[DINOv3 ViT] Loading Weights: {model_path}")
            print(f"[DINOv3 ViT] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            print(f"[DINOv3 ViT] Warning: Model path not found: {model_path}")
        
        # 获取特征维度
        embed_dim = self.backbone.num_features  # 通常是1024 for ViT Large
        
        # 设置微调策略
        if use_lora:
            # 获取 LoRA 模块选择配置
            # 如果未指定，默认微调所有模块 (qkv, proj, mlp)
            lora_modules = head_kwargs.get('lora_modules', None) if head_kwargs else None
            if lora_modules is None:
                lora_modules = ['qkv', 'proj', 'mlp']  # 默认微调所有模块
            elif isinstance(lora_modules, str):
                lora_modules = [lora_modules]
            
            modules_to_replace = {}
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    # 检查是否匹配指定的模块类型
                    should_replace = False
                    if 'qkv' in lora_modules and ('attn' in name and ('qkv' in name or name.endswith('.qkv'))):
                        should_replace = True
                    elif 'proj' in lora_modules and ('attn' in name and ('proj' in name or name.endswith('.proj'))):
                        should_replace = True
                    elif 'mlp' in lora_modules and ('mlp' in name and ('fc1' in name or 'fc2' in name)):
                        should_replace = True
                    
                    if should_replace:
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
            
            print(f"[DINOv3 ViT] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层")
            print(f"[DINOv3 ViT] LoRA 模块: {lora_modules}")
                
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[DINOv3 ViT] 完全冻结backbone，仅训练回归头")
            
        elif freeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
            print(f"[DINOv3 ViT] 冻结层: {freeze_layers}")
        else:
            print("[DINOv3 ViT] 全参数微调")
        
        # 根据pooling方案创建不同的头
        head_kwargs = head_kwargs or {}
        
        if pooling == 'gap':
            # 方案A: GAP + MLP
            # forward_features 返回 (B, N, C)，需要GAP得到 (B, C)
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[DINOv3 ViT] 方案A: GAP + {head_type} 回归头")
            
        elif pooling == 'attention':
            # 方案B: Attention Pooling
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[DINOv3 ViT] 方案B: Attention Pooling (heads={num_heads})")
            
        elif pooling == 'multiscale':
            # 方案C: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[DINOv3 ViT] 方案C: 多尺度融合 (layers={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的pooling方案: {pooling}。可选: 'gap', 'attention', 'multiscale'")
        
        print(f"[DINOv3 ViT] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            output: Regression output [B, num_outputs]
        """
        if self.pooling == 'multiscale':
            # 方案C: 提取多层特征
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        else:
            # 方案A和B: 使用forward_features获取空间特征
            features = self.backbone.forward_features(x)  # (B, N, C)
            
            if self.pooling == 'gap':
                # 方案A: Global Average Pooling
                # 跳过CLS token和register tokens，只对patch tokens做GAP
                # 假设第一个token是CLS，接下来可能有register tokens
                # 对于DINOv3 ViT，通常结构是: CLS + 4 registers + patches
                # 我们取所有tokens的平均（或者只取patches）
                features = features.mean(dim=1)  # (B, C)
                return self.head(features)
            else:
                # 方案B: Attention Pooling (保留空间特征)
                # features已经是 (B, N, C)，直接传给AttentionPoolingHead
                # DINOv3 ViT结构: CLS (1) + Registers (4) + Patches (N)
                # 跳过CLS和register tokens，只使用patch tokens
                # 前5个tokens是CLS+registers，其余是patches
                if features.shape[1] > 5:
                    patch_features = features[:, 5:, :]  # (B, N_patches, C)
                else:
                    # 如果tokens数量不足，使用所有tokens
                    patch_features = features
                return self.head(patch_features)
    
    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征（方案C）
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features_list: List of (B, N, C) tensors from different layers
        """
        features_list = []
        
        def hook_fn(module, input, output):
            # output 是 Block 的输出 (B, N, C)
            # 在训练时保持梯度，在eval时detach
            if self.training:
                features_list.append(output)
            else:
                features_list.append(output.detach())
        
        hooks = []
        
        # 注册hooks到指定层
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            for layer_idx in self.multiscale_layers:
                if 0 <= layer_idx < len(blocks):
                    hooks.append(blocks[layer_idx].register_forward_hook(hook_fn))
        else:
            # Fallback: 如果找不到blocks，尝试其他方式
            print("[DINOv3 ViT] Warning: 无法找到blocks，使用forward_features作为fallback")
            features = self.backbone.forward_features(x)
            return [features]  # 返回单层特征
        
        # 前向传播（保持梯度流）
        _ = self.backbone.forward_features(x)
        
        # 移除hooks
        for h in hooks:
            h.remove()
        
        return features_list
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (without passing through regression head)
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features: Extracted features [B, embed_dim] or [B, N, embed_dim] depending on pooling
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone.forward_features(x)  # (B, N, C)
            if self.pooling == 'gap':
                features = features.mean(dim=1)  # (B, C)
        return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total