import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import timm
from safetensors.torch import load_file

from .layers.regression import LinearRegressionHead,MLPRegressionHead,AttentionRegressionHead,GatedRegressionHead,ResidualRegressionHead
from .layers.aggregate import AttentionPoolingHead,MultiScaleFusionHead
from .layers.lora import LoRALinear


def create_regression_head(head_type: str, input_dim: int, output_dim: int = 1,
                          dropout: float = 0.5, **kwargs):
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
        raise ValueError(f"No Supported Regression Head Type: {head_type}")


class DINOv3RegressionModel(nn.Module):
    """基于DINOv3 ConvNeXt-Large的回归模型
    支持四种 pooling 方案：
    - 'linear': 直接线性层回归（使用 CLS token）
    - 'gap': Global Average Pooling
    - 'attention': Attention Pooling（将空间特征 reshape 成序列）
    - 'multiscale': 多尺度特征融合
    """
    
    def __init__(
        self,
        model_path: str,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'linear', 'gap', 'attention', 'multiscale'
        head_type: str = 'linear',  # 对于 linear pooling，head_type 会被忽略
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        head_kwargs: Optional[dict] = None,
        multiscale_layers: Optional[List[int]] = None,  # 用于 multiscale 方案，指定要提取的 stage 索引
    ):
        super().__init__()
        
        self.model_path = model_path
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.pooling = pooling
        self.multiscale_layers = multiscale_layers or [1, 2, 3]  # ConvNeXt 有4个 stage，默认提取后3个
        
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
            
            print(f"[DINOv3 ConvNeXt] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层 (MLP 的 fc1 和 fc2)")
                
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
        
        # 根据 pooling 方案创建不同的头
        head_kwargs = head_kwargs or {}
        
        if pooling == 'linear':
            # 方案1: 直接线性层回归
            # 与原始实现一致：使用 backbone(x) 获取 GAP 特征，然后通过线性层
            self.head = nn.Linear(embed_dim, num_outputs)
            print(f"[DINOv3 ConvNeXt] 方案: Linear (直接线性层回归)")
        elif pooling == 'gap':
            # 方案2: Global Average Pooling
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[DINOv3 ConvNeXt] 方案: GAP + {head_type}")
        elif pooling == 'attention':
            # 方案3: Attention Pooling
            # 将空间特征 reshape 成序列，然后使用 AttentionPoolingHead
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[DINOv3 ConvNeXt] 方案: Attention Pooling (heads={num_heads})")
        elif pooling == 'multiscale':
            # 方案4: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[DINOv3 ConvNeXt] 方案: Multiscale (stages={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的 pooling 方案: {pooling}。可选: 'linear', 'gap', 'attention', 'multiscale'")
        
        print(f"[DINOv3 ConvNeXt] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'multiscale':
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        elif self.pooling == 'linear':
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
            return self.head(features)
        elif self.pooling == 'attention':
            features = self.backbone(x)  # (B, C, H, W)
            features_seq = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
            return self.head(features_seq)
        else:
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
            return self.head(features)
    
    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features_list: List of (B, N, C) tensors from different stages
        """
        features_list = []
        
        def hook_fn(module, input, output):
            # output 是 stage 的输出 (B, C, H, W)
            # 转换为序列格式 (B, H*W, C)
            if len(output.shape) == 4:
                B, C, H, W = output.shape
                output_seq = output.flatten(2).transpose(1, 2)  # (B, H*W, C)
                features_list.append(output_seq)
            else:
                features_list.append(output)
        
        hooks = []
        
        # 注册 hooks 到指定的 stages
        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
            for stage_idx in self.multiscale_layers:
                if 0 <= stage_idx < len(stages):
                    # Hook 到 stage 的最后一个 block
                    stage = stages[stage_idx]
                    if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
                        last_block = stage.blocks[-1]
                        hooks.append(last_block.register_forward_hook(hook_fn))
                    else:
                        hooks.append(stage.register_forward_hook(hook_fn))
        else:
            # Fallback: 如果找不到 stages，使用 forward_features
            print("[DINOv3 ConvNeXt] Warning: 无法找到 stages，使用 forward_features 作为 fallback")
            feat_dict = self.backbone.forward_features(x)
            if isinstance(feat_dict, dict):
                # 提取 patch tokens
                patch_tokens = feat_dict.get('x_norm_patchtokens', None)
                if patch_tokens is not None:
                    return [patch_tokens]
            return [self.backbone(x)]
        
        # 前向传播（保持梯度流）
        _ = self.backbone(x)
        
        # 移除 hooks
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
            if self.pooling == 'linear':
                # 与 forward 一致：使用 backbone(x)
                features = self.backbone(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])
                return features
            elif self.pooling == 'gap':
                features = self.backbone(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
                return features
            elif self.pooling == 'attention':
                # 返回序列特征
                features = self.backbone(x)  # (B, C, H, W)
                B, C, H, W = features.shape
                features_seq = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
                return features_seq
            else:  # multiscale
                features_list = self._extract_multiscale_features(x)
                # 返回第一个特征（可以修改为返回所有特征）
                if features_list:
                    return features_list[0]
                else:
                    features = self.backbone(x)
                    if len(features.shape) > 2:
                        features = features.mean(dim=[2, 3])
                    return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class DINOv3ViTRegressionModel(nn.Module):
    """基于DINOv3 ViT Large的回归模型
    支持四种 pooling 方案：
    - 'linear': 直接线性层回归（使用 CLS token）
    - 'gap': Global Average Pooling
    - 'attention': Attention Pooling（保留空间特征）
    - 'multiscale': 多尺度特征融合
    """
    
    def __init__(
        self,
        model_path: str,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'linear', 'gap', 'attention', 'multiscale'
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
            module_counts = {'qkv': 0, 'proj': 0, 'mlp': 0}
            
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    # 检查是否匹配指定的模块类型
                    should_replace = False
                    module_type = None
                    
                    if 'qkv' in lora_modules and ('attn' in name and ('qkv' in name or name.endswith('.qkv'))):
                        should_replace = True
                        module_type = 'qkv'
                    elif 'proj' in lora_modules and ('attn' in name and ('proj' in name or name.endswith('.proj'))):
                        should_replace = True
                        module_type = 'proj'
                    elif 'mlp' in lora_modules and ('mlp' in name and ('fc1' in name or 'fc2' in name)):
                        should_replace = True
                        module_type = 'mlp'
                    
                    if should_replace:
                        modules_to_replace[name] = module
                        module_counts[module_type] += 1
            
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
            
            # 打印详细信息
            print(f"[DINOv3 ViT] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层")
            print(f"[DINOv3 ViT] LoRA 模块配置: {lora_modules}")
            print(f"[DINOv3 ViT] LoRA 应用详情: qkv={module_counts['qkv']}, proj={module_counts['proj']}, mlp={module_counts['mlp']}")
                
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
        
        if pooling == 'linear':
            # 方案1: 直接线性层回归（使用 CLS token）
            self.head = nn.Linear(embed_dim, num_outputs)
            print(f"[DINOv3 ViT] 方案: Linear (直接使用 CLS token)")
        elif pooling == 'gap':
            # 方案2: GAP + MLP
            # forward_features 返回 (B, N, C)，需要GAP得到 (B, C)
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[DINOv3 ViT] 方案: GAP + {head_type}")
        elif pooling == 'attention':
            # 方案3: Attention Pooling
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[DINOv3 ViT] 方案: Attention Pooling (heads={num_heads})")
        elif pooling == 'multiscale':
            # 方案4: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[DINOv3 ViT] 方案: Multiscale (layers={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的pooling方案: {pooling}。可选: 'linear', 'gap', 'attention', 'multiscale'")
        
        print(f"[DINOv3 ViT] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            output: Regression output [B, num_outputs]
        """
        if self.pooling == 'multiscale':
            # 方案4: 提取多层特征
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        elif self.pooling == 'linear':
            # 方案1: 直接使用 CLS token
            features = self.backbone.forward_features(x)  # (B, N, C)
            cls_token = features[:, 0, :]  # (B, C) - 第一个 token 是 CLS
            return self.head(cls_token)
        else:
            features = self.backbone.forward_features(x)  # (B, N, C)
            
            if self.pooling == 'gap':
                features = features.mean(dim=1)  # (B, C)
                return self.head(features)
            else:
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
            if self.pooling == 'linear':
                features = self.backbone.forward_features(x)  # (B, N, C)
                features = features[:, 0, :]  # (B, C) - CLS token
            elif self.pooling == 'gap':
                features = self.backbone.forward_features(x)  # (B, N, C)
                features = features.mean(dim=1)  # (B, C)
            elif self.pooling == 'attention':
                features = self.backbone.forward_features(x)  # (B, N, C)
                if features.shape[1] > 5:
                    features = features[:, 5:, :]  # (B, N_patches, C)
            else:  # multiscale
                features_list = self._extract_multiscale_features(x)
                features = features_list[0] if features_list else self.backbone.forward_features(x)
        return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class ResNetRegressionModel(nn.Module):
    """基于ResNet的回归模型
    支持四种 pooling 方案：
    - 'linear': 直接线性层回归
    - 'gap': Global Average Pooling
    - 'attention': Attention Pooling（将空间特征 reshape 成序列）
    - 'multiscale': 多尺度特征融合
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',  # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        pretrained: bool = True,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'linear', 'gap', 'attention', 'multiscale'
        head_type: str = 'linear',
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        head_kwargs: Optional[dict] = None,
        multiscale_layers: Optional[List[int]] = None,  # 用于 multiscale 方案，指定要提取的 stage 索引
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        self.multiscale_layers = multiscale_layers or [2, 3, 4]  # ResNet 有5个 stage (0-4)，默认提取后3个
        
        # 创建ResNet结构
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # 移除分类头
        )
        
        # 获取特征维度
        embed_dim = self.backbone.num_features
        
        # 设置微调策略（ResNet 是 CNN 架构，主要由卷积层组成，不支持 LoRA）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[ResNet] 完全冻结backbone，仅训练回归头")
        elif freeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
            print(f"[ResNet] 冻结层: {freeze_layers}")
        else:
            print(f"[ResNet] 全参数微调")
        
        # 根据 pooling 方案创建不同的头
        head_kwargs = head_kwargs or {}
        
        if pooling == 'linear':
            # 方案1: 直接线性层回归
            self.head = nn.Linear(embed_dim, num_outputs)
            print(f"[ResNet] 方案: Linear (直接线性层回归)")
        elif pooling == 'gap':
            # 方案2: Global Average Pooling
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[ResNet] 方案: GAP + {head_type}")
        elif pooling == 'attention':
            # 方案3: Attention Pooling
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[ResNet] 方案: Attention Pooling (heads={num_heads})")
        elif pooling == 'multiscale':
            # 方案4: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[ResNet] 方案: Multiscale (stages={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的 pooling 方案: {pooling}。可选: 'linear', 'gap', 'attention', 'multiscale'")
        
        print(f"[ResNet] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'multiscale':
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        elif self.pooling == 'linear':
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
            return self.head(features)
        elif self.pooling == 'attention':
            features = self.backbone(x)  # (B, C, H, W)
            features_seq = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
            return self.head(features_seq)
        else:
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
            return self.head(features)
    
    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features_list: List of (B, N, C) tensors from different stages
        """
        features_list = []
        
        def hook_fn(module, input, output):
            # output 是 stage 的输出 (B, C, H, W)
            # 转换为序列格式 (B, H*W, C)
            if len(output.shape) == 4:
                B, C, H, W = output.shape
                output_seq = output.flatten(2).transpose(1, 2)  # (B, H*W, C)
                features_list.append(output_seq)
            else:
                features_list.append(output)
        
        hooks = []
        
        # 注册 hooks 到指定的 stages
        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
            for stage_idx in self.multiscale_layers:
                if 0 <= stage_idx < len(stages):
                    stage = stages[stage_idx]
                    if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
                        last_block = stage.blocks[-1]
                        hooks.append(last_block.register_forward_hook(hook_fn))
                    else:
                        hooks.append(stage.register_forward_hook(hook_fn))
        elif hasattr(self.backbone, 'layer1'):
            # ResNet 使用 layer1, layer2, layer3, layer4
            layers = [self.backbone.layer1, self.backbone.layer2, 
                     self.backbone.layer3, self.backbone.layer4]
            for stage_idx in self.multiscale_layers:
                # 调整索引：multiscale_layers 中的索引对应 layer1-4 (索引1-4)
                # 但我们需要映射到 layers 列表 (索引0-3)
                if 1 <= stage_idx <= 4:
                    layer_idx = stage_idx - 1
                    if 0 <= layer_idx < len(layers):
                        layer = layers[layer_idx]
                        if hasattr(layer, '__iter__'):
                            # 如果是 Sequential 或 ModuleList，hook 到最后一个 block
                            last_block = list(layer.children())[-1]
                            hooks.append(last_block.register_forward_hook(hook_fn))
                        else:
                            hooks.append(layer.register_forward_hook(hook_fn))
        else:
            # Fallback: 使用 forward_features
            print("[ResNet] Warning: 无法找到 stages 或 layers，使用 forward_features 作为 fallback")
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.flatten(2).transpose(1, 2)
            return [features]
        
        # 前向传播（保持梯度流）
        _ = self.backbone(x)
        
        # 移除 hooks
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
            if self.pooling == 'linear':
                features = self.backbone(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])
                return features
            elif self.pooling == 'gap':
                features = self.backbone(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
                return features
            elif self.pooling == 'attention':
                features = self.backbone(x)  # (B, C, H, W)
                B, C, H, W = features.shape
                features_seq = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
                return features_seq
            else:  # multiscale
                features_list = self._extract_multiscale_features(x)
                if features_list:
                    return features_list[0]
                else:
                    features = self.backbone(x)
                    if len(features.shape) > 2:
                        features = features.mean(dim=[2, 3])
                    return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class SwinTRegressionModel(nn.Module):
    """基于Swin Transformer的回归模型
    支持四种 pooling 方案：
    - 'linear': 直接线性层回归
    - 'gap': Global Average Pooling
    - 'attention': Attention Pooling（保留空间特征）
    - 'multiscale': 多尺度特征融合
    """
    
    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',  # 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224'
        pretrained: bool = True,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'linear', 'gap', 'attention', 'multiscale'
        head_type: str = 'mlp',
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        head_kwargs: Optional[dict] = None,
        multiscale_layers: Optional[List[int]] = None,  # 用于 multiscale 方案，指定要提取的 stage 索引
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.pooling = pooling
        self.multiscale_layers = multiscale_layers or [1, 2, 3]  # Swin-T 有4个 stage，默认提取后3个
        
        # 创建Swin Transformer结构
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            dynamic_img_size=True  # 支持动态输入尺寸
        )
        
        # 获取特征维度
        embed_dim = self.backbone.num_features
        
        # 设置微调策略
        if use_lora:
            # Swin Transformer 的 LoRA 主要应用于 attention 和 MLP 中的 Linear 层
            lora_modules = head_kwargs.get('lora_modules', None) if head_kwargs else None
            if lora_modules is None:
                lora_modules = ['qkv', 'proj', 'mlp']  # 默认微调所有模块
            elif isinstance(lora_modules, str):
                lora_modules = [lora_modules]
            
            modules_to_replace = {}
            module_counts = {'qkv': 0, 'proj': 0, 'mlp': 0}
            
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    should_replace = False
                    module_type = None
                    
                    if 'qkv' in lora_modules and ('attn' in name and ('qkv' in name or name.endswith('.qkv'))):
                        should_replace = True
                        module_type = 'qkv'
                    elif 'proj' in lora_modules and ('attn' in name and ('proj' in name or name.endswith('.proj'))):
                        should_replace = True
                        module_type = 'proj'
                    elif 'mlp' in lora_modules and ('mlp' in name and ('fc1' in name or 'fc2' in name)):
                        should_replace = True
                        module_type = 'mlp'
                    
                    if should_replace:
                        modules_to_replace[name] = module
                        module_counts[module_type] += 1
            
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
            
            print(f"[Swin-T] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层")
            print(f"[Swin-T] LoRA 模块配置: {lora_modules}")
            print(f"[Swin-T] LoRA 应用详情: qkv={module_counts['qkv']}, proj={module_counts['proj']}, mlp={module_counts['mlp']}")
                
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[Swin-T] 完全冻结backbone，仅训练回归头")
            
        elif freeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
            print(f"[Swin-T] 冻结层: {freeze_layers}")
        else:
            print(f"[Swin-T] 全参数微调")
        
        # 根据pooling方案创建不同的头
        head_kwargs = head_kwargs or {}
        
        if pooling == 'linear':
            # 方案1: 直接线性层回归
            self.head = nn.Linear(embed_dim, num_outputs)
            print(f"[Swin-T] 方案: Linear (直接线性层回归)")
        elif pooling == 'gap':
            # 方案2: GAP + MLP
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[Swin-T] 方案: GAP + {head_type}")
        elif pooling == 'attention':
            # 方案3: Attention Pooling
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[Swin-T] 方案: Attention Pooling (heads={num_heads})")
        elif pooling == 'multiscale':
            # 方案4: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[Swin-T] 方案: Multiscale (stages={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的pooling方案: {pooling}。可选: 'linear', 'gap', 'attention', 'multiscale'")
        
        print(f"[Swin-T] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            output: Regression output [B, num_outputs]
        """
        if self.pooling == 'multiscale':
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        elif self.pooling == 'linear':
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
            return self.head(features)
        else:
            # 方案2和3: 使用forward_features获取空间特征
            features = self.backbone.forward_features(x)  # (B, N, C) 或 (B, C, H, W)
            
            if self.pooling == 'gap':
                # 方案2: Global Average Pooling
                if len(features.shape) == 3:
                    # (B, N, C) -> (B, C)
                    features = features.mean(dim=1)
                elif len(features.shape) == 4:
                    # (B, C, H, W) -> (B, C)
                    features = features.mean(dim=[2, 3])
                return self.head(features)
            else:
                # 方案3: Attention Pooling (保留空间特征)
                if len(features.shape) == 4:
                    # (B, C, H, W) -> (B, H*W, C)
                    B, C, H, W = features.shape
                    features = features.flatten(2).transpose(1, 2)
                # features 已经是 (B, N, C)
                return self.head(features)
    
    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features_list: List of (B, N, C) tensors from different stages
        """
        features_list = []
        
        def hook_fn(module, input, output):
            # output 可能是 (B, N, C) 或 (B, C, H, W)
            if len(output.shape) == 3:
                # 已经是序列格式
                features_list.append(output)
            elif len(output.shape) == 4:
                # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = output.shape
                output_seq = output.flatten(2).transpose(1, 2)
                features_list.append(output_seq)
            else:
                features_list.append(output)
        
        hooks = []
        
        # 注册 hooks 到指定的 stages
        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
            for stage_idx in self.multiscale_layers:
                if 0 <= stage_idx < len(stages):
                    stage = stages[stage_idx]
                    if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
                        last_block = stage.blocks[-1]
                        hooks.append(last_block.register_forward_hook(hook_fn))
                    else:
                        hooks.append(stage.register_forward_hook(hook_fn))
        else:
            # Fallback: 使用 forward_features
            print("[Swin-T] Warning: 无法找到 stages，使用 forward_features 作为 fallback")
            features = self.backbone.forward_features(x)
            if len(features.shape) == 4:
                features = features.flatten(2).transpose(1, 2)
            return [features]
        
        # 前向传播（保持梯度流）
        _ = self.backbone.forward_features(x)
        
        # 移除 hooks
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
            if self.pooling == 'linear':
                features = self.backbone(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])
                return features
            elif self.pooling == 'gap':
                features = self.backbone.forward_features(x)
                if len(features.shape) == 3:
                    features = features.mean(dim=1)
                elif len(features.shape) == 4:
                    features = features.mean(dim=[2, 3])
                return features
            elif self.pooling == 'attention':
                features = self.backbone.forward_features(x)
                if len(features.shape) == 4:
                    B, C, H, W = features.shape
                    features = features.flatten(2).transpose(1, 2)
                return features
            else:  # multiscale
                features_list = self._extract_multiscale_features(x)
                if features_list:
                    return features_list[0]
                else:
                    features = self.backbone.forward_features(x)
                    if len(features.shape) == 4:
                        features = features.flatten(2).transpose(1, 2)
                    return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class ViTRegressionModel(nn.Module):
    """基于普通ViT（非DINOv3）的回归模型
    支持四种 pooling 方案：
    - 'linear': 直接线性层回归（使用 CLS token）
    - 'gap': Global Average Pooling
    - 'attention': Attention Pooling（保留空间特征）
    - 'multiscale': 多尺度特征融合
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',  # 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224' 等
        pretrained: bool = True,
        num_outputs: int = 1,
        pooling: str = 'gap',  # 'linear', 'gap', 'attention', 'multiscale'
        head_type: str = 'mlp',
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        head_kwargs: Optional[dict] = None,
        multiscale_layers: Optional[List[int]] = None,  # 用于 multiscale 方案，指定要提取的层索引
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.pooling = pooling
        self.multiscale_layers = multiscale_layers or [6, 12, 18]  # 默认提取3个不同深度的层（根据模型深度调整）
        
        # 创建ViT结构
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            dynamic_img_size=True  # 支持动态输入尺寸
        )
        
        # 获取特征维度
        embed_dim = self.backbone.num_features
        
        # 设置微调策略
        if use_lora:
            # ViT 的 LoRA 主要应用于 attention 和 MLP 中的 Linear 层
            lora_modules = head_kwargs.get('lora_modules', None) if head_kwargs else None
            if lora_modules is None:
                lora_modules = ['qkv', 'proj', 'mlp']  # 默认微调所有模块
            elif isinstance(lora_modules, str):
                lora_modules = [lora_modules]
            
            modules_to_replace = {}
            module_counts = {'qkv': 0, 'proj': 0, 'mlp': 0}
            
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    should_replace = False
                    module_type = None
                    
                    if 'qkv' in lora_modules and ('attn' in name and ('qkv' in name or name.endswith('.qkv'))):
                        should_replace = True
                        module_type = 'qkv'
                    elif 'proj' in lora_modules and ('attn' in name and ('proj' in name or name.endswith('.proj'))):
                        should_replace = True
                        module_type = 'proj'
                    elif 'mlp' in lora_modules and ('mlp' in name and ('fc1' in name or 'fc2' in name)):
                        should_replace = True
                        module_type = 'mlp'
                    
                    if should_replace:
                        modules_to_replace[name] = module
                        module_counts[module_type] += 1
            
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
            
            print(f"[ViT] LoRA 成功应用到 {len(modules_to_replace)} 个 Linear 层")
            print(f"[ViT] LoRA 模块配置: {lora_modules}")
            print(f"[ViT] LoRA 应用详情: qkv={module_counts['qkv']}, proj={module_counts['proj']}, mlp={module_counts['mlp']}")
                
        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[ViT] 完全冻结backbone，仅训练回归头")
            
        elif freeze_layers:
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
            print(f"[ViT] 冻结层: {freeze_layers}")
        else:
            print(f"[ViT] 全参数微调")
        
        # 根据pooling方案创建不同的头
        head_kwargs = head_kwargs or {}
        
        if pooling == 'linear':
            # 方案1: 直接线性层回归（使用 CLS token）
            self.head = nn.Linear(embed_dim, num_outputs)
            print(f"[ViT] 方案: Linear (直接使用 CLS token)")
        elif pooling == 'gap':
            # 方案2: GAP + MLP
            linear_in_dim = embed_dim
            self.head = create_regression_head(
                head_type, linear_in_dim, num_outputs, dropout, **head_kwargs
            )
            print(f"[ViT] 方案: GAP + {head_type}")
        elif pooling == 'attention':
            # 方案3: Attention Pooling
            num_heads = head_kwargs.get('num_heads', 8)
            hidden_dim = head_kwargs.get('hidden_dim', None)
            self.head = AttentionPoolingHead(
                embed_dim, num_outputs, dropout, num_heads, hidden_dim
            )
            print(f"[ViT] 方案: Attention Pooling (heads={num_heads})")
        elif pooling == 'multiscale':
            # 方案4: 多尺度特征融合
            num_scales = len(self.multiscale_layers)
            hidden_dims = head_kwargs.get('hidden_dims', [embed_dim * 2, embed_dim])
            self.head = MultiScaleFusionHead(
                embed_dim, num_outputs, dropout, num_scales, hidden_dims
            )
            print(f"[ViT] 方案: Multiscale (layers={self.multiscale_layers})")
        else:
            raise ValueError(f"不支持的pooling方案: {pooling}。可选: 'linear', 'gap', 'attention', 'multiscale'")
        
        print(f"[ViT] 特征维度: {embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            output: Regression output [B, num_outputs]
        """
        if self.pooling == 'multiscale':
            features_list = self._extract_multiscale_features(x)
            return self.head(features_list)
        elif self.pooling == 'linear':
            # 方案1: 直接使用 CLS token
            features = self.backbone.forward_features(x)  # (B, N, C)
            cls_token = features[:, 0, :]  # (B, C) - 第一个 token 是 CLS
            return self.head(cls_token)
        else:
            # 方案2和3: 使用forward_features获取空间特征
            features = self.backbone.forward_features(x)  # (B, N, C)
            
            if self.pooling == 'gap':
                # 方案2: Global Average Pooling
                # 取所有tokens的平均（包括CLS和patches）
                features = features.mean(dim=1)  # (B, C)
                return self.head(features)
            else:
                # 方案3: Attention Pooling (保留空间特征)
                # features已经是 (B, N, C)，直接传给AttentionPoolingHead
                # 普通ViT结构: CLS (1) + Patches (N)
                # 跳过CLS token，只使用patch tokens
                if features.shape[1] > 1:
                    patch_features = features[:, 1:, :]  # (B, N_patches, C)
                else:
                    # 如果tokens数量不足，使用所有tokens
                    patch_features = features
                return self.head(patch_features)
    
    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            features_list: List of (B, N, C) tensors from different layers
        """
        features_list = []
        
        def hook_fn(module, input, output):
            # output 是 Block 的输出 (B, N, C)
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
            print("[ViT] Warning: 无法找到blocks，使用forward_features作为fallback")
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
            if self.pooling == 'linear':
                features = self.backbone.forward_features(x)  # (B, N, C)
                features = features[:, 0, :]  # (B, C) - CLS token
            elif self.pooling == 'gap':
                features = self.backbone.forward_features(x)  # (B, N, C)
                features = features.mean(dim=1)  # (B, C)
            elif self.pooling == 'attention':
                features = self.backbone.forward_features(x)  # (B, N, C)
                if features.shape[1] > 1:
                    features = features[:, 1:, :]  # (B, N_patches, C)
            else:  # multiscale
                features_list = self._extract_multiscale_features(x)
                # 返回第一个特征（可以修改为返回所有特征）
                features = features_list[0] if features_list else self.backbone.forward_features(x)
        return features
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total