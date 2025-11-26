from typing import Optional, List
import torch.nn as nn
from .model import DINOv3RegressionModel, DINOv3ViTRegressionModel


def create_dinov3_model(
    backbone_type: str = 'convnext',  # 'convnext' 或 'vit'
    model_path: str = '',
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
    lora_modules: Optional[List[str]] = None,  # 仅 ViT: ['qkv', 'proj', 'mlp'] 或其组合
    head_kwargs: Optional[dict] = None,
    multiscale_layers: Optional[List[int]] = None,  # multiscale 方案的层索引
) -> nn.Module:
    """
    Args:
        backbone_type: 'convnext' 或 'vit'
        model_path: 模型权重路径
        num_outputs: 输出维度
        pooling: pooling 方式 ('linear', 'gap', 'attention', 'multiscale')
        head_type: 回归头类型 (仅用于 gap 方案)
        dropout: dropout 率
        freeze_backbone: 是否冻结 backbone
        freeze_layers: 要冻结的层列表
        use_lora: 是否使用 LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_modules: LoRA 模块选择 (仅 ViT): ['qkv', 'proj', 'mlp'] 或其组合
        head_kwargs: 回归头的额外参数
        multiscale_layers: 多尺度融合的层索引
            - ConvNeXt: stage 索引，例如 [1, 2, 3] (默认)
            - ViT: block 索引，例如 [6, 12, 18, 23] (默认)
    
    Returns:
        DINOv3RegressionModel 或 DINOv3ViTRegressionModel 实例
    """
    # 准备 head_kwargs
    if head_kwargs is None:
        head_kwargs = {}
    
    # 对于 ViT，将 lora_modules 添加到 head_kwargs
    if backbone_type == 'vit' and lora_modules is not None:
        head_kwargs = head_kwargs.copy()
        head_kwargs['lora_modules'] = lora_modules
    
    if backbone_type == 'vit':
        return DINOv3ViTRegressionModel(
            model_path=model_path,
            num_outputs=num_outputs,
            pooling=pooling,
            head_type=head_type,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            head_kwargs=head_kwargs,
            multiscale_layers=multiscale_layers,
        )
    else:
        return DINOv3RegressionModel(
            model_path=model_path,
            num_outputs=num_outputs,
            pooling=pooling,  
            head_type=head_type,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            head_kwargs=head_kwargs,
        )

