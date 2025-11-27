from typing import Optional, List
import torch.nn as nn
from .model import DINOv3RegressionModel, DINOv3ViTRegressionModel, ResNetRegressionModel, SwinTRegressionModel, ViTRegressionModel

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


def create_dinov3_model(
    backbone_type: str = 'convnext',  # 'convnext', 'dinov3-vit', 'vit', 'resnet', 'swin-t'
    model_path: str = '',
    model_name: str = 'resnet50',  # 用于 vit, resnet 和 swin-t
    pretrained: bool = True,  # 用于 vit, resnet 和 swin-t
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
    lora_modules: Optional[List[str]] = None,  # 仅 ViT 和 Swin-T: ['qkv', 'proj', 'mlp'] 或其组合
    head_kwargs: Optional[dict] = None,
    multiscale_layers: Optional[List[int]] = None,  # multiscale 方案的层索引
) -> nn.Module:
    """
    Args:
        backbone_type: 
            - 'convnext': DINOv3 ConvNeXt-Large
            - 'dinov3-vit': DINOv3 ViT Large (需要 model_path)
            - 'vit': 普通 ViT (使用 timm 预训练，如 vit_base_patch16_224)
            - 'resnet': ResNet (使用 timm 预训练)
            - 'swin-t': Swin Transformer (使用 timm 预训练)
        model_path: 模型权重路径 (仅用于 DINOv3 模型: 'convnext' 和 'dinov3-vit')
        model_name: 模型名称 (用于 vit, resnet 和 swin-t)
            - vit: 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224' 等
            - resnet: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            - swin-t: 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224'
        pretrained: 是否使用预训练权重 (用于 vit, resnet 和 swin-t)
        num_outputs: 输出维度
        pooling: pooling 方式 ('linear', 'gap', 'attention', 'multiscale')
        head_type: 回归头类型 (仅用于 gap 方案)
        dropout: dropout 率
        freeze_backbone: 是否冻结 backbone
        freeze_layers: 要冻结的层列表
        use_lora: 是否使用 LoRA (支持: 'convnext', 'dinov3-vit', 'vit', 'swin-t'；不支持: 'resnet')
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_modules: LoRA 模块选择 (仅 ViT 和 Swin-T): ['qkv', 'proj', 'mlp'] 或其组合
        head_kwargs: 回归头的额外参数
        multiscale_layers: 多尺度融合的层索引
            - ConvNeXt: stage 索引，例如 [1, 2, 3] (默认)
            - DINOv3 ViT: block 索引，例如 [6, 12, 18, 23] (默认)
            - ViT: block 索引，例如 [6, 12, 18] (默认)
            - ResNet: layer 索引，例如 [2, 3, 4] (默认)
            - Swin-T: stage 索引，例如 [1, 2, 3] (默认)
    
    Returns:
        相应的回归模型实例
    """
    # 准备 head_kwargs
    if head_kwargs is None:
        head_kwargs = {}
    else:
        # 如果 head_kwargs 是 OmegaConf 对象，转换为普通字典
        if OMEGACONF_AVAILABLE and isinstance(head_kwargs, OmegaConf):
            head_kwargs = OmegaConf.to_container(head_kwargs, resolve=True)
        # 确保是字典类型
        if not isinstance(head_kwargs, dict):
            head_kwargs = dict(head_kwargs)
        else:
            head_kwargs = head_kwargs.copy()
    
    # 转换可能为 OmegaConf 对象的参数
    def _convert_omegaconf_to_python(obj):
        """将 OmegaConf 对象转换为 Python 原生类型"""
        if obj is None:
            return None
        if OMEGACONF_AVAILABLE:
            try:
                from omegaconf import ListConfig, DictConfig
                if isinstance(obj, (ListConfig, DictConfig)):
                    return OmegaConf.to_container(obj, resolve=True)
            except ImportError:
                pass
        return obj
    
    # 转换参数
    lora_modules = _convert_omegaconf_to_python(lora_modules)
    multiscale_layers = _convert_omegaconf_to_python(multiscale_layers)
    freeze_layers = _convert_omegaconf_to_python(freeze_layers)
    
    # 对于 ViT 和 Swin-T，将 lora_modules 添加到 head_kwargs
    if backbone_type in ['vit', 'dinov3-vit', 'swin-t'] and lora_modules is not None:
        head_kwargs['lora_modules'] = lora_modules
    
    if backbone_type == 'dinov3-vit':
        # DINOv3 ViT Large
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
    elif backbone_type == 'vit':
        # 普通 ViT (使用 timm 预训练)
        return ViTRegressionModel(
            model_name=model_name,
            pretrained=pretrained,
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
    elif backbone_type == 'resnet':
        # ResNet 是 CNN 架构，不支持 LoRA
        return ResNetRegressionModel(
            model_name=model_name,
            pretrained=pretrained,
            num_outputs=num_outputs,
            pooling=pooling,
            head_type=head_type,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers,
            head_kwargs=head_kwargs,
            multiscale_layers=multiscale_layers,
        )
    elif backbone_type == 'swin-t':
        return SwinTRegressionModel(
            model_name=model_name,
            pretrained=pretrained,
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
    else:  # 'convnext' 或其他
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

