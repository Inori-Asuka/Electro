"""
可视化 DINOv3 ViT 的 Attention Maps
参考: E:\Code\Proj\Elect\EndoVGGT\visualization\vis_demo\vis_attention.ipynb
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import argparse
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import DINOv3ViTRegressionModel


def get_dino_attention_map(model, img_tensor, device, dtype=torch.float32):
    """
    获取DINOv3 ViT最后一层的Attention Map。
    
    Args:
        model: DINOv3ViTRegressionModel 或 timm ViT 模型
        img_tensor: 输入图像 tensor，shape: (B, 3, H, W)
        device: 设备
        dtype: 数据类型
    
    Returns:
        attentions: shape (B, num_heads, N_tokens, N_tokens)
    """
    attentions = []

    def hook_fn(module, input, output):
        x = input[0]  # (B, N, C)
        B, N, C = x.shape
        
        # 获取attention模块的属性
        if hasattr(module, 'qkv'):
            qkv = module.qkv(x)
            if qkv.dim() == 3:
                # timm ViT的标准格式
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                q, k, v = torch.unbind(qkv, 2)  # 每个都是 (B, N, num_heads, head_dim)
                q, k = [t.transpose(1, 2) for t in [q, k]]  # 每个都是 (B, num_heads, N, head_dim)
            else:
                # 可能已经是reshape后的格式
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
        else:
            # 如果没有qkv，尝试其他方式
            return
        
        # 计算attention scores
        if hasattr(module, 'scale'):
            scale = module.scale
        else:
            scale = (C // module.num_heads) ** -0.5
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        attentions.append(attn.detach().cpu())

    # 获取backbone（如果是DINOv3ViTRegressionModel）或直接使用模型
    if isinstance(model, DINOv3ViTRegressionModel):
        backbone = model.backbone
    else:
        backbone = model
    
    # 定位最后一层 Attention
    if hasattr(backbone, 'blocks'):
        target_layer = backbone.blocks[-1].attn
    elif hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
        target_layer = backbone.blocks[-1].attn
    else:
        print("Warning: 无法找到blocks，尝试查找attn模块")
        # Fallback: 查找所有attn模块，取最后一个
        attn_modules = []
        for name, module in backbone.named_modules():
            if 'attn' in name.lower() and isinstance(module, nn.Module):
                attn_modules.append((name, module))
        if attn_modules:
            target_layer = attn_modules[-1][1]
        else:
            raise ValueError("无法找到attention层")

    handle = target_layer.register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        if isinstance(model, DINOv3ViTRegressionModel):
            # 使用backbone的forward_features
            _ = model.backbone.forward_features(img_tensor)
        else:
            # 直接使用模型的forward_features
            if hasattr(model, 'forward_features'):
                _ = model.forward_features(img_tensor)
            else:
                _ = model(img_tensor)

    handle.remove()
    return attentions[0] if attentions else None


def visualize_dino_attention(
    image_paths: List[str],
    model,
    model_type: str = 'dinov3',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    threshold: Optional[float] = None,
    target_size: int = 512,
    patch_size: int = 16,
    save_path: Optional[str] = None
):
    """
    可视化DINOv3的attention maps。
    
    Args:
        image_paths: 图像路径列表
        model: DINOv3ViTRegressionModel 或 timm ViT 模型
        model_type: 'dinov2' 或 'dinov3'
        device: 设备
        dtype: 数据类型
        threshold: 可选的阈值，用于mask可视化
        target_size: 目标图像尺寸
        patch_size: Patch大小
        save_path: 保存路径（可选）
    
    Returns:
        fig: matplotlib figure
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append(img_tensor)
    
    images_batch = torch.cat(images, dim=0)  # (B, 3, H, W)
    
    print(f"Processing {len(image_paths)} images for {model_type} attention...")
    
    # 获取attention maps
    attn_maps = get_dino_attention_map(model, images_batch, device, dtype)  # (B, num_heads, N_tokens, N_tokens)
    
    if attn_maps is None:
        print("Failed to extract attention maps")
        return None
    
    B, num_heads, N_tokens, _ = attn_maps.shape
    
    # 计算register tokens数量
    expected_patches = (target_size // patch_size) ** 2
    num_registers = N_tokens - 1 - expected_patches  # 减去CLS token
    
    # 确定保存目录
    if save_path:
        if os.path.isdir(save_path):
            save_dir = save_path
        else:
            save_dir = os.path.dirname(save_path)
            if not save_dir:
                save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    
    # 为每张图片单独创建和保存可视化
    saved_paths = []
    for frame_idx in range(len(image_paths)):
        if frame_idx >= B:
            break
            
        # 获取该帧的attention: (num_heads, N_tokens, N_tokens)
        frame_attn = attn_maps[frame_idx]
        
        # 提取CLS token对patch tokens的attention: (num_heads, N_patches)
        # CLS token是第0个，跳过register tokens (1:1+num_registers)
        cls_attn = frame_attn[:, 0, 1 + num_registers:]  # (num_heads, N_patches)
        
        # 对所有head取平均
        mean_attn = cls_attn.mean(dim=0)  # (N_patches,)
        
        # Reshape为2D feature map
        N_patches = mean_attn.shape[0]
        feat_dim = int(np.sqrt(N_patches))
        if feat_dim * feat_dim != N_patches:
            # 如果不是完全平方数，尝试其他形状
            print(f"Warning: N_patches={N_patches} is not a perfect square, using approximate shape")
            feat_dim = int(np.sqrt(N_patches))
        
        attn_2d = mean_attn.reshape(feat_dim, feat_dim).numpy()
        
        # 读取原始图像
        img_path = image_paths[frame_idx]
        original_img = np.array(Image.open(img_path).convert('RGB'))
        H_orig, W_orig = original_img.shape[:2]
        
        # 上采样attention到原图尺寸
        attn_resized = cv2.resize(attn_2d, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
        
        # 归一化
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # 为每张图片创建单独的figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原图（左列）
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image\n{os.path.basename(img_path)}", fontsize=12)
        axes[0].axis('off')
        
        # 显示可视化图（右列）
        if threshold is not None:
            # 使用阈值创建mask
            mask = attn_resized > threshold
            axes[1].imshow(original_img)
            axes[1].imshow(mask.astype(float), cmap='jet', alpha=0.5)
            axes[1].set_title(f"Attention Map (threshold={threshold:.2f})", fontsize=12)
        else:
            # 显示纯attention map
            axes[1].imshow(attn_resized, cmap='jet')
            axes[1].set_title("Attention Heatmap", fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # 保存每张图片
        if save_dir:
            # 生成文件名
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(save_dir, f'dino_attention_{img_basename}_{frame_idx}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    # 返回第一张图片的figure（用于显示，如果不需要可以返回None）
    return None if not saved_paths else saved_paths


def get_dino_multiscale_features(model, img_tensor, dtype, multiscale_layers: List[int] = [6, 12, 18, 23]):
    """
    获取DINO模型不同深度的特征图。
    选取模型指定层的 Block 输出。
    
    Args:
        model: DINOv3ViTRegressionModel 或 timm ViT 模型
        img_tensor: 输入图像 tensor (B, 3, H, W)
        dtype: 数据类型
        multiscale_layers: 要提取的层索引列表
    
    Returns:
        features_list: list of tensors, 每个 tensor shape: (B, N_tokens, C)
    """
    features_list = []
    
    def hook_fn(module, input, output):
        # output 是 Block 的输出 (B, N, C)
        features_list.append(output.detach().cpu())

    hooks = []
    
    # 获取backbone
    if isinstance(model, DINOv3ViTRegressionModel):
        backbone = model.backbone
    else:
        backbone = model
    
    # 自动确定要 Hook 的层索引
    if hasattr(backbone, 'blocks'):
        n_blocks = len(backbone.blocks)
        # 确保层索引在有效范围内
        valid_layers = [idx for idx in multiscale_layers if 0 <= idx < n_blocks]
        if not valid_layers:
            # Fallback: 均匀选择4层
            valid_layers = [int(n_blocks * (i+1) / 4) - 1 for i in range(4)]
        target_layers = [backbone.blocks[i] for i in valid_layers]
    else:
        # Fallback: 如果找不到 blocks，尝试获取最后一层
        print("Warning: 无法找到blocks，使用forward_features作为fallback")
        features = backbone.forward_features(img_tensor) if hasattr(backbone, 'forward_features') else backbone(img_tensor)
        return [features.detach().cpu()]

    # 注册 Hooks
    for layer in target_layers:
        if layer is not None:
            hooks.append(layer.register_forward_hook(hook_fn))

    # 前向传播
    with torch.no_grad():
        if isinstance(model, DINOv3ViTRegressionModel):
            _ = model.backbone.forward_features(img_tensor)
        else:
            if hasattr(model, 'forward_features'):
                _ = model.forward_features(img_tensor)
            else:
                _ = model(img_tensor)

    # 移除 Hooks
    for h in hooks:
        h.remove()
        
    return features_list


def visualize_dino_multiscale_features(
    image_paths: List[str],
    model,
    model_type: str = 'dinov3',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    target_size: int = 512,
    patch_size: int = 16,
    multiscale_layers: List[int] = [6, 12, 18, 23],
    save_path: Optional[str] = None
):
    """
    可视化DINO的多尺度PCA特征图。
    
    Args:
        image_paths: 图像路径列表
        model: DINOv3ViTRegressionModel 或 timm ViT 模型
        model_type: 'dinov2' 或 'dinov3'
        device: 设备
        dtype: 数据类型
        target_size: 目标图像尺寸
        patch_size: Patch大小
        multiscale_layers: 要可视化的层索引列表
        save_path: 保存路径（可选）
    """
    from sklearn.decomposition import PCA
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append(img_tensor)
    
    if not images:
        print("No valid images found.")
        return None

    images_batch = torch.cat(images, dim=0)
    
    print(f"Extracting multiscale features for {len(image_paths)} images...")
    
    # 获取多层特征: List[Tensor(B, N, C)]
    multiscale_feats = get_dino_multiscale_features(model, images_batch, dtype, multiscale_layers)
    
    if not multiscale_feats:
        print("Failed to extract features.")
        return None

    n_layers = len(multiscale_feats)
    n_frames = len(image_paths)
    
    # 确定保存目录
    if save_path:
        if os.path.isdir(save_path):
            save_dir = save_path
        else:
            save_dir = os.path.dirname(save_path)
            if not save_dir:
                save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    
    # 计算 Token 信息 (用于去除 CLS/Registers)
    _, N_tokens, _ = multiscale_feats[0].shape
    expected_patches = (target_size // patch_size) ** 2
    num_registers = N_tokens - 1 - expected_patches

    saved_paths = []
    # 为每张图片单独创建和保存可视化
    for img_idx in range(n_frames):
        # 为每张图片创建单独的figure
        fig, axes = plt.subplots(1, n_layers + 1, figsize=((n_layers + 1) * 4, 4))
        if n_layers == 0:
            axes = [axes]
        
        # --- A. 显示原图 ---
        img_path = image_paths[img_idx]
        original_img = np.array(Image.open(img_path).convert('RGB'))
        
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original\n{os.path.basename(img_path)}", fontsize=10)
        axes[0].axis('off')

        # --- B. 处理每一层的特征并显示 PCA ---
        for layer_idx, feats in enumerate(multiscale_feats):
            # feats: (B, N, C) -> 取当前图片 (N, C)
            feat = feats[img_idx] 
            
            # 去除 Special Tokens (CLS + Registers)
            # 只保留 Patch Tokens: (N_patches, C)
            patch_feat = feat[1 + num_registers:, :]
            
            # --- PCA 降维 (C -> 3) ---
            x = patch_feat.float().numpy()
            
            # 训练 PCA
            pca = PCA(n_components=3)
            pca_feat = pca.fit_transform(x)  # (N_patches, 3)
            
            # 归一化到 [0, 1] 用于显示 RGB
            feat_min = pca_feat.min(axis=0)
            feat_max = pca_feat.max(axis=0)
            pca_feat = (pca_feat - feat_min) / (feat_max - feat_min + 1e-6)
            
            # --- 重塑回图像尺寸 ---
            N_patches = pca_feat.shape[0]
            grid_dim = int(np.sqrt(N_patches))
            
            if grid_dim * grid_dim != N_patches:
                print(f"Warning: Layer {layer_idx} patches {N_patches} not square.")
                continue
                
            # (N_patches, 3) -> (H_grid, W_grid, 3)
            feat_map = pca_feat.reshape(grid_dim, grid_dim, 3)
            
            # 上采样到原图尺寸
            H_orig, W_orig = original_img.shape[:2]
            feat_resized = cv2.resize(feat_map, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            
            # 显示
            ax = axes[layer_idx + 1]
            ax.imshow(feat_resized)
            ax.set_title(f"Layer {multiscale_layers[layer_idx]}\nSemantic PCA", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        
        # 保存每张图片
        if save_dir:
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(save_dir, f'dino_multiscale_{img_basename}_{img_idx}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")
        
        plt.close()
    
    return saved_paths if saved_paths else None


def main():
    parser = argparse.ArgumentParser(description="可视化 DINOv3 ViT Attention Maps")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='DINOv3模型权重路径 (.safetensors)')
    parser.add_argument('--image_paths', type=str, nargs='+', default=None,
                       help='要可视化的图像路径列表（如果提供，将使用这些图像）')
    parser.add_argument('--config_path', type=str, default=None,
                       help='配置文件路径（用于从数据集选择样本）')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='从数据集中随机选择的样本数量（仅当使用 --config_path 时有效）')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='输出目录')
    parser.add_argument('--model_type', type=str, default='dinov3', choices=['dinov2', 'dinov3'],
                       help='模型类型')
    parser.add_argument('--target_size', type=int, default=512,
                       help='目标图像尺寸')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch大小')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Attention阈值（可选）')
    parser.add_argument('--mode', type=str, default='both', choices=['attention', 'multiscale', 'both'],
                       help='可视化模式')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查参数组合
    if args.image_paths is None and args.config_path is None:
        parser.error("必须提供 --image_paths 或 --config_path 参数之一")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # 检查模型文件类型
    model_path = args.checkpoint_path
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
    
    model_ext = os.path.splitext(model_path)[1].lower()
    
    # 如果是 .pth 文件，使用训练后的模型（需要配置文件）
    if model_ext == '.pth':
        if args.config_path is None:
            print("Error: When loading .pth checkpoint, --config_path is required")
            return
        
        from omegaconf import OmegaConf
        from model.model_factory import create_dinov3_model
        
        cfg = OmegaConf.load(args.config_path)
        
        # 禁用 LoRA 以避免兼容性问题
        use_lora = False
        
        # 创建模型
        model = create_dinov3_model(
            backbone_type=cfg.dinov3.get('backbone_type', 'vit'),
            model_path=cfg.dinov3.model_path,  # 原始 DINOv3 权重路径
            num_outputs=1,
            pooling=cfg.dinov3.get('pooling', 'gap'),
            head_type=cfg.dinov3.head_type,
            dropout=cfg.training.dropout,
            freeze_backbone=cfg.dinov3.freeze_backbone,
            freeze_layers=cfg.dinov3.get('freeze_layers', None),
            use_lora=use_lora,
            lora_r=cfg.dinov3.get('lora_r', 16),
            lora_alpha=cfg.dinov3.get('lora_alpha', 32),
            lora_dropout=cfg.dinov3.get('lora_dropout', 0.1),
            lora_modules=cfg.dinov3.get('lora_modules', None),
            head_kwargs=cfg.dinov3.get('head_kwargs', {}),
            multiscale_layers=cfg.dinov3.get('multiscale_layers', None),
        ).to(device)
        
        # 加载训练后的权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded trained model from: {model_path}")
        # 对于可视化，我们只需要 backbone
        if hasattr(model, 'backbone'):
            model = model.backbone
        else:
            print("Warning: Model does not have 'backbone' attribute, using model directly")
    
    # 如果是 .safetensors 文件，加载原始 DINOv3 权重
    elif model_ext == '.safetensors':
        from safetensors.torch import load_file
        import timm
        
        model_name = "vit_large_patch16_dinov3.lvd1689m" if args.model_type == 'dinov3' else "vit_large_patch14_dinov2"
        model = timm.create_model(model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        
        state_dict = load_file(model_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("backbone.", "").replace("model.", "")
            cleaned_state_dict[new_k] = v
            
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"{args.model_type} Loading Weights Result:")
        print(f" - Missing keys: {len(missing_keys)}")
        print(f" - Unexpected keys: {len(unexpected_keys)}")
        
        model = model.to(device)
    
    else:
        print(f"Error: Unsupported model file format: {model_ext}")
        print("Supported formats: .pth (PyTorch checkpoint), .safetensors (DINOv3 weights)")
        return
    
    model.eval()
    print(f"Model loaded successfully!")
    
    # 确定要可视化的图像路径
    image_paths = args.image_paths
    
    # 如果提供了 image_paths，直接使用；否则从数据集中选择样本
    if image_paths is None or len(image_paths) == 0:
        # 如果没有提供 image_paths，需要从数据集选择
        if args.config_path is None:
            print("Error: 必须提供 --image_paths 或 --config_path 参数之一")
            return
        
        from omegaconf import OmegaConf
        from dataset import TEMRegressionDataset, get_transforms
        
        # 加载配置文件（如果之前没有加载过）
        try:
            cfg  # 检查是否已定义
        except NameError:
            cfg = OmegaConf.load(args.config_path)
        
        excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
        
        # 创建数据集
        dataset = TEMRegressionDataset(
            cfg.data.data_root,
            excel_path,
            material_ids=None,
            transform=get_transforms('test', cfg.data.image_size),
            use_cleaned=True,
            normalizer=None
        )
        
        # 随机选择样本
        if len(dataset) == 0:
            print("Error: Dataset is empty")
            return
        
        num_samples = min(args.num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        # 获取图像路径
        image_paths = [dataset.data_list[idx]['image_path'] for idx in indices]
        print(f"从数据集中随机选择了 {num_samples} 个样本")
    else:
        # 使用指定的图像路径
        print(f"使用指定的 {len(image_paths)} 张图像进行可视化")
    
    if image_paths is None or len(image_paths) == 0:
        print("Error: No images to visualize")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 可视化
    if args.mode in ['attention', 'both']:
        save_dir = os.path.join(args.output_dir, 'dino_attention')
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = visualize_dino_attention(
            image_paths, model, args.model_type, str(device), dtype,
            args.threshold, args.target_size, args.patch_size, save_dir
        )
        if saved_paths:
            print(f"\nSaved {len(saved_paths)} attention visualizations to: {save_dir}")
    
    if args.mode in ['multiscale', 'both']:
        save_dir = os.path.join(args.output_dir, 'dino_multiscale')
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = visualize_dino_multiscale_features(
            image_paths, model, args.model_type, str(device), dtype,
            args.target_size, args.patch_size, multiscale_layers=[6, 12, 18, 23], save_path=save_dir
        )
        if saved_paths:
            print(f"\nSaved {len(saved_paths)} multiscale visualizations to: {save_dir}")
    
    print(f"\nVisualization completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

