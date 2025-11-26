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
    
    # 绘图设置
    n_frames = len(image_paths)
    cols = min(4, n_frames)
    rows = math.ceil(n_frames / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for frame_idx in range(n_frames):
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
        
        # 可视化
        ax = axes[frame_idx]
        ax.imshow(original_img)
        
        if threshold is not None:
            # 使用阈值创建mask
            mask = attn_resized > threshold
            ax.imshow(mask.astype(float), cmap='jet', alpha=0.5)
        else:
            ax.imshow(attn_resized, cmap='jet', alpha=0.5)
        
        ax.set_title(f"Image {frame_idx}\n{os.path.basename(img_path)}", fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的subplot
    for j in range(n_frames, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    
    return fig


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
    
    # 绘图设置 (Rows=Images, Cols=Layers+Original)
    fig, axes = plt.subplots(n_frames, n_layers + 1, figsize=((n_layers + 1) * 4, n_frames * 4))
    if n_frames == 1:
        axes = axes.reshape(1, -1)
    
    # 计算 Token 信息 (用于去除 CLS/Registers)
    _, N_tokens, _ = multiscale_feats[0].shape
    expected_patches = (target_size // patch_size) ** 2
    num_registers = N_tokens - 1 - expected_patches

    for img_idx in range(n_frames):
        # --- A. 显示原图 ---
        img_path = image_paths[img_idx]
        original_img = np.array(Image.open(img_path).convert('RGB'))
        
        ax_orig = axes[img_idx, 0]
        ax_orig.imshow(original_img)
        ax_orig.set_title(f"Original\n{os.path.basename(img_path)}", fontsize=10)
        ax_orig.axis('off')

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
            ax = axes[img_idx, layer_idx + 1]
            ax.imshow(feat_resized)
            ax.set_title(f"Layer {multiscale_layers[layer_idx]}\nSemantic PCA", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multiscale visualization saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="可视化 DINOv3 ViT Attention Maps")
    parser.add_argument('--model_path', type=str, required=True,
                       help='DINOv3模型权重路径 (.safetensors)')
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                       help='要可视化的图像路径列表')
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
    parser.add_argument('--mode', type=str, default='attention', choices=['attention', 'multiscale', 'both'],
                       help='可视化模式')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # 加载模型
    from safetensors.torch import load_file
    import timm
    
    model_name = "vit_large_patch16_dinov3.lvd1689m" if args.model_type == 'dinov3' else "vit_large_patch14_dinov2"
    model = timm.create_model(model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
    
    if os.path.exists(args.model_path):
        state_dict = load_file(args.model_path)
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
    else:
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    model = model.to(device)
    model.eval()
    print(f"{args.model_type} model loaded successfully!")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 可视化
    if args.mode in ['attention', 'both']:
        save_path = os.path.join(args.output_dir, 'dino_attention.png')
        fig = visualize_dino_attention(
            args.image_paths, model, args.model_type, str(device), dtype,
            args.threshold, args.target_size, args.patch_size, save_path
        )
        if fig:
            plt.show()
    
    if args.mode in ['multiscale', 'both']:
        save_path = os.path.join(args.output_dir, 'dino_multiscale.png')
        fig = visualize_dino_multiscale_features(
            args.image_paths, model, args.model_type, str(device), dtype,
            args.target_size, args.patch_size, save_path=save_path
        )
        if fig:
            plt.show()
    
    print(f"\nVisualization completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

