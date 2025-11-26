"""
Visualize multiscale features from ConvNeXt stages
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms
from model import DINOv3RegressionModel


class MultiscaleFeatureExtractor:
    """Extract features from multiple stages of ConvNeXt"""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.handles = []
        
    def hook_layers(self):
        """Register hooks to capture features from different stages"""
        def make_hook(stage_name):
            def hook(module, input, output):
                # Store the output feature map
                if isinstance(output, torch.Tensor):
                    self.features[stage_name] = output.detach()
                else:
                    self.features[stage_name] = output
            return hook
        
        backbone = self.model.backbone
        
        # Hook different stages
        if hasattr(backbone, 'stages'):
            stages = backbone.stages
            for i, stage in enumerate(stages):
                stage_name = f'stage{i+1}'
                handle = stage.register_forward_hook(make_hook(stage_name))
                self.handles.append(handle)
        else:
            print("Warning: Cannot find stages in backbone")
    
    def extract_features(self, input_tensor):
        """Extract features from all stages"""
        self.features = {}
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return self.features
    
    def __del__(self):
        """Remove hooks"""
        for handle in self.handles:
            if handle is not None:
                try:
                    handle.remove()
                except:
                    pass


def process_feature_map(features, original_size):
    """
    Process feature map to create heatmap similar to Grad-CAM
    
    Args:
        features: Feature tensor or numpy array
        original_size: (width, height) of original image
    
    Returns:
        cam: Normalized CAM heatmap [H, W]
    """
    # Handle different feature shapes
    if isinstance(features, torch.Tensor):
        if len(features.shape) == 4:
            # [B, C, H, W] -> take first sample and average over channels
            features = features[0].mean(0).cpu().numpy()
        elif len(features.shape) == 3:
            # [C, H, W] -> average over channels
            features = features.mean(0).cpu().numpy()
        elif len(features.shape) == 2:
            # [H, W]
            features = features.cpu().numpy()
        else:
            return None
    else:
        if isinstance(features, np.ndarray):
            if len(features.shape) > 2:
                features = features.mean(0)
        else:
            return None
    
    # Normalize to [0, 1]
    if features.max() > features.min():
        cam = (features - features.min()) / (features.max() - features.min())
    else:
        cam = np.zeros_like(features)
    
    # Resize to original image size
    cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
    
    return cam_resized


def visualize_multiscale_features(feature_maps, original_image=None, output_path=None):
    """
    Visualize multiscale features from different stages with heatmap overlay
    Keep original layout (2x2 grid for 4 stages) but use same heatmap style as visualize_attention.py
    
    Args:
        feature_maps: Dictionary of feature maps from different stages
        original_image: PIL Image or numpy array of original image (optional)
        output_path: Path to save the visualization
    """
    # Get available stages
    stage_names = sorted([k for k in feature_maps.keys() if k.startswith('stage')])
    
    if len(stage_names) == 0:
        print("No stage features found")
        return
    
    # Get original image size
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            original_size = original_image.size
            image_array = np.array(original_image)
        else:
            original_size = (original_image.shape[1], original_image.shape[0])
            image_array = original_image
    else:
        # Use feature map size
        first_features = feature_maps[stage_names[0]]
        if isinstance(first_features, torch.Tensor):
            if len(first_features.shape) >= 2:
                h, w = first_features.shape[-2:]
            else:
                h, w = 224, 224
        else:
            if len(first_features.shape) >= 2:
                h, w = first_features.shape[-2:]
            else:
                h, w = 224, 224
        original_size = (w, h)
        image_array = None
    
    # Create subplots: 2x2 grid for 4 stages (original layout)
    n_stages = len(stage_names)
    n_cols = 2
    n_rows = (n_stages + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    if n_stages == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, stage_name in enumerate(stage_names):
        if i >= len(axes):
            break
            
        features = feature_maps[stage_name]
        
        # Process feature map to CAM
        cam = process_feature_map(features, original_size)
        if cam is None:
            print(f"Warning: Failed to process features for {stage_name}")
            continue
        
        # Create heatmap using same method as visualize_attention.py
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image (same as visualize_attention.py)
        if image_array is not None:
            overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
        else:
            overlay = heatmap
        
        # Display overlay (same style as visualize_attention.py)
        axes[i].imshow(overlay)
        axes[i].set_title(f'{stage_name.upper()} Features', fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(stage_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multiscale features visualization saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_single_image_multiscale(model, image_path, output_path, image_size, device):
    """Visualize multiscale features for a single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess
    transform = get_transforms('test', image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract multiscale features
    extractor = MultiscaleFeatureExtractor(model)
    extractor.hook_layers()
    feature_maps = extractor.extract_features(input_tensor)
    
    # Visualize with original image
    visualize_multiscale_features(feature_maps, original_image=image, output_path=output_path)


def visualize_dataset_samples_multiscale(model, dataset, output_dir, image_size, device, num_samples=5):
    """Visualize multiscale features for dataset samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    extractor = MultiscaleFeatureExtractor(model)
    extractor.hook_layers()
    
    for idx in indices:
        image, label = dataset[idx]
        image_path = dataset.data_list[idx]['image_path']
        entry_id = dataset.data_list[idx]['material_id']
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        transform = get_transforms('test', image_size)
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(input_tensor).item()
        
        # Extract features
        feature_maps = extractor.extract_features(input_tensor)
        
        # Get stage names
        stage_names = sorted([k for k in feature_maps.keys() if k.startswith('stage')])
        n_stages = len(stage_names)
        
        # Create visualization: 2x2 grid for 4 stages (original layout)
        n_cols = 2
        n_rows = (n_stages + 1) // 2
        
        fig = plt.figure(figsize=(20, 10))
        
        # Original image at top
        ax1 = plt.subplot(2, 3, 1)
        image_array = np.array(original_image)
        ax1.imshow(original_image)
        ax1.set_title(f'Entry {entry_id}\nTrue: {label:.4f}, Pred: {pred:.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Multiscale features in 2x2 grid (original layout)
        for i, stage_name in enumerate(stage_names[:4]):  # Show up to 4 stages
            ax = plt.subplot(2, 3, i + 2)
            
            features = feature_maps[stage_name]
            
            # Process feature map to CAM
            cam = process_feature_map(features, original_image.size)
            if cam is None:
                print(f"Warning: Failed to process features for {stage_name}")
                continue
            
            # Create heatmap using same method as visualize_attention.py
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay on original image (same as visualize_attention.py)
            overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
            
            # Display overlay (same style as visualize_attention.py)
            ax.imshow(overlay)
            ax.set_title(f'{stage_name.upper()} Features', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'entry_{entry_id}_multiscale_{idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multiscale features from ConvNeXt")
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--image_path', type=str, default=None, help='Single image path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Model checkpoint path')
    
    args = parser.parse_args()
    
    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.misc.checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    # For visualization, always disable LoRA to avoid compatibility issues
    # LoRA with PEFT library has issues with ConvNeXt (CNN models)
    use_lora = False
    
    model = DINOv3RegressionModel(
        model_path=cfg.dinov3.model_path,
        num_outputs=1,
        head_type=cfg.dinov3.head_type,
        dropout=cfg.training.dropout,
        freeze_backbone=cfg.dinov3.freeze_backbone,
        freeze_layers=cfg.dinov3.get('freeze_layers', None),
        use_lora=use_lora,
        lora_r=cfg.dinov3.get('lora_r', 16),
        lora_alpha=cfg.dinov3.get('lora_alpha', 32),
        lora_dropout=cfg.dinov3.get('lora_dropout', 0.1),
        head_kwargs=cfg.dinov3.get('head_kwargs', {}),
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Use strict=False to allow missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (may be due to backbone structure differences)")
        if len(missing_keys) <= 10:
            print(f"Missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys")
    print(f"Model loaded: {checkpoint_path}")
    
    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(cfg.misc.checkpoint_dir, 'multiscale_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize single image
    if args.image_path and os.path.exists(args.image_path):
        output_path = os.path.join(output_dir, 'single_image_multiscale.png')
        visualize_single_image_multiscale(model, args.image_path, output_path, cfg.data.image_size, device)
    
    # Visualize dataset samples
    excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
    dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=None,
        transform=get_transforms('test', cfg.data.image_size),
        use_cleaned=True,
        normalizer=None
    )
    
    samples_dir = os.path.join(output_dir, 'multiscale_samples')
    visualize_dataset_samples_multiscale(model, dataset, samples_dir, cfg.data.image_size, device, args.num_samples)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

