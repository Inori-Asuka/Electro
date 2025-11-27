"""
Visualize attention regions of DINOv3 ConvNeXt model on original images (Grad-CAM)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
from omegaconf import OmegaConf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms
from model import DINOv3RegressionModel


class GradCAM:
    """Grad-CAM visualization class"""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.handles = []
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            # Save activations
            if isinstance(output, torch.Tensor):
                self.activations.append(output.detach())
            else:
                self.activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            # Save gradients
            if grad_output[0] is not None:
                self.gradients.append(grad_output[0].detach())
        
        # Get target layer
        target_module = self._get_target_layer()
        if target_module is not None:
            handle_f = target_module.register_forward_hook(forward_hook)
            handle_b = target_module.register_full_backward_hook(backward_hook)
            self.handles = [handle_f, handle_b]
    
    def _get_target_layer(self):
        """Get target layer (last convolutional layer of ConvNeXt)"""
        backbone = self.model.backbone
        
        # ConvNeXt structure is usually: stages[0-3], each stage contains multiple blocks
        if hasattr(backbone, 'stages'):
            stages = backbone.stages
            if len(stages) > 0:
                last_stage = stages[-1]
                # Get the last block of the last stage
                if hasattr(last_stage, 'blocks') and len(last_stage.blocks) > 0:
                    last_block = last_stage.blocks[-1]
                    # Return the last convolutional layer in the block
                    for name, module in last_block.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            return module
                    return last_block
                return last_stage
        
        # Fallback: iterate through all modules to find the last convolutional layer
        last_conv = None
        for module in backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    
    def generate_cam(self, input_image):
        """
        Generate CAM heatmap
        
        Args:
            input_image: Input image [1, C, H, W]
        
        Returns:
            cam: CAM heatmap [H, W]
        """
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # Forward pass
        output = self.model(input_image)
        
        # For regression task, use output gradient
        self.model.zero_grad()
        output.backward(torch.ones_like(output), retain_graph=True)
        
        # Get gradients and activations
        if len(self.activations) == 0 or len(self.gradients) == 0:
            # If not captured, return zeros
            print("Warning: Failed to capture activations or gradients, returning zero CAM")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        activations = self.activations[0]  # [1, C, H, W]
        gradients = self.gradients[0]  # [1, C, H, W]
        
        # Process spatial features
        if len(activations.shape) == 4:
            # Calculate weights (global average pooling of gradients)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            
            # Weighted sum
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize to [0, 1]
            cam = cam.squeeze().detach().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            # If features are already globally pooled, cannot generate spatial CAM
            print("Warning: Features are already globally pooled, cannot generate spatial CAM")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        return cam
    
    def __del__(self):
        """Clean up hooks"""
        for handle in self.handles:
            if handle is not None:
                try:
                    handle.remove()
                except:
                    pass


def visualize_single_image(model, image_path, output_path, cfg, device):
    """Visualize single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Preprocess
    transform = get_transforms('test', cfg.data.image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate CAM
    target_layer = None
    gradcam = GradCAM(model, target_layer)
    
    cam = gradcam.generate_cam(input_tensor)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    image_array = np.array(image)
    overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization result saved: {output_path}")


def visualize_dataset_samples(model, dataset, output_dir, cfg, device, num_samples=10):
    """Visualize samples from dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    gradcam = GradCAM(model)
    
    for idx in indices:
        image, label = dataset[idx]
        image_path = dataset.data_list[idx]['image_path']
        entry_id = dataset.data_list[idx]['material_id']
        
        # Preprocess
        transform = get_transforms('test', cfg.data.image_size)
        input_tensor = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(input_tensor).item()
        
        # Generate CAM
        try:
            cam = gradcam.generate_cam(input_tensor)
            
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            
            # Resize CAM
            cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            image_array = np.array(original_image)
            overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(image_array)
            axes[0].set_title(f'Entry {entry_id}\nTrue: {label:.4f}', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Attention Heatmap', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay\nPred: {pred:.4f}', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'entry_{entry_id}_sample_{idx}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Sample {idx} (Entry {entry_id}) CAM generation failed: {e}")
            # Only save original image and prediction
            original_image = Image.open(image_path).convert('RGB')
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(np.array(original_image))
            ax.set_title(f'Entry {entry_id}\nTrue: {label:.4f}, Pred: {pred:.4f}', fontsize=12)
            ax.axis('off')
            output_path = os.path.join(output_dir, f'entry_{entry_id}_sample_{idx}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize model attention regions (Grad-CAM)")
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Single image path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Model checkpoint path')
    
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"Model loaded: {checkpoint_path}")
    
    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(cfg.misc.checkpoint_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize single image
    image_path = args.image_path
    if image_path and os.path.exists(image_path):
        output_path = os.path.join(output_dir, 'single_image_cam.png')
        visualize_single_image(model, image_path, output_path, cfg, device)
    
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
    
    samples_dir = os.path.join(output_dir, 'samples')
    visualize_dataset_samples(model, dataset, samples_dir, cfg, device, args.num_samples)
    
    print(f"\nAll visualization results saved to: {output_dir}")


if __name__ == '__main__':
    main()

