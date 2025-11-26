# Electro_dino_ml

DINOv3-based regression model for TEM image analysis with traditional ML support.

## 模型配置

### Backbone
- **`backbone_type: convnext`** : DINOv3 ConvNeXt-Large
- **`backbone_type: vit`**:  DINOv3 ViT Large

### Pooling
#### 1：Linear (`pooling: linear`)
```yaml
dinov3:
  backbone_type: convnext  # vit
  pooling: linear
```

#### 2: GAP 
```yaml
dinov3:
  backbone_type: convnext  # vit
  pooling: gap
  head_type: mlp  # linear, mlp, attention, gated, residual
```

#### 3: Attention (`pooling: attention`)
```yaml
dinov3:
  backbone_type: vit 
  pooling: attention
  head_kwargs:
    num_heads: 8
    hidden_dim: null  # null表示使用backbone的特征维度
```

#### 4: Multiscale (`pooling: multiscale`)
```yaml
# ConvNeXt
dinov3:
  backbone_type: convnext
  pooling: multiscale
  multiscale_layers: [1, 2, 3]  # ConvNeXt：stage (0-3)
  head_kwargs:
    hidden_dims: [3072, 1536]

# ViT
dinov3:
  backbone_type: vit
  pooling: multiscale
  multiscale_layers: [6, 12, 18, 23]  # ViT Large： (0-23)
  head_kwargs:
    hidden_dims: [2048, 1024]
```

### LoRA
**ConvNeXt LoRA：**
```yaml
dinov3:
  backbone_type: convnext
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # ConvNeXt 自动应用到所有 MLP 的 fc1 和 fc2 层
```

**ViT LoRA：**

```yaml
dinov3:
  backbone_type: vit
  use_lora: true
  lora_modules: ['qkv', 'mlp']  # 只微调 QKV 和 MLP
```


使用配置文件：
```bash
cd train
python train.py --config-name=config_convnext_attention
python train.py --config-name=config_vit_multiscale
python train.py --config-name=config_vit_lora_qkv_mlp
```

## Usage

### Training

```bash
cd train
python train.py
```

### Testing

```bash
python run_test.py <model_path> [config_path]
```

### Feature Extraction

```bash
cd train
python extract_features.py
```

### Traditional ML Regression

```bash
cd train
python ml_regression.py --features_dir ../checkpoints/my_experiment/features --output_dir ../checkpoints/my_experiment/ml_results
```

### Visualization

#### Attention Visualization (Grad-CAM)

```bash
cd visualization
python visualize_attention.py --config_path ../checkpoints/my_experiment/config.yaml --checkpoint_path ../checkpoints/my_experiment/best_model.pth --output_dir ../checkpoints/my_experiment/visualization --num_samples 10
```

#### Multi-scale Features Visualization

```bash
cd visualization
python visualize_multiscale_features.py --config_path ../checkpoints/my_experiment/config.yaml --checkpoint_path ../checkpoints/my_experiment/best_model.pth --output_dir ../checkpoints/my_experiment/visualization --num_samples 5
```

#### Feature Analysis

```bash
cd visualization
python analyze_features.py --features_dir ../checkpoints/my_experiment/features --output_dir ../checkpoints/my_experiment/feature_analysis
```

#### ML Model Visualization

```bash
cd visualization
python visualize_ml_models.py --features_dir ../checkpoints/my_experiment/features --output_dir ../checkpoints/my_experiment/ml_visualization
```

#### DINO Attention Visualization
```bash
cd visualization
python visualize_dino_attention.py \
    --model_path /path/to/dinov3_vitl16.safetensors \
    --image_paths img1.jpg img2.jpg \
    --output_dir ./visualizations \
    --mode both  # 'attention', 'multiscale', 或 'both'
```
