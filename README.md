# Electro_dino_ml

DINOv3-based regression model for TEM image analysis with traditional ML support.

## 模型配置

### Backbone 类型

在 `train/conf/config.yaml` 中可以配置使用不同的 backbone：

- **`backbone_type: convnext`** (默认): 使用 DINOv3 ConvNeXt-Large
- **`backbone_type: vit`**: 使用 DINOv3 ViT Large

### Pooling 方案

ConvNeXt 和 ViT 都支持四种 pooling 方案，通过 `pooling` 参数选择：

#### 方案 1: Linear (`pooling: linear`)
- **适用场景**: 直接使用 CLS token 进行回归，最简单快速
- **实现**: 
  - ConvNeXt: 使用 `forward_features` 返回的 CLS token
  - ViT: 使用第一个 token (CLS token)
- **用法**:

```yaml
dinov3:
  backbone_type: convnext  # 或 vit
  pooling: linear
```

#### 方案 2: GAP (`pooling: gap`)
- **适用场景**: 活性值与材料的整体性质有关（例如整体孔隙率、均匀度）
- **实现**: 
  - ConvNeXt: 对空间特征图做全局平均池化
  - ViT: 对所有 tokens (包括 CLS 和 patches) 做平均
- **用法**:

```yaml
dinov3:
  backbone_type: convnext  # 或 vit
  pooling: gap
  head_type: mlp  # 可选: linear, mlp, attention, gated, residual
```

#### 方案 3: Attention (`pooling: attention`)
- **适用场景**: 活性由某个具体的缺陷（如裂纹、杂质点）决定
- **实现**: 
  - ConvNeXt: 将空间特征 reshape 成序列，使用 Attention 机制学习关键区域
  - ViT: 保留空间特征，使用 Attention 机制学习关键 patch 的权重
- **用法**:

```yaml
dinov3:
  backbone_type: convnext  # 或 vit
  pooling: attention
  head_kwargs:
    num_heads: 8
    hidden_dim: null  # null表示使用backbone的特征维度
```

#### 方案 4: Multiscale (`pooling: multiscale`)
- **适用场景**: 材料活性既取决于宏观结构（深层特征），也取决于微观边缘（浅层特征）
- **实现**: 
  - ConvNeXt: 提取多个 stage 的特征（默认：stage 1, 2, 3）
  - ViT: 提取多个 block 的特征（默认：第 6, 12, 18, 23 层）
- **用法**:

```yaml
# ConvNeXt
dinov3:
  backbone_type: convnext
  pooling: multiscale
  multiscale_layers: [1, 2, 3]  # ConvNeXt 有4个 stage (0-3)
  head_kwargs:
    hidden_dims: [3072, 1536]

# ViT
dinov3:
  backbone_type: vit
  pooling: multiscale
  multiscale_layers: [6, 12, 18, 23]  # ViT Large 有24层 (0-23)
  head_kwargs:
    hidden_dims: [2048, 1024]
```

### LoRA 微调

ConvNeXt 和 ViT 都支持 LoRA 微调。当启用 LoRA 时：

- **ConvNeXt**: 自动应用到所有 `.fc1` 和 `.fc2` 的 Linear 层（MLP 模块）
- **ViT**: 可以通过 `lora_modules` 参数选择要微调的模块

**ConvNeXt LoRA 配置：**
```yaml
dinov3:
  backbone_type: convnext
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # ConvNeXt 自动应用到所有 MLP 的 fc1 和 fc2 层
```

**ViT LoRA 模块选择：**
- **`['qkv']`**: 只微调 attention 的 QKV 投影层
- **`['mlp']`**: 只微调 MLP 的 fc1 和 fc2 层
- **`['qkv', 'mlp']`**: 微调 QKV 和 MLP（不包括 attention 的 proj 层）
- **`['qkv', 'proj', 'mlp']`** 或 **`null`**: 微调所有模块（默认）

```yaml
dinov3:
  backbone_type: vit
  use_lora: true
  lora_modules: ['qkv', 'mlp']  # 只微调 QKV 和 MLP
```

### 示例配置文件

项目提供了多个示例配置文件：

**ConvNeXt 示例：**
- `train/conf/config_convnext_linear.yaml`: Linear
- `train/conf/config.yaml`: GAP (默认)
- `train/conf/config_convnext_attention.yaml`: Attention
- `train/conf/config_convnext_multiscale.yaml`: Multiscale

**ViT 示例：**
- `train/conf/config_vit_linear.yaml`: Linear
- `train/conf/config_vit_gap.yaml`: GAP
- `train/conf/config_vit_attention.yaml`: Attention
- `train/conf/config_vit_multiscale.yaml`: Multiscale

**LoRA 模块选择示例：**
- `train/conf/config_vit_lora_qkv.yaml`: 只微调 QKV
- `train/conf/config_vit_lora_mlp.yaml`: 只微调 MLP
- `train/conf/config_vit_lora_qkv_mlp.yaml`: 微调 QKV 和 MLP

使用示例配置文件：
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

可视化 DINOv3 ViT 的 Attention Maps：

```bash
cd visualization
python visualize_dino_attention.py \
    --model_path /path/to/dinov3_vitl16.safetensors \
    --image_paths img1.jpg img2.jpg \
    --output_dir ./visualizations \
    --mode both  # 'attention', 'multiscale', 或 'both'
```

或者使用 Python API：

```python
from visualization import visualize_dino_attention
import torch

# 加载模型
from model import DINOv3ViTRegressionModel
model = DINOv3ViTRegressionModel(
    model_path="path/to/dinov3_vitl16.safetensors",
    pooling='gap'
).eval()

# 可视化
fig = visualize_dino_attention(
    image_paths=['img1.jpg', 'img2.jpg'],
    model=model,
    model_type='dinov3',
    device='cuda'
)
```
