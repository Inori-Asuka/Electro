# Electro_dino_ml

DINOv3-based regression model for TEM image analysis with traditional ML support.

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
