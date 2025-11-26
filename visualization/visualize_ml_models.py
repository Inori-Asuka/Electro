"""
Visualize trained machine learning models
Including: tree model structure, feature importance, decision paths, etc.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
import graphviz

try:
    from sklearn.tree import export_graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def visualize_tree_structure(model, model_name, output_dir, max_depth=3, feature_names=None):
    """Visualize tree model structure"""
    print(f"\nVisualizing {model_name} tree structure...")
    
    if isinstance(model, RandomForestRegressor):
        # Random Forest: visualize first tree
        tree = model.estimators_[0]
        tree_dir = os.path.join(output_dir, f'{model_name.lower()}_tree')
        os.makedirs(tree_dir, exist_ok=True)
        
        # Use matplotlib to plot
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tree, max_depth=max_depth, feature_names=feature_names, 
                 filled=True, rounded=True, ax=ax, fontsize=8)
        plt.title(f'{model_name} - First Decision Tree (Max Depth={max_depth})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(tree_dir, 'tree_structure.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Export text format
        tree_text = export_text(tree, max_depth=max_depth, feature_names=feature_names)
        with open(os.path.join(tree_dir, 'tree_structure.txt'), 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        print(f"Tree structure saved to: {tree_dir}")
        
        # If graphviz is installed, generate prettier graph
        if GRAPHVIZ_AVAILABLE:
            try:
                dot_data = export_graphviz(
                    tree, out_file=None, max_depth=max_depth,
                    feature_names=feature_names, filled=True,
                    rounded=True, special_characters=True
                )
                graph = graphviz.Source(dot_data)
                graph.render(os.path.join(tree_dir, 'tree_structure'), format='png', cleanup=True)
                print(f"Graphviz tree graph saved")
            except Exception as e:
                print(f"Graphviz generation failed: {e}")
    
    elif XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor):
        # XGBoost: visualize first tree
        tree_dir = os.path.join(output_dir, f'{model_name.lower()}_tree')
        os.makedirs(tree_dir, exist_ok=True)
        
        # Use XGBoost's built-in visualization
        try:
            xgb.plot_tree(model, num_trees=0, max_depth=max_depth, 
                         rankdir='LR', ax=plt.gca())
            plt.title(f'{model_name} - First Decision Tree', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(tree_dir, 'tree_structure.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"XGBoost tree structure saved to: {tree_dir}")
        except Exception as e:
            print(f"XGBoost visualization failed: {e}")
    
    elif LIGHTGBM_AVAILABLE and isinstance(model, lgb.LGBMRegressor):
        # LightGBM: visualize first tree
        tree_dir = os.path.join(output_dir, f'{model_name.lower()}_tree')
        os.makedirs(tree_dir, exist_ok=True)
        
        try:
            ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10), max_depth=max_depth)
            plt.title(f'{model_name} - First Decision Tree', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(tree_dir, 'tree_structure.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"LightGBM tree structure saved to: {tree_dir}")
        except Exception as e:
            print(f"LightGBM visualization failed: {e}")


def visualize_feature_importance_comparison(ml_results_dir, output_dir):
    """Compare feature importance across different models"""
    print("\nComparing feature importance across different models...")
    
    models = ['ridge', 'lasso', 'elastic', 'svr', 'randomforest', 'gradientboosting']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    
    importance_data = {}
    
    for model_name in models:
        model_path = os.path.join(ml_results_dir, model_name, 'model.pkl')
        if not os.path.exists(model_path):
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models use absolute value of coefficients
                importance_data[model_name] = np.abs(model.coef_)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    if not importance_data:
        print("No available models found")
        return
    
    # Create comparison plot
    n_models = len(importance_data)
    n_features = len(list(importance_data.values())[0])
    
    # Select top features
    avg_importance = np.mean(list(importance_data.values()), axis=0)
    top_k = min(30, n_features)
    top_indices = np.argsort(avg_importance)[-top_k:][::-1]
    
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4*n_models))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, importances) in enumerate(importance_data.items()):
        top_importances = importances[top_indices]
        axes[idx].barh(range(top_k), top_importances[::-1])
        axes[idx].set_yticks(range(top_k))
        axes[idx].set_yticklabels([f'Feature {idx}' for idx in top_indices[::-1]], fontsize=8)
        axes[idx].set_xlabel('Feature Importance', fontsize=10)
        axes[idx].set_title(f'{model_name.capitalize()} - Top {top_k} Important Features', fontsize=12)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_importance_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Feature importance comparison plot saved")


def visualize_prediction_distribution(ml_results_dir, output_dir):
    """Visualize prediction distribution"""
    print("\nVisualizing prediction distribution...")
    
    models = ['ridge', 'randomforest', 'gradientboosting']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models[:4]):
        pred_path = os.path.join(ml_results_dir, model_name, 'predictions.csv')
        if not os.path.exists(pred_path):
            continue
        
        df = pd.read_csv(pred_path)
        
        # Scatter plot: predicted vs true values
        axes[idx].scatter(df['true_label'], df['pred_label'], alpha=0.6, s=30)
        
        # Add diagonal line
        min_val = min(df['true_label'].min(), df['pred_label'].min())
        max_val = max(df['true_label'].max(), df['pred_label'].max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(df['true_label'], df['pred_label'])
        
        axes[idx].set_xlabel('True Value', fontsize=12)
        axes[idx].set_ylabel('Predicted Value', fontsize=12)
        axes[idx].set_title(f'{model_name.capitalize()} (R² = {r2:.4f})', fontsize=12)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Prediction distribution plot saved")


def visualize_residuals(ml_results_dir, output_dir):
    """Visualize residual analysis"""
    print("\nVisualizing residual analysis...")
    
    models = ['ridge', 'randomforest', 'gradientboosting']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    
    fig, axes = plt.subplots(len(models), 2, figsize=(16, 5*len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    for row, model_name in enumerate(models):
        pred_path = os.path.join(ml_results_dir, model_name, 'predictions.csv')
        if not os.path.exists(pred_path):
            continue
        
        df = pd.read_csv(pred_path)
        residuals = df['pred_label'] - df['true_label']
        
        # Residual distribution
        axes[row, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[row, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[row, 0].set_xlabel('Residual (Predicted - True)', fontsize=12)
        axes[row, 0].set_ylabel('Frequency', fontsize=12)
        axes[row, 0].set_title(f'{model_name.capitalize()} - Residual Distribution', fontsize=12)
        axes[row, 0].grid(alpha=0.3)
        
        # Residual vs predicted value
        axes[row, 1].scatter(df['pred_label'], residuals, alpha=0.6, s=30)
        axes[row, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[row, 1].set_xlabel('Predicted Value', fontsize=12)
        axes[row, 1].set_ylabel('Residual', fontsize=12)
        axes[row, 1].set_title(f'{model_name.capitalize()} - Residual vs Predicted', fontsize=12)
        axes[row, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Residual analysis plot saved")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained machine learning models")
    parser.add_argument('--ml_results_dir', type=str, required=True,
                       help='ML results directory (containing subdirectories for each model)')
    parser.add_argument('--output_dir', type=str, default='./ml_visualizations',
                       help='Output directory for visualization results')
    parser.add_argument('--max_depth', type=int, default=3,
                       help='Maximum depth for tree visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features (for feature names)
    features_dir = os.path.dirname(args.ml_results_dir)  # Assume features are in the same parent directory
    if os.path.exists(os.path.join(features_dir, 'features', 'train_features.npy')):
        train_features = np.load(os.path.join(features_dir, 'features', 'train_features.npy'))
        feature_names = [f'Feature_{i}' for i in range(train_features.shape[1])]
    else:
        feature_names = None
    
    # 1. Visualize tree structure
    models_to_visualize = ['randomforest', 'gradientboosting']
    if XGBOOST_AVAILABLE:
        models_to_visualize.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models_to_visualize.append('lightgbm')
    
    for model_name in models_to_visualize:
        model_path = os.path.join(args.ml_results_dir, model_name, 'model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                visualize_tree_structure(model, model_name, args.output_dir, 
                                       args.max_depth, feature_names)
            except Exception as e:
                print(f"Failed to visualize {model_name}: {e}")
    
    # 2. Feature importance comparison
    visualize_feature_importance_comparison(args.ml_results_dir, args.output_dir)
    
    # 3. Prediction distribution
    visualize_prediction_distribution(args.ml_results_dir, args.output_dir)
    
    # 4. Residual analysis
    visualize_residuals(args.ml_results_dir, args.output_dir)
    
    print(f"\nAll visualization results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

