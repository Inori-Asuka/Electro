"""
Analyze the relationship between extracted features and activity values
Including: feature importance, correlation analysis, dimensionality reduction visualization, etc.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import argparse
from sklearn.ensemble import RandomForestRegressor
import pickle

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


def load_features(features_dir):
    """Load features"""
    train_features = np.load(os.path.join(features_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))
    train_entry_ids = np.load(os.path.join(features_dir, 'train_entry_ids.npy'))
    
    test_features = np.load(os.path.join(features_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))
    test_entry_ids = np.load(os.path.join(features_dir, 'test_entry_ids.npy'))
    
    return (train_features, train_labels, train_entry_ids), \
           (test_features, test_labels, test_entry_ids)


def analyze_feature_importance(train_features, train_labels, test_features, test_labels, output_dir):
    """Analyze feature importance"""
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Use multiple models to get feature importance
    importances_dict = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train_features_scaled, train_labels)
    importances_dict['RandomForest'] = rf.feature_importances_
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        xgb_model.fit(train_features_scaled, train_labels)
        importances_dict['XGBoost'] = xgb_model.feature_importances_
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        lgb_model.fit(train_features_scaled, train_labels)
        importances_dict['LightGBM'] = lgb_model.feature_importances_
    
    # Calculate average importance
    importances_array = np.array(list(importances_dict.values()))
    avg_importance = np.mean(importances_array, axis=0)
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature_idx': range(len(avg_importance)),
        'avg_importance': avg_importance,
    })
    for model_name, importance in importances_dict.items():
        importance_df[model_name] = importance
    
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Visualize top features
    top_k = min(50, len(avg_importance))
    top_indices = np.argsort(avg_importance)[-top_k:][::-1]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top feature importance
    axes[0].barh(range(top_k), avg_importance[top_indices][::-1])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels([f'Feature {idx}' for idx in top_indices[::-1]])
    axes[0].set_xlabel('Average Importance', fontsize=12)
    axes[0].set_title(f'Top {top_k} Important Features', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Feature importance distribution
    axes[1].hist(avg_importance, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Feature Importance', fontsize=12)
    axes[1].set_ylabel('Number of Features', fontsize=12)
    axes[1].set_title('Feature Importance Distribution', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Top 10 important feature indices: {top_indices[:10]}")
    print(f"Feature importance saved: {os.path.join(output_dir, 'feature_importance.csv')}")
    
    return top_indices, avg_importance


def analyze_correlation(train_features, train_labels, output_dir, top_features=None):
    """Analyze correlation between features and labels"""
    print("\n" + "="*60)
    print("Feature-Label Correlation Analysis")
    print("="*60)
    
    # Calculate Pearson and Spearman correlation coefficients
    n_features = train_features.shape[1]
    if top_features is not None:
        feature_indices = top_features[:min(100, len(top_features))]  # Analyze top 100
    else:
        feature_indices = range(min(100, n_features))  # Analyze first 100 features
    
    correlations_pearson = []
    correlations_spearman = []
    
    for idx in feature_indices:
        feat = train_features[:, idx]
        pearson_r, pearson_p = pearsonr(feat, train_labels)
        spearman_r, spearman_p = spearmanr(feat, train_labels)
        correlations_pearson.append((idx, pearson_r, pearson_p))
        correlations_spearman.append((idx, spearman_r, spearman_p))
    
    # Save correlation results
    corr_df = pd.DataFrame({
        'feature_idx': [c[0] for c in correlations_pearson],
        'pearson_r': [c[1] for c in correlations_pearson],
        'pearson_p': [c[2] for c in correlations_pearson],
        'spearman_r': [c[1] for c in correlations_spearman],
        'spearman_p': [c[2] for c in correlations_spearman],
    })
    corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)
    corr_df.to_csv(os.path.join(output_dir, 'feature_correlations.csv'), index=False)
    
    # Visualize correlations
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Pearson correlation coefficients
    top_corr = corr_df.head(30)
    axes[0].barh(range(len(top_corr)), top_corr['pearson_r'].values[::-1])
    axes[0].set_yticks(range(len(top_corr)))
    axes[0].set_yticklabels([f'Feature {idx}' for idx in top_corr['feature_idx'].values[::-1]])
    axes[0].set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    axes[0].set_title('Top 30 Features: Pearson Correlation with Labels', fontsize=14)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Correlation distribution
    axes[1].hist(corr_df['pearson_r'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    axes[1].set_ylabel('Number of Features', fontsize=12)
    axes[1].set_title('Feature-Label Correlation Distribution', fontsize=14)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Strongest positive correlation: Feature {corr_df.iloc[0]['feature_idx']} (r={corr_df.iloc[0]['pearson_r']:.4f})")
    print(f"Strongest negative correlation: Feature {corr_df.iloc[-1]['feature_idx']} (r={corr_df.iloc[-1]['pearson_r']:.4f})")


def visualize_dimension_reduction(train_features, train_labels, test_features, test_labels, output_dir):
    """Dimensionality reduction visualization"""
    print("\n" + "="*60)
    print("Dimensionality Reduction Visualization")
    print("="*60)
    
    # Standardize
    scaler = StandardScaler()
    all_features = np.vstack([train_features, test_features])
    all_features_scaled = scaler.fit_transform(all_features)
    all_labels = np.hstack([train_labels, test_labels])
    
    # PCA dimensionality reduction
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features_scaled)
    
    print(f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    
    # t-SNE dimensionality reduction (if feature dimension is not too large)
    if all_features_scaled.shape[1] <= 1000:
        print("Performing t-SNE dimensionality reduction (may take some time)...")
        # First reduce to 50 dimensions using PCA, then t-SNE
        pca_50 = PCA(n_components=50)
        features_pca = pca_50.fit_transform(all_features_scaled)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(features_pca)
    else:
        print("Feature dimension too large, skipping t-SNE")
        tsne_result = None
    
    # Visualization
    n_train = len(train_labels)
    
    fig, axes = plt.subplots(1, 2 if tsne_result is not None else 1, figsize=(16, 6))
    if tsne_result is None:
        axes = [axes]
    
    # PCA visualization
    scatter = axes[0].scatter(
        pca_result[:n_train, 0], pca_result[:n_train, 1],
        c=train_labels, cmap='viridis', alpha=0.6, s=50, label='Train Set'
    )
    axes[0].scatter(
        pca_result[n_train:, 0], pca_result[n_train:, 1],
        c=test_labels, cmap='viridis', alpha=0.6, s=50, marker='^', label='Test Set'
    )
    axes[0].set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[0].set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    axes[0].set_title('PCA Dimensionality Reduction', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Activity Value (Eapp)')
    
    # t-SNE visualization
    if tsne_result is not None:
        scatter = axes[1].scatter(
            tsne_result[:n_train, 0], tsne_result[:n_train, 1],
            c=train_labels, cmap='viridis', alpha=0.6, s=50, label='Train Set'
        )
        axes[1].scatter(
            tsne_result[n_train:, 0], tsne_result[n_train:, 1],
            c=test_labels, cmap='viridis', alpha=0.6, s=50, marker='^', label='Test Set'
        )
        axes[1].set_xlabel('t-SNE 1', fontsize=12)
        axes[1].set_ylabel('t-SNE 2', fontsize=12)
        axes[1].set_title('t-SNE Dimensionality Reduction', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Activity Value (Eapp)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Dimensionality reduction visualization saved")


def visualize_feature_label_relationship(train_features, train_labels, output_dir, top_features=None):
    """Visualize relationship between features and labels"""
    print("\n" + "="*60)
    print("Feature-Label Relationship Visualization")
    print("="*60)
    
    if top_features is None:
        top_features = range(min(9, train_features.shape[1]))  # Default: show first 9
    
    n_features = min(9, len(top_features))
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, feat_idx in enumerate(top_features[:n_features]):
        feat = train_features[:, feat_idx]
        
        # Scatter plot
        axes[i].scatter(feat, train_labels, alpha=0.5, s=20)
        axes[i].set_xlabel(f'Feature {feat_idx}', fontsize=10)
        axes[i].set_ylabel('Eapp (Activity Value)', fontsize=10)
        axes[i].set_title(f'Feature {feat_idx} vs Eapp', fontsize=12)
        axes[i].grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(feat, train_labels, 1)
        p = np.poly1d(z)
        axes[i].plot(feat, p(feat), "r--", alpha=0.8, linewidth=2, label=f'Trend Line (r={np.corrcoef(feat, train_labels)[0,1]:.3f})')
        axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_label_relationships.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Feature-label relationship plot saved")


def main():
    parser = argparse.ArgumentParser(description="Analyze relationship between features and activity values")
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='./feature_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features_dir}")
    (train_features, train_labels, train_entry_ids), \
    (test_features, test_labels, test_entry_ids) = load_features(args.features_dir)
    
    print(f"Train set: {train_features.shape[0]} samples, {train_features.shape[1]} feature dimensions")
    print(f"Test set: {test_features.shape[0]} samples, {test_features.shape[1]} feature dimensions")
    
    # 1. Feature importance analysis
    top_features, importances = analyze_feature_importance(
        train_features, train_labels, test_features, test_labels, args.output_dir
    )
    
    # 2. Feature-label correlation analysis
    analyze_correlation(train_features, train_labels, args.output_dir, top_features)
    
    # 3. Dimensionality reduction visualization
    visualize_dimension_reduction(train_features, train_labels, test_features, test_labels, args.output_dir)
    
    # 4. Feature-label relationship visualization
    visualize_feature_label_relationship(train_features, train_labels, args.output_dir, top_features)
    
    print(f"\nAll analysis results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
