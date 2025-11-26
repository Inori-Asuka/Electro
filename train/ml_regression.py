"""
使用传统机器学习方法进行回归
支持多种ML算法：SVM、Random Forest、XGBoost、LightGBM等
"""
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] XGBoost 未安装，将跳过 XGBoost 回归")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[警告] LightGBM 未安装，将跳过 LightGBM 回归")


def load_features(features_dir):
    """加载提取的特征"""
    train_features = np.load(os.path.join(features_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(features_dir, 'train_labels.npy'))
    train_entry_ids = np.load(os.path.join(features_dir, 'train_entry_ids.npy'))
    
    test_features = np.load(os.path.join(features_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(features_dir, 'test_labels.npy'))
    test_entry_ids = np.load(os.path.join(features_dir, 'test_entry_ids.npy'))
    
    return (train_features, train_labels, train_entry_ids), \
           (test_features, test_labels, test_entry_ids)


def evaluate_by_entry(predictions, labels, entry_ids):
    """按Entry聚合评估"""
    entry_dict = {}
    for pred, label, entry_id in zip(predictions, labels, entry_ids):
        if entry_id not in entry_dict:
            entry_dict[entry_id] = {'preds': [], 'labels': []}
        entry_dict[entry_id]['preds'].append(pred)
        entry_dict[entry_id]['labels'].append(label)
    
    entry_preds = []
    entry_labels = []
    for entry_id in sorted(entry_dict.keys()):
        entry_preds.append(np.mean(entry_dict[entry_id]['preds']))
        entry_labels.append(np.mean(entry_dict[entry_id]['labels']))
    
    entry_preds = np.array(entry_preds)
    entry_labels = np.array(entry_labels)
    
    mae = mean_absolute_error(entry_labels, entry_preds)
    rmse = np.sqrt(mean_squared_error(entry_labels, entry_preds))
    r2 = r2_score(entry_labels, entry_preds)
    
    return mae, rmse, r2, len(entry_preds)


def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, 
                      train_entry_ids, test_entry_ids, use_scaler=True):
    """训练模型并评估"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")
    
    # 特征标准化
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # 训练
    print(f"训练中...")
    model.fit(X_train_scaled, y_train)
    
    # 预测
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # 图像级评估
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    # Entry级评估
    train_entry_mae, train_entry_rmse, train_entry_r2, n_train_entries = evaluate_by_entry(
        train_pred, y_train, train_entry_ids
    )
    test_entry_mae, test_entry_rmse, test_entry_r2, n_test_entries = evaluate_by_entry(
        test_pred, y_test, test_entry_ids
    )
    
    # 打印结果
    print(f"\n训练集 (图像级):")
    print(f"  MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}")
    print(f"训练集 (Entry级, N={n_train_entries}):")
    print(f"  MAE: {train_entry_mae:.6f}, RMSE: {train_entry_rmse:.6f}, R²: {train_entry_r2:.6f}")
    
    print(f"\n测试集 (图像级):")
    print(f"  MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")
    print(f"测试集 (Entry级, N={n_test_entries}):")
    print(f"  MAE: {test_entry_mae:.6f}, RMSE: {test_entry_rmse:.6f}, R²: {test_entry_r2:.6f}")
    
    results = {
        'model_name': model_name,
        'train': {
            'image_level': {'mae': float(train_mae), 'rmse': float(train_rmse), 'r2': float(train_r2)},
            'entry_level': {'mae': float(train_entry_mae), 'rmse': float(train_entry_rmse), 
                          'r2': float(train_entry_r2), 'n_entries': n_train_entries}
        },
        'test': {
            'image_level': {'mae': float(test_mae), 'rmse': float(test_rmse), 'r2': float(test_r2)},
            'entry_level': {'mae': float(test_entry_mae), 'rmse': float(test_entry_rmse), 
                          'r2': float(test_entry_r2), 'n_entries': n_test_entries}
        }
    }
    
    return results, model, scaler, test_pred


def main():
    parser = argparse.ArgumentParser(description="使用传统机器学习方法进行回归")
    parser.add_argument('--features_dir', type=str, required=True,
                       help='特征文件目录（包含train_features.npy等文件）')
    parser.add_argument('--output_dir', type=str, default='./ml_results',
                       help='结果保存目录')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['ridge', 'lasso', 'elastic', 'svr', 'rf', 'gbm', 'xgb', 'lgb'],
                       choices=['ridge', 'lasso', 'elastic', 'svr', 'rf', 'gbm', 'xgb', 'lgb'],
                       help='要训练的模型列表')
    parser.add_argument('--use_scaler', action='store_true', default=True,
                       help='是否使用特征标准化')
    
    args = parser.parse_args()
    
    # 加载特征
    print(f"加载特征 from {args.features_dir}")
    (train_features, train_labels, train_entry_ids), \
    (test_features, test_labels, test_entry_ids) = load_features(args.features_dir)
    
    print(f"训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 维特征")
    print(f"测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 维特征")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义模型
    models = {}
    
    if 'ridge' in args.models:
        models['Ridge'] = Ridge(alpha=1.0)
    if 'lasso' in args.models:
        models['Lasso'] = Lasso(alpha=0.1)
    if 'elastic' in args.models:
        models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
    if 'svr' in args.models:
        models['SVR'] = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    if 'rf' in args.models:
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
    if 'gbm' in args.models:
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    if 'xgb' in args.models:
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
            )
        else:
            print("[跳过] XGBoost 未安装")
    if 'lgb' in args.models:
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
            )
        else:
            print("[跳过] LightGBM 未安装")
    
    # 训练和评估所有模型
    all_results = {}
    best_model_name = None
    best_test_entry_mae = float('inf')
    
    for model_name, model in models.items():
        results, trained_model, scaler, test_pred = train_and_evaluate(
            model, model_name,
            train_features, train_labels,
            test_features, test_labels,
            train_entry_ids, test_entry_ids,
            use_scaler=args.use_scaler
        )
        
        all_results[model_name] = results
        
        # 保存模型和预测结果
        model_dir = os.path.join(args.output_dir, model_name.lower())
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(trained_model, f)
        if scaler is not None:
            with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        
        # 保存预测结果
        pred_df = pd.DataFrame({
            'entry_id': test_entry_ids,
            'true_label': test_labels,
            'pred_label': test_pred
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
        
        # 更新最佳模型
        if results['test']['entry_level']['mae'] < best_test_entry_mae:
            best_test_entry_mae = results['test']['entry_level']['mae']
            best_model_name = model_name
    
    # 保存所有结果
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("所有模型结果总结")
    print(f"{'='*60}")
    print(f"{'模型':<20} {'测试集Entry MAE':<20} {'测试集Entry RMSE':<20} {'测试集Entry R²':<20}")
    print("-" * 80)
    
    for model_name, results in sorted(all_results.items(), 
                                      key=lambda x: x[1]['test']['entry_level']['mae']):
        test_entry = results['test']['entry_level']
        marker = " ★" if model_name == best_model_name else ""
        print(f"{model_name:<20} {test_entry['mae']:<20.6f} {test_entry['rmse']:<20.6f} {test_entry['r2']:<20.6f}{marker}")
    
    print(f"\n最佳模型: {best_model_name} (测试集Entry MAE: {best_test_entry_mae:.6f})")
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()

