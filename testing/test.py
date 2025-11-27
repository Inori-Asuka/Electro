"""
简化的测试脚本
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import json
import pickle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms, LabelNormalizer
from model import DINOv3RegressionModel
from train.train_utils import evaluate_model, evaluate_by_group, derive_groups_from_dataset

EXCLUDED_IDS = [23, 24]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """将配置文件转换为args对象"""
    class Args:
        pass
    
    args = Args()
    
    data_cfg = config.get('data', {})
    args.data_root = data_cfg.get('data_root', '.')
    args.image_size = data_cfg.get('image_size', 512)
    
    dinov3_cfg = config.get('dinov3', {})
    args.dinov3_model_path = dinov3_cfg.get('model_path', '')
    args.head_type = dinov3_cfg.get('head_type', 'mlp')
    args.head_kwargs = dinov3_cfg.get('head_kwargs', {})
    args.dropout = config.get('training', {}).get('dropout', 0.5)
    
    misc_cfg = config.get('misc', {})
    args.num_workers = misc_cfg.get('num_workers', 4)
    args.batch_size = misc_cfg.get('batch_size', 8)
    
    return args


def test_model(model_path, config_path, test_ids=None):
    """测试模型"""
    # 优先使用checkpoint目录中的config.yaml
    checkpoint_config_path = os.path.join(model_path, 'config.yaml')
    if os.path.exists(checkpoint_config_path):
        print(f"使用checkpoint目录中的配置文件: {checkpoint_config_path}")
        config = load_config(checkpoint_config_path)
    else:
        print(f"使用指定的配置文件: {config_path}")
        config = load_config(config_path)
    
    args = config_to_args(config)
    
    excel_path = os.path.join(args.data_root, 'TEM-EA.xlsx')
    
    # 读取数据
    df = pd.read_excel(excel_path)
    
    # 使用与训练时相同的数据划分方式（直接使用code/science的预期Entry列表）
    if test_ids is None:
        _, test_ids = split_train_test(
            df, 
            excluded_ids=EXCLUDED_IDS,
            use_science_split=True  # 直接使用code/science的预期划分
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print(f"测试集材料数量: {len(test_ids)}")
    
    # 加载模型
    checkpoint_path = os.path.join(model_path, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    print(f"正在加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载归一化器参数
    normalizer = None
    if 'normalizer_params' in checkpoint and checkpoint['normalizer_params'] is not None:
        normalizer = LabelNormalizer()
        normalizer.set_params(checkpoint['normalizer_params'])
        print(f"使用checkpoint中保存的归一化参数")
    else:
        normalizer_path = os.path.join(model_path, 'normalizer.pkl')
        if os.path.exists(normalizer_path):
            with open(normalizer_path, 'rb') as f:
                normalizer_params = pickle.load(f)
            normalizer = LabelNormalizer()
            normalizer.set_params(normalizer_params)
            print(f"从normalizer.pkl加载归一化参数")
    
    # 创建模型
    model = DINOv3RegressionModel(
        model_path=args.dinov3_model_path,
        num_outputs=1,
        head_type=args.head_type,
        dropout=args.dropout,
        head_kwargs=args.head_kwargs,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功 (Epoch: {checkpoint.get('epoch', 'unknown')})")
    
    # 创建测试数据集
    test_dataset = TEMRegressionDataset(
        args.data_root,
        excel_path,
        material_ids=test_ids,
        transform=get_transforms('test', args.image_size),
        use_cleaned=True,
        normalizer=normalizer
    )
    
    print(f"测试集样本数量: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 获取Entry分组
    test_groups = derive_groups_from_dataset(test_dataset, args.data_root)
    
    # 评估模型（图片级）
    criterion = torch.nn.MSELoss()
    test_loss, test_mae_norm, test_mae_denorm, test_r2 = evaluate_model(
        model, test_loader, criterion, device,
        normalizer=normalizer,
    )
    test_rmse = np.sqrt(test_loss) if test_loss > 0 else 0.0
    
    # 评估模型（Entry级）
    test_entry_mae_norm, test_entry_mae_denorm, test_entry_rmse_denorm, test_entry_r2, n_test_entries = evaluate_by_group(
        model, test_loader, test_groups, device,
        agg="mean",
        normalizer=normalizer,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果（图片级）")
    print("=" * 60)
    print(f"MSE:  {test_loss:.6f}")
    print(f"MAE (归一化): {test_mae_norm:.6f}")
    print(f"MAE (反归一化): {test_mae_denorm:.6f} ★")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"R²:   {test_r2:.6f}")
    
    print("\n" + "=" * 60)
    print("测试结果（Entry级）")
    print("=" * 60)
    print(f"MAE (归一化): {test_entry_mae_norm:.6f}")
    print(f"MAE (反归一化): {test_entry_mae_denorm:.6f} ★")
    print(f"RMSE (反归一化): {test_entry_rmse_denorm:.6f}")
    print(f"R²:   {test_entry_r2:.6f}")
    print(f"Entry数量: {n_test_entries}")
    
    # 返回结果（重点关注反归一化后的MAE）
    results = {
        'image_level': {
            'mse': float(test_loss),
            'rmse': float(test_rmse),
            'mae_normalized': float(test_mae_norm),
            'mae_denormalized': float(test_mae_denorm),  # 重点关注
            'r2': float(test_r2)
        },
        'entry_level': {
            'mae_normalized': float(test_entry_mae_norm),
            'mae_denormalized': float(test_entry_mae_denorm),  # 重点关注
            'rmse_denormalized': float(test_entry_rmse_denorm),
            'r2': float(test_entry_r2),
            'n_entries': n_test_entries
        },
        'test_material_ids': test_ids
    }
    
    return results


def main():
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python test.py <model_path> <config_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    
    if os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)
    
    results = test_model(model_path, config_path)
    
    # 保存结果
    results_path = os.path.join(model_path, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n测试结果已保存到: {results_path}")


if __name__ == '__main__':
    main()

