"""
完善的测试脚本
支持配置文件、保存路径参数，并保存 GT 和预测值到 CSV
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import pickle
from omegaconf import OmegaConf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms, split_train_test, LabelNormalizer
from model.model_factory import create_dinov3_model
from training.train_utils import evaluate_model, evaluate_by_group, derive_groups_from_dataset

EXCLUDED_IDS = [23, 24]


def get_predictions_and_labels(model, data_loader, device, normalizer=None):
    """获取所有预测值和真实值"""
    model.eval()
    all_preds = []
    all_labels = []
    all_entry_ids = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="获取预测值")):
            images = images.to(device)
            
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            else:
                labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # 收集预测值和标签
            preds_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            if isinstance(preds_np, float):
                all_preds.append(preds_np)
                all_labels.append(labels_np)
            else:
                all_preds.extend(preds_np.tolist())
                all_labels.extend(labels_np.tolist())
            
            # 获取对应的 Entry ID
            batch_size = images.size(0)
            start_idx = batch_idx * data_loader.batch_size
            end_idx = min(start_idx + batch_size, len(data_loader.dataset.data_list))
            for i in range(start_idx, end_idx):
                if i < len(data_loader.dataset.data_list):
                    entry_id = data_loader.dataset.data_list[i]['material_id']
                    all_entry_ids.append(entry_id)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 反归一化
    if normalizer is not None:
        preds_denorm = normalizer.inverse_transform(all_preds)
        labels_denorm = normalizer.inverse_transform(all_labels)
    else:
        preds_denorm = all_preds
        labels_denorm = all_labels
    
    return all_preds, all_labels, preds_denorm, labels_denorm, all_entry_ids


def get_entry_level_predictions(model, data_loader, groups, device, normalizer=None):
    """获取 Entry 级别的预测值和真实值"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="获取 Entry 级预测值")):
            images = images.to(device)
            
            if torch.is_tensor(labels):
                labels = labels.to(device).float()
            else:
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    
    if len(all_preds) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    groups = np.asarray(groups)
    
    # 按 Entry 分组聚合
    uniq = np.unique(groups)
    agg_pred = []
    agg_true = []
    entry_ids = []
    
    for g in uniq:
        mask = (groups == g)
        p = preds[mask]
        y = labels[mask]
        
        agg_pred.append(np.mean(p))
        agg_true.append(np.mean(y))
        entry_ids.append(g)
    
    agg_pred = np.array(agg_pred)
    agg_true = np.array(agg_true)
    
    # 反归一化
    if normalizer is not None:
        agg_pred_denorm = normalizer.inverse_transform(agg_pred)
        agg_true_denorm = normalizer.inverse_transform(agg_true)
    else:
        agg_pred_denorm = agg_pred
        agg_true_denorm = agg_true
    
    return agg_pred, agg_true, agg_pred_denorm, agg_true_denorm, entry_ids


def test_model(config_path, checkpoint_path=None, output_dir=None):
    """测试模型"""
    # 加载配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    print(f"使用配置文件: {config_path}")
    
    # 如果指定了 checkpoint_path，优先使用 checkpoint 目录中的 config.yaml
    if checkpoint_path is None:
        checkpoint_path = cfg.misc.checkpoint_dir
    
    checkpoint_config_path = os.path.join(checkpoint_path, 'config.yaml')
    if os.path.exists(checkpoint_config_path):
        print(f"使用 checkpoint 目录中的配置文件: {checkpoint_config_path}")
        cfg = OmegaConf.load(checkpoint_config_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = checkpoint_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果保存路径: {output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
    df = pd.read_excel(excel_path)
    
    # 划分训练集和测试集（与训练时保持一致）
    train_ids, test_ids = split_train_test(
        df,
        excluded_ids=EXCLUDED_IDS
    )
    
    # 确保 test_ids 是 numpy 数组
    if not isinstance(test_ids, np.ndarray):
        test_ids = np.array(test_ids)
    
    print(f"训练集材料数: {len(train_ids)}")
    print(f"测试集材料数: {len(test_ids)}")
    
    # 加载模型权重
    model_checkpoint_path = checkpoint_path
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {model_checkpoint_path}")
    
    print(f"正在加载模型: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    # 加载归一化器参数
    normalizer = None
    if 'normalizer_params' in checkpoint and checkpoint['normalizer_params'] is not None:
        normalizer = LabelNormalizer()
        normalizer.set_params(checkpoint['normalizer_params'])
        print(f"使用 checkpoint 中保存的归一化参数")
    else:
        normalizer_path = os.path.join(checkpoint_path, 'normalizer.pkl')
        if os.path.exists(normalizer_path):
            with open(normalizer_path, 'rb') as f:
                normalizer_params = pickle.load(f)
            normalizer = LabelNormalizer()
            normalizer.set_params(normalizer_params)
            print(f"从 normalizer.pkl 加载归一化参数")
    
    # 创建模型（使用统一的创建函数，支持所有 backbone）
    use_lora = False  # 测试时不使用 LoRA
    model = create_dinov3_model(
        backbone_type=cfg.dinov3.get('backbone_type', 'convnext'),
        model_path=cfg.dinov3.get('model_path', ''),
        model_name=cfg.dinov3.get('model_name', 'resnet50'),
        pretrained=cfg.dinov3.get('pretrained', True),
        num_outputs=1,
        pooling=cfg.dinov3.get('pooling', 'gap'),
        head_type=cfg.dinov3.head_type,
        dropout=cfg.training.dropout,
        freeze_backbone=cfg.dinov3.freeze_backbone,
        freeze_layers=cfg.dinov3.get('freeze_layers', None),
        use_lora=use_lora,
        lora_r=cfg.dinov3.get('lora_r', 16),
        lora_alpha=cfg.dinov3.get('lora_alpha', 32),
        lora_dropout=cfg.dinov3.get('lora_dropout', 0.1),
        lora_modules=cfg.dinov3.get('lora_modules', None),
        head_kwargs=cfg.dinov3.get('head_kwargs', {}),
        multiscale_layers=cfg.dinov3.get('multiscale_layers', None),
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"模型加载成功 (Epoch: {checkpoint.get('epoch', 'unknown')})")
    
    # 创建测试数据集
    test_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=test_ids,
        transform=get_transforms('test', cfg.data.image_size),
        use_cleaned=True,
        normalizer=normalizer
    )
    
    print(f"测试集样本数量: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 获取 Entry 分组
    test_groups = derive_groups_from_dataset(test_dataset, cfg.data.data_root)
    
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
    
    # 获取预测值和真实值
    print("\n正在收集预测值...")
    preds_norm, labels_norm, preds_denorm, labels_denorm, entry_ids = get_predictions_and_labels(
        model, test_loader, device, normalizer
    )
    
    # 获取 Entry 级别的预测值
    entry_preds_norm, entry_labels_norm, entry_preds_denorm, entry_labels_denorm, entry_ids_agg = get_entry_level_predictions(
        model, test_loader, test_groups, device, normalizer
    )
    
    # 保存图片级结果到 CSV
    image_level_df = pd.DataFrame({
        'entry_id': entry_ids,
        'predicted_normalized': preds_norm,
        'ground_truth_normalized': labels_norm,
        'predicted_denormalized': preds_denorm,
        'ground_truth_denormalized': labels_denorm,
        'error': labels_denorm - preds_denorm,
        'absolute_error': np.abs(labels_denorm - preds_denorm),
    })
    os.makedirs(os.path.join(output_dir, 'test_results'),exist_ok=True)
    
    image_level_csv_path = os.path.join(output_dir, 'test_results','test_results_image_level.csv')
    image_level_df.to_csv(image_level_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n图片级结果已保存到: {image_level_csv_path}")
    
    # 保存 Entry 级结果到 CSV
    entry_level_df = pd.DataFrame({
        'entry_id': entry_ids_agg,
        'predicted_normalized': entry_preds_norm,
        'ground_truth_normalized': entry_labels_norm,
        'predicted_denormalized': entry_preds_denorm,
        'ground_truth_denormalized': entry_labels_denorm,
        'error': entry_labels_denorm - entry_preds_denorm,
        'absolute_error': np.abs(entry_labels_denorm - entry_preds_denorm),
    })
    entry_level_csv_path = os.path.join(output_dir,'test_results', 'test_results_entry_level.csv')
    entry_level_df.to_csv(entry_level_csv_path, index=False, encoding='utf-8-sig')
    print(f"Entry 级结果已保存到: {entry_level_csv_path}")
    
    # 保存评估指标到 JSON
    results = {
        'image_level': {
            'mse': float(test_loss),
            'rmse': float(test_rmse),
            'mae_normalized': float(test_mae_norm),
            'mae_denormalized': float(test_mae_denorm),
            'r2': float(test_r2),
            'n_samples': len(preds_denorm)
        },
        'entry_level': {
            'mae_normalized': float(test_entry_mae_norm),
            'mae_denormalized': float(test_entry_mae_denorm),
            'rmse_denormalized': float(test_entry_rmse_denorm),
            'r2': float(test_entry_r2),
            'n_entries': n_test_entries
        },
        'test_material_ids': test_ids.tolist() if isinstance(test_ids, np.ndarray) else test_ids
    }
    
    results_json_path = os.path.join(output_dir, 'test_results','test_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {results_json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试模型并保存预测结果')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（YAML 格式）')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型 checkpoint 目录路径（如果未指定，将使用配置文件中的 checkpoint_dir）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='结果保存目录（如果未指定，将保存到 checkpoint 目录）')
    
    args = parser.parse_args()
    
    results = test_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"图片级 MAE (反归一化): {results['image_level']['mae_denormalized']:.6f}")
    print(f"Entry 级 MAE (反归一化): {results['entry_level']['mae_denormalized']:.6f}")


if __name__ == '__main__':
    main()
