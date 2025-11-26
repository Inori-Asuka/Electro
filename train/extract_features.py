"""
提取特征并保存，用于传统机器学习方法
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms, split_train_test
from model import create_dinov3_model, DINOv3ViTRegressionModel

EXCLUDED_IDS = [23, 24]


def extract_features_from_config(config_path: str):
    """
    从指定配置文件路径提取特征
    
    Args:
        config_path: 配置文件路径（绝对路径或相对路径）
    """
    # 加载配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"使用配置文件: {config_path}")
    
    excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
    df = pd.read_excel(excel_path)
    
    _, test_ids = split_train_test(
        df, 
        excluded_ids=EXCLUDED_IDS
    )
    
    if EXCLUDED_IDS:
        df = df[~df['Entry'].isin(EXCLUDED_IDS)]
    
    all_material_ids = df['Entry'].unique()
    train_ids = np.array([id for id in all_material_ids if id not in test_ids])
    
    print(f"训练集材料数: {len(train_ids)}")
    print(f"测试集材料数: {len(test_ids)}")
    
    # 加载模型
    checkpoint_path = os.path.join(cfg.misc.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # For feature extraction, disable LoRA to avoid compatibility issues
    use_lora = False
    
    # 使用统一的创建函数
    model = create_dinov3_model(
        backbone_type=cfg.dinov3.get('backbone_type', 'convnext'),
        model_path=cfg.dinov3.model_path,
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print(f"已加载模型: {checkpoint_path}")
    
    # 创建数据集
    train_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=train_ids,
        transform=get_transforms('test', cfg.data.image_size), 
        use_cleaned=True,
        normalizer=None 
    )
    
    test_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=test_ids,
        transform=get_transforms('test', cfg.data.image_size),
        use_cleaned=True,
        normalizer=None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # 不shuffle，保持顺序
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # 提取特征
    def extract_from_loader(loader, dataset, dataset_name):
        all_features = []
        all_labels = []
        all_entry_ids = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Extract {dataset_name} Feature")):
                images = images.to(device)
                features = model.extract_features(images)  # [B, embed_dim]
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # 获取对应的Entry ID
                start_idx = batch_idx * loader.batch_size
                end_idx = min(start_idx + len(labels), len(dataset.data_list))
                for i in range(start_idx, end_idx):
                    entry_id = dataset.data_list[i]['material_id']
                    all_entry_ids.append(entry_id)
        
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        entry_ids_array = np.array(all_entry_ids)
        
        return features_array, labels_array, entry_ids_array
    
    # 提取训练集特征
    train_features, train_labels, train_entry_ids = extract_from_loader(
        train_loader, train_dataset, "Train Dataset"
    )
    
    # 提取测试集特征
    test_features, test_labels, test_entry_ids = extract_from_loader(
        test_loader, test_dataset, "Test Dataset"
    )
    
    # 保存特征
    output_dir = os.path.join(cfg.misc.checkpoint_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为numpy格式
    np.save(os.path.join(output_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'train_entry_ids.npy'), train_entry_ids)
    
    np.save(os.path.join(output_dir, 'test_features.npy'), test_features)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    np.save(os.path.join(output_dir, 'test_entry_ids.npy'), test_entry_ids)
    
    # 保存为CSV格式（便于查看）
    train_df = pd.DataFrame({
        'entry_id': train_entry_ids,
        'label': train_labels,
    })
    train_df.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
    
    test_df = pd.DataFrame({
        'entry_id': test_entry_ids,
        'label': test_labels,
    })
    test_df.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    
    # 保存特征矩阵（用于ML）
    # 将特征作为列添加到DataFrame
    feature_cols = [f'feature_{i}' for i in range(train_features.shape[1])]
    train_features_df = pd.DataFrame(train_features, columns=feature_cols)
    train_features_df['entry_id'] = train_entry_ids
    train_features_df['label'] = train_labels
    train_features_df.to_csv(os.path.join(output_dir, 'train_features_full.csv'), index=False)
    
    test_features_df = pd.DataFrame(test_features, columns=feature_cols)
    test_features_df['entry_id'] = test_entry_ids
    test_features_df['label'] = test_labels
    test_features_df.to_csv(os.path.join(output_dir, 'test_features_full.csv'), index=False)
    
    print(f"\n特征提取完成！")
    print(f"训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 维特征")
    print(f"测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 维特征")
    print(f"特征保存路径: {output_dir}")
    print(f"\n文件列表:")
    print(f"  - train_features.npy / train_features_full.csv")
    print(f"  - test_features.npy / test_features_full.csv")
    print(f"  - train_labels.npy / train_entry_ids.npy")
    print(f"  - test_labels.npy / test_entry_ids.npy")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def extract_features(cfg: DictConfig):
    """
    提取特征并保存，用于传统机器学习方法
    
    使用方法:
        python extract_features.py                          # 使用默认配置 config.yaml
        python extract_features.py --config-name=config_vit_lora_qkv  # 使用指定配置
        python extract_features.py --config-path=/path/to/config.yaml  # 直接指定配置文件路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
    df = pd.read_excel(excel_path)
    
    _, test_ids = split_train_test(
        df, 
        excluded_ids=EXCLUDED_IDS
    )
    
    if EXCLUDED_IDS:
        df = df[~df['Entry'].isin(EXCLUDED_IDS)]
    
    all_material_ids = df['Entry'].unique()
    train_ids = np.array([id for id in all_material_ids if id not in test_ids])
    
    print(f"训练集材料数: {len(train_ids)}")
    print(f"测试集材料数: {len(test_ids)}")
    
    # 加载模型
    checkpoint_path = os.path.join(cfg.misc.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # For feature extraction, disable LoRA to avoid compatibility issues
    use_lora = False
    
    # 使用统一的创建函数
    model = create_dinov3_model(
        backbone_type=cfg.dinov3.get('backbone_type', 'convnext'),
        model_path=cfg.dinov3.model_path,
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print(f"已加载模型: {checkpoint_path}")
    
    # 创建数据集
    train_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=train_ids,
        transform=get_transforms('test', cfg.data.image_size), 
        use_cleaned=True,
        normalizer=None 
    )
    
    test_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=test_ids,
        transform=get_transforms('test', cfg.data.image_size),
        use_cleaned=True,
        normalizer=None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # 不shuffle，保持顺序
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # 提取特征
    def extract_from_loader(loader, dataset, dataset_name):
        all_features = []
        all_labels = []
        all_entry_ids = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Extract {dataset_name} Feature")):
                images = images.to(device)
                features = model.extract_features(images)  # [B, embed_dim]
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # 获取对应的Entry ID
                start_idx = batch_idx * loader.batch_size
                end_idx = min(start_idx + len(labels), len(dataset.data_list))
                for i in range(start_idx, end_idx):
                    entry_id = dataset.data_list[i]['material_id']
                    all_entry_ids.append(entry_id)
        
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        entry_ids_array = np.array(all_entry_ids)
        
        return features_array, labels_array, entry_ids_array
    
    # 提取训练集特征
    train_features, train_labels, train_entry_ids = extract_from_loader(
        train_loader, train_dataset, "Train Dataset"
    )
    
    # 提取测试集特征
    test_features, test_labels, test_entry_ids = extract_from_loader(
        test_loader, test_dataset, "Test Dataset"
    )
    
    # 保存特征
    output_dir = os.path.join(cfg.misc.checkpoint_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为numpy格式
    np.save(os.path.join(output_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'train_entry_ids.npy'), train_entry_ids)
    
    np.save(os.path.join(output_dir, 'test_features.npy'), test_features)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    np.save(os.path.join(output_dir, 'test_entry_ids.npy'), test_entry_ids)
    
    # 保存为CSV格式（便于查看）
    train_df = pd.DataFrame({
        'entry_id': train_entry_ids,
        'label': train_labels,
    })
    train_df.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
    
    test_df = pd.DataFrame({
        'entry_id': test_entry_ids,
        'label': test_labels,
    })
    test_df.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    
    # 保存特征矩阵（用于ML）
    # 将特征作为列添加到DataFrame
    feature_cols = [f'feature_{i}' for i in range(train_features.shape[1])]
    train_features_df = pd.DataFrame(train_features, columns=feature_cols)
    train_features_df['entry_id'] = train_entry_ids
    train_features_df['label'] = train_labels
    train_features_df.to_csv(os.path.join(output_dir, 'train_features_full.csv'), index=False)
    
    test_features_df = pd.DataFrame(test_features, columns=feature_cols)
    test_features_df['entry_id'] = test_entry_ids
    test_features_df['label'] = test_labels
    test_features_df.to_csv(os.path.join(output_dir, 'test_features_full.csv'), index=False)
    
    print(f"\n特征提取完成！")
    print(f"训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 维特征")
    print(f"测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 维特征")
    print(f"特征保存路径: {output_dir}")
    print(f"\n文件列表:")
    print(f"  - train_features.npy / train_features_full.csv")
    print(f"  - test_features.npy / test_features_full.csv")
    print(f"  - train_labels.npy / train_entry_ids.npy")
    print(f"  - test_labels.npy / test_entry_ids.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提取特征并保存')
    parser.add_argument('--config-path', type=str, default=None,
                       help='配置文件路径（绝对路径或相对路径）。如果指定，将直接使用该配置文件，而不是 Hydra 配置系统')
    
    # 解析参数（Hydra 会处理自己的参数，我们需要先检查是否有 --config-path）
    args, unknown = parser.parse_known_args()
    
    if args.config_path:
        # 直接使用指定路径的配置文件
        extract_features_from_config(args.config_path)
    else:
        # 使用 Hydra 配置系统
        extract_features()

