"""
使用 Hydra 管理配置的 DINOv3 训练脚本
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pickle
from datetime import datetime
import json
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TEMRegressionDataset, get_transforms, split_train_test, LabelNormalizer, create_normalizer
from model import create_dinov3_model
from train_utils import train_one_epoch, evaluate_model, evaluate_by_group, derive_groups_from_dataset

EXCLUDED_IDS = [23, 24]


def save_config(config: DictConfig, save_path: str):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(config=config_dict, f=f)
    print(f"config save to: {save_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    torch.manual_seed(cfg.misc.seed)
    np.random.seed(cfg.misc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.misc.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    excel_path = os.path.join(cfg.data.data_root, 'TEM-EA.xlsx')
    df = pd.read_excel(excel_path)
    
    # 划分训练集和测试集
    train_ids, test_ids = split_train_test(
        df, 
        excluded_ids=EXCLUDED_IDS
    )
    
    checkpoint_dir = cfg.misc.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
    save_config(cfg, config_save_path)
    
    normalizer = None
    if cfg.normalization.method is not None and cfg.normalization.method != 'null':
        train_df = df[df['Entry'].isin(train_ids)]
        train_labels = train_df['Eapp'].values
        normalizer = create_normalizer(method=cfg.normalization.method, labels=train_labels)
        print(f"归一化方法: {cfg.normalization.method}")
        print(f"归一化参数: {normalizer.get_params()}")
        
        # 保存归一化器参数
        normalizer_path = os.path.join(checkpoint_dir, 'normalizer.pkl')
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer.get_params(), f)
        print(f"归一化器参数已保存到: {normalizer_path}")
    
    
    # 创建数据集
    train_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=train_ids,
        transform=get_transforms('train', cfg.data.image_size),
        use_cleaned=True,
        normalizer=normalizer
    )
    
    # 将测试集作为验证集
    val_dataset = TEMRegressionDataset(
        cfg.data.data_root,
        excel_path,
        material_ids=test_ids,
        transform=get_transforms('test', cfg.data.image_size),
        use_cleaned=True,
        normalizer=normalizer
    )
    
    # test_dataset和val_dataset是同一个
    test_dataset = val_dataset
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.misc.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
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
    
    val_groups = derive_groups_from_dataset(val_dataset, cfg.data.data_root)
    test_groups = val_groups
    
    # 创建模型（使用统一的创建函数）
    model = create_dinov3_model(
        backbone_type=cfg.dinov3.get('backbone_type', 'convnext'),
        model_path=cfg.dinov3.model_path,
        num_outputs=1,
        pooling=cfg.dinov3.get('pooling', 'gap'),
        head_type=cfg.dinov3.head_type,
        dropout=cfg.training.dropout,
        freeze_backbone=cfg.dinov3.freeze_backbone,
        freeze_layers=cfg.dinov3.get('freeze_layers', None),
        use_lora=cfg.dinov3.get('use_lora', False),
        lora_r=cfg.dinov3.get('lora_r', 16),
        lora_alpha=cfg.dinov3.get('lora_alpha', 32),
        lora_dropout=cfg.dinov3.get('lora_dropout', 0.1),
        lora_modules=cfg.dinov3.get('lora_modules', None),  # 仅 ViT: ['qkv', 'proj', 'mlp']
        head_kwargs=cfg.dinov3.get('head_kwargs', {}),
        multiscale_layers=cfg.dinov3.get('multiscale_layers', None),
    ).to(device)
    
    trainable_params, total_params = model.get_trainable_parameters()
    backbone_type = cfg.dinov3.get('backbone_type', 'convnext')
    print(f"\n模型: DINOv3 {'ViT Large' if backbone_type == 'vit' else 'ConvNeXt-Large'}")
    if backbone_type == 'vit':
        pooling = cfg.dinov3.get('pooling', 'gap')
        print(f"预测方案: {pooling}")
        if pooling == 'multiscale':
            print(f"多尺度层: {cfg.dinov3.get('multiscale_layers', [6, 12, 18, 23])}")
        if cfg.dinov3.get('use_lora', False) and cfg.dinov3.get('lora_modules'):
            print(f"LoRA 模块: {cfg.dinov3.get('lora_modules')}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    
    if cfg.loss.type == 'mse':
        criterion = nn.MSELoss()
    elif cfg.loss.type == 'smooth_l1':
        criterion = nn.SmoothL1Loss(beta=cfg.loss.smooth_l1_beta)
    elif cfg.loss.type == 'huber':
        criterion = nn.HuberLoss(delta=cfg.loss.huber_delta)
    else:
        criterion = nn.MSELoss()
    
    
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    
    if cfg.training.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg.training.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    scheduler = None
    scheduler_type = cfg.scheduler.get('type', None)
    if scheduler_type is None or scheduler_type == 'null':
        scheduler = None
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
    elif scheduler_type == 'cosine':
        warmup_epochs = cfg.scheduler.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (cfg.training.epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs, eta_min=lr * 0.001
            )
    elif scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=cfg.training.epochs,
            steps_per_epoch=len(train_loader), pct_start=0.3,
            div_factor=25.0, final_div_factor=10000.0
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    # TensorBoard
    writer = None
    if cfg.misc.use_tensorboard:
        log_dir = cfg.misc.log_dir.replace('${exp}', cfg.exp)
        writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    best_val_rmse = float('inf')
    best_entry_rmse = float('inf')
    patience_counter = 0
    best_epoch = -1
    
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")
        print("-" * 60)
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler if scheduler_type == 'onecycle' else None,
            max_grad_norm=cfg.misc.max_grad_norm,
        )
        
        val_loss, val_mae_norm, val_mae_denorm, val_r2 = evaluate_model(
            model, val_loader, criterion, device,
            normalizer=normalizer,
        )
        val_rmse = np.sqrt(val_loss) if val_loss > 0 else 0.0
        
        entry_mae_norm, entry_mae_denorm, entry_rmse_denorm, entry_r2, n_entries = evaluate_by_group(
            model, val_loader, val_groups, device,
            agg="mean",
            normalizer=normalizer,
        )
        
        if scheduler is not None:
            if scheduler_type == 'plateau':
                metric = entry_mae_denorm 
                scheduler.step(metric)
            elif scheduler_type in ['cosine', 'onecycle']:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        is_best = False
        best_metric = cfg.training.best_metric
        if best_metric == 'entry_mae' or best_metric == 'entry_mae_denorm':
            if entry_mae_denorm < best_entry_rmse: 
                best_entry_rmse = entry_mae_denorm
                best_val_rmse = val_mae_denorm
                patience_counter = 0
                best_epoch = epoch
                is_best = True
            else:
                patience_counter += 1
        elif best_metric == 'entry_rmse':
            if entry_rmse_denorm < best_entry_rmse:
                best_entry_rmse = entry_rmse_denorm
                best_val_rmse = val_mae_denorm
                patience_counter = 0
                best_epoch = epoch
                is_best = True
            else:
                patience_counter += 1
        else:
            if val_mae_denorm < best_val_rmse:
                best_val_rmse = val_mae_denorm
                best_entry_rmse = entry_mae_denorm
                patience_counter = 0
                best_epoch = epoch
                is_best = True
            else:
                patience_counter += 1
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae_denorm': val_mae_denorm,
                'entry_mae_denorm': entry_mae_denorm,
                'entry_rmse_denorm': entry_rmse_denorm,
                'val_r2': val_r2,
                'entry_r2': entry_r2,
                'normalizer_params': normalizer.get_params() if normalizer is not None else None,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        print(f"Train Loss: {train_loss:.4f} | "
              f"Val MAE(denorm): {val_mae_denorm:.6f}, Val R²: {val_r2:.4f} | "
              f"Entry MAE(denorm): {entry_mae_denorm:.6f}, Entry RMSE(denorm): {entry_rmse_denorm:.6f}, Entry R²: {entry_r2:.4f} (N={n_entries}) | "
              f"LR: {current_lr:.2e}", end='')
        if is_best:
            print(" ★")
        else:
            print()
        
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('MAE/Val_Normalized', val_mae_norm, epoch)
            writer.add_scalar('MAE/Val_Denormalized', val_mae_denorm, epoch)
            writer.add_scalar('MAE/Entry_Normalized', entry_mae_norm, epoch)
            writer.add_scalar('MAE/Entry_Denormalized', entry_mae_denorm, epoch)
            writer.add_scalar('RMSE/Entry_Denormalized', entry_rmse_denorm, epoch)
            writer.add_scalar('R2/Val', val_r2, epoch)
            writer.add_scalar('R2/Entry', entry_r2, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 早停
        if patience_counter >= cfg.training.patience:
            print(f"验证性能在{cfg.training.patience}个epoch内未改善，提前停止训练")
            break
    
    print("\n" + "=" * 60)
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae_norm, test_mae_denorm, test_r2 = evaluate_model(
        model, test_loader, criterion, device,
        normalizer=normalizer,
    )
    test_rmse = np.sqrt(test_loss) if test_loss > 0 else 0.0
    
    test_entry_mae_norm, test_entry_mae_denorm, test_entry_rmse_denorm, test_entry_r2, n_test_entries = evaluate_by_group(
        model, test_loader, test_groups, device,
        agg="mean",
        normalizer=normalizer,
    )
    
    print("\n" + "=" * 60)
    print("Image Level")
    print("=" * 60)
    print(f"Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}")
    print(f"MAE (归一化): {test_mae_norm:.6f}")
    print(f"MAE (反归一化): {test_mae_denorm:.6f} ")
    print(f"R²: {test_r2:.6f}")
    
    print("\n" + "=" * 60)
    print("Entry Level")
    print("=" * 60)
    print(f"MAE (归一化): {test_entry_mae_norm:.6f}")
    print(f"MAE (反归一化): {test_entry_mae_denorm:.6f} ★")
    print(f"RMSE (反归一化): {test_entry_rmse_denorm:.6f}")
    print(f"R²: {test_entry_r2:.6f}")
    print(f"Entry数量: {n_test_entries}")
    
    test_results = {
        'image_level': {
            'loss': float(test_loss),
            'rmse': float(test_rmse),
            'mae_normalized': float(test_mae_norm),
            'mae_denormalized': float(test_mae_denorm),  
            'r2': float(test_r2),
        },
        'entry_level': {
            'mae_normalized': float(test_entry_mae_norm),
            'mae_denormalized': float(test_entry_mae_denorm), 
            'rmse_denormalized': float(test_entry_rmse_denorm),
            'r2': float(test_entry_r2),
            'n_entries': n_test_entries,
        },
        'best_epoch': best_epoch,
        'best_entry_mae_denormalized': float(best_entry_rmse),
    }
    
    results_path = os.path.join(checkpoint_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    if writer:
        writer.close()
     
    return test_entry_mae_denorm, test_entry_rmse_denorm, test_entry_r2


if __name__ == '__main__':
    train()

