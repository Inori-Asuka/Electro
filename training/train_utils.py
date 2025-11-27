"""
训练工具模块
支持Entry级评估
"""
import os
import numpy as np
import torch
from typing import Tuple, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    max_grad_norm=1.0,
):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(data_loader):
        images, labels = batch_data
        
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
        else:
            labels = labels.to(device).float()
        
        images = images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        if scheduler is not None and hasattr(scheduler, 'step') and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
    
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    normalizer=None,
):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            images, labels = batch_data
            
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            else:
                labels = labels.to(device).float()
            
            images = images.to(device)
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            
            if outputs.dim() == 0:
                pred_value = outputs.item()
                true_value = labels.item()
            else:
                pred_value = outputs.cpu().numpy()
                true_value = labels.cpu().numpy()
            
            if isinstance(pred_value, float):
                all_preds.append(pred_value)
                all_labels.append(true_value)
            else:
                all_preds.extend(pred_value.tolist())
                all_labels.extend(true_value.tolist())
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    
    if total_samples > 0:
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        mae_normalized = float(np.mean(np.abs(preds_arr - labels_arr)))
        
        if normalizer is not None:
            preds_denorm = normalizer.inverse_transform(preds_arr)
            labels_denorm = normalizer.inverse_transform(labels_arr)
            mae_denormalized = float(np.mean(np.abs(preds_denorm - labels_denorm)))
        else:
            mae_denormalized = mae_normalized 
        
        if labels_arr.size > 0:
            mean_label = np.mean(labels_arr)
            ss_tot = np.sum((labels_arr - mean_label) ** 2)
            ss_res = np.sum((labels_arr - preds_arr) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        else:
            r2 = 0.0
    else:
        mae_normalized = 0.0
        mae_denormalized = 0.0
        r2 = 0.0
    
    return avg_loss, mae_normalized, mae_denormalized, r2


def evaluate_by_group(
    model,
    data_loader,
    groups: List[str],
    device,
    agg: str = "mean",
    normalizer=None,
) :
    model.eval()
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            images, labels = batch_data
            
            if torch.is_tensor(labels):
                labels = labels.to(device).float()
            else:
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            images = images.to(device)
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            batch_size = images.size(0)
            start_idx = batch_idx * data_loader.batch_size
            end_idx = min(start_idx + batch_size, len(groups))
            batch_groups = groups[start_idx:end_idx]
            all_groups.extend(batch_groups[:batch_size])
    
    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    groups = np.asarray(all_groups)
    
    if not (len(preds) == len(labels) == len(groups)):
        raise ValueError("evaluate_by_group: preds/labels/groups 长度不一致")
    
    uniq = np.unique(groups)
    agg_pred, agg_true = [], []
    
    for g in uniq:
        mask = (groups == g)
        p = preds[mask]
        y = labels[mask]
        
        if agg == "median":
            agg_pred.append(np.median(p))
            agg_true.append(np.median(y))
        else:
            agg_pred.append(np.mean(p))
            agg_true.append(np.mean(y))
    
    agg_pred = np.asarray(agg_pred)
    agg_true = np.asarray(agg_true)
    
    mae_normalized = float(np.mean(np.abs(agg_pred - agg_true)))
    
    if normalizer is not None:
        agg_pred_denorm = normalizer.inverse_transform(agg_pred)
        agg_true_denorm = normalizer.inverse_transform(agg_true)
        mae_denormalized = float(np.mean(np.abs(agg_pred_denorm - agg_true_denorm)))
        mse_denormalized = float(np.mean((agg_pred_denorm - agg_true_denorm) ** 2))
        rmse_denormalized = float(np.sqrt(mse_denormalized))
    else:
        mae_denormalized = mae_normalized
        mse_denormalized = float(np.mean((agg_pred - agg_true) ** 2))
        rmse_denormalized = float(np.sqrt(mse_denormalized))
    
    if len(agg_true) > 1:
        ss_tot = float(np.sum((agg_true - np.mean(agg_true)) ** 2))
        ss_res = float(np.sum((agg_true - agg_pred) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    else:
        r2 = 0.0
    
    return mae_normalized, mae_denormalized, rmse_denormalized, r2, int(len(uniq))


def derive_groups_from_dataset(dataset, data_root: Optional[str] = None) -> List[str]:
    groups = []
    for item in dataset.data_list:
        image_path = item['image_path']
        if data_root:
            try:
                rel = os.path.relpath(image_path, data_root)
            except ValueError:
                rel = image_path
            group = rel.replace("\\", "/").split("/")[0]
        else:
            group = os.path.basename(os.path.dirname(image_path))
        groups.append(group)
    return groups

