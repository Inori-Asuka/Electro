# prepare_tem_abmil_data.py
# -*- coding: utf-8 -*-
"""
TEM 多尺度 (Mixed-Scale) ABMIL 数据预处理脚本 - 无测试集版

配置特点：
1. **无测试集**：所有数据按比例划分为 Train / Val (用于有独立外部测试集的场景)。
2. **分层采样**：基于 Eapp 的分位数 (Quantile) 进行分层，保证 Train/Val 分布一致。
3. **混合尺度**：默认 128 (Stride 64) + 256 (Stride 128)。
"""

import os
import argparse
import json
import random
import shutil
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# 防止处理超大图报错
Image.MAX_IMAGE_PIXELS = None

ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def is_processed_image(fname: str) -> bool:
    stem, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext not in ALLOWED_EXTS:
        return False
    if not stem.endswith("_c"):
        return False
    if stem.endswith("_c_mask") or "_mask" in stem:
        return False
    return True


def load_mask(mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as m:
        m = m.convert("L")
        arr = np.array(m)
        fg = (arr > 0).astype(np.uint8)
    return fg


def create_patches_for_image(
        img_path: str,
        mask_path: str,
        out_dir_small: str,
        out_dir_large: str,
        small_size: int,
        small_stride: int,
        large_size: int,
        large_stride: int,
        min_fg_ratio: float,
) -> Tuple[int, int]:
    # 使用上下文管理器打开图像
    with Image.open(img_path) as img_obj:
        if img_obj.mode not in ("L", "RGB"):
            img = img_obj.convert("L")
        else:
            img = img_obj.copy()

    mask_fg = load_mask(mask_path)
    h, w = mask_fg.shape

    if (w, h) != img.size:
        raise ValueError(f"尺寸不一致: {img_path} vs {mask_path}")

    n_small = 0
    n_large = 0

    # --- Small Scale ---
    ps = small_size
    if ps is not None and small_stride is not None:
        ensure_dir(out_dir_small)
        for top in range(0, h - ps + 1, small_stride):
            for left in range(0, w - ps + 1, small_stride):
                mask_patch = mask_fg[top:top + ps, left:left + ps]
                if mask_patch.mean() < min_fg_ratio:
                    continue

                patch_img = img.crop((left, top, left + ps, top + ps))
                patch_name = f"patch_{top}_{left}.png"
                patch_img.save(os.path.join(out_dir_small, patch_name))
                n_small += 1

    # --- Large Scale ---
    pl = large_size
    if pl is not None and large_stride is not None:
        ensure_dir(out_dir_large)
        for top in range(0, h - pl + 1, large_stride):
            for left in range(0, w - pl + 1, large_stride):
                mask_patch = mask_fg[top:top + pl, left:left + pl]
                if mask_patch.mean() < min_fg_ratio:
                    continue

                patch_img = img.crop((left, top, left + pl, top + pl))
                patch_name = f"patch_{top}_{left}.png"
                patch_img.save(os.path.join(out_dir_large, patch_name))
                n_large += 1

    return n_small, n_large


def assign_label_bins(entry_df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    基于 Eapp 进行分箱 (Quantile Binning)，用于分层采样。
    """
    try:
        # 尝试使用分位数分箱 (保证每个 bin 数量大致相等)
        entry_df["bin_id"] = pd.qcut(entry_df["Eapp"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # 如果数据太少或数值重复太多，回退到等宽分箱
        entry_df["bin_id"] = pd.cut(entry_df["Eapp"], bins=n_bins, labels=False)
    return entry_df


def stratified_train_val_split(
        entry_df: pd.DataFrame,
        val_ratio: float = 0.2,
        seed: int = 42,
) -> Dict[str, List[str]]:
    """
    分层划分 Train / Val (无 Test)。
    """
    assert 0 <= val_ratio < 1.0

    rng = random.Random(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": []}

    # 按 bin_id 分组进行划分，实现分层采样
    for bin_id, group in entry_df.groupby("bin_id"):
        entries = list(group["entry_id"].astype(str).unique())
        rng.shuffle(entries)
        n = len(entries)
        if n == 0:
            continue

        # 计算验证集数量
        n_val = int(round(n * val_ratio))

        # 边界处理：如果 val_ratio > 0 但计算结果为 0，且样本数足够，强制分 1 个给 val
        if n_val == 0 and val_ratio > 0 and n > 1:
            n_val = 1

        # 剩下的全部给 train
        val_entries = entries[:n_val]
        train_entries = entries[n_val:]

        splits["val"].extend(val_entries)
        splits["train"].extend(train_entries)

    # 排序去重
    for k in splits:
        splits[k] = sorted(set(splits[k]))

    # 兜底检查：如果有漏网之鱼（极少情况），放入 Train
    all_assigned = set(splits["train"]) | set(splits["val"])
    all_entries = set(entry_df["entry_id"].astype(str).unique())
    missing = all_entries - all_assigned
    if missing:
        print(f"[WARN] 未分配 Entry {missing}，默认放入 Train")
        splits["train"].extend(sorted(missing))

    return splits


def compute_bin_weights(entry_df: pd.DataFrame, train_entries: List[str], alpha: float = 1.0) -> pd.DataFrame:
    train_mask = entry_df["entry_id"].astype(str).isin(set(train_entries))
    train_bins = entry_df.loc[train_mask, "bin_id"]

    counts = train_bins.value_counts().sort_index()
    bin_ids = counts.index.tolist()
    n_b = counts.values.astype(float)
    n_b = np.maximum(n_b, 1.0)

    raw_w = 1.0 / np.power(n_b, alpha)
    raw_w = raw_w * (len(raw_w) / raw_w.sum())

    weight_map = {int(b): float(w) for b, w in zip(bin_ids, raw_w)}
    entry_df["bin_weight"] = entry_df["bin_id"].map(weight_map).fillna(1.0)
    return entry_df


def main():
    parser = argparse.ArgumentParser(description="TEM ABMIL 数据预处理 (Train/Val Only)")
    parser.add_argument("--excel_path", type=str, default="./processed/TEM-EA.xlsx")
    parser.add_argument("--processed_root", type=str, default="./processed")
    parser.add_argument("--out_root", type=str, default="./tem_abmil_data")

    # 混合尺度参数
    parser.add_argument("--small_size", type=int, default=128)
    parser.add_argument("--small_stride", type=int, default=64)
    parser.add_argument("--large_size", type=int, default=256)
    parser.add_argument("--large_stride", type=int, default=128)

    parser.add_argument("--min_fg_ratio", type=float, default=0.3)
    parser.add_argument("--n_bins", type=int, default=4, help="分层采样的分箱数")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例 (剩余为训练集)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_patches", action="store_true")
    args = parser.parse_args()

    # --- 目录准备 ---
    ensure_dir(args.out_root)
    patches_small_root = os.path.join(args.out_root, "patches_small")
    patches_large_root = os.path.join(args.out_root, "patches_large")
    meta_root = os.path.join(args.out_root, "meta")
    ensure_dir(meta_root)

    if args.overwrite_patches:
        if os.path.exists(patches_small_root): shutil.rmtree(patches_small_root)
        if os.path.exists(patches_large_root): shutil.rmtree(patches_large_root)
    ensure_dir(patches_small_root)
    ensure_dir(patches_large_root)

    # --- 1. 读取 Excel ---
    if not os.path.isfile(args.excel_path):
        raise FileNotFoundError(f"Excel not found: {args.excel_path}")

    label_df = pd.read_excel(args.excel_path)
    label_df["Entry"] = label_df["Entry"].astype(str).str.strip()
    entry_to_eapp = dict(zip(label_df["Entry"], label_df["Eapp"]))

    # --- 2. 图像处理 ---
    rows = []
    entry_dirs = sorted([d for d in os.listdir(args.processed_root)
                         if os.path.isdir(os.path.join(args.processed_root, d))])

    print(f"[Config] Val Ratio: {args.val_ratio} (No Test Split)")
    print(f"[Config] Scales: Small={args.small_size}, Large={args.large_size}")

    for entry_name in entry_dirs:
        if entry_name not in entry_to_eapp:
            continue

        eapp = float(entry_to_eapp[entry_name])
        entry_dir = os.path.join(args.processed_root, entry_name)
        img_files = [f for f in os.listdir(entry_dir) if is_processed_image(f)]

        if not img_files: continue

        print(f"Processing Entry: {entry_name} ...")

        for fname in img_files:
            img_path = os.path.join(entry_dir, fname)
            mask_path = os.path.join(entry_dir, f"{os.path.splitext(fname)[0]}_mask.png")

            if not os.path.isfile(mask_path): continue

            out_small = os.path.join(patches_small_root, entry_name, os.path.splitext(fname)[0])
            out_large = os.path.join(patches_large_root, entry_name, os.path.splitext(fname)[0])

            try:
                n_s, n_l = create_patches_for_image(
                    img_path, mask_path, out_small, out_large,
                    args.small_size, args.small_stride,
                    args.large_size, args.large_stride,
                    args.min_fg_ratio
                )
            except Exception as e:
                print(f"Error {fname}: {e}")
                continue

            if n_s == 0 and n_l == 0: continue

            rows.append({
                "entry_id": entry_name,
                "image_id": f"{entry_name}_{os.path.splitext(fname)[0]}",
                "image_fname": fname,
                "patch_dir_small": out_small,
                "patch_dir_large": out_large,
                "n_patches_small": n_s,
                "n_patches_large": n_l,
                "Eapp": eapp
            })

    img_df = pd.DataFrame(rows)
    if img_df.empty: raise RuntimeError("No patches generated.")

    # --- 3. 分层划分 (Stratified Split) ---
    entry_df = img_df[["entry_id", "Eapp"]].drop_duplicates("entry_id").reset_index(drop=True)

    # 关键步骤：根据 Eapp 分箱
    entry_df = assign_label_bins(entry_df, n_bins=args.n_bins)

    # 关键步骤：基于 Bin 进行划分
    splits = stratified_train_val_split(entry_df, val_ratio=args.val_ratio, seed=args.seed)

    print("\n[Split Result]")
    print(f"  Train: {len(splits['train'])} entries")
    print(f"  Val:   {len(splits['val'])} entries")

    # --- 4. 计算权重 & 标准化 ---
    entry_df = compute_bin_weights(entry_df, splits["train"], args.alpha)

    split_map = {}
    for k, v in splits.items():
        for e in v: split_map[e] = k
    entry_df["split"] = entry_df["entry_id"].map(split_map)

    img_df = img_df.merge(entry_df[["entry_id", "bin_id", "bin_weight", "split"]], on="entry_id", how="left")

    # 计算 Z-score (仅基于 Train)
    train_vals = img_df.loc[img_df["split"] == "train", "Eapp"].values.astype(float)
    mu, sigma = (float(train_vals.mean()), float(train_vals.std())) if len(train_vals) > 0 else (0.0, 1.0)
    img_df["Eapp_z"] = (img_df["Eapp"] - mu) / sigma

    # --- 5. 保存 ---
    img_df.to_csv(os.path.join(meta_root, "images_meta.csv"), index=False)
    entry_df.to_csv(os.path.join(meta_root, "entries_meta.csv"), index=False)
    with open(os.path.join(meta_root, "entries_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(meta_root, "label_norm_stats.json"), "w") as f:
        json.dump({"train_mean": mu, "train_std": sigma}, f, indent=2)

    print("\n[Done] Success.")


if __name__ == "__main__":
    main()
