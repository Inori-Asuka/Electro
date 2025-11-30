# extract_features.py
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from tqdm import tqdm

# 防止截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PatchDataset(Dataset):
    """
    用于加载单个 Entry 下的所有 Patch
    """

    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')  # ResNet 需要 RGB
            if self.transform:
                img = self.transform(img)
        return img


# extract_features.py 的 get_feature_extractor 函数修改如下：

def get_feature_extractor():
    print("Constructing ResNet50...")
    model = models.resnet50(weights=None)  # 必须是 None，不要 ImageNet
    modules = list(model.children())[:-1]
    backbone = nn.Sequential(*modules)

    # 指向刚才训练好的权重文件
    ckpt_path = "resnet50_tem_debug.pth"

    if os.path.exists(ckpt_path):
        print(f"Loading SUCCESSFUL SimCLR weights from {ckpt_path} ...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # 加载权重 (注意：我们保存的是 backbone.state_dict()，所以直接 load 即可)
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded: {msg}")
    else:
        raise FileNotFoundError(f"没找到权重: {ckpt_path}")

    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def main():
    parser = argparse.ArgumentParser(description="TEM ABMIL 特征提取")
    parser.add_argument("--meta_csv", type=str, default="./tem_abmil_data/meta/images_meta.csv")
    parser.add_argument("--out_dir", type=str, default="./tem_abmil_data/features")
    parser.add_argument("--batch_size", type=int, default=128, help="提取特征时的 Batch Size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 准备模型 ---
    model = get_feature_extractor().to(device)
    model.eval()

    # --- 2. 准备数据转换 ---
    # 无论 Patch 是 128 还是 256，都统一 Resize 到 224 输入 ResNet
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- 3. 读取元数据 ---
    df = pd.read_csv(args.meta_csv)
    # 确保 ID 是字符串
    df["entry_id"] = df["entry_id"].astype(str)

    os.makedirs(args.out_dir, exist_ok=True)

    # 按 Entry 分组处理
    entries = df["entry_id"].unique()
    print(f"Total entries to process: {len(entries)}")

    for entry_id in tqdm(entries):
        save_path = os.path.join(args.out_dir, f"{entry_id}.pt")

        # 断点续传：如果文件已存在，跳过
        if os.path.exists(save_path):
            continue

        entry_df = df[df["entry_id"] == entry_id]

        # --- 提取 Small Scale 特征 ---
        # 收集该 Entry 下所有 small patch 的路径
        small_paths = []
        for _, row in entry_df.iterrows():
            # 这里的路径是目录，我们需要找到目录下具体的 patch 文件
            # 注意：prepare 脚本里存的是 patch_dir_small (目录)
            # 但实际上我们可能需要遍历那个目录，或者如果 images_meta 每一行对应一个原始图像
            # 那么我们需要去 patch_dir_small 下找所有 png

            # 更正逻辑：images_meta.csv 的每一行是一个原始大图 (Image)
            # 我们需要把这个 Image 下切出来的所有 Patch 都找出来
            p_dir = row["patch_dir_small"]
            if os.path.isdir(p_dir):
                files = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.endswith(".png")]
                small_paths.extend(files)

        # --- 提取 Large Scale 特征 ---
        large_paths = []
        for _, row in entry_df.iterrows():
            p_dir = row["patch_dir_large"]
            if os.path.isdir(p_dir):
                files = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.endswith(".png")]
                large_paths.extend(files)

        # 定义一个内部函数来批量提取
        def extract_feats(paths):
            if not paths:
                return torch.tensor([])
            ds = PatchDataset(paths, transform=eval_transform)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            feats_list = []
            with torch.no_grad():
                for imgs in dl:
                    imgs = imgs.to(device)
                    # model 输出 (B, 2048, 1, 1) -> squeeze -> (B, 2048)
                    feats = model(imgs).squeeze(-1).squeeze(-1)
                    feats_list.append(feats.cpu())

            if feats_list:
                return torch.cat(feats_list, dim=0)
            else:
                return torch.tensor([])

        # 执行提取
        feats_small = extract_feats(small_paths)
        feats_large = extract_feats(large_paths)

        # --- 保存 ---
        # 存为一个字典，包含两种尺度的特征
        data_to_save = {
            "small": feats_small,  # Shape: [N_small, 2048]
            "large": feats_large,  # Shape: [N_large, 2048]
            "label": float(entry_df.iloc[0]["Eapp"]),  # 顺便把标签也存了
            "entry_id": entry_id
        }

        torch.save(data_to_save, save_path)

    print("特征提取完成！")


if __name__ == "__main__":
    main()
