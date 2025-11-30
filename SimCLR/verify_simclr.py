# verify_simclr.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm


def load_features_and_labels(feature_dir, meta_csv):
    print("Loading features...")
    df = pd.read_csv(meta_csv)

    X_bags = []  # 存放每个包的平均特征
    y = []  # 存放 Eapp

    # 遍历 CSV
    valid_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        eid = str(row["entry_id"])
        label = float(row["Eapp"])
        path = os.path.join(feature_dir, f"{eid}.pt")

        if os.path.exists(path):
            try:
                data = torch.load(path, map_location="cpu")
                # 处理字典或Tensor
                if isinstance(data, dict):
                    if "features" in data:
                        feats = data["features"]
                    elif "feats" in data:
                        feats = data["feats"]
                    else:
                        feats = list(data.values())[0]
                else:
                    feats = data

                if feats.dim() == 3: feats = feats.squeeze(0)

                # --- 关键：取平均 (Mean Pooling) ---
                # 如果 SimCLR 学得好，平均特征也应该包含足够的信息
                feat_avg = feats.mean(dim=0).numpy()

                X_bags.append(feat_avg)
                y.append(label)
                valid_count += 1
            except Exception as e:
                print(f"Error loading {eid}: {e}")

    print(f"Loaded {valid_count} samples.")
    return np.array(X_bags), np.array(y)


def run_linear_probing(X, y):
    print("\n--- 1. Linear Probing (线性探测) ---")
    print("原理：测试特征是否足够好，好到只需要一个线性层就能预测 Eapp。")

    # 简单的划分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用 Ridge Regression (带正则化的线性回归)
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)

    r2_train = r2_score(y_train, pred_train)
    r2_val = r2_score(y_val, pred_val)

    print(f"Train R2: {r2_train:.4f}")
    print(f"Val R2:   {r2_val:.4f}")

    if r2_val > 0.2:
        print("✅ 结论：SimCLR 特征非常有效！线性层都能跑出正结果。")
    elif r2_val > 0:
        print("⚠️ 结论：特征有效，但可能需要非线性模型 (MIL) 才能发挥威力。")
    else:
        print("❌ 结论：特征区分度不够，或者 Mean Pooling 损失了太多信息。")


def run_tsne(X, y):
    print("\n--- 2. t-SNE Visualization ---")
    print("正在降维 (2048 -> 2)... 这可能需要几分钟...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    # 根据 Eapp 数值上色
    sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Eapp')
    plt.title("t-SNE of SimCLR Features (Colored by Eapp)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    save_path = "simclr_tsne.png"
    plt.savefig(save_path)
    print(f"✅ t-SNE 图已保存为: {save_path}")
    print("请打开图片查看：颜色的渐变是否有规律？")
    print("  - 理想情况：深色点聚一边，亮色点聚一边。")
    print("  - 糟糕情况：颜色完全随机杂乱。")


if __name__ == "__main__":
    # 配置路径
    FEATURE_DIR = "./tem_abmil_data/features"
    META_CSV = "./tem_abmil_data/meta/images_meta.csv"

    X, y = load_features_and_labels(FEATURE_DIR, META_CSV)

    if len(X) > 0:
        run_linear_probing(X, y)
        run_tsne(X, y)
    else:
        print("没有加载到数据，请检查路径。")
