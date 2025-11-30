# train_simclr_debug.py
# -*- coding: utf-8 -*-
import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import models, transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# --- 1. TEM 增强 (无模糊，有旋转) ---
class TEMSimCLRTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x = x.convert("RGB")
        return self.transform(x), self.transform(x)


class PatchFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        # 过滤 mask
        self.files = [f for f in self.files if "_mask" not in f]
        print(f"Found {len(self.files)} images.")
        self.transform = TEMSimCLRTransform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with Image.open(path) as img:
                x_i, x_j = self.transform(img)
            return x_i, x_j
        except Exception:
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)


# --- 2. 模型 ---
class SimCLR(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),  # 加个 BN 防止坍塌
            nn.ReLU(),
            nn.Linear(2048, 128)
        )

    def forward(self, x):
        h = self.backbone(x).squeeze()
        z = self.projector(h)
        return h, z


# --- 3. 带诊断功能的 Loss ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1, device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.mask = self.mask_correlated_samples(batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        返回: loss, accuracy, pos_sim, neg_sim
        """
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # 计算相似度矩阵
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # 提取正样本对的相似度
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # 提取负样本对的相似度
        negative_samples = sim[self.mask].reshape(N, -1)

        # 构建 Logits 和 Labels
        # Label 全是 0，因为我们在 logits 拼接时，把正样本放在了第0列
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        # --- 诊断指标计算 ---
        with torch.no_grad():
            # 1. Accuracy: 预测概率最大的那个是不是第0列(正样本)?
            pred = torch.argmax(logits, dim=1)
            correct = (pred == labels).float().sum()
            acc = correct / N

            # 2. Similarity Stats (还原 temperature 缩放前的值)
            mean_pos_sim = positive_samples.mean() * self.temperature
            mean_neg_sim = negative_samples.mean() * self.temperature

        return loss, acc, mean_pos_sim, mean_neg_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_root", default="./tem_abmil_data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    # 数据
    dataset = PatchFolderDataset(args.patch_root)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=8, pin_memory=True, drop_last=True)

    # 模型
    resnet = models.resnet50(weights=None)
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)

    model = SimCLR(backbone).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = NTXentLoss(args.batch_size, temperature=0.1, device=device)

    if local_rank == 0:
        print(f"Start Debugging SimCLR... Global Batch: {args.batch_size * 8}")
        print(
            f"{'Epoch':<6} | {'Step':<6} | {'Loss':<8} | {'Acc(%)':<8} | {'PosSim':<8} | {'NegSim':<8} | {'FeatStd':<8}")
        print("-" * 70)

    start_time = time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for step, (x_i, x_j) in enumerate(loader):
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()
            h, z_i = model(x_i)
            _, z_j = model(x_j)

            loss, acc, pos_sim, neg_sim = criterion(z_i, z_j)

            loss.backward()
            optimizer.step()

            # --- 打印详细诊断信息 (每 10 步) ---
            if local_rank == 0 and step % 10 == 0:
                # 计算特征标准差，检查是否坍塌
                with torch.no_grad():
                    # z_i shape: [B, 128]
                    # 计算每个维度上的标准差，然后取平均
                    feat_std = torch.std(z_i, dim=0).mean().item()

                print(
                    f"{epoch + 1:<6} | {step:<6} | {loss.item():.4f}   | {acc.item() * 100:.2f}     | {pos_sim.item():.4f}   | {neg_sim.item():.4f}   | {feat_std:.4f}")

        if local_rank == 0:
            torch.save(model.module.backbone.state_dict(), "resnet50_tem_debug.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
