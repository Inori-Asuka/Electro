import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from .normalization import LabelNormalizer

# 固定测试集
EXPECTED_TEST_ENTRIES_FROM_SCIENCE = [
    '10', '11', '20', '22', '32', '40', '41', '42', '44', '49', '56', '62', '80', '91',
    '101', '107', '109', '112', '114', '116', '121', '127', '131', '132', '136', '140', 
    '141', '143', '148', '160', '161', '162'
]
EXCLUDED_IDS = [23, 24]


class TEMRegressionDataset(Dataset):
    """TEM图像回归数据集"""
    def __init__(self, data_root, excel_path, material_ids=None, transform=None, 
                 use_cleaned=True, normalizer=None):

        self.data_root = data_root
        self.transform = transform
        self.use_cleaned = use_cleaned
        self.normalizer = normalizer
        
        df = pd.read_excel(excel_path)
        
        excluded_ids = [23, 24]
        df = df[~df['Entry'].isin(excluded_ids)]
        
        if material_ids is not None:
            df = df[df['Entry'].isin(material_ids)]
        
        self.data_list = []
        for _, row in df.iterrows():
            material_id = str(int(row['Entry']))
            eapp = float(row['Eapp'])
            
            material_dir = os.path.join(data_root, material_id)
            if not os.path.exists(material_dir):
                continue
            
            if use_cleaned:
                pattern = os.path.join(material_dir, '*_c.png')
            else:
                pattern = os.path.join(material_dir, '*.png')
            
            image_files = glob.glob(pattern)
            
            image_files = [f for f in image_files if '_mask' not in f]
            
            if len(image_files) == 0:
                continue
            
            for img_path in image_files:
                self.data_list.append({
                    'image_path': img_path,
                    'material_id': material_id,
                    'eapp': eapp
                })
        print(f"数据集大小: {len(self.data_list)} 个样本")
        print(f"材料数量: {len(df)} 个")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
    
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = item['eapp']
        if self.normalizer is not None:
            label = self.normalizer.transform(label)
            if isinstance(label, np.ndarray):
                label = torch.tensor(label, dtype=torch.float32)
            elif not isinstance(label, torch.Tensor):
                label = torch.tensor(float(label), dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        
        return image, label


def get_transforms(mode='train', image_size=512):
    """
    Args:
        mode: 'train' 或 'test'
        image_size: 最终输入网络的图像大小
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def split_train_test(df, excluded_ids=EXCLUDED_IDS):
    """
    划分训练集和测试集
    Args:
        df: 包含Entry和Eapp列的DataFrame
        excluded_ids: 要排除的材料ID列表

    Returns:
        train_ids, test_ids
    """
    if excluded_ids:
        df = df[~df['Entry'].isin(excluded_ids)]
    
    test_entry_names = EXPECTED_TEST_ENTRIES_FROM_SCIENCE
    test_ids = np.array([int(name) for name in test_entry_names])
    all_material_ids = df['Entry'].unique()
    valid_test_ids = [id for id in test_ids if id in all_material_ids]
    test_ids = np.array(valid_test_ids)
    train_ids = np.array([id for id in all_material_ids if id not in test_ids])
    
    
    print(f"训练集材料数: {len(train_ids)}")
    print(f"测试集材料数: {len(test_ids)}")
    
    return train_ids, test_ids

