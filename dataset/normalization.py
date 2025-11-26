"""
标签归一化工具
"""
import numpy as np
import torch


class LabelNormalizer:
    def __init__(self, method='standard', **kwargs):
        """
        Args:
            method: 归一化方法
                - 'standard': 标准化 (x - mean) / std
                - 'minmax': MinMax归一化 (x - min) / (max - min)
                - 'robust': Robust归一化 (x - median) / IQR
        """
        self.method = method
        self.params = kwargs
        self._fitted = False
    
    def fit(self, labels):
        labels = np.array(labels).flatten()
        
        if self.method == 'standard':
            self.params['mean'] = np.mean(labels)
            self.params['std'] = np.std(labels)
            if self.params['std'] == 0:
                self.params['std'] = 1.0
        elif self.method == 'minmax':
            self.params['min'] = np.min(labels)
            self.params['max'] = np.max(labels)
            if self.params['max'] == self.params['min']:
                self.params['max'] = self.params['min'] + 1.0
        elif self.method == 'robust':
            self.params['median'] = np.median(labels)
            q75, q25 = np.percentile(labels, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            self.params['iqr'] = iqr
        
        self._fitted = True
        return self
    
    def transform(self, labels):
        
        is_tensor = isinstance(labels, torch.Tensor)
        if is_tensor:
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.array(labels)
        
        original_shape = labels_np.shape
        labels_flat = labels_np.flatten()
        
        if self.method == 'standard':
            normalized = (labels_flat - self.params['mean']) / self.params['std']
        elif self.method == 'minmax':
            normalized = (labels_flat - self.params['min']) / (self.params['max'] - self.params['min'])
        elif self.method == 'robust':
            normalized = (labels_flat - self.params['median']) / self.params['iqr']
        
        normalized = normalized.reshape(original_shape)
        
        if is_tensor:
            return torch.tensor(normalized, dtype=labels.dtype, device=labels.device)
        else:
            return normalized
    
    def inverse_transform(self, normalized_labels):
        is_tensor = isinstance(normalized_labels, torch.Tensor)
        if is_tensor:
            labels_np = normalized_labels.detach().cpu().numpy()
        else:
            labels_np = np.array(normalized_labels)
        
        original_shape = labels_np.shape
        labels_flat = labels_np.flatten()
        
        if self.method == 'standard':
            original = labels_flat * self.params['std'] + self.params['mean']
        elif self.method == 'minmax':
            original = labels_flat * (self.params['max'] - self.params['min']) + self.params['min']
        elif self.method == 'robust':
            original = labels_flat * self.params['iqr'] + self.params['median']
        
        original = original.reshape(original_shape)
        
        if is_tensor:
            return torch.tensor(original, dtype=normalized_labels.dtype, device=normalized_labels.device)
        else:
            return original
    
    def get_params(self):
        return {
            'method': self.method,
            **self.params
        }
    
    def set_params(self, params):
        self.method = params['method']
        self.params = {k: v for k, v in params.items() if k != 'method'}
        self._fitted = True


def create_normalizer(method='standard', labels=None, **kwargs):
    normalizer = LabelNormalizer(method=method, **kwargs)
    if labels is not None:
        normalizer.fit(labels)
    return normalizer

