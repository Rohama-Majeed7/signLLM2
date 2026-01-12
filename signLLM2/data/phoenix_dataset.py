"""
Gloss-Free Dataset Loader - No gloss annotations
"""
import os
import torch
import numpy as np
import random

class GlossFreeDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', config=None):
        self.split = split
        self.config = config
        self.max_length = config.max_seq_length
        
        # Load features only (no gloss)
        self.features = self._load_features()
        
        print(f"📊 Gloss-Free {split}: {len(self.features)} samples")
    
    def _load_features(self):
        """Load only features, no gloss"""
        features_dir = os.path.join(self.config.features_dir, self.split)
        
        if not os.path.exists(features_dir):
            return []
        
        # Get all .npy files
        files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
        
        # Shuffle for randomness
        random.shuffle(files)
        
        # Limit samples
        if self.split == 'train' and self.config.max_train_samples:
            files = files[:self.config.max_train_samples]
        elif self.split in ['val', 'test'] and self.config.max_val_samples:
            files = files[:self.config.max_val_samples]
        
        features = []
        
        for filename in files:
            file_path = os.path.join(features_dir, filename)
            
            try:
                arr = np.load(file_path, allow_pickle=True)
                
                if isinstance(arr, np.ndarray):
                    tensor = torch.from_numpy(arr).float()
                    
                    # Ensure (T, 1024)
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() > 2:
                        tensor = tensor.reshape(-1, 1024)
                    
                    # Pad/truncate
                    T = tensor.shape[0]
                    if T > self.max_length:
                        # Random crop for augmentation
                        start = random.randint(0, T - self.max_length)
                        tensor = tensor[start:start + self.max_length]
                    elif T < self.max_length:
                        pad = self.max_length - T
                        tensor = torch.cat([tensor, torch.zeros(pad, 1024)], dim=0)
                    
                    features.append(tensor)
                    
            except:
                continue
        
        return features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]  # Only features, no text/gloss