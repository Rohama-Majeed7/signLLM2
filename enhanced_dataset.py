"""
Enhanced Dataset with Feature Augmentation
"""
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class EnhancedGlossFreeDataset(Dataset):
    def __init__(self, split='train', config=None):
        self.split = split
        self.config = config
        self.max_length = config.max_seq_length
        self.min_length = config.min_seq_length
        
        # Load features with augmentation
        self.features, self.file_names = self._load_features_with_augmentation()
        
        print(f"📊 Enhanced {split}: {len(self.features)} samples")
        
        # Calculate statistics for reporting
        self._calculate_statistics()
    
    def _load_features_with_augmentation(self):
        """Load features with multiple augmentations"""
        features_dir = os.path.join(self.config.features_dir, self.split)
        
        if not os.path.exists(features_dir):
            return [], []
        
        # Get all files
        all_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
        
        # Shuffle and limit
        random.shuffle(all_files)
        
        if self.split == 'train' and self.config.max_train_samples:
            all_files = all_files[:self.config.max_train_samples]
        elif self.split in ['val', 'test']:
            if self.split == 'val' and self.config.max_val_samples:
                all_files = all_files[:self.config.max_val_samples]
            elif self.split == 'test' and self.config.max_test_samples:
                all_files = all_files[:self.config.max_test_samples]
        
        features = []
        file_names = []
        
        for filename in all_files:
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
                    
                    # Apply augmentations for training
                    if self.split == 'train' and self.config.use_feature_augmentation:
                        tensor = self._augment_features(tensor)
                    
                    # Adaptive length handling
                    tensor = self._adaptive_length_handling(tensor)
                    
                    features.append(tensor)
                    file_names.append(filename)
                    
            except Exception as e:
                continue
        
        return features, file_names
    
    def _augment_features(self, features):
        """Apply feature-level augmentations"""
        augmented = features.clone()
        
        # 1. Temporal masking (mask random time steps)
        if random.random() < 0.3:
            mask_len = random.randint(1, int(features.shape[0] * 0.2))
            mask_start = random.randint(0, features.shape[0] - mask_len)
            augmented[mask_start:mask_start+mask_len] = 0
        
        # 2. Feature dropout
        if random.random() < 0.2:
            dropout_mask = torch.rand_like(augmented) > 0.9
            augmented[dropout_mask] = 0
        
        # 3. Gaussian noise
        if random.random() < 0.4:
            noise = torch.randn_like(augmented) * 0.05
            augmented = augmented + noise
        
        # 4. Temporal scaling (speed variation)
        if random.random() < 0.25:
            scale = random.uniform(0.8, 1.2)
            new_len = int(features.shape[0] * scale)
            if new_len > 10:
                # Simple interpolation
                augmented = torch.nn.functional.interpolate(
                    augmented.unsqueeze(0).unsqueeze(0),
                    size=(new_len, features.shape[1]),
                    mode='bilinear'
                ).squeeze()
        
        return augmented
    
    def _adaptive_length_handling(self, features):
        """Handle variable lengths intelligently"""
        T = features.shape[0]
        
        if T > self.max_length:
            # Multiple strategies for long sequences
            strategy = random.choice(['random_crop', 'center_crop', 'temporal_divide'])
            
            if strategy == 'random_crop':
                start = random.randint(0, T - self.max_length)
                features = features[start:start + self.max_length]
            elif strategy == 'center_crop':
                start = (T - self.max_length) // 2
                features = features[start:start + self.max_length]
            else:  # temporal_divide
                # Divide and take mean of segments
                segments = []
                segment_len = self.max_length // 2
                for i in range(0, T, segment_len):
                    if len(segments) < 2:
                        segment = features[i:i+segment_len]
                        if len(segment) > 0:
                            segments.append(segment.mean(dim=0, keepdim=True))
                if segments:
                    features = torch.cat(segments, dim=0)
                features = features[:self.max_length]
        
        elif T < self.min_length:
            # Repeat pattern for very short sequences
            repeats = (self.min_length + T - 1) // T
            features = features.repeat(repeats, 1)[:self.min_length]
        
        elif T < self.max_length:
            # Smart padding with reflection
            pad_len = self.max_length - T
            if pad_len > 0:
                # Pad with reflection of the sequence
                pad_features = torch.flip(features[-pad_len:], dims=[0])
                features = torch.cat([features, pad_features], dim=0)
        
        return features
    
    def _calculate_statistics(self):
        """Calculate dataset statistics"""
        if len(self.features) == 0:
            return
        
        lengths = [f.shape[0] for f in self.features]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        print(f"   Sequence length: {min(lengths)}-{max(lengths)} (mean: {mean_len:.1f} ± {std_len:.1f})")
        print(f"   Feature dimension: {self.features[0].shape[1]}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'filename': self.file_names[idx],
            'length': self.features[idx].shape[0]
        }