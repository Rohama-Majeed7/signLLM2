# File: signllm/data/dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from signLLM2.configs.config import config

class PhoenixI3DDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.feature_path = os.path.join(config.features_dir, split)
        
        # Load all .npy files
        self.files = []
        if os.path.exists(self.feature_path):
            self.files = [f for f in os.listdir(self.feature_path) if f.endswith('.npy')]
        else:
            print(f"❌ Error: Path not found {self.feature_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.feature_path, file_name)
        
        try:
            # Load Feature
            data = np.load(file_path, allow_pickle=True)
            tensor = torch.from_numpy(data).float()
            
            # Shape check (T, 1024)
            if tensor.dim() == 1: 
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2:
                # Agar shape (C, T, D) ho to average kar lo
                tensor = tensor.mean(dim=0) 
            
            return {'features': tensor, 'name': file_name}
        except Exception as e:
            return None

# --- ERROR FIXING PART ---
def collate_fn(batch):
    """
    Ye function variable length features ko handle karega
    aur Runtime Error se bachayega.
    """
    # Remove None items (agar koi file load na ho paye)
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    features_list = [item['features'] for item in batch]
    names_list = [item['name'] for item in batch]
    
    # 1. Lengths save karo (Masking ke liye)
    lengths = torch.tensor([f.shape[0] for f in features_list])
    
    # 2. Padding (Sabse lambi sequence ke barabar 0 add karna)
    # Output shape: (Batch, Max_Time, 1024)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # 3. Mask banana (Taake model padding walay 0s ko ignore kare)
    # True jahan real data hai, False jahan padding hai
    mask = torch.arange(features_padded.size(1))[None, :] < lengths[:, None]
    
    return {
        'features': features_padded,
        'mask': mask,  # Ye model mein jayega
        'names': names_list
    }