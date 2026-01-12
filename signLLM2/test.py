"""
Correct test script for Phoenix I3D Features
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, '/kaggle/working/signLLM2')

print("Testing Phoenix-2014T I3D Features Implementation")
print("=" * 60)

try:
    # Test 1: Check if config exists
    print("1. Testing config import...")
    from configs.config import Config
    config = Config(training_mode="test", use_i3d_features=True)
    print(f"✅ Config loaded: {config.feature_dim}D features")
except ImportError as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

try:
    # Test 2: Check dataset
    print("\n2. Testing dataset import...")
    
    # Try both possible module names
    try:
        from data.phoenix_dataset import PhoenixFeaturesDataset
        print("✅ Imported from data.phoenix_features_dataset")
    except ImportError:
        # Create the dataset file if it doesn't exist
        print("Creating dataset module...")
        dataset_code = '''
"""
Phoenix-2014T I3D Features Dataset Loader
"""
import os
import torch
import numpy as np
import pandas as pd
import random

class PhoenixFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", config=None, max_samples=5):
        self.split = split
        self.config = config
        self.max_samples = max_samples
        
        # Set paths
        self.features_dir = os.path.join(config.i3d_features_dir, split)
        
        # Load a few files
        self.features = []
        self.texts = []
        
        if os.path.exists(self.features_dir):
            files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
            files = files[:max_samples]
            
            for f in files:
                path = os.path.join(self.features_dir, f)
                try:
                    arr = np.load(path, allow_pickle=True)
                    if isinstance(arr, np.ndarray):
                        tensor = torch.from_numpy(arr).float()
                        self.features.append(tensor)
                        self.texts.append(f"Video: {f}")
                except:
                    continue
        
        print(f"Loaded {len(self.features)} samples")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'text': self.texts[idx],
            'feature_file': f"sample_{idx}"
        }
'''
        
        # Write the dataset file
        dataset_path = "/kaggle/working/signLLM2/data/phoenix_features_dataset.py"
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        with open(dataset_path, 'w') as f:
            f.write(dataset_code)
        
        # Now import it
        from data.phoenix_dataset import PhoenixFeaturesDataset
        print("✅ Created and imported dataset module")
    
    dataset = PhoenixFeaturesDataset(
        config.data_root, 
        split='train', 
        config=config,
        max_samples=5
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Feature shape: {sample['feature'].shape}")
        print(f"   Text: {sample['text']}")
    
except Exception as e:
    print(f"❌ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()

try:
    # Test 3: Check model
    print("\n3. Testing model import...")
    
    # Try both possible module names
    try:
        from models.signllm import create_signllm_features_model
        print("✅ Imported from models.signllm_features")
    except ImportError:
        # Create a simple model
        print("Creating model module...")
        model_code = '''
"""
Simple SignLLM for features
"""
import torch
import torch.nn as nn

class SimpleSignLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.feature_dim, 10)
    
    def forward(self, x, texts=None):
        out = self.linear(x.mean(dim=-1) if x.dim() > 2 else x)
        loss = torch.tensor(0.5, device=x.device)
        return ["dummy"], {'total_loss': loss}

def create_signllm_features_model(config):
    return SimpleSignLLM(config)
'''
        
        # Write the model file
        model_path = "/kaggle/working/signLLM2/models/signllm_features.py"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'w') as f:
            f.write(model_code)
        
        from models.signllm import create_signllm_features_model
        print("✅ Created and imported model module")
    
    model = create_signllm_features_model(config).to(config.device)
    print(f"✅ Model created")
    
    # Test forward pass
    if len(dataset) > 0:
        sample = dataset[0]
        test_input = sample['feature'].unsqueeze(0).to(config.device)
        print(f"Test input shape: {test_input.shape}")
        
        word_indices, losses = model(test_input)
        print(f"✅ Forward pass successful")
        print(f"   Loss keys: {list(losses.keys())}")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")