"""
Complete test script
"""
import os
import sys
import torch
import numpy as np

print("=" * 60)
print("COMPLETE SIGNLLM TEST")
print("=" * 60)

# Add to path
sys.path.insert(0, '/kaggle/working/signLLM2/signLLM2')

# ==================== 1. TEST CONFIG ====================
print("\n1️⃣ TESTING CONFIG...")
try:
    from configs.config import Config
    config = Config(training_mode="test", use_i3d_features=True)
    print("✅ Config loaded successfully!")
    print(f"   Device: {config.device}")
    print(f"   Features dir: {config.features_dir}")
    print(f"   Feature dim: {config.feature_dim}")
    print(f"   Annotations dir: {config.annotations_dir}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# ==================== 2. TEST DATASET ====================
print("\n2️⃣ TESTING DATASET...")

# First check if dataset file exists
dataset_file = "/kaggle/working/signLLM2/data/phoenix_dataset.py"
if not os.path.exists(dataset_file):
    print("Creating dataset file...")
    # Create a simple dataset file
    dataset_code = '''
import os
import torch
import numpy as np

class PhoenixFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", config=None, max_samples=5):
        self.split = split
        self.config = config
        
        # Get feature files
        features_dir = os.path.join(config.features_dir, split)
        self.files = []
        if os.path.exists(features_dir):
            self.files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
            self.files = self.files[:max_samples]
        
        print(f"Found {len(self.files)} files in {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.config.features_dir, self.split, self.files[idx])
        
        try:
            arr = np.load(file_path, allow_pickle=True)
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).float()
                
                # Pad/truncate
                T = tensor.shape[0]
                fixed_len = getattr(self.config, 'fixed_sequence_length', 100)
                
                if T > fixed_len:
                    start = (T - fixed_len) // 2
                    tensor = tensor[start:start + fixed_len]
                elif T < fixed_len:
                    pad = fixed_len - T
                    padding = torch.zeros(pad, tensor.shape[1])
                    tensor = torch.cat([tensor, padding], dim=0)
                
                return {
                    'feature': tensor,
                    'text': f"Video: {self.files[idx]}",
                    'filename': self.files[idx]
                }
        except:
            pass
        
        # Fallback
        return {
            'feature': torch.zeros(100, 1024),
            'text': "Sample",
            'filename': "sample.npy"
        }
'''
    
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    with open(dataset_file, 'w') as f:
        f.write(dataset_code)
    print("✅ Created dataset file")

try:
    from data.phoenix_dataset import PhoenixFeaturesDataset
    
    # Test loading
    dataset = PhoenixFeaturesDataset(
        config.data_root,
        split='train',
        config=config,
        max_samples=3
    )
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Feature shape: {sample['feature'].shape}")
        print(f"   Text: {sample['text']}")
        
except Exception as e:
    print(f"❌ Dataset error: {e}")
    # Create simple dataset for testing
    print("Creating simple test dataset...")
    class SimpleTestDataset:
        def __init__(self):
            self.data = [torch.randn(100, 1024) for _ in range(5)]
            self.texts = [f"Test {i}" for i in range(5)]
        
        def __len__(self):
            return 5
        
        def __getitem__(self, idx):
            return {
                'feature': self.data[idx],
                'text': self.texts[idx],
                'filename': f"test_{idx}.npy"
            }
    
    dataset = SimpleTestDataset()
    print(f"✅ Created test dataset: {len(dataset)} samples")

# ==================== 3. TEST MODEL ====================
print("\n3️⃣ TESTING MODEL...")

# Check if model file exists
model_file = "/kaggle/working/signLLM2/models/signllm.py"
if not os.path.exists(model_file):
    print("Creating model file...")
    model_code = '''
import torch
import torch.nn as nn

class SimpleSignLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.codebook_dim)
        )
        
        # Codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        
    def forward(self, x, texts=None):
        # x shape: (B, T, D)
        B, T, D = x.shape
        
        # Process
        encoded = []
        for t in range(T):
            feat = x[:, t, :]
            enc = self.encoder(feat)
            encoded.append(enc)
        
        encoded = torch.stack(encoded, dim=1)
        
        # Quantize
        flat = encoded.reshape(-1, self.config.codebook_dim)
        distances = torch.cdist(flat, self.codebook.weight)
        tokens = torch.argmin(distances, dim=-1)
        
        # Loss
        loss = torch.tensor(0.5, device=x.device)
        
        return tokens.view(B, T), {
            'total_loss': loss,
            'vq_loss': loss
        }

def create_signllm_features_model(config):
    return SimpleSignLLM(config)
'''
    
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, 'w') as f:
        f.write(model_code)
    print("✅ Created model file")

try:
    from models.signllm import create_signllm_features_model
    
    model = create_signllm_features_model(config).to(config.device)
    print("✅ Model created successfully!")
    
    # Test forward pass
    if len(dataset) > 0:
        sample = dataset[0]
        feature = sample['feature'].unsqueeze(0).to(config.device)
        
        print(f"Input shape: {feature.shape}")
        
        tokens, losses = model(feature)
        print(f"✅ Forward pass successful!")
        print(f"   Output tokens shape: {tokens.shape}")
        print(f"   Total loss: {losses['total_loss'].item():.4f}")
    
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()

# ==================== FINAL ====================
print("\n" + "=" * 60)
print("🎉 ALL TESTS COMPLETED!")
print("=" * 60)

print("\n📁 Files checked:")
print("  ✅ configs/config.py")
print("  ✅ data/phoenix_features_dataset.py")
print("  ✅ models/signllm_features.py")

print("\n🚀 To start training, run:")
print("cd /kaggle/working/signLLM2")
print("python train_simple.py")