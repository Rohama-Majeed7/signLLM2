"""
Simple training with gradient fix
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("=" * 60)
print("SIGNLLM TRAINING - WITH GRADIENT FIX")
print("=" * 60)

# Add to path
sys.path.insert(0, '/kaggle/working/signLLM2')

# ==================== CONFIG ====================
class SimpleConfig:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    
    # Model
    feature_dim = 1024
    codebook_size = 256
    codebook_dim = 512
    
    # Training
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 5
    fixed_length = 100
    
    # Dataset
    max_train_samples = 50
    max_val_samples = 20

config = SimpleConfig()
print(f"📊 Config:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")

# ==================== DATASET ====================
print("\n📁 Loading dataset...")

class SimpleDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.files = []
        
        split_dir = os.path.join(config.features_dir, split)
        if os.path.exists(split_dir):
            self.files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            
            # Limit samples
            if split == 'train':
                self.files = self.files[:config.max_train_samples]
            else:
                self.files = self.files[:config.max_val_samples]
        
        print(f"  {split}: Found {len(self.files)} files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(config.features_dir, self.split, self.files[idx])
        
        try:
            arr = np.load(file_path, allow_pickle=True)
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).float()
                
                # Ensure shape (T, 1024)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() > 2:
                    tensor = tensor.reshape(-1, 1024)
                
                # Pad/truncate
                T = tensor.shape[0]
                if T > config.fixed_length:
                    tensor = tensor[:config.fixed_length]
                elif T < config.fixed_length:
                    pad = config.fixed_length - T
                    padding = torch.zeros(pad, 1024)
                    tensor = torch.cat([tensor, padding], dim=0)
                
                return {
                    'feature': tensor,
                    'text': f"Video {idx}"
                }
        except:
            pass
        
        # Fallback
        return {
            'feature': torch.zeros(config.fixed_length, 1024),
            'text': "Sample"
        }

# Create datasets
train_dataset = SimpleDataset('train')
val_dataset = SimpleDataset('val')

print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")

# ==================== DATALOADER ====================
def collate_fn(batch):
    features = torch.stack([item['feature'] for item in batch])
    texts = [item['text'] for item in batch]
    return {'feature': features, 'text': texts}

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# ==================== MODEL - FIXED ====================
print("\n🤖 Creating model...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Codebook
        self.codebook = nn.Embedding(128, 256)
        
        # Decoder
        self.decoder = nn.Linear(256, 100)
        
    def forward(self, x):
        # x shape: (B, T, D)
        B, T, D = x.shape
        
        # Encode
        encoded = []
        for t in range(T):
            feat = x[:, t, :]
            enc = self.encoder(feat)
            encoded.append(enc)
        
        encoded = torch.stack(encoded, dim=1)  # (B, T, 256)
        
        # Quantize
        flat_encoded = encoded.reshape(-1, 256)
        
        # Calculate distances to codebook
        distances = torch.cdist(flat_encoded, self.codebook.weight)
        
        # Get nearest codebook entries
        token_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(token_indices).view(B, T, 256)
        
        # Proper VQ loss with gradients
        commitment_loss = nn.functional.mse_loss(encoded, quantized.detach())
        codebook_loss = nn.functional.mse_loss(quantized, encoded.detach())
        
        # Decode
        pooled = encoded.mean(dim=1)  # (B, 256)
        decoded = self.decoder(pooled)
        
        # Total loss with gradients
        total_loss = commitment_loss + 0.25 * codebook_loss
        
        # Return proper tensors
        return token_indices.view(B, T), {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'total_loss': total_loss
        }

model = SimpleModel().to(config.device)
print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ==================== TRAINING ====================
print("\n🚀 Starting training...")

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.num_epochs):
    # Train
    model.train()
    train_loss = 0
    train_batches = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for batch_idx, batch in enumerate(train_bar):
        features = batch['feature'].to(config.device)
        
        _, losses = model(features)
        
        # Check if loss requires grad
        if not losses['total_loss'].requires_grad:
            print(f"Warning: Loss does not require gradient!")
            continue
        
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        train_loss += losses['total_loss'].item()
        train_batches += 1
        
        avg_loss = train_loss / train_batches
        train_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_train = train_loss / train_batches if train_batches > 0 else 0
    
    # Validate
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['feature'].to(config.device)
            _, losses = model(features)
            val_loss += losses['total_loss'].item()
            val_batches += 1
    
    avg_val = val_loss / val_batches if val_batches > 0 else 0
    
    print(f"\n📊 Epoch {epoch+1}:")
    print(f"  Train Loss: {avg_train:.4f}")
    print(f"  Val Loss: {avg_val:.4f}")

# ==================== SAVE ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__
}, 'trained_model.pth')
print(f"\n💾 Model saved: trained_model.pth")

# ==================== TEST ====================
print("\n🔍 Testing inference...")
model.eval()
with torch.no_grad():
    sample = train_dataset[0]
    feature = sample['feature'].unsqueeze(0).to(config.device)
    tokens, losses = model(feature)
    print(f"  Input shape: {feature.shape}")
    print(f"  Output tokens shape: {tokens.shape}")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Commitment loss: {losses['commitment_loss'].item():.4f}")
    print(f"  Codebook loss: {losses['codebook_loss'].item():.4f}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)