"""
Simple training script with all fixes
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("=" * 60)
print("SIGNLLM - Simple Training with Fixes")
print("=" * 60)

# ==================== CONFIG ====================
class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset paths
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    features_dir = i3d_features_dir
    feature_dim = 1024
    
    # Model
    codebook_size = 256
    codebook_dim = 512
    lambda_mmd = 0.5
    lambda_sim = 1.0
    
    # Training
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 3
    max_train_samples = 100
    max_val_samples = 30
    fixed_length = 100  # Fixed sequence length
    
    # Splits
    train_split = 'train'
    val_split = 'val'

config = Config()
print(f"Device: {config.device}")
print(f"Fixed length: {config.fixed_length}")

# ==================== DATASET ====================
class SimpleDataset(Dataset):
    def __init__(self, split='train', config=None, max_samples=None):
        self.split = split
        self.config = config
        self.max_samples = max_samples or 50
        self.fixed_length = config.fixed_length
        
        # Get files
        features_dir = os.path.join(config.features_dir, split)
        self.files = []
        
        if os.path.exists(features_dir):
            self.files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
            self.files = self.files[:self.max_samples]
        
        print(f"📁 {split}: Found {len(self.files)} files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.config.features_dir, self.split, self.files[idx])
        
        try:
            # Load feature
            arr = np.load(file_path, allow_pickle=True)
            
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).float()
                
                # Ensure shape (T, 1024)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() > 2:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                
                # Pad/truncate
                T = tensor.shape[0]
                if T > self.fixed_length:
                    # Center crop
                    start = (T - self.fixed_length) // 2
                    tensor = tensor[start:start + self.fixed_length]
                elif T < self.fixed_length:
                    # Pad
                    pad_size = self.fixed_length - T
                    padding = torch.zeros(pad_size, self.config.feature_dim)
                    tensor = torch.cat([tensor, padding], dim=0)
                
                return {
                    'feature': tensor,
                    'text': f"Video: {os.path.splitext(self.files[idx])[0]}"
                }
        
        except:
            pass
        
        # Fallback
        return {
            'feature': torch.zeros(self.fixed_length, self.config.feature_dim),
            'text': "Sample"
        }

# ==================== MODEL ====================
class SimpleVQSign(nn.Module):
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
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)
    
    def forward(self, x):
        # x shape: (B, T, D)
        B, T, D = x.shape
        
        # Process each time step
        encoded = []
        for t in range(T):
            feat = x[:, t, :]
            enc = self.encoder(feat)
            encoded.append(enc)
        
        encoded = torch.stack(encoded, dim=1)  # (B, T, codebook_dim)
        
        # Quantize
        flat_encoded = encoded.reshape(-1, self.config.codebook_dim)
        distances = torch.cdist(flat_encoded, self.codebook.weight)
        token_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(token_indices).view(B, T, self.config.codebook_dim)
        
        # Losses
        commitment_loss = nn.functional.mse_loss(encoded, quantized.detach())
        codebook_loss = nn.functional.mse_loss(quantized, encoded.detach())
        
        return token_indices.view(B, T), quantized, {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'vq_loss': commitment_loss + 0.25 * codebook_loss
        }

class SimpleSignLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # VQ module
        self.vq_sign = SimpleVQSign(config)
        
        # Decoder
        self.decoder = nn.Linear(config.codebook_dim, 1000)
    
    def forward(self, x, texts=None):
        # VQ encoding
        tokens, quantized, vq_losses = self.vq_sign(x)
        
        # Simple decoding (use mean of quantized features)
        pooled = quantized.mean(dim=1)  # (B, codebook_dim)
        decoded = self.decoder(pooled)
        
        # Dummy translation loss
        translation_loss = torch.tensor(0.0, device=x.device)
        if texts is not None:
            translation_loss = torch.tensor(0.1, device=x.device) * x.shape[0]
        
        # Total loss
        total_loss = vq_losses['vq_loss'] + translation_loss
        
        return tokens, {
            **vq_losses,
            'translation_loss': translation_loss,
            'total_loss': total_loss
        }

# ==================== TRAINING ====================
def main():
    # Create datasets
    print("\n📁 Creating datasets...")
    train_dataset = SimpleDataset('train', config, max_samples=config.max_train_samples)
    val_dataset = SimpleDataset('val', config, max_samples=config.max_val_samples)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Dataloaders
    def collate_fn(batch):
        return {
            'feature': torch.stack([item['feature'] for item in batch]),
            'text': [item['text'] for item in batch]
        }
    
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
    
    # Model
    print("\n🤖 Creating model...")
    model = SimpleSignLLM(config).to(config.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training
    print("\n🚀 Starting training...")
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_bar):
            features = batch['feature'].to(config.device)
            
            _, losses = model(features)
            
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            train_loss += losses['total_loss'].item()
            train_bar.set_postfix({'loss': f'{train_loss/(batch_idx+1):.4f}'})
        
        avg_train = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['feature'].to(config.device)
                _, losses = model(features)
                val_loss += losses['total_loss'].item()
        
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"\n📊 Epoch {epoch+1}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    
    # Save
    torch.save(model.state_dict(), 'signllm_trained.pth')
    print(f"\n💾 Model saved: signllm_trained.pth")
    
    # Test
    print("\n🔍 Testing inference...")
    model.eval()
    with torch.no_grad():
        sample = train_dataset[0]
        feature = sample['feature'].unsqueeze(0).to(config.device)
        tokens, losses = model(feature)
        print(f"  Input shape: {feature.shape}")
        print(f"  Output tokens shape: {tokens.shape}")
        print(f"  Loss: {losses['total_loss'].item():.4f}")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()