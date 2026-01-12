"""
Gloss-Free Training for SignLLM
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

print("=" * 70)
print("GLOSS-FREE SIGNLLM TRAINING")
print("=" * 70)

# Config
class GlossFreeConfig:
    # Gloss-free settings
    gloss_free = True
    use_contrastive_loss = True
    use_reconstruction_loss = True
    
    # Model
    feature_dim = 1024
    codebook_size = 512
    codebook_dim = 768
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 30
    warmup_epochs = 5
    max_seq_length = 150
    
    # Loss weights
    lambda_commitment = 1.0
    lambda_codebook = 0.25
    lambda_reconstruction = 1.0
    lambda_contrastive = 0.5
    
    # Dataset
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    max_train_samples = 500
    max_val_samples = 100

config = GlossFreeConfig()
print(f"📊 Gloss-Free Configuration:")
print(f"  Codebook: {config.codebook_size} tokens, dim {config.codebook_dim}")
print(f"  Contrastive Learning: {'Yes' if config.use_contrastive_loss else 'No'}")
print(f"  Reconstruction Loss: {'Yes' if config.use_reconstruction_loss else 'No'}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Epochs: {config.num_epochs}")

# Dataset
print(f"\n📁 Loading gloss-free dataset...")

def load_gloss_free_features(split='train'):
    features_dir = os.path.join(config.features_dir, split)
    features = []
    
    if os.path.exists(features_dir):
        files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
        
        if split == 'train':
            files = files[:config.max_train_samples]
        else:
            files = files[:config.max_val_samples]
        
        for filename in files:
            file_path = os.path.join(features_dir, filename)
            try:
                arr = np.load(file_path, allow_pickle=True)
                if isinstance(arr, np.ndarray):
                    tensor = torch.from_numpy(arr).float()
                    
                    # Reshape
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() > 2:
                        tensor = tensor.reshape(-1, 1024)
                    
                    # Fixed length
                    T = tensor.shape[0]
                    if T > config.max_seq_length:
                        tensor = tensor[:config.max_seq_length]
                    elif T < config.max_seq_length:
                        pad = config.max_seq_length - T
                        tensor = nn.functional.pad(tensor, (0, 0, 0, pad))
                    
                    features.append(tensor)
            except:
                continue
    
    print(f"  {split}: {len(features)} samples")
    return features

# Load data
train_features = load_gloss_free_features('train')
val_features = load_gloss_free_features('val')

# Simple model for gloss-free
class SimpleGlossFreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # Codebook
        self.codebook = nn.Embedding(512, 512)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 1024)
        )
        
        # Contrastive projection
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Encode
        encoded = []
        for t in range(T):
            enc = self.encoder(x[:, t, :])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)
        
        # Quantize
        flat = encoded.reshape(-1, 512)
        distances = torch.cdist(flat, self.codebook.weight)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(indices).view(B, T, 512)
        
        # Straight-through
        quantized_st = encoded + (quantized - encoded).detach()
        
        # Decode
        decoded = []
        for t in range(T):
            dec = self.decoder(quantized_st[:, t, :])
            decoded.append(dec)
        decoded = torch.stack(decoded, dim=1)
        
        # Losses
        recon_loss = nn.functional.mse_loss(decoded, x)
        commitment_loss = nn.functional.mse_loss(encoded, quantized.detach())
        codebook_loss = nn.functional.mse_loss(quantized, encoded.detach())
        
        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if config.use_contrastive_loss and B > 1:
            proj = self.projection(quantized_st.mean(dim=1))
            norm_proj = nn.functional.normalize(proj, dim=-1)
            sim = torch.mm(norm_proj, norm_proj.T) / 0.1
            labels = torch.arange(B, device=x.device)
            contrastive_loss = nn.functional.cross_entropy(sim, labels)
        
        # Total
        total_loss = (
            recon_loss * config.lambda_reconstruction +
            commitment_loss * config.lambda_commitment +
            codebook_loss * config.lambda_codebook +
            contrastive_loss * config.lambda_contrastive
        )
        
        return indices.view(B, T), {
            'recon_loss': recon_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss
        }

# Create model
model = SimpleGlossFreeModel().to(config.device)
print(f"\n🤖 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Training
print(f"\n🚀 Starting gloss-free training...")

# Optimizer with warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

# Learning rate scheduler
def lr_lambda(epoch):
    if epoch < config.warmup_epochs:
        return float(epoch) / float(max(1, config.warmup_epochs))
    return max(0.0, float(config.num_epochs - epoch) / float(max(1, config.num_epochs - config.warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    recon_loss = 0
    vq_loss = 0
    contrastive_loss = 0
    
    # Shuffle training data
    indices = list(range(len(train_features)))
    np.random.shuffle(indices)
    
    num_batches = len(indices) // config.batch_size
    
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx in pbar:
        # Get batch
        start = batch_idx * config.batch_size
        end = start + config.batch_size
        batch_indices = indices[start:end]
        
        batch = torch.stack([train_features[i] for i in batch_indices]).to(config.device)
        
        # Forward
        _, losses = model(batch)
        
        # Backward
        optimizer.zero_grad()
        model.module.total_loss.backward() if hasattr(model, 'module') else losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += losses['total_loss'].item()
        recon_loss += losses['recon_loss']
        vq_loss += losses['commitment_loss'] + losses['codebook_loss']
        contrastive_loss += losses['contrastive_loss']
        
        # Update progress
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'recon': f'{recon_loss/(batch_idx+1):.4f}',
            'vq': f'{vq_loss/(batch_idx+1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Update LR
    scheduler.step()
    
    # Validation
    model.eval()
    val_total = 0
    
    with torch.no_grad():
        for i in range(0, len(val_features), config.batch_size):
            batch = val_features[i:i+config.batch_size]
            if len(batch) > 0:
                batch_tensor = torch.stack(batch).to(config.device)
                _, losses = model(batch_tensor)
                val_total += losses['total_loss'].item()
    
    avg_train = total_loss / num_batches if num_batches > 0 else 0
    avg_val = val_total / (len(val_features) // config.batch_size) if len(val_features) > 0 else 0
    
    print(f"\n📊 Epoch {epoch+1}:")
    print(f"  Train Loss: {avg_train:.4f} (Recon: {recon_loss/num_batches:.4f}, VQ: {vq_loss/num_batches:.4f})")
    print(f"  Val Loss: {avg_val:.4f}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
            'config': config.__dict__
        }, f'gloss_free_checkpoint_epoch_{epoch+1}.pth')
        print(f"  💾 Saved checkpoint")

# Final save
torch.save(model.state_dict(), 'gloss_free_final_model.pth')
print(f"\n💾 Final model saved: gloss_free_final_model.pth")

print("\n" + "=" * 70)
print("🎉 GLOSS-FREE TRAINING COMPLETED!")
print("=" * 70)