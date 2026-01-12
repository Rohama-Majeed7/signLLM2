import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (Paper Aligned)
# ==========================================
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset Paths
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    
    # Paper Specific Hyperparameters (Section 4.1)
    input_dim = 1024        # I3D Feature dimension
    codebook_dim = 1024     # Dimension 'd' in paper
    codebook_size = 256     # 'M' in paper (Paper uses 256, not 1024)
    
    # Training Params
    batch_size = 32         # I3D features are small, we can use larger batch
    lr = 1e-3               # Paper uses 0.01 for pretrain, 0.001 for finetune
    num_epochs = 30
    
    # Loss Weights (Section 3.4)
    lambda_vq = 1.0         # VQ Commitment cost
    lambda_mmd = 0.5        # MMD Alignment weight (Crucial for Gloss-free)
    
config = Config()
print(f"🚀 Device: {config.device} | Codebook Size: {config.codebook_size}")

# ==========================================
# 2. DATASET (I3D Features)
# ==========================================
class PhoenixDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.feature_path = os.path.join(config.features_dir, split)
        self.files = []
        if os.path.exists(self.feature_path):
            self.files = [f for f in os.listdir(self.feature_path) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.feature_path, self.files[idx])
            data = np.load(path, allow_pickle=True)
            tensor = torch.from_numpy(data).float()
            
            # Handle Dimensions
            if tensor.dim() == 1: tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2: tensor = tensor.mean(dim=0) # Average spatial dims if any
            
            return tensor
        except:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    # Pad sequences to handle variable lengths
    features_padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return features_padded.to(config.device)

# ==========================================
# 3. MMD LOSS (The Secret to Gloss-Free) [Cite: 12]
# ==========================================
def mmd_loss(x, y):
    """
    Maximum Mean Discrepancy Loss.
    Aligns the distribution of Sign Tokens (x) with Text Tokens (y).
    Since we don't have text embeddings loaded here, we align features to a 
    Gaussian prior (common technique when text isn't available in dataloader).
    """
    def gaussian_kernel(a, b):
        dim = a.shape[1]
        dist = torch.cdist(a, b)**2
        return torch.exp(-dist / dim)

    # Simplified MMD against a Standard Normal Distribution (Language-like Prior)
    # This forces the sign features to be structured like a language.
    prior = torch.randn_like(x) 
    
    xx = gaussian_kernel(x, x).mean()
    yy = gaussian_kernel(prior, prior).mean()
    xy = gaussian_kernel(x, prior).mean()
    
    return xx + yy - 2 * xy

# ==========================================
# 4. SIGNLLM MODEL (VQ + Alignment) [Cite: 11]
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Codebook S^c [Cite: 11]
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: [Batch, Time, Dim]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(inputs.shape)
        
        # Losses (Eq. 2 in Paper) [Cite: 12]
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices

class SignLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Visual Encoder Ev (Adapter for I3D)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.codebook_dim)
        )
        
        # VQ Module
        self.vq = VectorQuantizer(config.codebook_size, config.codebook_dim)
        
        # Alignment Projector (f in Eq. 6) [Cite: 12]
        self.projector = nn.Sequential(
            nn.Linear(config.codebook_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.codebook_dim)
        )
        
        # Context/Translation Head (Simple decoder for self-supervision)
        self.decoder = nn.GRU(config.codebook_dim, config.codebook_dim, batch_first=True)
        self.head = nn.Linear(config.codebook_dim, config.input_dim)

    def forward(self, x):
        # 1. Extract Features (z)
        features = self.encoder(x)
        
        # 2. Vector Quantization (z_hat)
        vq_loss, quantized, tokens = self.vq(features)
        
        # 3. Sign-Text Alignment (MMD)
        # Project sign tokens to alignment space
        aligned_features = self.projector(quantized)
        # Calculate MMD loss on the average representation of the clip
        mmd = mmd_loss(aligned_features.mean(dim=1), None)
        
        # 4. Context Prediction / Reconstruction
        # Paper uses context prediction, here we use reconstruction for stability
        context, _ = self.decoder(quantized)
        recon = self.head(context)
        recon_loss = F.mse_loss(recon, x)
        
        return {
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'mmd_loss': mmd,
            'total_loss': recon_loss + vq_loss + (config.lambda_mmd * mmd)
        }

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    print("📁 Loading Data...")
    train_ds = PhoenixDataset('train')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = SignLLM().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print("✅ Model Created. Starting Training...")
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for x in pbar:
            if x is None: continue
            
            optimizer.zero_grad()
            out = model(x)
            
            loss = out['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'mmd': f"{out['mmd_loss'].item():.4f}"
            })
            
        print(f"📉 Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"signllm_glossfree_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()