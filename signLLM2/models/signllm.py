# File: signllm/models/signllm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from signLLM2.configs.config import config

class SignLLM_VQ(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Feature Compression
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, config.codebook_dim)
        )
        
        # 2. VQ Codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        # Init codebook properly
        nn.init.uniform_(self.codebook.weight, -1/config.codebook_size, 1/config.codebook_size)
        
        # 3. Context Modeling (Transformer) - Best for Gloss Free
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.codebook_dim,
            nhead=config.transformer_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        
        # 4. Decoder (Reconstruction)
        self.decoder = nn.Linear(config.codebook_dim, config.input_dim)

    def forward(self, x, mask=None):
        # x: (Batch, Time, 1024)
        
        # Encode -> (Batch, Time, 512)
        z_e = self.encoder(x)
        
        # --- Vector Quantization (VQ) ---
        z_e_flat = z_e.view(-1, config.codebook_dim)
        
        # Distances calculate karna
        d = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - \
            2 * torch.matmul(z_e_flat, self.codebook.weight.t())
            
        min_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_indices).view(z_e.shape)
        
        # Straight Through Estimator (Gradient flow ke liye zaroori hai)
        z_q = z_e + (z_q - z_e).detach()
        
        # --- Transformer Context ---
        # Masking: Padding positions (False in mask) should be ignored
        if mask is not None:
            # Transformer expects True for padding to be IGNORED
            padding_mask = ~mask
            context = self.transformer(z_q, src_key_padding_mask=padding_mask)
        else:
            context = self.transformer(z_q)
            
        # Reconstruct
        recon = self.decoder(context)
        
        # --- Losses ---
        # 1. Reconstruction Loss (Sirf valid frames par, padding par nahi)
        if mask is not None:
            recon_loss = F.mse_loss(recon[mask], x[mask])
            commit_loss = F.mse_loss(z_e[mask], z_q.detach()[mask])
            codebook_loss = F.mse_loss(z_q[mask], z_e.detach()[mask])
        else:
            recon_loss = F.mse_loss(recon, x)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            
        total_loss = recon_loss + commit_loss + 0.25 * codebook_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'tokens': min_indices.view(x.shape[0], -1)
        }