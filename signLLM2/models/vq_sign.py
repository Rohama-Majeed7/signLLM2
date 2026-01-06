"""
Vector-Quantized Visual Sign (VQ-Sign) Module
Converts sign videos to discrete character-level tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VQSign(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Visual encoder (ResNet18 + Conv3D)
        self.visual_encoder = self._build_visual_encoder()
        
        # Character-level codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        self.codebook.weight.data.uniform_(-1.0/config.codebook_size, 
                                           1.0/config.codebook_size)
        
        # Autoregressive model for context prediction
        self.context_predictor = nn.GRU(
            input_size=config.codebook_dim,
            hidden_size=config.codebook_dim,
            batch_first=True
        )
        
        # Projection for contrastive loss
        self.projection = nn.Linear(config.codebook_dim, config.codebook_dim)
        
    def _build_visual_encoder(self):
        """Build visual encoder with ResNet18 + Conv3D"""
        from torchvision import models
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=self.config.pretrained_resnet)
        
        # Remove final layers
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add Conv3D layers for temporal processing
        conv3d = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, self.config.codebook_dim, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        return nn.Sequential(backbone, conv3d)
    
    def extract_features(self, x):
        """
        Extract compact features from sign video
        x: (B, T, C, H, W) video tensor
        Returns: (B, T//n, d) feature sequence
        """
        B, T, C, H, W = x.shape
        
        # Process video as overlapping clips
        clips = []
        for t in range(0, T - self.config.clip_frames + 1, self.config.clip_stride):
            clip = x[:, t:t+self.config.clip_frames]
            # (B, C, clip_frames, H, W) -> (B, d, 1, 1, 1)
            feature = self.visual_encoder(clip.permute(0, 2, 1, 3, 4))
            clips.append(feature.squeeze())
        
        # Stack features: (num_clips, B, d) -> (B, num_clips, d)
        features = torch.stack(clips, dim=1)
        return features
    
    def quantize(self, features):
        """
        Quantize features to discrete tokens using codebook
        features: (B, L, d) feature sequence
        Returns: (B, L) token indices, (B, L, d) quantized features
        """
        B, L, d = features.shape
        
        # Flatten for distance computation
        flat_features = features.reshape(-1, d)
        
        # Compute distances to codebook vectors
        distances = torch.cdist(flat_features, self.codebook.weight)
        
        # Find nearest codebook entries
        token_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(token_indices).view(B, L, d)
        
        return token_indices.view(B, L), quantized
    
    def context_prediction_loss(self, features, tokens, quantized):
        """
        Compute context prediction contrastive loss
        """
        B, L, d = features.shape
        
        # Get context representations
        context_rep, _ = self.context_predictor(quantized)
        context_rep = self.projection(context_rep)
        
        losses = []
        for k in range(1, min(self.config.num_predict_clips + 1, L)):
            positive_sim = torch.bmm(
                context_rep[:, :-k].reshape(-1, 1, d),
                features[:, k:].reshape(-1, d, 1)
            ).squeeze()
            
            # Negative sampling (in-batch negatives)
            negative_sim = torch.bmm(
                context_rep[:, :-k].reshape(-1, 1, d),
                features.roll(shifts=1, dims=0)[:, k:].reshape(-1, d, 1)
            ).squeeze()
            
            loss = -torch.log(torch.sigmoid(positive_sim)) - \
                   self.config.lambda_vq * torch.log(1 - torch.sigmoid(negative_sim))
            losses.append(loss.mean())
        
        return torch.stack(losses).mean()
    
    def forward(self, x, compute_loss=True):
        """
        Forward pass through VQ-Sign
        x: Input video (B, T, C, H, W)
        Returns: token indices, quantized features, losses
        """
        # Extract features
        features = self.extract_features(x)
        
        # Quantize to discrete tokens
        token_indices, quantized = self.quantize(features)
        
        losses = {}
        if compute_loss:
            # Context prediction loss
            cp_loss = self.context_prediction_loss(features, token_indices, quantized)
            
            # Commitment loss (Eq. 2)
            commitment_loss = F.mse_loss(features, quantized.detach())
            codebook_loss = F.mse_loss(quantized, features.detach())
            
            losses = {
                'cp_loss': cp_loss,
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'total_vq_loss': cp_loss + commitment_loss + 
                                self.config.lambda_vq * codebook_loss
            }
        
        return token_indices, quantized, losses