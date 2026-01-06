"""
SignLLM Model optimized for Phoenix-2014T dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VQSign(nn.Module):
    """Vector-Quantized Visual Sign Module for Phoenix dataset"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 3D CNN Encoder (optimized for sign language)
        self.encoder = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Conv Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Conv Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((config.clip_frames, 4, 4))
        )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # Projection to codebook dimension
        self.projection = nn.Linear(256 * 4 * 4, config.codebook_dim)
        
        # Codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/config.codebook_size, 1.0/config.codebook_size)
        
        # Context predictor (for self-supervised learning)
        self.context_predictor = nn.GRU(
            config.codebook_dim,
            config.codebook_dim,
            batch_first=True,
            num_layers=2
        )
    
    def extract_features(self, x):
        """Extract features from video"""
        B, C, T, H, W = x.shape
        
        # Process as clips
        num_clips = (T - self.config.clip_frames) // self.config.clip_stride + 1
        features = []
        
        for i in range(num_clips):
            start = i * self.config.clip_stride
            end = start + self.config.clip_frames
            clip = x[:, :, start:end]
            
            # Encode clip
            feat = self.encoder(clip)
            feat = self.temporal_conv(feat)
            
            # Flatten and project
            feat = feat.view(B, -1)
            feat = self.projection(feat)
            features.append(feat)
        
        # Stack: (B, num_clips, codebook_dim)
        if features:
            return torch.stack(features, dim=1)
        else:
            # Handle case when no clips can be extracted
            dummy = torch.zeros(B, 1, self.config.codebook_dim, device=x.device)
            return dummy
    
    def quantize(self, features):
        """Quantize features using codebook"""
        B, L, D = features.shape
        
        # Reshape for quantization
        flat_features = features.reshape(-1, D)
        
        # Find nearest codebook entries
        distances = torch.cdist(flat_features, self.codebook.weight)
        token_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(token_indices).view(B, L, D)
        
        return token_indices.view(B, L), quantized
    
    def forward(self, x):
        """Forward pass"""
        # Extract features
        features = self.extract_features(x)
        
        # Quantize
        token_indices, quantized = self.quantize(features)
        
        # Compute losses
        commitment_loss = F.mse_loss(features, quantized.detach())
        codebook_loss = F.mse_loss(quantized, features.detach())
        
        # Context prediction loss (simplified)
        context_loss = torch.tensor(0.0, device=x.device)
        if features.shape[1] > 1:
            context_features, _ = self.context_predictor(quantized[:, :-1])
            context_loss = F.mse_loss(context_features, features[:, 1:])
        
        losses = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'context_loss': context_loss,
            'vq_loss': commitment_loss + 0.25 * codebook_loss + 0.1 * context_loss
        }
        
        return token_indices, quantized, losses

class SimpleCRA(nn.Module):
    """Simplified Codebook Reconstruction and Alignment"""
    def __init__(self, config, char_codebook):
        super().__init__()
        self.config = config
        
        # Word codebook (smaller than character codebook)
        word_size = config.codebook_size // 4
        self.word_codebook = nn.Embedding(word_size, config.codebook_dim)
        nn.init.uniform_(self.word_codebook.weight, -1.0/word_size, 1.0/word_size)
        
        # Learnable grouping
        self.grouping = nn.Sequential(
            nn.Linear(config.codebook_dim * 2, config.codebook_dim),
            nn.ReLU(),
            nn.Linear(config.codebook_dim, 1)
        )
        
        # Alignment projection
        self.alignment_proj = nn.Linear(config.codebook_dim, config.codebook_dim)
    
    def group_characters(self, char_tokens, char_embeddings):
        """Group characters into words"""
        B, L, D = char_embeddings.shape
        
        # Simple fixed grouping: every 2 characters = 1 word
        word_length = 2
        word_embeddings = []
        word_indices = []
        
        for b in range(B):
            words_b = []
            indices_b = []
            
            for i in range(0, L, word_length):
                if i + word_length <= L:
                    # Average character embeddings
                    char_group = char_embeddings[b, i:i+word_length]
                    word_emb = char_group.mean(dim=0)
                    
                    # Find nearest word in codebook
                    distances = torch.cdist(word_emb.unsqueeze(0), self.word_codebook.weight)
                    word_idx = torch.argmin(distances)
                    
                    words_b.append(self.word_codebook(word_idx))
                    indices_b.append(word_idx.item())
            
            if words_b:
                word_embeddings.append(torch.stack(words_b))
                word_indices.append(indices_b)
            else:
                word_embeddings.append(torch.zeros(0, D, device=char_embeddings.device))
                word_indices.append([])
        
        return word_indices, word_embeddings
    
    def forward(self, char_tokens, char_embeddings):
        """Forward pass"""
        word_indices, word_embeddings = self.group_characters(char_tokens, char_embeddings)
        
        # Alignment loss (simplified)
        alignment_loss = torch.tensor(0.0, device=char_embeddings.device)
        
        # Project for alignment
        if word_embeddings and len(word_embeddings[0]) > 0:
            projected = self.alignment_proj(torch.cat(word_embeddings))
            alignment_loss = F.mse_loss(projected, torch.cat(word_embeddings).detach())
        
        return word_indices, word_embeddings, {'alignment_loss': alignment_loss}

class SignLLM(nn.Module):
    """Complete SignLLM Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modules
        self.vq_sign = VQSign(config)
        self.cra = SimpleCRA(config, self.vq_sign.codebook)
        
        # Text decoder (vocabulary size based on Phoenix dataset)
        vocab_size = 3000  # Phoenix has ~2887 German words
        self.text_decoder = nn.Sequential(
            nn.Linear(config.codebook_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, vocab_size)
        )
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, config.codebook_dim)
        
        # Loss weights
        self.lambda_mmd = config.lambda_mmd
        self.lambda_sim = config.lambda_sim
    
    def forward(self, videos, texts=None, text_tokens=None):
        """
        Forward pass
        videos: (B, C, T, H, W)
        texts: list of strings (optional)
        text_tokens: (B, L) tokenized texts (optional)
        """
        # VQ-Sign: Video -> Character tokens
        char_tokens, char_embeddings, vq_losses = self.vq_sign(videos)
        
        # CRA: Character -> Word tokens
        word_indices, word_embeddings, cra_losses = self.cra(char_tokens, char_embeddings)
        
        # Text generation
        translation_loss = torch.tensor(0.0, device=videos.device)
        
        if text_tokens is not None:
            # Use provided text tokens
            batch_size = videos.shape[0]
            for b in range(batch_size):
                if word_embeddings[b].shape[0] > 0:
                    # Use mean of word embeddings as context
                    context = word_embeddings[b].mean(dim=0, keepdim=True)
                    logits = self.text_decoder(context)
                    
                    # Simple cross-entropy with first token
                    if text_tokens[b].numel() > 0:
                        target = text_tokens[b][0] if len(text_tokens[b]) > 0 else 0
                        translation_loss += F.cross_entropy(logits, target.unsqueeze(0))
        
        elif texts is not None and len(texts) > 0:
            # Simple dummy loss for now
            translation_loss = torch.tensor(0.1, device=videos.device)
        
        # Combine losses
        total_loss = (
            vq_losses.get('vq_loss', 0.0) +
            cra_losses.get('alignment_loss', 0.0) * self.lambda_mmd +
            translation_loss * self.lambda_sim
        )
        
        losses = {
            **vq_losses,
            **cra_losses,
            'translation_loss': translation_loss,
            'total_loss': total_loss
        }
        
        return word_indices, losses

# Helper function to create model
def create_signllm_model(config):
    """Create and initialize SignLLM model"""
    model = SignLLM(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model