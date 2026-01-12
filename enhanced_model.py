"""
Enhanced Model with Transformer and InfoNCE Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EMACodebook(nn.Module):
    """Exponential Moving Average Codebook for better training stability"""
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        
        # Codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=0.02)
        
        # EMA statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', torch.zeros(num_embeddings, embedding_dim))
        
        # Initialize with first batch
        self._is_initialized = False
    
    def forward(self, z_e):
        """Forward with EMA update"""
        B, T, D = z_e.shape
        flat_z_e = z_e.reshape(-1, D)
        
        # Calculate distances
        distances = torch.cdist(flat_z_e, self.codebook.weight)
        
        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=-1)
        
        if self.training:
            # One-hot encoding of indices
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # Update EMA statistics
            if not self._is_initialized:
                # Initialize with first batch
                self.ema_cluster_size = encodings.sum(0)
                self.ema_weight = torch.matmul(encodings.T, flat_z_e)
                self._is_initialized = True
            else:
                # EMA update
                self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
                self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * torch.matmul(encodings.T, flat_z_e)
            
            # Laplace smoothing and normalization
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.eps) /
                (n + self.num_embeddings * self.eps) * n
            )
            
            # Normalize codebook
            codebook_normalized = self.ema_weight / smoothed_cluster_size.unsqueeze(1)
            
            # Update codebook weights
            self.codebook.weight.data = codebook_normalized
        
        # Get quantized vectors
        quantized = self.codebook(encoding_indices).view(B, T, D)
        
        # Straight-through estimator
        quantized_st = z_e + (quantized - z_e).detach()
        
        return encoding_indices.view(B, T), quantized, quantized_st

class EnhancedEncoder(nn.Module):
    """Enhanced encoder with residual connections"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-layer encoder with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 if i == 0 else config.codebook_dim, config.codebook_dim),
                nn.LayerNorm(config.codebook_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for i in range(4)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.codebook_dim, config.codebook_dim)
    
    def forward(self, x):
        """Forward with residual connections"""
        # x: (B, T, 1024)
        B, T, D = x.shape
        
        # Process each time step
        encoded = []
        for t in range(T):
            h = x[:, t, :]
            
            # Residual encoding
            for i, layer in enumerate(self.layers):
                residual = h if i == 0 else torch.zeros_like(h)
                h = layer(h) + residual
            
            encoded.append(h)
        
        encoded = torch.stack(encoded, dim=1)  # (B, T, codebook_dim)
        encoded = self.output_proj(encoded)
        
        return encoded

class SignTransformer(nn.Module):
    """Transformer for sign language sequence modeling"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.codebook_dim, max_len=500)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.codebook_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(config.codebook_dim)
    
    def forward(self, x):
        """Forward pass"""
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Normalize
        x = self.norm(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for better representation learning"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features):
        """
        features: (B, D) normalized features
        """
        B = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Labels
        labels = torch.arange(B, device=features.device)
        
        # InfoNCE loss
        loss = self.criterion(sim_matrix, labels)
        
        return loss

class EnhancedGlossFreeModel(nn.Module):
    """Complete enhanced model for high gloss-free performance"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced components
        self.encoder = EnhancedEncoder(config)
        self.codebook = EMACodebook(config.codebook_size, config.codebook_dim)
        
        # Transformer context model
        if config.use_transformer:
            self.context_model = SignTransformer(config)
        else:
            self.context_model = nn.GRU(
                config.codebook_dim,
                config.codebook_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(config.codebook_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1024)
        )
        
        # Projection heads for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.codebook_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Loss modules
        self.infonce_loss = InfoNCELoss(temperature=0.1)
        
        # Text decoder (simplified)
        self.text_decoder = nn.Linear(config.codebook_dim, 3000)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass with all losses"""
        B, T, D = x.shape
        
        # 1. Encode features
        encoded = self.encoder(x)  # (B, T, codebook_dim)
        
        # 2. Quantize with EMA codebook
        tokens, quantized, quantized_st = self.codebook(encoded)
        
        # 3. Context modeling
        if self.config.use_transformer:
            context = self.context_model(quantized_st)  # (B, T, codebook_dim)
        else:
            context, _ = self.context_model(quantized_st)
        
        # 4. Decode for reconstruction
        reconstructed = []
        for t in range(T):
            recon_t = self.decoder(quantized_st[:, t, :])
            reconstructed.append(recon_t)
        reconstructed = torch.stack(reconstructed, dim=1)  # (B, T, 1024)
        
        # 5. Project for contrastive learning
        projected = self.projection_head(context.mean(dim=1))  # (B, 256)
        
        # 6. Calculate losses
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Commitment loss
        commitment_loss = F.mse_loss(encoded, quantized.detach())
        
        # Codebook loss
        codebook_loss = F.mse_loss(quantized, encoded.detach())
        
        # InfoNCE contrastive loss
        contrastive_loss = self.infonce_loss(projected)
        
        # Codebook diversity loss
        diversity_loss = torch.tensor(0.0, device=x.device)
        if self.config.use_codebook_diversity:
            # Encourage uniform codebook usage
            codebook_usage = torch.bincount(
                tokens.flatten(),
                minlength=self.config.codebook_size
            ).float()
            codebook_usage = codebook_usage / codebook_usage.sum()
            diversity_loss = -torch.sum(codebook_usage * torch.log(codebook_usage + 1e-10))
        
        # Total loss
        total_loss = (
            recon_loss * self.config.lambda_reconstruction +
            commitment_loss * self.config.lambda_commitment +
            codebook_loss * self.config.lambda_codebook +
            contrastive_loss * self.config.lambda_contrastive +
            diversity_loss * self.config.lambda_diversity
        )
        
        # Calculate metrics for reporting
        metrics = {
            'recon_loss': recon_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
            'total_loss': total_loss.item(),
            'codebook_usage_unique': len(torch.unique(tokens)),
            'codebook_usage_rate': len(torch.unique(tokens)) / self.config.codebook_size
        }
        
        return tokens, metrics, {
            'encoded': encoded,
            'quantized': quantized,
            'context': context,
            'reconstructed': reconstructed,
            'projected': projected
        }