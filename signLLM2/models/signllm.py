"""
SignLLM Model for I3D Features - Fixed shape handling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQSignFeatures(nn.Module):
    """Vector-Quantized Visual Sign Module for I3D Features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input feature dimension
        self.input_dim = config.feature_dim
        
        # Feature processor (for temporal features)
        self.feature_processor = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.codebook_dim),
            nn.ReLU()
        )
        
        # Temporal convolution for sequence processing
        self.temporal_conv = nn.Conv1d(
            config.codebook_dim, 
            config.codebook_dim, 
            kernel_size=3, 
            padding=1
        )
        
        # Codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/config.codebook_size, 1.0/config.codebook_size)
        
        # Context predictor
        self.context_predictor = nn.GRU(
            config.codebook_dim,
            config.codebook_dim,
            batch_first=True,
            num_layers=1
        )
    
    def process_features(self, x):
        """
        Process input features with flexible shape handling
        x shape could be: (B, D), (B, T, D), (B, C, T, D), etc.
        Returns: (B, T, codebook_dim)
        """
        B = x.shape[0]
        
        print(f"Debug: Input shape: {x.shape}")
        
        # Handle different input shapes
        if x.dim() == 2:
            # (B, D) -> single feature vector
            x = x.unsqueeze(1)  # (B, 1, D)
            print(f"Debug: Reshaped to: {x.shape}")
        
        elif x.dim() == 4:
            # (B, C, T, D) -> flatten C and T
            B, C, T, D = x.shape
            x = x.permute(0, 2, 1, 3)  # (B, T, C, D)
            x = x.reshape(B, T * C, D)  # (B, T*C, D)
            print(f"Debug: Reshaped 4D to: {x.shape}")
        
        elif x.dim() > 4:
            # Flatten all dimensions except batch and last
            x = x.reshape(B, -1, x.shape[-1])
            print(f"Debug: Flattened to: {x.shape}")
        
        # x should now be (B, T, D)
        B, T, D = x.shape
        
        # Process each temporal step
        processed = []
        for t in range(T):
            feature_slice = x[:, t, :]  # (B, D)
            processed_slice = self.feature_processor(feature_slice)  # (B, codebook_dim)
            processed.append(processed_slice)
        
        # Stack: (B, T, codebook_dim)
        features = torch.stack(processed, dim=1)
        print(f"Debug: After processing: {features.shape}")
        
        # Apply temporal convolution if T > 1
        if T > 1:
            features = features.permute(0, 2, 1)  # (B, codebook_dim, T)
            features = self.temporal_conv(features)
            features = features.permute(0, 2, 1)  # (B, T, codebook_dim)
            print(f"Debug: After temporal conv: {features.shape}")
        
        return features
    
    def quantize(self, features):
        """Quantize features using codebook"""
        B, T, D = features.shape
        
        # Reshape for quantization
        flat_features = features.reshape(-1, D)
        
        # Find nearest codebook entries
        distances = torch.cdist(flat_features, self.codebook.weight)
        token_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook(token_indices).view(B, T, D)
        
        return token_indices.view(B, T), quantized
    
    def forward(self, x):
        """Forward pass"""
        # Process features
        features = self.process_features(x)
        
        # Quantize
        token_indices, quantized = self.quantize(features)
        
        # Compute losses
        commitment_loss = F.mse_loss(features, quantized.detach())
        codebook_loss = F.mse_loss(quantized, features.detach())
        
        # Context prediction loss
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

class SimpleCRAFeatures(nn.Module):
    """Codebook Reconstruction and Alignment for Features"""
    def __init__(self, config, char_codebook):
        super().__init__()
        self.config = config
        
        # Word codebook
        word_size = max(32, config.codebook_size // 8)
        self.word_codebook = nn.Embedding(word_size, config.codebook_dim)
        nn.init.uniform_(self.word_codebook.weight, -1.0/word_size, 1.0/word_size)
        
        # Alignment projection
        self.alignment_proj = nn.Linear(config.codebook_dim, config.codebook_dim)
    
    def group_characters(self, char_tokens, char_embeddings):
        """Group character tokens into word tokens"""
        B, T, D = char_embeddings.shape
        
        word_embeddings = []
        word_indices = []
        
        for b in range(B):
            words_b = []
            indices_b = []
            
            # Simple grouping: every 2 time steps = 1 word
            word_length = 2
            for i in range(0, T, word_length):
                if i + word_length <= T:
                    char_group = char_embeddings[b, i:i+word_length]
                    word_emb = char_group.mean(dim=0)
                    
                    distances = torch.cdist(word_emb.unsqueeze(0), self.word_codebook.weight)
                    word_idx = torch.argmin(distances)
                    
                    words_b.append(self.word_codebook(word_idx))
                    indices_b.append(word_idx.item())
                elif i < T:  # Handle leftover
                    word_emb = char_embeddings[b, i:i+1].mean(dim=0)
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
        
        alignment_loss = torch.tensor(0.0, device=char_embeddings.device)
        
        if word_embeddings and len(word_embeddings[0]) > 0:
            all_word_embs = torch.cat(word_embeddings)
            projected = self.alignment_proj(all_word_embs)
            alignment_loss = F.mse_loss(projected, all_word_embs.detach())
        
        return word_indices, word_embeddings, {'alignment_loss': alignment_loss}

class TextDecoderFeatures(nn.Module):
    """Text decoder for features"""
    def __init__(self, input_dim, vocab_size=5000):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, vocab_size)
        )
    
    def forward(self, x):
        return self.decoder(x)

class SignLLMFeatures(nn.Module):
    """Complete SignLLM Model for I3D Features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modules
        self.vq_sign = VQSignFeatures(config)
        self.cra = SimpleCRAFeatures(config, self.vq_sign.codebook)
        
        # Text decoder
        vocab_size = 5000  # Adjust based on your dataset
        self.text_decoder = TextDecoderFeatures(config.codebook_dim, vocab_size)
        
        # Loss weights
        self.lambda_mmd = config.lambda_mmd
        self.lambda_sim = config.lambda_sim
    
    def forward(self, features, texts=None):
        """
        Forward pass
        features: Input features (B, ...)
        texts: Optional texts for training
        """
        B = features.shape[0]
        
        # VQ-Sign: Features -> Character tokens
        char_tokens, char_embeddings, vq_losses = self.vq_sign(features)
        
        # CRA: Character -> Word tokens
        word_indices, word_embeddings, cra_losses = self.cra(char_tokens, char_embeddings)
        
        # Text generation loss
        translation_loss = torch.tensor(0.0, device=features.device)
        
        if texts is not None:
            # Simple translation loss (you can implement proper loss here)
            translation_loss = torch.tensor(0.1, device=features.device) * B
        
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

# Helper function
def create_signllm_features_model(config):
    """Create and initialize SignLLM model for features"""
    model = SignLLMFeatures(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model