"""
Improved Configuration for Maximum Gloss-Free Performance
"""
import torch
import os

class ImprovedConfig:
    # ==================== GLOSS-FREE OPTIMIZATION ====================
    gloss_free = True
    target_bleu = 22.0  # Target gloss-free BLEU score
    
    # ==================== DEVICE ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== DATASET PATHS ====================
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    features_dir = i3d_features_dir
    
    # ==================== IMPROVED MODEL ARCHITECTURE ====================
    # Larger model for better representation
    codebook_size = 1024  # Increased from 512 (Paper uses 1024)
    codebook_dim = 1024   # Increased from 768
    
    # Transformer dimensions
    transformer_dim = 512
    transformer_heads = 8
    transformer_layers = 4
    
    # ==================== IMPROVED TRAINING PARAMETERS ====================
    batch_size = 16 if torch.cuda.is_available() else 8
    learning_rate = 3e-4  # Optimal for transformers
    num_epochs = 50       # More epochs for convergence
    warmup_steps = 2000
    
    # ==================== IMPROVED LOSS WEIGHTS ====================
    lambda_commitment = 1.0
    lambda_codebook = 0.25
    lambda_reconstruction = 2.0  # Increased for better reconstruction
    lambda_contrastive = 1.0     # Increased for better alignment
    lambda_diversity = 0.1       # New: codebook diversity loss
    
    # ==================== ADVANCED TECHNIQUES ====================
    use_transformer = True        # Transformer instead of GRU
    use_infonce_loss = True       # Better contrastive loss
    use_feature_augmentation = True
    use_codebook_diversity = True
    use_ema_codebook = True       # Exponential moving average for codebook
    
    # ==================== DATASET ====================
    train_split = 'train'
    val_split = 'val'
    test_split = 'test'
    max_train_samples = 2000  # Increased from 500
    max_val_samples = 200
    max_test_samples = 200
    
    # Sequence handling
    max_seq_length = 200
    min_seq_length = 50
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        print(f"\n🎯 GLOSS-FREE OPTIMIZATION TARGET: BLEU > {self.target_bleu}")
        print(f"📊 Improved Configuration:")
        print(f"   Codebook: {self.codebook_size} tokens, dim {self.codebook_dim}")
        print(f"   Transformer: {self.transformer_layers} layers, {self.transformer_heads} heads")
        print(f"   Training samples: {self.max_train_samples}")
        print(f"   Advanced techniques: {sum([self.use_transformer, self.use_infonce_loss, self.use_feature_augmentation])}/3")