"""
Configuration for Gloss-Free SignLLM
"""
import torch
import os

class Config:
    # ==================== GLOSS-FREE SETTINGS ====================
    gloss_free = True  # No gloss annotations used
    use_text_only = True  # Use only text supervision (no gloss)
    
    # ==================== DEVICE ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== DATASET PATHS ====================
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    features_dir = i3d_features_dir
    
    # ==================== MODEL ARCHITECTURE ====================
    # Paper settings for gloss-free
    codebook_size = 512  # Increased for better representation
    codebook_dim = 768   # Increased dimension
    feature_dim = 1024   # I3D features
    
    # ==================== TRAINING PARAMETERS ====================
    # Gloss-free requires more careful training
    batch_size = 8
    learning_rate = 1e-4  # Lower LR for gloss-free
    num_epochs = 30       # More epochs for gloss-free
    warmup_epochs = 5     # Warmup for stable training
    
    # Loss weights (gloss-free specific)
    lambda_commitment = 1.0
    lambda_codebook = 0.25
    lambda_reconstruction = 1.0
    lambda_contrastive = 0.5  # For contrastive learning
    
    # ==================== GLOSS-FREE TRAINING TRICKS ====================
    use_contrastive_loss = True      # Contrastive learning helps gloss-free
    use_reconstruction_loss = True   # Reconstruction from quantized features
    use_curriculum_learning = True   # Start easy, get harder
    max_seq_length = 150             # Fixed sequence length
    
    # ==================== DATASET ====================
    train_split = 'train'
    val_split = 'val'
    test_split = 'test'
    max_train_samples = 500  # Use more samples for gloss-free
    max_val_samples = 100
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        print(f"\n🔧 GLOSS-FREE CONFIGURATION")
        print(f"   Mode: {'GLOSS-FREE' if self.gloss_free else 'Gloss-based'}")
        print(f"   Device: {self.device}")
        print(f"   Codebook: {self.codebook_size} tokens, dim {self.codebook_dim}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.num_epochs}")