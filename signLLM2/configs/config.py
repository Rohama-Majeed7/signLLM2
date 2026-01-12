# File: signllm/configs/config.py
import torch
import os

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    
    # Model Hyperparameters (Paper Optimized)
    input_dim = 1024        # I3D feature size
    codebook_dim = 512      # Dimension of discrete tokens
    codebook_size = 1024    # Number of tokens (Paper uses 1024)
    transformer_layers = 4  # Better context modeling
    transformer_heads = 8
    
    # Training
    batch_size = 8          # Memory safe for Kaggle
    lr = 3e-4               # Standard for Transformer
    num_epochs = 30         # Gloss-free needs time to converge
    
    # Output
    checkpoint_dir = "signllm/checkpoints"

config = Config()