"""
Optimized Configuration for Phoenix-2014T
"""
import torch
import os

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset paths
    data_root = "/kaggle/input/phoenixweather2014t-3rd-attempt"
    videos_dir = os.path.join(data_root, "videos_phoenix/videos")
    
    # Optimized parameters for Kaggle memory
    video_frames = 16  # Reduced for memory (Phoenix videos are ~25fps)
    img_size = (112, 112)  # Reduced resolution
    clip_frames = 8
    clip_stride = 4
    
    # Model parameters
    codebook_size = 128  # Reduced for memory
    codebook_dim = 256
    
    # Training
    batch_size = 2  # Small for Kaggle CPU
    learning_rate = 0.001
    num_epochs = 5  # Fewer epochs for testing
    lambda_mmd = 0.5
    lambda_sim = 1.0
    
    # Dataset splits
    train_split = 'train'
    val_split = 'dev'
    test_split = 'test'
    
    # Limit samples for quick testing (set to None for full training)
    max_train_samples = 100  # Small subset for testing
    max_val_samples = 20
    max_test_samples = 20
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Verify dataset exists
        if not os.path.exists(self.videos_dir):
            print(f"⚠️ Warning: Videos directory not found: {self.videos_dir}")
        else:
            print(f"✅ Dataset found at: {self.videos_dir}")
            
        # Auto-detect if using CPU or GPU
        if self.device.type == 'cpu':
            print("⚠️ Training on CPU - using smaller parameters")
            self.batch_size = 2
            self.video_frames = 16
            self.img_size = (112, 112)
            self.max_train_samples = 50  # Even smaller for CPU