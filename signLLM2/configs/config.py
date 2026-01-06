"""
Configuration for Phoenix-2014T Dataset
"""
import torch
import os

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset paths (updated for your specific dataset)
    data_root = "/kaggle/input/phoenixweather2014t-3rd-attempt"
    videos_dir = os.path.join(data_root, "videos_phoenix/videos")
    annotations_dir = data_root  # Annotations are in the root
    
    # Video parameters (optimized for Phoenix dataset)
    video_frames = 32  # Phoenix videos are typically 25-30fps
    img_size = (224, 224)  # Original Phoenix resolution
    clip_frames = 16
    clip_stride = 8
    
    # Model parameters
    codebook_size = 256  # Larger for real data
    codebook_dim = 512
    
    # Training
    batch_size = 4  # Can increase to 8 if you have enough memory
    learning_rate = 0.001
    num_epochs = 5
    lambda_mmd = 0.5
    lambda_sim = 1.0
    
    # Dataset splits
    train_split = 'train'
    val_split = 'dev'
    test_split = 'test'
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Verify dataset exists
        if not os.path.exists(self.videos_dir):
            print(f"⚠️ Warning: Videos directory not found: {self.videos_dir}")
        else:
            print(f"✅ Dataset found at: {self.videos_dir}")