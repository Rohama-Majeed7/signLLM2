"""
Configuration file for SignLLM
"""
import torch

class Config:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data parameters
    video_frames = 64  # Total frames per video
    clip_frames = 13   # Frames per clip
    clip_stride = 4    # Stride between clips
    img_size = (224, 224)  # Video frame size
    
    # VQ-Sign parameters
    codebook_size = 256     # M - number of character-level tokens
    codebook_dim = 1024     # d - token dimension
    num_predict_clips = 3   # K - future clips to predict
    
    # CRA parameters
    word_increment = 32     # m - codebook size increment
    max_word_length = 8     # Maximum characters per word
    
    # Training parameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 20
    lambda_vq = 0.25       # γ in Eq. 2
    lambda_mmd = 0.5       # λ1 in fine-tuning
    lambda_sim = 1.0       # λ2 in fine-tuning
    
    # Model paths
    pretrained_resnet = True
    llm_model = "decapoda-research/llama-7b-hf"  # Or use smaller model for Kaggle
    
    # Dataset paths (Kaggle specific)
    data_root = "/kaggle/input/phoenix2014t"
    
config = Config()