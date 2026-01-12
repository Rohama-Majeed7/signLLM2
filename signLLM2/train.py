"""
Training script for Phoenix-2014T I3D Features
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from models.signllm import create_signllm_features_model
from data.phoenix_dataset import PhoenixFeaturesDataset
from data.phoenix_dataset import features_collate_fn

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def train_model():
    """Main training function"""
    print_header("SIGNLLM TRAINING - Phoenix-2014T I3D Features")
    
    # Configuration
    config = Config(training_mode="quick", use_i3d_features=True)
    
    print(f"\n📊 CONFIGURATION:")
    print(f"  Device: {config.device}")
    print(f"  Mode: {config.training_mode}")
    print(f"  Features: {'I3D' if config.use_i3d_features else 'MediaPipe'}")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    
    # Create datasets
    print(f"\n📁 LOADING DATASETS:")
    
    train_dataset = features_collate_fn(
        config.data_root,
        split=config.train_split,
        config=config,
        max_samples=config.max_train_samples
    )
    
    val_dataset = PhoenixFeaturesDataset(
        config.data_root,
        split=config.val_split,
        config=config,
        max_samples=config.max_val_samples
    )
    
    print(f"  Training: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=features_collate_fn  # Add this line
)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=features_collate_fn  # Add this line
    )
    
    # Create model
    print(f"\n🤖 CREATING MODEL...")
    model = create_signllm_features_model(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Training
    print_header(f"TRAINING - {config.num_epochs} EPOCHS")
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_bar):
            features = batch['feature'].to(config.device)
            
            # Forward pass
            _, losses = model(features)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += losses['total_loss'].item()
            train_batches += 1
            
            # Update progress
            avg_loss = train_loss / train_batches
            train_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
            for batch in val_bar:
                features = batch['feature'].to(config.device)
                _, losses = model(features)
                val_loss += losses['total_loss'].item()
                val_batches += 1
                
                avg_val = val_loss / val_batches if val_batches > 0 else 0
                val_bar.set_postfix({'val_loss': f'{avg_val:.4f}'})
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__
            }, 'best_model_features.pth')
            print(f"  💾 Saved best model (val loss: {avg_val_loss:.4f})")
    
    # Training complete
    total_time = time.time() - start_time
    
    print_header("TRAINING COMPLETE")
    print(f"\n🎉 Training completed in {total_time:.1f}s")
    print(f"📈 Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model_features.pth')
    print(f"💾 Models saved: best_model_features.pth, final_model_features.pth")
    
    # Test inference
    print(f"\n🔍 TEST INFERENCE:")
    model.eval()
    with torch.no_grad():
        sample = train_dataset[0]
        feature = sample['feature'].unsqueeze(0).to(config.device)
        
        word_indices, losses = model(feature)
        print(f"  Input shape: {feature.shape}")
        print(f"  Sample loss: {losses['total_loss'].item():.4f}")
        if word_indices and len(word_indices[0]) > 0:
            print(f"  Generated {len(word_indices[0])} word tokens")
    
    return model

if __name__ == "__main__":
    model = train_model()