"""
Training script for Phoenix-2014T dataset
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from models.signllm import create_signllm_model
from data.phoenix_dataset import PhoenixDataset

def setup_environment():
    """Setup training environment"""
    print("=" * 70)
    print("SIGNLLM - Phoenix-2014T Training")
    print("=" * 70)
    
    # Configuration
    config = Config()
    
    print(f"\n📊 CONFIGURATION:")
    print(f"  Device: {config.device}")
    print(f"  Dataset: Phoenix-2014T")
    print(f"  Videos directory: {config.videos_dir}")
    print(f"  Video frames: {config.video_frames}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    
    return config

def create_datasets(config):
    """Create train, validation, and test datasets"""
    print(f"\n📁 LOADING DATASETS:")
    
    # Training dataset
    print(f"  Loading training split: '{config.train_split}'")
    train_dataset = PhoenixDataset(
        config.data_root,
        split=config.train_split,
        config=config
    )
    
    # Validation dataset
    print(f"  Loading validation split: '{config.val_split}'")
    val_dataset = PhoenixDataset(
        config.data_root,
        split=config.val_split,
        config=config
    )
    
    # Test dataset
    print(f"  Loading test split: '{config.test_split}'")
    test_dataset = PhoenixDataset(
        config.data_root,
        split=config.test_split,
        config=config
    )
    
    print(f"\n  📈 DATASET STATS:")
    print(f"    Training samples: {len(train_dataset)}")
    print(f"    Validation samples: {len(val_dataset)}")
    print(f"    Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(model, dataloader, optimizer, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_translation_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        videos = batch['video'].to(config.device)
        texts = batch['text']
        
        # Forward pass
        _, losses = model(videos, texts=texts)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total_loss'].item()
        total_vq_loss += losses.get('vq_loss', 0.0).item()
        total_translation_loss += losses.get('translation_loss', 0.0).item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'vq': f'{total_vq_loss/(batch_idx+1):.4f}',
            'trans': f'{total_translation_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
    epoch_vq_loss = total_vq_loss / num_batches if num_batches > 0 else 0
    epoch_trans_loss = total_translation_loss / num_batches if num_batches > 0 else 0
    
    return epoch_loss, epoch_vq_loss, epoch_trans_loss

def validate(model, dataloader, config):
    """Validation step"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            videos = batch['video'].to(config.device)
            texts = batch['text']
            
            _, losses = model(videos, texts=texts)
            total_loss += losses['total_loss'].item()
            
            progress_bar.set_postfix({
                'val_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

def main():
    """Main training function"""
    # Setup
    config = setup_environment()
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Kaggle
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print(f"\n🤖 CREATING MODEL:")
    model = create_signllm_model(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\n🚀 STARTING TRAINING:")
    print("=" * 70)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_vq_loss': [],
        'train_trans_loss': []
    }
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_vq_loss, train_trans_loss = train_epoch(
            model, train_loader, optimizer, epoch, config
        )
        
        # Validate
        val_loss = validate(model, val_loader, config)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_vq_loss'].append(train_vq_loss)
        history['train_trans_loss'].append(train_trans_loss)
        
        # Print epoch summary
        print(f"\n📊 EPOCH {epoch+1}/{config.num_epochs} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f} (VQ: {train_vq_loss:.4f}, Trans: {train_trans_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
                'history': history
            }, 'best_model.pth')
            print(f"  💾 Saved BEST model (val loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    # Training completed
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    
    # Save final model
    torch.save(model.state_dict(), 'signllm_final.pth')
    print(f"\n💾 Models saved:")
    print(f"  best_model.pth - Best validation model")
    print(f"  signllm_final.pth - Final trained model")
    
    # Test on test set
    print(f"\n🧪 TESTING ON TEST SET:")
    test_loss = validate(model, test_loader, config)
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Print final summary
    print(f"\n📈 FINAL SUMMARY:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final test loss: {test_loss:.4f}")
    print(f"  Total training samples: {len(train_dataset)}")
    print(f"  Total epochs: {config.num_epochs}")
    
    # Sample inference
    print(f"\n🔍 SAMPLE INFERENCE:")
    model.eval()
    with torch.no_grad():
        # Get a sample from test set
        sample = test_dataset[0]
        video = sample['video'].unsqueeze(0).to(config.device)
        text = sample['text']
        
        print(f"  Input video: {sample['video_path']}")
        print(f"  Ground truth: {text}")
        
        # Generate
        word_indices, losses = model(video)
        print(f"  Generated {len(word_indices[0]) if word_indices else 0} word tokens")
        print(f"  Inference loss: {losses['total_loss'].item():.4f}")

if __name__ == "__main__":
    main()