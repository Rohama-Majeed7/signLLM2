"""
Training script with memory optimizations
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from models.signllm import create_signllm_model
from data.phoenix_dataset import PhoenixDataset

def setup_environment():
    """Setup training environment"""
    print("=" * 70)
    print("SIGNLLM - Phoenix-2014T Training (Optimized)")
    print("=" * 70)
    
    # Configuration
    config = Config()
    
    print(f"\n📊 CONFIGURATION:")
    print(f"  Device: {config.device}")
    print(f"  Video frames: {config.video_frames}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max train samples: {config.max_train_samples}")
    
    # Clear cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return config

def create_datasets(config):
    """Create datasets with limited samples"""
    print(f"\n📁 LOADING DATASETS (Limited samples):")
    
    datasets = {}
    for split, max_samples in [
        (config.train_split, config.max_train_samples),
        (config.val_split, config.max_val_samples),
        (config.test_split, config.max_test_samples)
    ]:
        print(f"  Loading {split} split...")
        dataset = PhoenixDataset(
            config.data_root,
            split=split,
            config=config,
            max_samples=max_samples
        )
        datasets[split] = dataset
        print(f"    Loaded {len(dataset)} samples")
    
    return datasets['train'], datasets['val'], datasets['test']

def train_epoch(model, dataloader, optimizer, epoch, config):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    total_vq_loss = 0
    
    # Gradient accumulation steps (for effective larger batch size)
    accum_steps = 2
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move data to device
            videos = batch['video'].to(config.device)
            
            # Forward pass
            _, losses = model(videos, texts=batch.get('text', []))
            
            # Scale loss for gradient accumulation
            loss = losses['total_loss'] / accum_steps
            loss.backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_vq_loss += losses.get('vq_loss', 0.0).item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'vq': f'{total_vq_loss/(batch_idx+1):.4f}'
            })
            
            # Clear memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            continue
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
    epoch_vq_loss = total_vq_loss / num_batches if num_batches > 0 else 0
    
    return epoch_loss, epoch_vq_loss

def validate(model, dataloader, config):
    """Validation step"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                videos = batch['video'].to(config.device)
                
                _, losses = model(videos, texts=batch.get('text', []))
                total_loss += losses['total_loss'].item()
                batch_count += 1
                
                progress_bar.set_postfix({
                    'val_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
            except Exception as e:
                print(f"\nValidation error in batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss

def main():
    """Main training function"""
    # Setup
    config = setup_environment()
    
    # Create datasets with limited samples
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Disable for CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print(f"\n🤖 CREATING MODEL:")
    model = create_signllm_model(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Training loop
    print(f"\n🚀 STARTING TRAINING:")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_vq_loss = train_epoch(
            model, train_loader, optimizer, epoch, config
        )
        
        # Validate
        val_loss = validate(model, val_loader, config)
        
        # Print epoch summary
        print(f"\n📊 EPOCH {epoch+1}/{config.num_epochs} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f} (VQ: {train_vq_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, 'best_model.pth')
            print(f"  💾 Saved BEST model (val loss: {val_loss:.4f})")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Training completed
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    
    # Save final model
    torch.save(model.state_dict(), 'signllm_final.pth')
    print(f"\n💾 Models saved:")
    print(f"  best_model.pth - Best validation model")
    print(f"  signllm_final.pth - Final trained model")
    
    # Print summary
    print(f"\n📈 FINAL SUMMARY:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Total training samples: {len(train_dataset)}")
    print(f"  Total epochs: {config.num_epochs}")
    
    # Test inference
    print(f"\n🔍 TEST INFERENCE:")
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample = train_dataset[0]
        video = sample['video'].unsqueeze(0).to(config.device)
        
        word_indices, losses = model(video)
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Generated word tokens: {len(word_indices[0]) if word_indices else 0}")
        print(f"  Sample loss: {losses['total_loss'].item():.4f}")

if __name__ == "__main__":
    main()