"""
Training script for SignLLM
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import wandb

from configs.config import Config
from data.phoenix_dataset import PhoenixDataset
from models.signllm import SignLLM

def train_epoch(model, dataloader, optimizer, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        videos = batch['video'].to(config.device)
        texts = batch['text']
        
        # Forward pass
        _, losses = model(videos, texts)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # Logging
        total_loss += losses['total_loss'].item()
        
        pbar.set_postfix({
            'loss': losses['total_loss'].item(),
            'vq_loss': losses.get('total_vq_loss', 0).item(),
            'mmd_loss': losses.get('mmd_loss', 0).item(),
            'trans_loss': losses.get('translation_loss', 0).item()
        })
        
        # Wandb logging
        if batch_idx % 10 == 0:
            wandb.log({
                'train/loss': losses['total_loss'].item(),
                'train/vq_loss': losses.get('total_vq_loss', 0).item(),
                'train/step': epoch * len(dataloader) + batch_idx
            })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, config):
    """Validation step"""
    model.eval()
    total_bleu = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            videos = batch['video'].to(config.device)
            texts = batch['text']
            
            # Generate translations
            translations, _ = model(videos)
            
            # Compute BLEU score (simplified)
            for pred, true in zip(translations, texts):
                # Simple word overlap (replace with proper BLEU)
                pred_words = set(pred.lower().split())
                true_words = set(true.lower().split())
                if true_words:
                    overlap = len(pred_words & true_words) / len(true_words)
                    total_bleu += overlap
                    num_samples += 1
    
    return total_bleu / max(num_samples, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/phoenixweather2014t-3rd-attempt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.data_root = args.data_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='signllm', config=vars(config))
    
    # Dataset and dataloader
    train_dataset = PhoenixDataset(config.data_root, 'train', config=config)
    val_dataset = PhoenixDataset(config.data_root, 'dev', config=config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # Model
    model = SignLLM(config).to(config.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    )
    
    # Training loop
    best_bleu = 0
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, config)
        
        # Validate
        bleu_score = validate(model, val_loader, config)
        
        print(f'Epoch {epoch}: Loss={train_loss:.4f}, BLEU={bleu_score:.4f}')
        
        # Save best model
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bleu_score': bleu_score,
                'config': config
            }, 'best_model.pth')
            
            print(f'Saved best model with BLEU: {bleu_score:.4f}')
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss_epoch': train_loss,
                'val/bleu': bleu_score,
                'val/best_bleu': best_bleu
            })
    
    print(f'Training completed. Best BLEU: {best_bleu:.4f}')

if __name__ == '__main__':
    main()