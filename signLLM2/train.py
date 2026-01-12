# File: train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Imports from our files
from signLLM2.configs.config import config
from signLLM2.data.phoenix_dataset import PhoenixI3DDataset, collate_fn
from signLLM2.models.signllm import SignLLM_VQ

def main():
    print(f"🚀 Starting Training on {config.device}")
    
    # 1. Load Data
    print("📁 Loading Datasets...")
    train_dataset = PhoenixI3DDataset(split='train')
    val_dataset = PhoenixI3DDataset(split='test') # Using test as val for now
    
    if len(train_dataset) == 0:
        print("❌ Dataset empty. Please check paths in config.py")
        return

    # DataLoader with our custom collate_fn (Fixes stack error)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 2. Init Model
    model = SignLLM_VQ().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    print(f"✅ Model Created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            if batch is None: continue
            
            # Data to GPU
            x = batch['features'].to(config.device)
            mask = batch['mask'].to(config.device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(x, mask)
            loss = output['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        # Validation (Optional but recommended)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                x = batch['features'].to(config.device)
                mask = batch['mask'].to(config.device)
                output = model(x, mask)
                val_loss += output['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"🔍 Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{config.checkpoint_dir}/best_model.pth")
            print("💾 Best Model Saved!")

if __name__ == "__main__":
    main()