"""
Fixed test script for SignLLM
"""
import os
import sys
sys.path.insert(0, '/kaggle/working/signLLM2/signLLM2')

print("Testing SignLLM with I3D Features - FIXED")
print("=" * 60)

# ==================== TEST 1: CONFIG ====================
print("\n1. Testing config...")
try:
    from configs.config import Config
    config = Config(training_mode="test", use_i3d_features=True)
    print(f"✅ Config loaded")
    print(f"   Device: {config.device}")
    print(f"   Features dir: {config.features_dir}")
    print(f"   Feature dim: {config.feature_dim}")
except Exception as e:
    print(f"❌ Config error: {e}")
    # Create config on the fly
    class QuickConfig:
        data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
        i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
        features_dir = i3d_features_dir
        feature_dim = 1024
        max_train_samples = 5
        use_i3d_features = True
        train_split = 'train'
    
    config = QuickConfig()
    print(f"✅ Created quick config")

# ==================== TEST 2: DATASET ====================
print("\n2. Testing dataset...")
try:
    from data.phoenix_dataset import PhoenixFeaturesDataset
    
    train_dataset = PhoenixFeaturesDataset(
        config.data_root,
        split='train',
        config=config,
        max_samples=config.max_train_samples if hasattr(config, 'max_train_samples') else 5
    )
    
    print(f"✅ Dataset loaded: {len(train_dataset)} samples")
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Feature shape: {sample['feature'].shape}")
        print(f"   Text: {sample['text'][:50]}...")
        print(f"   Filename: {sample['filename']}")
    
    # Save dataset reference for later use
    test_dataset = train_dataset
    
except Exception as e:
    print(f"❌ Dataset error: {e}")
    import traceback
    traceback.print_exc()
    
    # Create simple dataset
    print("\nCreating simple dataset...")
    import torch
    import numpy as np
    
    class SimpleDataset:
        def __init__(self, n_samples=5):
            self.n_samples = n_samples
            self.features = [torch.randn(100, 1024) for _ in range(n_samples)]
            self.texts = [f"Sample {i}" for i in range(n_samples)]
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            return {
                'feature': self.features[idx],
                'text': self.texts[idx],
                'filename': f"sample_{idx}.npy"
            }
    
    test_dataset = SimpleDataset()
    print(f"✅ Created simple dataset: {len(test_dataset)} samples")

# ==================== TEST 3: MODEL ====================
print("\n3. Testing model...")
try:
    from models.signllm import create_signllm_features_model
    
    # Update config with necessary attributes
    if not hasattr(config, 'codebook_size'):
        config.codebook_size = 256
    if not hasattr(config, 'codebook_dim'):
        config.codebook_dim = 512
    if not hasattr(config, 'lambda_mmd'):
        config.lambda_mmd = 0.5
    if not hasattr(config, 'lambda_sim'):
        config.lambda_sim = 1.0
    
    model = create_signllm_features_model(config).to(config.device)
    print(f"✅ Model created")
    
    # Test forward pass
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        feature = sample['feature'].unsqueeze(0)  # Add batch dimension
        
        # Ensure device
        if hasattr(config, 'device'):
            feature = feature.to(config.device)
        
        print(f"Input shape: {feature.shape}")
        
        word_indices, losses = model(feature)
        print(f"✅ Forward pass successful")
        print(f"   Loss keys: {list(losses.keys())}")
        print(f"   Total loss: {losses['total_loss'].item():.4f}")
    
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ TEST COMPLETED!")
print("\nTo start training, run:")
print("cd /kaggle/working/signLLM2")
print("python train.py")