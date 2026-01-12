"""
Test script for Phoenix I3D Features
"""
import os
import sys
sys.path.insert(0, '/kaggle/working/signLLM2')

print("Testing Phoenix-2014T I3D Features Implementation")
print("=" * 60)

# Test config
from configs.config import Config
config = Config(training_mode="test", use_i3d_features=True)
print(f"✅ Config loaded: {config.feature_dim}D features")

# Test dataset
from data.phoenix_dataset import PhoenixFeaturesDataset
dataset = PhoenixFeaturesDataset(
    config.data_root, 
    split='train', 
    config=config,
    max_samples=5
)
print(f"✅ Dataset loaded: {len(dataset)} samples")

# Test a sample
sample = dataset[0]
print(f"✅ Sample loaded:")
print(f"   Feature shape: {sample['feature'].shape}")
print(f"   Text: {sample['text'][:50]}...")
print(f"   File: {sample['feature_file']}")

# Test model
from models.signllm import create_signllm_features_model
model = create_signllm_features_model(config).to(config.device)
print(f"✅ Model created")

# Test forward pass
test_input = sample['feature'].unsqueeze(0).to(config.device)
word_indices, losses = model(test_input)
print(f"✅ Forward pass successful")
print(f"   Losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")

print("\n" + "=" * 60)
print("✅ All tests passed! Run: python train_features.py")