"""
Debug test script for Phoenix I3D Features
"""
import os
import sys
sys.path.insert(0, '/kaggle/working/signLLM2')

print("Debug Testing Phoenix-2014T I3D Features")
print("=" * 60)

# First check the actual files
import numpy as np

data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
i3d_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t/train")

print(f"Checking I3D directory: {i3d_dir}")
if os.path.exists(i3d_dir):
    files = os.listdir(i3d_dir)[:5]  # First 5 files
    print(f"Found {len(os.listdir(i3d_dir))} files")
    print(f"First 5 files: {files}")
    
    # Load one file to check shape
    if files:
        sample_file = os.path.join(i3d_dir, files[0])
        print(f"\nLoading sample file: {sample_file}")
        sample_data = np.load(sample_file, allow_pickle=True)
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data dtype: {sample_data.dtype}")
        print(f"Sample data type: {type(sample_data)}")
        
        # If it's an object array, check what's inside
        if sample_data.dtype == 'object':
            print(f"It's an object array, checking first element...")
            if len(sample_data) > 0:
                print(f"First element shape: {sample_data[0].shape if hasattr(sample_data[0], 'shape') else 'No shape'}")
                print(f"First element type: {type(sample_data[0])}")

# Test config
from configs.config import Config
config = Config(training_mode="test", use_i3d_features=True)
print(f"\n✅ Config loaded: {config.feature_dim}D features")

# Test dataset
from data.phoenix_features_dataset import PhoenixFeaturesDataset
dataset = PhoenixFeaturesDataset(
    config.data_root, 
    split='train', 
    config=config,
    max_samples=2
)
print(f"✅ Dataset loaded: {len(dataset)} samples")

# Test a sample
if len(dataset) > 0:
    sample = dataset[0]
    print(f"\n✅ Sample loaded:")
    print(f"   Feature shape: {sample['feature'].shape}")
    print(f"   Feature dtype: {sample['feature'].dtype}")
    print(f"   Text: {sample['text'][:50]}...")
    print(f"   File: {sample['feature_file']}")
    
    # Test model
    from models.signllm import create_signllm_features_model
    model = create_signllm_features_model(config).to(config.device)
    print(f"\n✅ Model created")
    
    # Test forward pass
    test_input = sample['feature'].unsqueeze(0)  # Add batch dimension
    print(f"Test input shape: {test_input.shape}")
    
    word_indices, losses = model(test_input)
    print(f"✅ Forward pass successful")
    print(f"   Losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
else:
    print("❌ No samples loaded from dataset")

print("\n" + "=" * 60)
print("Debug test completed")