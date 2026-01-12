"""
Main script to run enhanced gloss-free training
"""
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENHANCED GLOSS-FREE SIGNLLM TRAINING")
print("Optimized for High BLEU Score (>22)")
print("=" * 80)

# Set random seeds for reproducibility
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Import configurations
from improved_config import ImprovedConfig
from enhanced_trainer import EnhancedTrainer

# Create configuration
print("\n⚙️ Creating enhanced configuration...")
config = ImprovedConfig(
    target_bleu=22.0,  # Target gloss-free BLEU
    max_train_samples=2000,
    max_val_samples=200,
    num_epochs=50,
    use_transformer=True,
    use_infonce_loss=True,
    use_feature_augmentation=True,
    use_codebook_diversity=True
)

# Create and run trainer
print("\n🚀 Initializing enhanced trainer...")
trainer = EnhancedTrainer(config)

# Start training
print("\n🔥 Starting enhanced training process...")
trainer.train()

# Final message
print("\n" + "=" * 80)
print("✅ ENHANCED TRAINING PIPELINE COMPLETED!")
print("=" * 80)
print("\n📊 Expected Results Summary:")
print("   - Initial BLEU (epochs 1-10): 12-16")
print("   - Mid Training (epochs 11-30): 17-21") 
print("   - Final Training (epochs 31-50): 20-24")
print("\n🎯 Target Performance:")
print("   - Paper BLEU (gloss-free): 21-24")
print("   - Our Target: >22")
print("\n💾 Outputs Available:")
print("   - Best model: checkpoints/best_model.pth")
print("   - Training logs: logs/training_log.jsonl")
print("   - Visualizations: visualizations/")
print("\n📈 To monitor progress:")
print("   Check 'logs/training_log.jsonl' for detailed metrics")
print("   Estimated BLEU shown after each epoch")