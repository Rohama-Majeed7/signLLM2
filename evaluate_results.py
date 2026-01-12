"""
Script to evaluate and display gloss-free results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def display_training_results():
    """Display training results and BLEU estimates"""
    
    # Load training log
    try:
        with open('logs/training_log.jsonl', 'r') as f:
            logs = [json.loads(line) for line in f]
    except:
        print("No training logs found. Run training first.")
        return
    
    print("=" * 80)
    print("GLOSS-FREE TRAINING RESULTS ANALYSIS")
    print("=" * 80)
    
    # Extract metrics
    epochs = [log['epoch'] for log in logs]
    bleu_estimates = [log['train']['bleu_estimate'] for log in logs]
    train_losses = [log['train']['loss'] for log in logs]
    val_losses = [log['validation']['loss'] for log in logs]
    codebook_usage = [log['train']['codebook_usage'] for log in logs]
    
    # Calculate statistics
    best_bleu = max(bleu_estimates)
    best_epoch = epochs[bleu_estimates.index(best_bleu)]
    final_bleu = bleu_estimates[-1]
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Best Estimated BLEU: {best_bleu:.2f} (epoch {best_epoch})")
    print(f"   Final Estimated BLEU: {final_bleu:.2f}")
    print(f"   Target BLEU: {logs[0]['target_bleu']}")
    
    print(f"\n📈 TRAINING PROGRESS:")
    print(f"   Initial BLEU (epoch 1): {bleu_estimates[0]:.2f}")
    print(f"   Mid Training BLEU (epoch {len(epochs)//2}): {bleu_estimates[len(epochs)//2]:.2f}")
    print(f"   Final Codebook Usage: {codebook_usage[-1]*100:.1f}%")
    
    print(f"\n🎯 GLOSS-FREE PERFORMANCE ASSESSMENT:")
    
    # Performance categories
    if best_bleu >= 22:
        print("   ✅ EXCELLENT: Exceeds paper performance (>22 BLEU)")
        print("   Expected actual BLEU: 21-24")
    elif best_bleu >= 20:
        print("   ✅ GOOD: Matches paper performance (20-22 BLEU)")
        print("   Expected actual BLEU: 19-22")
    elif best_bleu >= 18:
        print("   ⚠️  FAIR: Close to paper performance (18-20 BLEU)")
        print("   Expected actual BLEU: 17-20")
    else:
        print("   ❌ NEEDS IMPROVEMENT: Below paper performance (<18 BLEU)")
        print("   Suggestions: More epochs, larger model, more data")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # BLEU estimates
    axes[0, 0].plot(epochs, bleu_estimates, 'b-', linewidth=2)
    axes[0, 0].axhline(y=logs[0]['target_bleu'], color='r', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Estimated BLEU')
    axes[0, 0].set_title('Gloss-Free BLEU Estimate Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    axes[0, 1].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training and Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Codebook usage
    axes[1, 0].plot(epochs, [u*100 for u in codebook_usage], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Codebook Usage (%)')
    axes[1, 0].set_title('Codebook Utilization')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance summary
    performance_categories = ['<18', '18-20', '20-22', '>22']
    performance_counts = [0, 0, 0, 0]
    
    for bleu in bleu_estimates:
        if bleu < 18:
            performance_counts[0] += 1
        elif bleu < 20:
            performance_counts[1] += 1
        elif bleu < 22:
            performance_counts[2] += 1
        else:
            performance_counts[3] += 1
    
    colors = ['red', 'orange', 'yellow', 'green']
    axes[1, 1].bar(performance_categories, performance_counts, color=colors)
    axes[1, 1].set_xlabel('BLEU Range')
    axes[1, 1].set_ylabel('Number of Epochs')
    axes[1, 1].set_title('Performance Distribution')
    
    plt.tight_layout()
    plt.savefig('visualizations/gloss_free_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 VISUALIZATIONS SAVED:")
    print(f"   Performance plot: visualizations/gloss_free_performance.png")
    
    print(f"\n🎯 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
    if best_bleu < 20:
        print("   1. Increase training epochs to 100+")
        print("   2. Use more training data (5000+ samples)")
        print("   3. Increase model capacity (codebook_size=2048)")
        print("   4. Add text supervision if available")
    elif best_bleu < 22:
        print("   1. Fine-tune with lower learning rate (1e-5)")
        print("   2. Add more contrastive learning pairs")
        print("   3. Use transformer with more layers (8+)")
    else:
        print("   1. Maintain current configuration")
        print("   2. Consider ensemble methods")
        print("   3. Deploy for production use")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    display_training_results()