"""
Inference script for SignLLM
"""
import torch
from torch.utils.data import DataLoader
import argparse
from nltk.translate.bleu_score import sentence_bleu

from config.config import Config
from data.phoenix_dataset import PhoenixDataset
from models.signllm import SignLLM

def inference(model, dataloader, config):
    """Run inference and compute metrics"""
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            videos = batch['video'].to(config.device)
            texts = batch['text']
            
            # Generate translations
            translations, _ = model(videos)
            
            all_predictions.extend(translations)
            all_references.extend(texts)
    
    # Compute BLEU scores
    bleu_scores = []
    for pred, ref in zip(all_predictions, all_references):
        score = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # Print samples
    print("\nSample translations:")
    for i in range(min(5, len(all_predictions))):
        print(f"Reference: {all_references[i]}")
        print(f"Predicted: {all_predictions[i]}")
        print(f"BLEU: {bleu_scores[i]:.4f}\n")
    
    return avg_bleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/phoenix2014t')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.data_root = args.data_dir
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model = SignLLM(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Dataset
    test_dataset = PhoenixDataset(config.data_root, args.split, config=config)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Run inference
    bleu_score = inference(model, test_loader, config)
    print(f"Average BLEU score on {args.split} set: {bleu_score:.4f}")

if __name__ == '__main__':
    main()