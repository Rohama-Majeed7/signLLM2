# signllm_implementation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import cv2
import os
from tqdm import tqdm
import pickle

# =================== Configuration ===================
class Config:
    # Video settings
    img_size = 224
    num_frames = 64
    clip_length = 13
    clip_stride = 4
    
    # VQ-Sign settings
    codebook_size = 256  # M
    codebook_dim = 1024  # d
    num_predict_steps = 3  # K
    
    # CRA settings
    increment_size = 32  # m
    min_word_tokens = 64
    max_word_tokens = 512
    
    # Training settings
    batch_size = 4
    learning_rate = 0.01
    num_epochs_pretrain = 10
    num_epochs_finetune = 5
    gamma = 0.25  # for VQ loss
    lambda1 = 0.5  # MMD weight
    lambda2 = 1.0  # similarity weight
    
    # LLM settings
    use_llama = True  # Set to False for lightweight testing
    llm_embed_dim = 4096  # LLaMA embedding dimension

# =================== Visual Encoder ===================
class VisualEncoder(nn.Module):
    """ResNet18-based encoder with Conv3D layers"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet18
        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)
        
        # Remove final fully connected layer
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add Conv3D layers for temporal processing
        self.conv3d = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, config.codebook_dim, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(config.codebook_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling to get clip features
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, num_clips, codebook_dim) features
        """
        B, T, C, H, W = x.shape
        
        # Process each clip
        num_clips = (T - self.config.clip_length) // self.config.clip_stride + 1
        clip_features = []
        
        for i in range(num_clips):
            start = i * self.config.clip_stride
            clip = x[:, start:start+self.config.clip_length]  # (B, clip_len, C, H, W)
            
            # ResNet features for each frame
            frame_features = []
            for t in range(self.config.clip_length):
                frame = clip[:, t]  # (B, C, H, W)
                frame_feat = self.resnet_layers(frame)  # (B, 512, H/32, W/32)
                frame_features.append(frame_feat.unsqueeze(2))  # Add temporal dim
            
            # Stack and process with Conv3D
            temporal_features = torch.cat(frame_features, dim=2)  # (B, 512, clip_len, H', W')
            conv_features = self.conv3d(temporal_features)  # (B, d, T', H'', W'')
            
            # Global pooling
            pooled = self.adaptive_pool(conv_features)  # (B, d, 1, 1, 1)
            clip_features.append(pooled.squeeze())
        
        # Stack clip features
        z = torch.stack(clip_features, dim=1)  # (B, num_clips, d)
        return z

# =================== VQ-Sign Module ===================
class VQSign(nn.Module):
    """Vector-Quantized Visual Sign Module"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Character-level codebook
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        self.codebook.weight.data.uniform_(-1.0/config.codebook_size, 1.0/config.codebook_size)
        
        # Context prediction model (autoregressive)
        self.context_model = nn.GRU(
            input_size=config.codebook_dim,
            hidden_size=config.codebook_dim,
            batch_first=True,
            num_layers=2
        )
        
        # Projection for contrastive loss
        self.projection = nn.Linear(config.codebook_dim, config.codebook_dim)
        
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize continuous features to discrete tokens
        Returns:
            quantized: quantized features
            indices: token indices
        """
        # Flatten for distance computation
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*T, D)
        
        # Compute distances to codebook vectors
        codebook_vectors = self.codebook.weight  # (M, D)
        distances = torch.cdist(z_flat, codebook_vectors, p=2)  # (B*T, M)
        
        # Get nearest neighbors
        indices = torch.argmin(distances, dim=-1)  # (B*T,)
        quantized = self.codebook(indices).reshape(B, T, D)
        
        return quantized, indices.reshape(B, T)
    
    def context_prediction_loss(self, z: torch.Tensor, quantized: torch.Tensor, k: int = 3):
        """Context prediction contrastive loss (Eq. 1)"""
        B, T, D = z.shape
        
        # Get context representations
        context_rep, _ = self.context_model(quantized)  # (B, T, D)
        
        # Project context
        h = self.projection(context_rep)  # (B, T, D)
        
        # Positive and negative samples
        total_loss = 0
        for step in range(1, k + 1):
            pos_scores = torch.sum(z[:, step:] * h[:, :-step], dim=-1)  # (B, T-step)
            pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores) + 1e-8))
            
            # Negative samples (from same batch)
            neg_indices = torch.randint(0, B, (B, T-step))
            neg_samples = z[neg_indices, torch.arange(T-step).unsqueeze(0)]
            neg_scores = torch.sum(neg_samples * h[:, :-step], dim=-1)
            neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(neg_scores) + 1e-8))
            
            total_loss += pos_loss + self.config.gamma * neg_loss
        
        return total_loss / k
    
    def vq_loss(self, z: torch.Tensor, quantized: torch.Tensor):
        """VQ commitment loss (Eq. 2)"""
        # Stop-gradient operations
        commitment_loss = F.mse_loss(z.detach(), quantized)
        codebook_loss = F.mse_loss(z, quantized.detach())
        
        return commitment_loss + self.config.gamma * codebook_loss
    
    def forward(self, z: torch.Tensor):
        quantized, indices = self.quantize(z)
        return quantized, indices

# =================== CRA Module ===================
class CRA(nn.Module):
    """Codebook Reconstruction and Alignment Module"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Word-level codebook (initialized later)
        self.word_codebook = None
        self.word_to_chars = {}  # Mapping from word tokens to character sequences
        
        # Projection to LLM space
        self.projection = nn.Sequential(
            nn.Linear(config.codebook_dim, config.codebook_dim * 2),
            nn.ReLU(),
            nn.Linear(config.codebook_dim * 2, config.llm_embed_dim)
        )
        
    def preprocess_repeated_chars(self, char_sequences: List[List[int]], alpha: float = 2.0):
        """
        Preprocess repeated character sequences (Sec 3.3)
        Returns preprocessed sequences with 'slowing down' tokens
        """
        processed_seqs = []
        
        for seq in char_sequences:
            processed = []
            i = 0
            while i < len(seq):
                count = 1
                # Count repetitions
                while i + count < len(seq) and seq[i] == seq[i + count]:
                    count += 1
                
                # Keep first character
                processed.append(seq[i])
                
                # Add slowing token if too many repetitions
                if count > alpha:
                    processed.append(0)  # s0 token for slowing
                
                i += count
            
            processed_seqs.append(processed)
        
        return processed_seqs
    
    def compute_token_probabilities(self, sequences: List[List[int]]):
        """Compute token frequencies for entropy calculation"""
        from collections import Counter
        all_tokens = [token for seq in sequences for token in seq]
        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())
        
        probs = {token: count/total for token, count in token_counts.items()}
        return probs
    
    def compute_entropy(self, sequences: List[List[int]]):
        """Compute entropy of sequences (Eq. 3)"""
        probs = self.compute_token_probabilities(sequences)
        entropy = -sum(p * np.log(p + 1e-8) for p in probs.values())
        return entropy
    
    def construct_word_codebook(self, char_sequences: List[List[int]]):
        """
        Construct word-level codebook using optimal transport formulation
        Simplified implementation
        """
        # Preprocess sequences
        processed_seqs = self.preprocess_repeated_chars(char_sequences)
        
        # Find common character n-grams
        from collections import Counter
        ngram_counter = Counter()
        
        # Count n-grams of different lengths
        for seq in processed_seqs:
            for n in range(2, 6):  # Word lengths 2-5 characters
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i+n])
                    ngram_counter[ngram] += 1
        
        # Select top n-grams as word tokens
        top_ngrams = ngram_counter.most_common(self.config.max_word_tokens)
        
        # Create word codebook
        word_tokens = []
        word_to_chars = {}
        
        for idx, (ngram, count) in enumerate(top_ngrams):
            word_tokens.append(f"WORD_{idx}")
            word_to_chars[f"WORD_{idx}"] = list(ngram)
        
        # Compute entropy for different codebook sizes
        entropies = []
        sizes = range(self.config.min_word_tokens, 
                     self.config.max_word_tokens + 1, 
                     self.config.increment_size)
        
        # Simplified entropy computation
        for size in sizes:
            # Use first 'size' n-grams
            current_words = top_ngrams[:size]
            # Convert sequences to word tokens
            word_seqs = []
            for seq in processed_seqs:
                word_seq = []
                i = 0
                while i < len(seq):
                    # Try to match longest word
                    matched = False
                    for word_len in range(5, 1, -1):
                        if i + word_len <= len(seq):
                            ngram = tuple(seq[i:i+word_len])
                            # Check if this ngram is a word
                            for word_idx, (w_ngram, _) in enumerate(current_words):
                                if w_ngram == ngram:
                                    word_seq.append(f"WORD_{word_idx}")
                                    i += word_len
                                    matched = True
                                    break
                            if matched:
                                break
                    if not matched:
                        # Keep as character
                        word_seq.append(f"CHAR_{seq[i]}")
                        i += 1
                word_seqs.append(word_seq)
            
            # Compute entropy
            entropy = self.compute_entropy([[hash(w) % 1000 for w in seq] for seq in word_seqs])
            entropies.append((size, entropy))
        
        # Find optimal size (max gradient of entropy reduction)
        if len(entropies) > 1:
            gradients = []
            for i in range(1, len(entropies)):
                grad = (entropies[i-1][1] - entropies[i][1]) / (entropies[i][0] - entropies[i-1][0])
                gradients.append((entropies[i][0], grad))
            
            optimal_size = max(gradients, key=lambda x: x[1])[0]
        else:
            optimal_size = self.config.min_word_tokens
        
        # Select optimal word tokens
        optimal_words = top_ngrams[:optimal_size]
        self.word_codebook = {f"WORD_{i}": list(ngram) for i, (ngram, _) in enumerate(optimal_words)}
        self.word_to_chars = self.word_codebook
        
        return optimal_size
    
    def chars_to_words(self, char_sequences: List[List[int]]) -> List[List[str]]:
        """Convert character sequences to word sequences"""
        if self.word_codebook is None:
            raise ValueError("Word codebook not constructed. Call construct_word_codebook first.")
        
        word_sequences = []
        
        # Reverse mapping for fast lookup
        ngram_to_word = {tuple(chars): word for word, chars in self.word_codebook.items()}
        
        for seq in char_sequences:
            word_seq = []
            i = 0
            
            while i < len(seq):
                matched = False
                # Try to match longest word (5 to 2 chars)
                for word_len in range(5, 1, -1):
                    if i + word_len <= len(seq):
                        ngram = tuple(seq[i:i+word_len])
                        if ngram in ngram_to_word:
                            word_seq.append(ngram_to_word[ngram])
                            i += word_len
                            matched = True
                            break
                
                if not matched:
                    # Keep as character token
                    word_seq.append(f"CHAR_{seq[i]}")
                    i += 1
            
            word_sequences.append(word_seq)
        
        return word_sequences
    
    def mmd_loss(self, sign_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
        """
        Maximum Mean Discrepancy loss (Eq. 6)
        Simplified implementation with RBF kernel
        """
        def rbf_kernel(x, y, sigma=1.0):
            x_norm = torch.sum(x**2, dim=1, keepdim=True)
            y_norm = torch.sum(y**2, dim=1, keepdim=True)
            dist = x_norm + y_norm.T - 2 * torch.matmul(x, y.T)
            return torch.exp(-dist / (2 * sigma**2))
        
        # Compute kernel matrices
        K_xx = rbf_kernel(sign_embeddings, sign_embeddings)
        K_yy = rbf_kernel(text_embeddings, text_embeddings)
        K_xy = rbf_kernel(sign_embeddings, text_embeddings)
        
        # MMD^2
        n_s = sign_embeddings.shape[0]
        n_t = text_embeddings.shape[0]
        
        mmd = (K_xx.sum() / (n_s * n_s) + 
               K_yy.sum() / (n_t * n_t) - 
               2 * K_xy.sum() / (n_s * n_t))
        
        return mmd

# =================== SignLLM Model ===================
class SignLLM(nn.Module):
    """Complete SignLLM Framework"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Components
        self.visual_encoder = VisualEncoder(config)
        self.vq_sign = VQSign(config)
        self.cra = CRA(config)
        
        # Freeze LLM (simulated for now)
        if config.use_llama:
            self.llm_projection = nn.Linear(config.llm_embed_dim, 32000)  # LLaMA vocab size
        else:
            # Lightweight text generator (mBART-like)
            self.text_generator = nn.Transformer(
                d_model=config.llm_embed_dim,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=2048
            )
            self.text_head = nn.Linear(config.llm_embed_dim, 32000)  # Vocabulary
        
    def forward(self, video: torch.Tensor, text_prompt: Optional[torch.Tensor] = None):
        # Extract visual features
        z = self.visual_encoder(video)  # (B, T, d)
        
        # VQ-Sign quantization
        quantized, char_indices = self.vq_sign(z)  # (B, T, d), (B, T)
        
        # Convert to word tokens (during inference)
        word_tokens = []
        if not self.training:
            char_seqs = char_indices.cpu().numpy().tolist()
            word_tokens = self.cra.chars_to_words(char_seqs)
        
        # Project to LLM space
        if self.config.use_llama:
            # For LLaMA: project quantized features
            projected = self.cra.projection(quantized.mean(dim=1))  # (B, llm_embed_dim)
            logits = self.llm_projection(projected)
        else:
            # For lightweight text generator
            memory = self.text_generator.encoder(quantized)
            if text_prompt is not None:
                tgt = self.text_generator.decoder(text_prompt, memory)
                logits = self.text_head(tgt)
            else:
                logits = self.text_head(memory.mean(dim=1))
        
        return logits, char_indices, word_tokens
    
    def compute_losses(self, video: torch.Tensor, target_text: torch.Tensor):
        """Compute all losses for training"""
        # Forward pass
        z = self.visual_encoder(video)
        quantized, char_indices = self.vq_sign(z)
        
        # VQ-Sign losses
        cp_loss = self.vq_sign.context_prediction_loss(z, quantized, k=self.config.num_predict_steps)
        vq_loss_val = self.vq_sign.vq_loss(z, quantized)
        vq_total_loss = cp_loss + vq_loss_val
        
        # MMD loss (sign-text alignment)
        if self.config.use_llama:
            # Simulate text embeddings (in practice, use LLM embeddings)
            text_embeddings = torch.randn_like(quantized.mean(dim=1))
            sign_embeddings = self.cra.projection(quantized.mean(dim=1))
            mmd_loss = self.cra.mmd_loss(sign_embeddings, text_embeddings)
        else:
            mmd_loss = torch.tensor(0.0, device=video.device)
        
        # Text generation loss
        if target_text is not None:
            logits, _, _ = self.forward(video)
            sim_loss = F.cross_entropy(logits, target_text)
        else:
            sim_loss = torch.tensor(0.0, device=video.device)
        
        # Total loss
        total_loss = (vq_total_loss + 
                     self.config.lambda1 * mmd_loss + 
                     self.config.lambda2 * sim_loss)
        
        return {
            'total': total_loss,
            'vq': vq_total_loss,
            'mmd': mmd_loss,
            'sim': sim_loss
        }

# =================== Dataset ===================
class PhoenixDataset(Dataset):
    """Phoenix-2014T Dataset Loader for Kaggle"""
    def __init__(self, root_dir, split='train', config: Config = None):
        self.root_dir = root_dir
        self.split = split
        self.config = config or Config()
        
        # Load dataset info
        self.video_dir = os.path.join(root_dir, 'features', 'fullFrame-210x260px')
        self.transcript_file = os.path.join(root_dir, 'annotations', f'{split}.corpus.csv')
        
        # Load samples
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        with open(self.transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    video_id = parts[0].strip()
                    translation = parts[1].strip()
                    samples.append((video_id, translation))
        return samples
    
    def load_video_frames(self, video_id):
        """Load and preprocess video frames"""
        video_path = os.path.join(self.video_dir, f'{video_id}.mp4')
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and normalize
            frame = cv2.resize(frame, (self.config.img_size, self.config.img_size))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet normalization
            frames.append(frame)
        
        cap.release()
        
        # Pad or truncate to fixed length
        if len(frames) > self.config.num_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(frames)-1, self.config.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame
            padding = [frames[-1]] * (self.config.num_frames - len(frames))
            frames.extend(padding)
        
        return np.array(frames)  # (T, H, W, C)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, translation = self.samples[idx]
        
        # Load video
        video_frames = self.load_video_frames(video_id)  # (T, H, W, C)
        
        # Convert to tensor and rearrange dimensions
        video_tensor = torch.FloatTensor(video_frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Simple tokenization of translation (in practice, use proper tokenizer)
        text_tensor = torch.LongTensor([hash(word) % 1000 for word in translation.split()[:20]])
        
        return {
            'video': video_tensor,
            'text': text_tensor,
            'translation': translation,
            'video_id': video_id
        }

# =================== Training Pipeline ===================
class SignLLMTrainer:
    def __init__(self, config: Config, model: SignLLM, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs_pretrain
        )
        
    def pre_train_vq_sign(self, train_loader, val_loader=None):
        """Pre-train VQ-Sign module"""
        print("Pre-training VQ-Sign module...")
        
        for epoch in range(self.config.num_epochs_pretrain):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                video = batch['video'].to(self.device)
                
                # Forward and compute VQ losses only
                z = self.model.visual_encoder(video)
                quantized, _ = self.model.vq_sign(z)
                
                cp_loss = self.model.vq_sign.context_prediction_loss(
                    z, quantized, k=self.config.num_predict_steps
                )
                vq_loss = self.model.vq_sign.vq_loss(z, quantized)
                loss = cp_loss + vq_loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}: VQ-Sign Loss = {avg_loss:.4f}')
            
            # Validation
            if val_loader:
                self.validate(val_loader, stage='pretrain')
            
            self.scheduler.step()
    
    def construct_word_codebook(self, train_loader):
        """Construct word-level codebook using CRA"""
        print("Constructing word-level codebook...")
        
        char_sequences = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='Collecting character sequences'):
                video = batch['video'].to(self.device)
                z = self.model.visual_encoder(video)
                _, char_indices = self.model.vq_sign(z)
                
                # Convert to list
                char_seqs = char_indices.cpu().numpy().tolist()
                char_sequences.extend(char_seqs)
        
        # Construct codebook
        optimal_size = self.model.cra.construct_word_codebook(char_sequences)
        print(f"Optimal word codebook size: {optimal_size}")
        
        return optimal_size
    
    def fine_tune(self, train_loader, val_loader=None):
        """Fine-tune the complete model"""
        print("Fine-tuning SignLLM...")
        
        for epoch in range(self.config.num_epochs_finetune):
            self.model.train()
            total_losses = {'total': 0, 'vq': 0, 'mmd': 0, 'sim': 0}
            
            for batch in tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}'):
                video = batch['video'].to(self.device)
                text = batch['text'].to(self.device)
                
                # Compute all losses
                losses = self.model.compute_losses(video, text)
                
                # Backward
                self.optimizer.zero_grad()
                losses['total'].backward()
                self.optimizer.step()
                
                # Accumulate losses
                for key in losses:
                    if isinstance(losses[key], torch.Tensor):
                        total_losses[key] += losses[key].item()
            
            # Print average losses
            avg_losses = {k: v/len(train_loader) for k, v in total_losses.items()}
            print(f'Epoch {epoch+1}: ' + 
                  ' | '.join([f'{k}: {v:.4f}' for k, v in avg_losses.items()]))
            
            # Validation
            if val_loader:
                self.validate(val_loader, stage='finetune')
    
    def validate(self, val_loader, stage='pretrain'):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(self.device)
                
                if stage == 'pretrain':
                    # Only VQ loss for pretraining
                    z = self.model.visual_encoder(video)
                    quantized, _ = self.model.vq_sign(z)
                    cp_loss = self.model.vq_sign.context_prediction_loss(
                        z, quantized, k=self.config.num_predict_steps
                    )
                    vq_loss = self.model.vq_sign.vq_loss(z, quantized)
                    loss = cp_loss + vq_loss
                else:
                    # All losses for fine-tuning
                    text = batch['text'].to(self.device)
                    losses = self.model.compute_losses(video, text)
                    loss = losses['total']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f'Validation Loss ({stage}): {avg_loss:.4f}')
        
        return avg_loss
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")

# =================== Inference ===================
class SignLLMInference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.config = Config()
        self.model = SignLLM(self.config).to(device)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Tokenizer for text generation (simplified)
        self.tokenizer = self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """Create a simple word tokenizer (replace with proper tokenizer in practice)"""
        class SimpleTokenizer:
            def encode(self, text):
                return [hash(word) % 1000 for word in text.split()]
            
            def decode(self, tokens):
                return ' '.join([f'word_{t}' for t in tokens])
        
        return SimpleTokenizer()
    
    def translate_video(self, video_path):
        """Translate a sign language video"""
        # Load and preprocess video
        video_frames = self._load_video_frames(video_path)
        video_tensor = torch.FloatTensor(video_frames).unsqueeze(0).to(self.device)
        video_tensor = video_tensor.permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
        
        # Forward pass
        with torch.no_grad():
            logits, char_indices, word_tokens = self.model(video_tensor)
            
            # Generate text (simplified)
            if self.config.use_llama:
                # Use logits to generate text
                probs = F.softmax(logits, dim=-1)
                predicted_tokens = torch.argmax(probs, dim=-1)
                translation = self.tokenizer.decode(predicted_tokens.cpu().numpy())
            else:
                # Use word tokens directly
                translation = ' '.join(word_tokens[0])
        
        return translation, word_tokens
    
    def _load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.config.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (self.config.img_size, self.config.img_size))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        if len(frames) < self.config.num_frames:
            padding = [frames[-1]] * (self.config.num_frames - len(frames))
            frames.extend(padding)
        else:
            frames = frames[:self.config.num_frames]
        
        return np.array(frames)

# =================== Main Script ===================
def main():
    # Configuration
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (adjust paths for Kaggle)
    # Assuming Phoenix dataset is in /kaggle/input/phoenix2014t/
    data_root = '/kaggle/input/phoenix2014t'
    
    train_dataset = PhoenixDataset(
        root_dir=data_root, 
        split='train', 
        config=config
    )
    val_dataset = PhoenixDataset(
        root_dir=data_root, 
        split='dev', 
        config=config
    )
    
    # Create data loaders
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = SignLLM(config)
    
    # Create trainer
    trainer = SignLLMTrainer(config, model, device)
    
    # Training pipeline
    print("\n" + "="*50)
    print("Phase 1: Pre-train VQ-Sign")
    print("="*50)
    trainer.pre_train_vq_sign(train_loader, val_loader)
    
    print("\n" + "="*50)
    print("Phase 2: Construct Word Codebook")
    print("="*50)
    trainer.construct_word_codebook(train_loader)
    
    print("\n" + "="*50)
    print("Phase 3: Fine-tune SignLLM")
    print("="*50)
    trainer.fine_tune(train_loader, val_loader)
    
    # Save model
    trainer.save_checkpoint('/kaggle/working/signllm_checkpoint.pth')
    
    # Test inference
    print("\n" + "="*50)
    print("Testing Inference")
    print("="*50)
    
    # Load a sample video for testing
    test_sample = train_dataset[0]
    video_tensor = test_sample['video'].unsqueeze(0).to(device)
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3)  # Adjust dimensions
    
    # Translate
    model.eval()
    with torch.no_grad():
        logits, char_indices, word_tokens = model(video_tensor)
    
    print(f"Original translation: {test_sample['translation']}")
    print(f"Generated word tokens: {word_tokens[0][:10]}...")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()