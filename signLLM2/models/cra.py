"""
Codebook Reconstruction and Alignment (CRA) Module
Transforms character-level tokens to word-level tokens
"""
import torch
import torch.nn as nn
import numpy as np
from ot import sinkhorn  # POT library for optimal transport
from scipy.special import softmax

class CRA(nn.Module):
    def __init__(self, config, char_codebook):
        super().__init__()
        self.config = config
        self.char_codebook = char_codebook
        
        # Word-level codebook (initially empty)
        self.word_codebook = nn.Parameter(torch.randn(0, config.codebook_dim))
        
        # Projection for sign-text alignment
        self.projection = nn.Sequential(
            nn.Linear(config.codebook_dim, config.codebook_dim),
            nn.ReLU(),
            nn.Linear(config.codebook_dim, config.codebook_dim)
        )
        
    def preprocess_repeated_chars(self, char_sequences):
        """
        Preprocess repeated character tokens
        char_sequences: List of token sequences
        Returns: Preprocessed sequences
        """
        processed_seqs = []
        
        for seq in char_sequences:
            # Find average repetition length
            unique_tokens = []
            counts = []
            
            for token in seq:
                if not unique_tokens or token != unique_tokens[-1]:
                    unique_tokens.append(token)
                    counts.append(1)
                else:
                    counts[-1] += 1
            
            avg_repeat = np.mean(counts)
            
            # Process sequence
            new_seq = []
            for token, count in zip(unique_tokens, counts):
                # Keep first occurrence
                new_seq.append(token)
                
                # Add slowing token if significant repetition
                if count > avg_repeat * 1.5:
                    new_seq.append(0)  # Special slowing token
            
            processed_seqs.append(new_seq)
        
        return processed_seqs
    
    def compute_entropy(self, codebook_usage):
        """
        Compute entropy of codebook usage
        codebook_usage: Probability distribution over tokens
        """
        # Add small epsilon to avoid log(0)
        probs = codebook_usage + 1e-10
        probs = probs / probs.sum()
        
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy
    
    def optimal_transport_reconstruction(self, char_sequences):
        """
        Reconstruct word-level codebook using optimal transport
        Returns: word-level codebook
        """
        # Convert to numpy for OT computation
        char_seqs_np = [seq.cpu().numpy() for seq in char_sequences]
        
        # Build character distribution
        all_chars = np.concatenate(char_seqs_np)
        unique_chars, char_counts = np.unique(all_chars, return_counts=True)
        char_probs = char_counts / char_counts.sum()
        
        # Try different codebook sizes
        best_entropy = float('inf')
        best_codebook = None
        
        for r in range(1, 10):  # Try up to 10*m tokens
            word_size = r * self.config.word_increment
            
            # Initialize uniform word distribution
            word_probs = np.ones(word_size) / word_size
            
            # Cost matrix: -log P(char|word)
            cost_matrix = np.ones((len(unique_chars), word_size)) * 100
            
            # Simple heuristic: group consecutive characters
            for i, word_idx in enumerate(range(0, word_size)):
                # Simple grouping (in practice, use learned assignments)
                assigned_chars = unique_chars[i::word_size]
                for char_idx in assigned_chars:
                    if char_idx < len(unique_chars):
                        cost_matrix[char_idx, word_idx] = -np.log(1.0 / len(assigned_chars))
            
            # Compute optimal transport
            P = sinkhorn(char_probs, word_probs, cost_matrix, reg=0.1)
            
            # Compute entropy
            joint_probs = P / P.sum()
            entropy = -np.sum(joint_probs * np.log(joint_probs + 1e-10))
            
            if entropy < best_entropy:
                best_entropy = entropy
                best_codebook = self._construct_word_codebook(P, word_size)
        
        return best_codebook
    
    def _construct_word_codebook(self, transport_matrix, word_size):
        """
        Construct word-level codebook from transport matrix
        """
        # Get character assignments to words
        char_assignments = np.argmax(transport_matrix, axis=1)
        
        # Initialize word codebook
        word_embeddings = []
        
        for word_idx in range(word_size):
            # Get characters assigned to this word
            char_indices = np.where(char_assignments == word_idx)[0]
            
            if len(char_indices) > 0:
                # Average character embeddings
                char_embs = self.char_codebook.weight[char_indices]
                word_emb = char_embs.mean(dim=0)
            else:
                # Random initialization if no assignments
                word_emb = torch.randn(self.config.codebook_dim)
            
            word_embeddings.append(word_emb)
        
        return torch.stack(word_embeddings)
    
    def mmd_loss(self, sign_embeddings, text_embeddings):
        """
        Maximum Mean Discrepancy loss for sign-text alignment
        """
        # Radial basis function kernel
        def rbf_kernel(x, y, sigma=1.0):
            x_norm = torch.sum(x**2, dim=1, keepdim=True)
            y_norm = torch.sum(y**2, dim=1, keepdim=True)
            dist = x_norm + y_norm.T - 2 * torch.matmul(x, y.T)
            return torch.exp(-dist / (2 * sigma**2))
        
        # Compute MMD (Eq. 6)
        xx = rbf_kernel(sign_embeddings, sign_embeddings)
        yy = rbf_kernel(text_embeddings, text_embeddings)
        xy = rbf_kernel(sign_embeddings, text_embeddings)
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd
    
    def forward(self, char_tokens, char_embeddings, text_embeddings=None):
        """
        Convert character-level tokens to word-level representation
        """
        # Preprocess repeated characters
        processed_seqs = self.preprocess_repeated_chars(char_tokens)
        
        # Reconstruct word-level codebook (training only)
        if self.training and len(self.word_codebook) == 0:
            self.word_codebook.data = self.optimal_transport_reconstruction(
                processed_seqs
            )
        
        # Convert to word-level tokens (simple grouping)
        word_tokens = []
        word_embeddings = []
        
        for seq in processed_seqs:
            # Group into words of max_word_length
            words = []
            word_embs = []
            
            for i in range(0, len(seq), self.config.max_word_length):
                word_chars = seq[i:i+self.config.max_word_length]
                
                if word_chars:
                    # Find nearest word in codebook
                    char_emb = self.char_codebook.weight[word_chars].mean(dim=0)
                    distances = torch.cdist(
                        char_emb.unsqueeze(0), 
                        self.word_codebook
                    )
                    word_idx = torch.argmin(distances).item()
                    
                    words.append(word_idx)
                    word_embs.append(self.word_codebook[word_idx])
            
            word_tokens.append(words)
            word_embeddings.append(torch.stack(word_embs) if word_embs else 
                                  torch.zeros(0, self.config.codebook_dim))
        
        losses = {}
        if self.training and text_embeddings is not None:
            # Sign-text alignment loss
            all_word_embs = torch.cat(word_embeddings)
            projected_sign = self.projection(all_word_embs)
            
            mmd_loss = self.mmd_loss(projected_sign, text_embeddings)
            losses['mmd_loss'] = mmd_loss
        
        return word_tokens, word_embeddings, losses