"""
Main SignLLM Framework
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vq_sign import VQSign
from .cra import CRA

class SignLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # VQ-Sign module
        self.vq_sign = VQSign(config)
        
        # CRA module (initialized after VQ-Sign training)
        self.cra = None
        
        # LLM (frozen)
        self.llm, self.tokenizer = self._load_llm()
        
        # Projection to LLM embedding space
        self.llm_projection = nn.Sequential(
            nn.Linear(config.codebook_dim, config.codebook_dim),
            nn.ReLU(),
            nn.Linear(config.codebook_dim, self.llm.config.hidden_size)
        )
        
    def _load_llm(self):
        """Load frozen LLM"""
        # Use smaller model for Kaggle compatibility
        model_name = "gpt2"  # Switch to llama if you have enough memory
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        llm = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze LLM parameters
        for param in llm.parameters():
            param.requires_grad = False
        
        return llm, tokenizer
    
    def initialize_cra(self):
        """Initialize CRA module after VQ-Sign training"""
        self.cra = CRA(self.config, self.vq_sign.codebook)
    
    def forward(self, videos, target_texts=None):
        """
        Forward pass through SignLLM
        videos: (B, T, C, H, W) sign videos
        target_texts: List of target sentences (for training)
        """
        # VQ-Sign: Video -> Character tokens
        char_tokens, char_embeddings, vq_losses = self.vq_sign(videos)
        
        # CRA: Character -> Word tokens
        if self.cra is None:
            self.initialize_cra()
        
        text_embeddings = None
        if target_texts is not None:
            # Get text embeddings for alignment
            text_tokens = self.tokenizer(
                target_texts, 
                padding=True, 
                return_tensors='pt'
            ).to(self.config.device)
            text_embeddings = self.llm.get_input_embeddings()(text_tokens['input_ids'])
        
        word_tokens, word_embeddings, cra_losses = self.cra(
            char_tokens, char_embeddings, text_embeddings
        )
        
        # Project to LLM space
        projected_embeddings = []
        for emb in word_embeddings:
            if len(emb) > 0:
                proj_emb = self.llm_projection(emb)
                projected_embeddings.append(proj_emb)
        
        # Generate translations
        translations = []
        translation_loss = 0
        
        if target_texts is not None:
            # Training: compute cross-entropy loss
            for i, emb in enumerate(projected_embeddings):
                if len(emb) > 0:
                    # Prepare prompt
                    prompt = "Translate sign language to text: "
                    prompt_emb = self.llm.get_input_embeddings()(
                        self.tokenizer(prompt, return_tensors='pt')['input_ids']
                    ).to(self.config.device)
                    
                    # Concatenate prompt and sign embeddings
                    input_emb = torch.cat([prompt_emb.repeat(emb.shape[0], 1, 1), emb.unsqueeze(1)], dim=1)
                    
                    # Forward through LLM
                    outputs = self.llm(inputs_embeds=input_emb)
                    
                    # Compute loss
                    target_ids = self.tokenizer(target_texts[i], return_tensors='pt')['input_ids']
                    loss = nn.CrossEntropyLoss()(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        target_ids.view(-1)
                    )
                    translation_loss += loss
            
            translation_loss = translation_loss / len(projected_embeddings)
        
        else:
            # Inference: generate text
            for emb in projected_embeddings:
                if len(emb) > 0:
                    prompt = "Translate sign language to text: "
                    prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.config.device)
                    
                    # Generate with LLM
                    generated = self.llm.generate(
                        inputs_embeds=emb.unsqueeze(0),
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    translation = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    translations.append(translation)
        
        # Combine losses
        losses = {**vq_losses, **cra_losses}
        if target_texts is not None:
            losses['translation_loss'] = translation_loss
            losses['total_loss'] = (
                vq_losses.get('total_vq_loss', 0) +
                cra_losses.get('mmd_loss', 0) * self.config.lambda_mmd +
                translation_loss * self.config.lambda_sim
            )
        
        return translations, losses