# modelling/bridge_text_encoder.py 완전 수정 버전

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from contextlib import nullcontext

class BridgeTextEncoder(nn.Module):
    """CLIP 512차원 → SoftVQ 32차원 Bridge"""
    
    def __init__(self, 
                 num_latent_tokens=64, 
                 embed_dim=64,  # ← 32차원!
                 freeze_clip=True):
        super().__init__()
        
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim
        self.freeze_clip = freeze_clip
        
        print("Loading CLIP text model...")
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for param in self.clip_text.parameters():
                param.requires_grad = False
        
        # 🎯 차원 축소: 512 → 32
        self.feature_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),  # → 32
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 🎯 Learnable Query Tokens  
        self.learnable_queries = nn.Parameter(
            torch.randn(num_latent_tokens, embed_dim) * 0.02
        )
        
        # 🎯 Cross-Attention (32차원에 맞춤)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,  # 32
            num_heads=4,          # 32/4 = 8 (적절함)
            dropout=0.1,
            batch_first=True
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, return_attention=False):
        batch_size = input_ids.size(0)
        
        # Step 1: CLIP encoding
        context = torch.no_grad() if self.freeze_clip else nullcontext()
        with context:
            clip_outputs = self.clip_text(input_ids, attention_mask=attention_mask)
        
        clip_features = clip_outputs.last_hidden_state  # [B, 77, 512]
        
        # Step 2: 차원 축소 512 → 32
        projected_features = self.feature_projector(clip_features)  # [B, 77, 32]
        
        # Step 3: Query-based aggregation
        queries = self.learnable_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 64, 32]
        
        # Attention mask 처리
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        aligned_tokens, attention_weights = self.cross_attention(
            query=queries,
            key=projected_features,
            value=projected_features,
            key_padding_mask=key_padding_mask
        )
        
        # Step 4: Final processing
        output = self.final_norm(self.dropout(aligned_tokens))  # [B, 64, 32]
        
        if return_attention:
            return output, attention_weights
        return output

def get_tokenizer():
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")