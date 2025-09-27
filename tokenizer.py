# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


from modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from quantizers.vq import VectorQuantizer
from quantizers.kl import DiagonalGaussianDistribution
from quantizers.softvq import SoftVectorQuantizer

from timm import create_model

from bridge_text_encoder import BridgeTextEncoder, get_tokenizer

import numpy as np

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 64
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0
    
    # deocder cls token
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    
    # encoder token drop for mask modeling
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6

    # aux decoder model
    aux_dec_model: str = 'vit_tiny_patch14_dinov2_movq'
    aux_loss_mask: bool = False
    aux_dec_cls_token: bool = True
    aux_hog_dec: bool = True
    aux_dino_dec: bool = True
    aux_supcls_dec: bool = True
    aux_clip_dec: bool = True
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        if config.repa and config.enc_type == 'vit':
            self.repa_model = create_model(config.repa_model, pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            from lpips.lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1/0.07)))))
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)
    class AttnPool(nn.Module):
        def __init__(self, dim, heads=4, dropout=0.0):
            super().__init__()
            self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # learnable query
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        def forward(self, x):  # x: [B, T, D]
            q = self.q.expand(x.size(0), -1, -1)      # [B,1,D]
            y, _ = self.attn(q, x, x)                 # [B,1,D]
            return self.out(y.squeeze(1))             # [B,D]
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info

class AttnPool(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # learnable query
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):  # x: [B, T, D]
        q = self.q.expand(x.size(0), -1, -1)      # [B,1,D]
        y, _ = self.attn(q, x, x)                 # [B,1,D]
        return self.out(y.squeeze(1))             # [B,D]

class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.config = config
        
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)
        
        self.bridge_text_encoder = BridgeTextEncoder(
            num_latent_tokens=config.num_latent_tokens, #args.num_latent_tokens,  # 64
            embed_dim=config.codebook_embed_dim, #args.codebook_embed_dim,         # 32
            freeze_clip=True
        )
        self.tokenizer = get_tokenizer()
        print(f"✅ Bridge Text Encoder initialized: {config.num_latent_tokens} tokens, {config.codebook_embed_dim} dim")
        
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))
            
    def encode_text(self, texts):
        if isinstance(texts, (list, tuple)):
            # 문자열 리스트 → 토큰 변환
            try:
                tokens = self.tokenizer(
                    texts,
                    max_length=77,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # GPU로 이동
                device = next(self.parameters()).device
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                
            except Exception as e:
                raise RuntimeError(f"토크나이징 실패: {e}")
        else:
            # 이미 토큰화된 경우
            input_ids = texts
            attention_mask = None
        
        # Bridge Text Encoder로 인코딩
        text_tokens = self.bridge_text_encoder(input_ids, attention_mask)
        return text_tokens  # [batch, num_latent_tokens, codebook_embed_dim]
    
    def encode_image(self, x):
        """이미지를 SoftVQ latent space로 인코딩 (기존 함수명 명확화)"""
        # 기존 encode 함수 사용
        enc, _, _ = self.encode(x)
        return enc  # [batch, num_latent_tokens, codebook_embed_dim]
    
    def decode_image(self, latent_tokens):
        """Latent tokens을 이미지로 디코딩"""
        # 기존 decode 함수 사용
        dec, _, _ = self.decode(latent_tokens, None, None)  
        return dec
    
    def compute_text_image_similarity(self, text_tokens, visual_tokens):
        text_features = self.attnpool(text_tokens)    # [batch, embed_dim]
        visual_features = self.attnpool(visual_tokens)   # [batch, embed_dim]
        
        # L2 정규화
        text_norm = F.normalize(text_features, dim=-1)
        visual_norm = F.normalize(visual_features, dim=-1)
        
        # 코사인 유사도
        similarities = torch.sum(text_norm * visual_norm, dim=-1)  # [batch]
        
        return similarities
    
    def cross_modal_retrieval(self, query_tokens, database_tokens, mode='text2image'):
        """Cross-modal retrieval (텍스트→이미지 또는 이미지→텍스트)"""
        # Global pooling & 정규화
        query_features = F.normalize(self.attnpool(query_tokens))     # [N, embed_dim]
        database_features = F.normalize(self.attnpool(database_tokens)) # [M, embed_dim]
        
        # 유사도 행렬
        similarity_matrix = torch.matmul(query_features, database_features.T)  # [N, M]
        
        # Top-K 검색
        top_similarities, top_indices = similarity_matrix.topk(k=min(10, similarity_matrix.size(1)), dim=1)
        
        return {
            'similarities': top_similarities,
            'indices': top_indices,
            'similarity_matrix': similarity_matrix
        }
    
    def forward(self, x, texts=None, return_similarity=False):
        # 🎯 기존 SoftVQ 처리
        enc, diff, info = self.encode(x)  # diff는 SoftVQ 내부 loss들
        dec = self.decode(enc, x=x)
        # b, _, h, w = input.size()
        # quant, diff, info = self.encode(input)
        # self.quant = quant
        # dec = self.decode(quant, x=input, h=h, w=w)
        # return dec, diff, info
        # ✅ 올바른 결과 (codebook_loss 제거)
        result = {
            'visual_tokens': enc,           # [batch, num_latent_tokens, codebook_embed_dim]
            'reconstructed': dec,           # [batch, 3, H, W]
            'info': info,                   # SoftVQ 내부 정보
            'diff': diff                    # SoftVQ 내부 loss들 (entropy 등)
        }
        
        # 🆕 텍스트 처리
        try:
            text_tokens = self.encode_text(texts)
            result['text_tokens'] = text_tokens
            # print(f"shape:!!!!!{text_tokens.shape}")
            
        except Exception as e:
            print(f"⚠️ 텍스트 처리 중 오류: {e}")
        
        return result

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SoftVQ(**kwargs):
    return SoftVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))