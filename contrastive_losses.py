import torch
import torch.nn.functional as F
import torch.nn as nn

from tokenizer import AttnPool
from lpips import LPIPS

def _pool_tokens(tokens, mode="mean"):
    # tokens: [B, T, D] 또는 이미 [B, D]
    if tokens.dim() == 3:
        if mode == "mean":
            return tokens.mean(dim=1)  # [B, D]
        elif mode == "cls":            # 필요 시 확장
            return tokens[:, 0]
        else:
            raise ValueError(f"Unknown pool mode: {mode}")
    elif tokens.dim() == 2:
        return tokens
    else:
        raise ValueError(f"tokens shape not supported: {tokens.shape}")

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pool="mean", eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.pool = pool
        self.eps = eps
    
    def forward(self, text_tokens, visual_tokens, logit_scale_param=None, attnpool=None):
        """
        CLIP 스타일 InfoNCE
        - text_tokens, visual_tokens: [B, T, D] or [B, D]
        - logit_scale_param: nn.Parameter(log(1/T)) 형태 (곱하기 exp(logit_scale))
          제공되면 temperature는 무시됨.
        """
        # 1) 풀링 -> [B, D]
        if attnpool is not None:
            text_features = attnpool(text_tokens)
            visual_features = attnpool(visual_tokens)
        else:
            text_features = _pool_tokens(text_tokens, self.pool)
            visual_features = _pool_tokens(visual_tokens, self.pool)

        # 2) L2 정규화 (안정 위해 eps)
        text_features = F.normalize(text_features, dim=-1, eps=self.eps)
        visual_features = F.normalize(visual_features, dim=-1, eps=self.eps)

        # 3) 유사도 & 로그릿 (fp32)
        sim = torch.matmul(text_features, visual_features.t()).float()  # [B, B]
        if logit_scale_param is not None:
            # CLIP 방식: logits = sim * exp(logit_scale)
            logits = sim * logit_scale_param.exp().clamp(max=100).float()
        else:
            # 고정 온도: logits = sim / T
            logits = sim / float(self.temperature)

        # 4) 라벨
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device, dtype=torch.long)

        # 5) 양방향 CE
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_t2i + loss_i2t)

        # 6) 메트릭 (R@1 in-batch)
        with torch.no_grad():
            t2i_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            i2t_acc = (logits.t().argmax(dim=1) == labels).float().mean().item()

        return loss, {
            "contrastive_loss": loss.item(),
            "t2i_acc": t2i_acc,
            "i2t_acc": i2t_acc,
            "avg_acc": 0.5 * (t2i_acc + i2t_acc),
        }

class CrossModalReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, text_tokens, visual_tokens, decoder):
        """(선택) 텍스트/비주얼 토큰 상호 복원 일치"""
        try:
            img_from_text = decoder(text_tokens)
            img_from_visual = decoder(visual_tokens.detach())
            cross_recon = F.mse_loss(img_from_text, img_from_visual)
            return self.alpha * cross_recon
        except Exception:
            return torch.tensor(0.0, device=text_tokens.device)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.perceptual_loss = LPIPS().to(device).eval()
    def forward(self, real_image, reconstruction):
        return self.perceptual_loss(real_image, reconstruction)

class CombinedLoss(nn.Module):
    def __init__(self, device, perceptual_loss=None, contrastive_weight=0.1, perceptual_weight=1.0, temperature=0.07, pool="mean", attnpool = None):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.pool = pool
        self.attnpool = attnpool.to(device)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature, pool=pool)
        self.perceptual_loss = perceptual_loss
        self.perceptual_weight = perceptual_weight
        
    def forward(self, model_output, target_images, logit_scale_param=None):
        """
        SoftVQ-VAE용 총손실
        - model_output: {
            'reconstructed': [B, ...],
            'text_tokens': [B, Tt, D] or [B, D],
            'visual_tokens': [B, Tv, D] or [B, D],
            # (선택) 'logit_scale': nn.Parameter(log(1/T))
          }
        - logit_scale_param: nn.Parameter(log(1/T)) 있으면 전달(권장).
        """
        # 1) 재구성
        recon = F.mse_loss(model_output["reconstructed"], target_images)
        loss_dict = {"recon_loss": recon.item()}

        # 2) 대조 손실
        text_tok = model_output.get("text_tokens", None)
        visual_tok = model_output.get("visual_tokens", None)
        contr_loss, contr_metrics = self.contrastive_loss(
                text_tok, visual_tok,
                attnpool=self.attnpool,
                logit_scale_param=logit_scale_param
        )
        loss_dict.update(contr_metrics)
        
        # 3) perceptual loss
        percep_loss = self.perceptual_loss(target_images, model_output["reconstructed"])
        percep_loss = torch.mean(percep_loss)
        loss_dict["total_loss"] = percep_loss.item()

        # 3) 합계
        total = recon + self.contrastive_weight * contr_loss + self.perceptual_weight * percep_loss
        loss_dict["total_loss"] = total.item()
        
        return total, loss_dict

# 하위 호환성을 위한 함수들 (기존 코드가 이 함수들을 사용할 수 있도록)
def contrastive_loss(text_tokens, visual_tokens, temperature=0.07, logit_scale_param=None, pool="mean", eps=1e-8, attnpool=None):
    loss_fn = ContrastiveLoss(temperature=temperature, pool=pool, eps=eps)
    return loss_fn(text_tokens, visual_tokens, logit_scale_param=logit_scale_param, attnpool=attnpool)

def cross_modal_reconstruction_loss(text_tokens, visual_tokens, decoder, alpha=0.1):
    loss_fn = CrossModalReconstructionLoss(alpha=alpha)
    return loss_fn(text_tokens, visual_tokens, decoder)

def combined_loss(model_output, target_images, device, perceptual_loss=None, contrastive_weight=0.1, perceptual_weight=1.0, temperature=0.07, logit_scale_param=None, pool="mean", attnpool=None):
    loss_fn = CombinedLoss(contrastive_weight=contrastive_weight, temperature=temperature, pool=pool, perceptual_loss=perceptual_loss, attnpool=attnpool, device=device)
    return loss_fn(model_output, target_images, logit_scale_param=logit_scale_param)