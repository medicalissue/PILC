import torch
import torch.nn.functional as F
import torch.nn as nn

from lpips import LPIPS
from discriminators import PatchGANDiscriminator
from diff_aug import DiffAugment

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)        


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pool="mean", eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.pool = pool
        self.eps = eps
    
    def forward(self, text_tokens, visual_tokens, logit_scale_param=None):
        """
        CLIP 스타일 InfoNCE
        - text_tokens, visual_tokens: [B, T, D] or [B, D]
        - logit_scale_param: nn.Parameter(log(1/T)) 형태 (곱하기 exp(logit_scale))
          제공되면 temperature는 무시됨.
        """
        
        # 모든 배치 쌍에 대해 토큰별 유사도 계산
        # text_tokens[i]와 visual_tokens[j]의 토큰별 유사도
        text_expanded = text_tokens.unsqueeze(1)  # [B, 1, T, D]
        visual_expanded = visual_tokens.unsqueeze(0)  # [1, B, T, D]
        
        # 배치 간 같은 위치 토큰들의 코사인 유사도: [B, B, T]
        batch_token_similarities = F.cosine_similarity(text_expanded, visual_expanded, dim=-1)  # [B, B, T]
        
        # 각 (i,j) 쌍에 대해 softmax 가중합: [B, B, T] -> [B, B]
        batch_softmax_weights = F.softmax(batch_token_similarities, dim=-1)  # [B, B, T]
        sim = torch.sum(batch_token_similarities * batch_softmax_weights, dim=-1)  # [B, B]
        
        # Temperature/logit_scale 적용
        if logit_scale_param is not None:
            logits = sim * logit_scale_param.exp().clamp(max=100).float()
        else:
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

class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg


class CombinedLoss(nn.Module):
    def __init__(self, rec_weight = None, perceptual_loss=None, contrastive_weight=0.1,
                 perceptual_weight=1.0, lecam_loss_weight=None, temperature=0.07,
                 disc_start=None, disc_weight=None, disc_cr_loss_weight=None,
                 pool="mean"):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.pool = pool
        
        self.rec_weight = rec_weight
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature, pool=pool)
        
        self.perceptual_loss = perceptual_loss
        self.perceptual_weight = perceptual_weight
        
        self.discriminator = PatchGANDiscriminator(
            input_nc=3, 
            n_layers=3,
            ndf=64,
        )
        
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.disc_cr_loss_weight = disc_cr_loss_weight
        self.lecam_loss_weight = lecam_loss_weight
        self.lecam_ema = LeCAM_EMA()
        
    def forward(self, model_output, target_images, logit_scale_param=None,
                global_step=None, optimizer_idx=0):
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
        recon_loss = F.mse_loss(model_output["reconstructed"], target_images)

        # 2) 대조 손실
        text_tok = model_output.get("text_tokens", None)
        visual_tok = model_output.get("visual_tokens", None)
        contr_loss, contr_metrics = self.contrastive_loss(
                text_tok, visual_tok,
                logit_scale_param=logit_scale_param
        )
        
        # 3) perceptual loss
        percep_loss = self.perceptual_loss(target_images, model_output["reconstructed"])
        percep_loss = torch.mean(percep_loss)
                
        self.disc_loss = hinge_d_loss
        self.discriminator_iter_start = self.disc_start
        self.gen_adv_loss = hinge_gen_loss
        
        reconstructions = DiffAugment(model_output["reconstructed"], policy='color,translation,cutout_0.2', prob=0.5)
        logits_fake = self.discriminator(reconstructions.contiguous())
        generator_adv_loss = self.gen_adv_loss(logits_fake)
        
        disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)        
        
        if optimizer_idx == 0:
            loss = self.rec_weight * recon_loss + self.contrastive_weight * contr_loss + self.perceptual_weight * percep_loss + disc_weight * generator_adv_loss
            
            loss_dict = {"recon_loss": recon_loss.item(),
                         "percep_loss": percep_loss.item(),
                         "gen_loss": generator_adv_loss.item(),
                         "total_loss": loss.item()}
            loss_dict.update(contr_metrics)
            
            return loss, loss_dict
        elif optimizer_idx == 1:
            logits_real = self.discriminator(DiffAugment(target_images.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
            d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)

            logits_real_s = self.discriminator(DiffAugment(target_images.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
            logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
            disc_cr_loss_weight = self.disc_cr_loss_weight if global_step >= self.discriminator_iter_start else 0.0
            d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
            d_adversarial_loss += d_cr
            
            loss_dict = {"discriminator_adv_loss": d_adversarial_loss.item(), "disc_weight": disc_weight, "discriminator_cr_loss": d_cr.item()}

            return d_adversarial_loss, loss_dict

# 하위 호환성을 위한 함수들 (기존 코드가 이 함수들을 사용할 수 있도록)
def contrastive_loss(text_tokens, visual_tokens, temperature=0.07, logit_scale_param=None, pool="mean", eps=1e-8):
    loss_fn = ContrastiveLoss(temperature=temperature, pool=pool, eps=eps)
    return loss_fn(text_tokens, visual_tokens, logit_scale_param=logit_scale_param)

def cross_modal_reconstruction_loss(text_tokens, visual_tokens, decoder, alpha=0.1):
    loss_fn = CrossModalReconstructionLoss(alpha=alpha)
    return loss_fn(text_tokens, visual_tokens, decoder)

def combined_loss(model_output, target_images, rec_weight = None, perceptual_loss=None,
                  contrastive_weight=0.1,perceptual_weight=1.0, lecam_loss_weight=None,
                  temperature=0.07, disc_start=15000, logit_scale_param=None, pool="mean",
                  global_step=None, optimizer_idx=0):
    
    loss_fn = CombinedLoss(rec_weight=rec_weight, contrastive_weight=contrastive_weight,
                           perceptual_weight=perceptual_weight, lecam_loss_weight=lecam_loss_weight,
                           disc_start=disc_start, temperature=temperature, pool=pool, perceptual_loss=perceptual_loss)
    
    return loss_fn(model_output, target_images, logit_scale_param=logit_scale_param, global_step=global_step, optimizer_idx=optimizer_idx)
