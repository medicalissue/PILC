import torch
import torch.nn.functional as F

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

def contrastive_loss(
    text_tokens,
    visual_tokens,
    temperature: float = 0.07,
    logit_scale_param: torch.Tensor | None = None,
    pool: str = "mean",
    eps: float = 1e-8,
):
    """
    CLIP 스타일 InfoNCE
    - text_tokens, visual_tokens: [B, T, D] or [B, D]
    - temperature: float (고정 온도, 나누기)
    - logit_scale_param: nn.Parameter(log(1/T)) 형태 (곱하기 exp(logit_scale))
      제공되면 temperature는 무시됨.
    """
    # 1) 풀링 -> [B, D]
    text_features  = _pool_tokens(text_tokens,  pool)
    visual_features = _pool_tokens(visual_tokens, pool)

    # 2) L2 정규화 (안정 위해 eps)
    text_features  = F.normalize(text_features,  dim=-1, eps=eps)
    visual_features = F.normalize(visual_features, dim=-1, eps=eps)

    # 3) 유사도 & 로그릿 (fp32)
    sim = torch.matmul(text_features, visual_features.t()).float()  # [B, B]
    if logit_scale_param is not None:
        # CLIP 방식: logits = sim * exp(logit_scale)
        logits = sim * logit_scale_param.exp().clamp(max=100).float()
    else:
        # 고정 온도: logits = sim / T
        logits = sim / float(temperature)

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


def cross_modal_reconstruction_loss(text_tokens, visual_tokens, decoder, alpha=0.1):
    """(선택) 텍스트/비주얼 토큰 상호 복원 일치"""
    try:
        img_from_text   = decoder(text_tokens)
        img_from_visual = decoder(visual_tokens.detach())
        cross_recon = F.mse_loss(img_from_text, img_from_visual)
        return alpha * cross_recon
    except Exception:
        return torch.tensor(0.0, device=text_tokens.device)


def combined_loss(
    model_output: dict,
    target_images: torch.Tensor,
    contrastive_weight: float = 0.1,
    temperature: float | None = 0.07,
    logit_scale_param: torch.Tensor | None = None,
    pool: str = "mean",
):
    """
    SoftVQ-VAE용 총손실
    - model_output: {
        'reconstructed': [B, ...],
        'text_tokens': [B, Tt, D] or [B, D],
        'visual_tokens': [B, Tv, D] or [B, D],
        # (선택) 'logit_scale': nn.Parameter(log(1/T))
      }
    - temperature: float 고정 온도. logit_scale_param가 주어지면 무시.
    - logit_scale_param: nn.Parameter(log(1/T)) 있으면 전달(권장).
    """
    # 1) 재구성
    recon = F.mse_loss(model_output["reconstructed"], target_images)
    loss_dict = {"recon_loss": recon.item()}
    total = recon

    # 2) 대조 손실
    text_tok   = model_output.get("text_tokens", None)
    visual_tok = model_output.get("visual_tokens", None)

    if text_tok is not None and visual_tok is not None:
        # model_output에 logit_scale가 있으면 우선 사용
        lsp = logit_scale_param
        if lsp is None and isinstance(model_output.get("logit_scale", None), torch.Tensor):
            lsp = model_output["logit_scale"]

        contr_loss, contr_metrics = contrastive_loss(
            text_tok, visual_tok,
            temperature=(temperature if lsp is None else 0.07),  # lsp 있으면 무시됨
            logit_scale_param=lsp,
            pool=pool,
        )
        total = total + contrastive_weight * contr_loss
        loss_dict.update(contr_metrics)

    # 3) 합계
    loss_dict["total_loss"] = total.item()
    return total, loss_dict
