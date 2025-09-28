import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# 프로젝트 imports
from tokenizer import SoftVQModel, ModelArgs
from contrastive_losses import PerceptualLoss, CombinedLoss
from dataloader import create_dataloader

def train_epoch(model, dataloader, optimizer_g, optimizer_d, device, epoch, loss_fn, log_interval=50, global_step_start=0):
    """한 에폭 학습 + wandb 로깅"""
    model.train()
    loss_fn.discriminator.train()
    
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0
    global_step = global_step_start

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, texts) in enumerate(pbar):
        images = images.to(device)
        # texts가 텐서면 디바이스 이동 필요
        if torch.is_tensor(texts):
            texts = texts.to(device)

        optimizer_g.zero_grad()

        # Forward
        output = model(images, texts)

        # ⚠️ sanity 단계: temperature는 0.07 상수 권장
        #  - 나중에 learnable logit_scale 쓰려면 combined_loss/contrastive_loss에서
        #    logits = sim * logit_scale.exp() 구조로 바꿔주세요 (아래 '추가 패치' 참고).
        loss_g, loss_dict_g = loss_fn(
            output,
            images,
            logit_scale_param=model.logit_scale,
            global_step=global_step,
            optimizer_idx=0,
        )

        # Backward
        loss_g.backward()
        optimizer_g.step()
        
        
        optimizer_d.zero_grad()
        
        with torch.no_grad():
            output_d = model(images, texts)
        
        loss_d, loss_dict_d = loss_fn(
            output_d,
            images,
            logit_scale_param=model.logit_scale,
            global_step=global_step,
            optimizer_idx=1  # Discriminator 모드
        )
        
        loss_d.backward()
        optimizer_d.step()
        
        # 누적
        total_loss += float(loss_g.item())
        
        combined_loss_dict = {}
        for k, v in loss_dict_g.items():
            combined_loss_dict[f"gen_{k}"] = v
        for k, v in loss_dict_d.items():
            combined_loss_dict[f"disc_{k}"] = v
        
        for k, v in combined_loss_dict.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + float(v)

        num_batches += 1
        global_step += 1

        # 평균 메트릭 갱신
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics['epoch_loss'] = total_loss / num_batches

        # 진행바 & 주기적 wandb 로깅
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'g_loss': f"{loss_g.item():.4f}",
                'd_loss': f"{loss_d.item():.4f}",
                't2i_acc': f"{loss_dict_g.get('t2i_acc', 0.0):.3f}",
                'i2t_acc': f"{loss_dict_g.get('i2t_acc', 0.0):.3f}",
            })
            
            log_data = {f"train/{k}": v for k, v in combined_loss_dict.items()}
            log_data["train/gen_total_loss"] = loss_g.item()
            log_data["train/disc_total_loss"] = loss_d.item()

                # logit_scale이 있으면 exp() 값도 기록
            if hasattr(model, "logit_scale"):
                log_data[f"train/logit_scale_exp"] = model.logit_scale.exp().item()

            wandb.log(log_data, step=global_step)

    # 에폭 요약도 한 번 더 로깅
    wandb.log({f"train(per epoch)/{k}": v for k, v in avg_metrics.items()}, step=epoch)

    return avg_metrics, global_step


def evaluate_model(model, dataloader, device, loss_fn, epoch=None, global_step=None):
    """검증 + wandb 로깅"""
    model.eval()
    loss_fn.discriminator.eval()
    
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            if torch.is_tensor(texts):
                texts = texts.to(device)

            output = model(images, texts)
            
            _, loss_dict = loss_fn(
                output,
                images,
                logit_scale_param=model.logit_scale,
                global_step=global_step,
                optimizer_idx=0
            )

            for k, v in loss_dict.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + float(v)

            num_batches += 1

    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    payload = {f"val/{k}": v for k, v in avg_metrics.items()}
    if epoch is not None:
        payload["val/epoch"] = epoch
    if global_step is not None:
        wandb.log(payload, step=global_step)
    else:
        wandb.log(payload)

    return avg_metrics


def main():
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator learning rate (TTUR)')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='Discriminator learning rate (TTUR - usually 4x generator)')
    parser.add_argument('--dataset_size', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default=f'./checkpoints/{date}')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--wandb_project', type=str, default='PILC')
    parser.add_argument('--wandb_run_name', type=str, default='base')
    parser.add_argument('--wandb_entity', type=str, default='medicalissues')
    parser.add_argument('--device', type=str, default='cuda:2')  # 'cuda', 'cuda:1', 'cpu' 등
    
    parser.add_argument('--disc_start', type=int, default=0, help='Discriminator 시작 step')
    parser.add_argument('--disc_weight', type=float, default=1.0, help='Discriminator loss weight')
    parser.add_argument('--lecam_loss_weight', type=float, default=0.001, help='LeCAM loss weight')
    parser.add_argument('--disc_cr_loss_weight', type=float, default=1.0, help='Discriminator consistency regularization weight')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # wandb 시작
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args)
    )

    # 디바이스 설정
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 생성
    model_args = ModelArgs(
        image_size=256,
        codebook_size=16384,
        codebook_embed_dim=64,
        num_latent_tokens=32,
        enc_type='vit',
        encoder_model='vit_small_patch14_dinov2.lvd142m',
        dec_type='vit',
        decoder_model='vit_small_patch14_dinov2.lvd142m'#'vit_base_patch14_dinov2.lvd142m'#'vit_small_patch14_dinov2.lvd142m'
    )

    model = SoftVQModel(model_args)
    model.to(device)

    perceptual_loss = PerceptualLoss(device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"trainable_params": n_params}, allow_val_change=True)

    # 선택: 가중치/그래디언트 로깅
    wandb.watch(model, log="gradients", log_freq=100)

    # 데이터 로더
    train_loader = create_dataloader(
        data_root='/data/coco',
        dataset_type='coco',
        split='train',
        batch_size=args.batch_size
    )
    val_loader = create_dataloader(
        data_root='/data/coco',
        dataset_type='coco',
        split='val',
        batch_size=args.batch_size
    )

    loss_fn = CombinedLoss(
        rec_weight=1.0,
        contrastive_weight=0.1,
        perceptual_weight=2.0,
        perceptual_loss=perceptual_loss,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_cr_loss_weight=args.disc_cr_loss_weight,
        temperature=0.07,
        pool="mean"
    ).to(device)

    # Optimizer
    optimizer_g = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_g,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    optimizer_d = torch.optim.AdamW(
        loss_fn.discriminator.parameters(),
        lr=args.lr_d,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8
    )

    # 저장 디렉토리
    os.makedirs(args.save_dir, exist_ok=True)

    # 학습
    print("🚀 학습 시작!")
    train_history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer_g, optimizer_d, device, epoch, loss_fn,
            log_interval=args.log_interval, global_step_start=global_step
        )

        # Val
        val_metrics = evaluate_model(
            model, val_loader, device, loss_fn, epoch=epoch, global_step=global_step
        )

        # 콘솔 요약
        print(f"Train Gen Loss: {train_metrics.get('gen_total_loss', 0.0):.4f}")
        print(f"Train Disc Loss: {train_metrics.get('disc_discriminator_adv_loss', 0.0):.4f}")
        print(f"Train T2I Acc: {train_metrics.get('gen_t2i_acc', 0.0):.4f}")
        print(f"Train I2T Acc: {train_metrics.get('gen_i2t_acc', 0.0):.4f}")
        print(f"Val   Loss: {val_metrics.get('total_loss', 0.0):.4f}")
        print(f"Val   T2I Acc: {val_metrics.get('t2i_acc', 0.0):.4f}")
        print(f"Val   I2T Acc: {val_metrics.get('i2t_acc', 0.0):.4f}")

        # 히스토리
        epoch_log = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        train_history.append(epoch_log)

        # 체크포인트 (매 5 에폭)
        if epoch % 1 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': loss_fn.discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'train_history': train_history,
                'args': vars(args),
                'model_args': vars(model_args)
            }, ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")

    # 최종 저장
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': loss_fn.discriminator.state_dict(),
        'train_history': train_history,
        'args': args,
        'model_args': model_args
    }, final_path)

    # 로그 JSON 저장
    log_path = os.path.join(args.save_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    print(f"🎉 학습 완료! 모델 저장 위치: {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()