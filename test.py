#!/usr/bin/env python3
"""
SoftVQ-VAE Pure Image Reconstruction Test
checkpoint_epoch_10.pth를 사용하여 COCO 이미지 10개에 대한 순수 이미지 reconstruction 테스트
(텍스트 없이 이미지만 사용)
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 프로젝트 imports
from tokenizer import SoftVQModel, ModelArgs
from dataloader import create_dataloader

def load_model(checkpoint_path, device):
    """체크포인트에서 모델 로드"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # 모델 설정 (훈련 시와 동일하게)
    model_args = ModelArgs(
        image_size=256,
        codebook_size=16384,
        codebook_embed_dim=64,
        num_latent_tokens=64,
        tau=0.07,
        enc_type='vit',
        encoder_model='vit_small_patch14_dinov2.lvd142m',
        dec_type='vit',
        decoder_model='vit_small_patch14_dinov2.lvd142m'
    )
    
    model = SoftVQModel(model_args)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # state_dict 키 확인 및 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

def calculate_metrics(original, reconstructed):
    """reconstruction 메트릭 계산"""
    with torch.no_grad():
        # MSE Loss
        mse = F.mse_loss(reconstructed, original).item()
        
        # PSNR
        psnr = -10 * torch.log10(torch.mean((original - reconstructed) ** 2)).item()
        
        # L1 Loss  
        l1 = F.l1_loss(reconstructed, original).item()
        
    return {'mse': mse, 'psnr': psnr, 'l1': l1}

def tensor_to_image(tensor):
    """Tensor를 PIL Image로 변환"""
    # [0,1] 범위로 클램핑
    tensor = torch.clamp(tensor, 0, 1)
    # CPU로 이동 후 numpy 변환
    if tensor.is_cuda:
        tensor = tensor.cpu()
    image_np = tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def run_inference_test(checkpoint_path, data_root='/data/coco', num_samples=10, save_dir='inference_results'):
    """추론 테스트 실행"""
    
    # 디바이스 설정
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 결과 저장 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    model = load_model(checkpoint_path, device)
    
    # 데이터 로더 생성
    print("Creating dataloader...")
    val_loader = create_dataloader(
        data_root=data_root,
        dataset_type='coco',
        split='val',
        batch_size=1,  # 하나씩 처리
        shuffle=True
    )
    
    print(f"Starting inference on {num_samples} samples...")
    
    all_metrics = []
    
    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            print(f"Processing sample {i+1}/{num_samples}")
            
            # 이미지를 디바이스로 이동
            original_image = images.to(device)  # [1, 3, 256, 256]
            
            try:
                # Forward pass (텍스트 없이 이미지만)
                output = model(original_image, None)  # 텍스트는 None
                reconstructed = output['reconstructed']
                
                # 메트릭 계산
                metrics = calculate_metrics(original_image, reconstructed)
                all_metrics.append(metrics)
                
                print(f"  MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}dB, L1: {metrics['l1']:.6f}")
                
                # 이미지 저장
                original_pil = tensor_to_image(original_image[0])
                reconstructed_pil = tensor_to_image(reconstructed[0])
                
                # 비교 이미지 생성
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(original_pil)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(reconstructed_pil)
                axes[1].set_title(f'Reconstructed\nMSE: {metrics["mse"]:.6f}, PSNR: {metrics["psnr"]:.1f}dB')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'comparison_{i+1:02d}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # 개별 이미지도 저장
                original_pil.save(save_dir / f'original_{i+1:02d}.png')
                reconstructed_pil.save(save_dir / f'reconstructed_{i+1:02d}.png')
                
            except Exception as e:
                print(f"  Error processing sample {i+1}: {e}")
                continue
    
    # 전체 메트릭 요약
    if all_metrics:
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in all_metrics]),
            'psnr': np.mean([m['psnr'] for m in all_metrics]),
            'l1': np.mean([m['l1'] for m in all_metrics])
        }
        
        print(f"\n{'='*50}")
        print("RECONSTRUCTION METRICS SUMMARY")
        print(f"{'='*50}")
        print(f"Samples processed: {len(all_metrics)}")
        print(f"Average MSE: {avg_metrics['mse']:.6f}")
        print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"Average L1: {avg_metrics['l1']:.6f}")
        print(f"Results saved to: {save_dir}")
        
        # 메트릭을 텍스트 파일로 저장
        with open(save_dir / 'metrics_summary.txt', 'w') as f:
            f.write("SoftVQ-VAE Pure Image Reconstruction Test Results\n")
            f.write("="*50 + "\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Samples: {len(all_metrics)}\n")
            f.write(f"Average MSE: {avg_metrics['mse']:.6f}\n")
            f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
            f.write(f"Average L1: {avg_metrics['l1']:.6f}\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 20 + "\n")
            for i, m in enumerate(all_metrics):
                f.write(f"Sample {i+1:2d}: MSE={m['mse']:.6f}, PSNR={m['psnr']:5.1f}dB, L1={m['l1']:.6f}\n")
    
    else:
        print("No samples were successfully processed!")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SoftVQ-VAE Pure Image Reconstruction Test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_10.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--data_root', type=str, default='/data/coco',
                       help='Path to COCO dataset')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--save_dir', type=str, default='inference_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # 데이터셋 경로 확인
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found: {args.data_root}")
        print("Please make sure COCO dataset is available")
        return
    
    # 추론 테스트 실행
    run_inference_test(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()