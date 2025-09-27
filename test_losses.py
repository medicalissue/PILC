import torch
from contrastive_losses import contrastive_loss, combined_loss

def test_contrastive_loss():
    print("=== Contrastive Loss 테스트 ===")
    
    batch_size = 4
    num_tokens = 64
    embed_dim = 32
    
    # 더미 토큰들
    text_tokens = torch.randn(batch_size, num_tokens, embed_dim)
    visual_tokens = torch.randn(batch_size, num_tokens, embed_dim)
    
    # Contrastive loss 계산
    loss, metrics = contrastive_loss(text_tokens, visual_tokens)
    
    print(f"Contrastive loss: {loss.item():.4f}")
    print(f"T2I Accuracy: {metrics['t2i_acc']:.4f}")
    print(f"I2T Accuracy: {metrics['i2t_acc']:.4f}")
    print(f"Average Accuracy: {metrics['avg_acc']:.4f}")
    
    assert loss.item() > 0, "Loss가 양수가 아님!"
    assert 0 <= metrics['t2i_acc'] <= 1, "Accuracy 범위 오류!"
    
    # Combined loss 테스트
    dummy_output = {
        'reconstructed': torch.randn(batch_size, 3, 256, 256),
        'visual_tokens': visual_tokens,
        'text_tokens': text_tokens,
        'codebook_loss': torch.tensor(0.1)
    }
    
    target_images = torch.randn(batch_size, 3, 256, 256)
    
    total_loss, loss_dict = combined_loss(dummy_output, target_images)
    
    print(f"\nCombined Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    print("✅ Loss 함수 테스트 성공!")

if __name__ == "__main__":
    test_contrastive_loss()