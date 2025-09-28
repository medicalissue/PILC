import torch
import torch.nn.functional as F

if __name__ == "__main__":
    # 예시 데이터 생성
    batch_size = 3
    token_length = 6
    dim = 128
    
    # 랜덤 텐서 생성
    tensor1 = torch.randn(batch_size, token_length, dim)
    tensor2 = torch.randn(batch_size, token_length, dim)
    tensor3 = tensor1
    
    print(f"Input Tensor1 shape: {tensor1.shape}")  # [3, 6, 128]
    print(f"Input Tensor2 shape: {tensor2.shape}")  # [3, 6, 128]
    print()
    
    # 각 토큰별 유사도 계산
    similarities = F.cosine_similarity(tensor1, tensor2, dim=-1)
    sim2 = F.cosine_similarity(tensor1, tensor3, dim=-1)
    
    print(f"Output Similarities shape: {similarities.shape}")  # [3, 6]
    print(f"Output Similarities:\n{similarities}, {sim2}")
    print()


