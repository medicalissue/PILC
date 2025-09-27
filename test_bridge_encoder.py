import torch
from bridge_text_encoder import BridgeTextEncoder, get_tokenizer

def test_bridge_encoder():
    print("=== Bridge Text Encoder 테스트 (32차원) ===")
    
    # 1. 토크나이저 테스트
    tokenizer = get_tokenizer()
    texts = [
        "A red sports car",
        "A beautiful cat sitting on a table", 
        "Sunset over mountains with orange sky",
        "A dog playing in the park"
    ]
    
    tokens = tokenizer(
        texts,
        max_length=77,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    print(f"Input texts: {len(texts)}")
    print(f"Tokenized shape: {tokens['input_ids'].shape}")  # [4, 77]
    print(f"Attention mask shape: {tokens['attention_mask'].shape}")  # [4, 77]
    
    # 2. Bridge Encoder 테스트 (32차원!)
    bridge_encoder = BridgeTextEncoder(
        num_latent_tokens=64,
        embed_dim=32,  # ← 32차원으로 변경!
        freeze_clip=True
    )
    
    # 파라미터 수 확인
    trainable_params = sum(p.numel() for p in bridge_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in bridge_encoder.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"CLIP frozen: {not any(p.requires_grad for p in bridge_encoder.clip_text.parameters())}")
    
    # 3. Forward pass
    print("\n=== Forward Pass 테스트 ===")
    with torch.no_grad():
        output, attention = bridge_encoder(
            tokens['input_ids'],
            tokens['attention_mask'],
            return_attention=True
        )
    
    print(f"✅ Output shape: {output.shape}")  # [4, 64, 32]
    print(f"✅ Attention shape: {attention.shape}")  # [4, 64, 77]
    
    # 4. 차원 검증
    batch_size, num_tokens, embed_dim = output.shape
    assert batch_size == len(texts), f"Batch size mismatch: {batch_size} != {len(texts)}"
    assert num_tokens == 64, f"Token count mismatch: {num_tokens} != 64"
    assert embed_dim == 32, f"Embedding dim mismatch: {embed_dim} != 32"
    print("✅ 모든 차원 검증 통과!")
    
    # 5. 출력 값 범위 확인
    output_mean = output.mean().item()
    output_std = output.std().item()
    output_min = output.min().item()
    output_max = output.max().item()
    
    print(f"\n=== 출력 통계 ===")
    print(f"Mean: {output_mean:.4f}")
    print(f"Std: {output_std:.4f}")
    print(f"Min: {output_min:.4f}")
    print(f"Max: {output_max:.4f}")
    
    # 6. Attention 분석 (첫 번째 샘플)
    print(f"\n=== Attention 분석 ===")
    analyze_attention(texts[0], tokens['input_ids'][0], attention[0], tokenizer)
    
    # 7. 다양한 텍스트 길이 테스트
    print(f"\n=== 다양한 길이 텍스트 테스트 ===")
    test_various_lengths(bridge_encoder, tokenizer)
    
    print("\n🎉 Bridge Text Encoder 테스트 성공!")
    return bridge_encoder, tokenizer

def analyze_attention(text, token_ids, attention_weights, tokenizer):
    """Attention pattern 상세 분석"""
    tokens_list = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"분석 텍스트: '{text}'")
    print(f"토큰 리스트: {tokens_list[:10]}...")  # 처음 10개만
    
    # 처음 5개 query 분석
    for q in range(5):
        top_attention = attention_weights[q].topk(3)
        attended_info = []
        
        for idx, score in zip(top_attention.indices, top_attention.values):
            if idx < len(tokens_list):
                token = tokens_list[idx]
                attended_info.append(f"{token}({score:.3f})")
        
        print(f"  Query {q}: {attended_info}")
        
        # Query 역할 추정
        if q == 0:
            print(f"    → 추정 역할: 주요 객체/개념 식별")
        elif q == 1:
            print(f"    → 추정 역할: 속성/수식어 정보")
        elif q == 2:
            print(f"    → 추정 역할: 공간/맥락 정보")

def test_various_lengths(model, tokenizer):
    """다양한 길이의 텍스트 테스트"""
    test_texts = [
        "Cat",  # 매우 짧음
        "A red car driving fast on the highway",  # 중간
        "A beautiful sunset over the mountains with orange and pink clouds reflecting on the calm lake water",  # 긴 텍스트
        "",  # 빈 문자열
    ]
    
    for i, text in enumerate(test_texts):
        if text == "":
            text = "[EMPTY]"
            test_text = ""
        else:
            test_text = text
            
        tokens = tokenizer(
            [test_text], 
            max_length=77, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            output = model(tokens['input_ids'], tokens['attention_mask'])
        
        print(f"  Text {i+1} ('{text[:20]}...'): {output.shape} ✅")

if __name__ == "__main__":
    test_bridge_encoder()