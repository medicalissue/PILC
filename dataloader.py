import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import Optional, Dict, Any, List, Tuple


class COCOImageTextDataset(Dataset):
    """COCO 데이터셋을 위한 Image-Text 쌍 로더"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 256,
        annotation_file: Optional[str] = None,
        max_caption_length: int = 77
    ):
        """
        Args:
            data_root: COCO 데이터셋 루트 경로 (예: '/data/coco')
            split: 'train', 'val', 'test' 중 하나
            image_size: 이미지 크기 (정사각형)
            annotation_file: 어노테이션 파일 경로 (기본값은 자동 설정)
            max_caption_length: 최대 캡션 길이
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        
        # 경로 설정
        if split == 'train':
            self.image_dir = os.path.join(data_root, 'train2017')
            if annotation_file is None:
                annotation_file = os.path.join(data_root, 'annotations', 'captions_train2017.json')
        elif split == 'val':
            self.image_dir = os.path.join(data_root, 'val2017')
            if annotation_file is None:
                annotation_file = os.path.join(data_root, 'annotations', 'captions_val2017.json')
        else:
            raise ValueError(f"Unsupported split: {split}")
            
        self.annotation_file = annotation_file
        
        # 경로 검증
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # 어노테이션 로드
        print(f"Loading COCO annotations from {self.annotation_file}...")
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 이미지 ID와 파일명 매핑
        self.id_to_filename = {}
        for img_info in self.coco_data['images']:
            self.id_to_filename[img_info['id']] = img_info['file_name']
        
        # Image-Caption 쌍 생성
        self.image_caption_pairs = []
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in self.id_to_filename:
                filename = self.id_to_filename[image_id]
                caption = ann['caption']
                self.image_caption_pairs.append({
                    'image_id': image_id,
                    'filename': filename,
                    'caption': caption
                })
        
        print(f"Loaded {len(self.image_caption_pairs)} image-caption pairs from COCO {split} split")
        
        # 이미지 변환 설정
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위
        ])
    
    def __len__(self) -> int:
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image: 변환된 이미지 텐서 (C, H, W)
            caption: 텍스트 캡션
        """
        pair = self.image_caption_pairs[idx]
        
        # 이미지 로드
        image_path = os.path.join(self.image_dir, pair['filename'])
        
        try:
            with Image.open(image_path) as img:
                # RGB로 변환 (RGBA나 그레이스케일 처리)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 변환 적용
                image = self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 에러 발생 시 더미 이미지 반환
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # 캡션 전처리
        caption = pair['caption'].strip()
        if len(caption) > self.max_caption_length:
            caption = caption[:self.max_caption_length]
        
        return image, caption
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """이미지 메타데이터 반환"""
        pair = self.image_caption_pairs[idx]
        return {
            'image_id': pair['image_id'],
            'filename': pair['filename'],
            'caption': pair['caption']
        }


class DummyImageTextDataset(Dataset):
    """개발용 더미 데이터셋"""
    
    def __init__(self, num_samples=1000, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
        # 더미 캡션들
        self.captions = [
            "A red sports car driving on the highway",
            "A beautiful cat sitting on a wooden table", 
            "Sunset over the mountains with orange sky",
            "A dog playing in the green park",
            "Modern architecture building in the city",
            "Fresh fruits on a white plate",
            "Ocean waves crashing on the beach",
            "A person reading a book in the library"
        ] * (num_samples // 8 + 1)
        
        self.captions = self.captions[:num_samples]
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 더미 이미지 생성 (실제 파일 대신)
        dummy_image = torch.randn(3, self.image_size, self.image_size)
        # 정규화 적용
        dummy_image = (dummy_image + 1) / 2  # [0, 1] 범위로
        
        caption = self.captions[idx]
        
        return dummy_image, caption


def create_dataloader(
    data_root: Optional[str] = None,
    dataset_type: str = 'dummy',
    split: str = 'train',
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """데이터 로더 생성
    
    Args:
        data_root: 데이터셋 루트 경로
        dataset_type: 'coco' 또는 'dummy'
        split: 'train', 'val', 'test'
        batch_size: 배치 크기
        image_size: 이미지 크기
        num_workers: 워커 프로세스 수
        pin_memory: 메모리 고정 여부
        shuffle: 셔플 여부
        **kwargs: 추가 인자들
    
    Returns:
        DataLoader 객체
    """
    
    if dataset_type == 'coco':
        if data_root is None:
            raise ValueError("data_root must be specified for COCO dataset")
        dataset = COCOImageTextDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            **kwargs
        )
    elif dataset_type == 'dummy':
        dataset_size = kwargs.get('dataset_size', 1000)
        dataset = DummyImageTextDataset(
            num_samples=dataset_size,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 마지막 배치가 작을 경우 제거
    )
    
    return dataloader


def load_coco_dataset(
    data_root: str = '/data/coco',
    split: str = 'train',
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """COCO 데이터셋 로더
    
    Args:
        data_root: COCO 데이터셋 경로
        split: 'train' 또는 'val'
        batch_size: 배치 크기
        image_size: 이미지 크기
        num_workers: 워커 프로세스 수
        **kwargs: 추가 인자들
        
    Returns:
        DataLoader 객체
    """
    return create_dataloader(
        data_root=data_root,
        dataset_type='coco',
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        **kwargs
    )


# 사용 예시 및 테스트 함수
def test_dataloader():
    """데이터로더 테스트"""
    print("Testing COCO dataloader...")
    
    try:
        # COCO 데이터로더 테스트
        coco_loader = load_coco_dataset(
            data_root='/data/coco',
            split='train',
            batch_size=4,
            image_size=256,
            num_workers=2
        )
        
        print(f"COCO dataset loaded successfully!")
        print(f"Number of batches: {len(coco_loader)}")
        
        # 첫 번째 배치 확인
        for batch_idx, (images, captions) in enumerate(coco_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"  Captions: {captions[:2]}")  # 처음 2개 캡션만 출력
            break
            
    except FileNotFoundError as e:
        print(f"COCO dataset not found: {e}")
        print("Falling back to dummy dataset...")
        
        # 더미 데이터로더 테스트
        dummy_loader = create_dataloader(
            dataset_type='dummy',
            batch_size=4,
            image_size=256,
            dataset_size=100
        )
        
        print(f"Dummy dataset loaded successfully!")
        print(f"Number of batches: {len(dummy_loader)}")
        
        # 첫 번째 배치 확인
        for batch_idx, (images, captions) in enumerate(dummy_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"  Captions: {captions[:2]}")
            break


if __name__ == "__main__":
    test_dataloader()