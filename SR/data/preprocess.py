import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

class SRDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, scale=4):
        """
        Super-Resolution Dataset
        data_path: 데이터가 저장된 루트 경로
        split: 'train' 또는 'val'
        transform: 이미지 변환
        scale: 업스케일링 배율
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.scale = scale
        
        # 이미지 파일 경로 수집
        self.hr_images = self._collect_images()
        
    def _collect_images(self):
        """이미지 파일 경로들을 수집"""
        # 일반적인 이미지 확장자들
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        # split별 경로 설정
        split_path = os.path.join(self.data_path, self.split)
        
        if os.path.exists(split_path):
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(split_path, ext)))
                image_files.extend(glob.glob(os.path.join(split_path, '**', ext), recursive=True))
        else:
            # split 폴더가 없는 경우 전체 폴더에서 수집
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(self.data_path, ext)))
                
        return sorted(image_files)
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        """
        HR 이미지를 로드하고 LR 이미지를 생성
        """
        # HR 이미지 로드
        hr_path = self.hr_images[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        # LR 이미지 생성 (다운샘플링)
        lr_size = (hr_image.width // self.scale, hr_image.height // self.scale)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        
        # 다시 원래 크기로 업샘플링 (bicubic interpolation)
        lr_image = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
        
        # 변환 적용
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return lr_image, hr_image