"""
Super Resolution Dataset Module
DIV2K 데이터셋을 위한 데이터 로더
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    """DIV2K Dataset for Super Resolution"""
    
    def __init__(self, hr_dir, lr_dir=None, scale=2, patch_size=96, train=True):
        """
        Args:
            hr_dir (str): High resolution images directory
            lr_dir (str): Low resolution images directory (optional)
            scale (int): Scale factor for downsampling (2, 3, 4, 8)
            patch_size (int): HR patch size for training (None for full image)
            train (bool): Training mode (with augmentation)
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        
        # Get all image files
        self.hr_files = sorted([f for f in self.hr_dir.glob('*.png')])
        if self.lr_dir:
            self.lr_files = sorted([f for f in self.lr_dir.glob('*.png')])
        else:
            self.lr_files = None
            
        print(f"Found {len(self.hr_files)} HR images")
        if self.lr_files:
            print(f"Found {len(self.lr_files)} LR images")
            
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = self.hr_files[idx]
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Load or generate LR image
        if self.lr_files:
            lr_path = self.lr_files[idx]
            lr_img = Image.open(lr_path).convert('RGB')
        else:
            # Downsample HR to create LR
            hr_w, hr_h = hr_img.size
            lr_w, lr_h = hr_w // self.scale, hr_h // self.scale
            lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
        
        # Crop for both training and validation
        if self.patch_size:
            if self.train:
                # Random crop for training
                hr_img, lr_img = self._random_crop(hr_img, lr_img)
            else:
                # Center crop for validation
                hr_img, lr_img = self._center_crop(hr_img, lr_img)
        
        # Data augmentation for training only
        if self.train:
            hr_img, lr_img = self._augment(hr_img, lr_img)
        
        # Convert to tensor
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        return lr_tensor, hr_tensor
    
    def _random_crop(self, hr_img, lr_img):
        """Random crop both HR and LR images"""
        hr_w, hr_h = hr_img.size
        lr_w, lr_h = lr_img.size
        
        # Calculate crop size for LR
        lr_patch_size = self.patch_size // self.scale
        
        # Random crop position for LR
        lr_left = random.randint(0, lr_w - lr_patch_size)
        lr_top = random.randint(0, lr_h - lr_patch_size)
        
        # Corresponding HR crop position
        hr_left = lr_left * self.scale
        hr_top = lr_top * self.scale
        
        # Crop images
        lr_img = TF.crop(lr_img, lr_top, lr_left, lr_patch_size, lr_patch_size)
        hr_img = TF.crop(hr_img, hr_top, hr_left, self.patch_size, self.patch_size)
        
        return hr_img, lr_img
    
    def _center_crop(self, hr_img, lr_img):
        """Center crop both HR and LR images"""
        hr_w, hr_h = hr_img.size
        lr_w, lr_h = lr_img.size
        
        # Calculate crop size for LR
        lr_patch_size = self.patch_size // self.scale
        
        # Check if image is smaller than patch size - resize if needed
        if lr_w < lr_patch_size or lr_h < lr_patch_size:
            # Resize LR image to minimum required size
            new_lr_w = max(lr_w, lr_patch_size)
            new_lr_h = max(lr_h, lr_patch_size)
            lr_img = lr_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            
            # Resize HR accordingly
            new_hr_w = new_lr_w * self.scale
            new_hr_h = new_lr_h * self.scale
            hr_img = hr_img.resize((new_hr_w, new_hr_h), Image.BICUBIC)
            
            # Update dimensions
            lr_w, lr_h = new_lr_w, new_lr_h
            hr_w, hr_h = new_hr_w, new_hr_h
        
        # Center crop position for LR
        lr_left = (lr_w - lr_patch_size) // 2
        lr_top = (lr_h - lr_patch_size) // 2
        
        # Corresponding HR crop position
        hr_left = lr_left * self.scale
        hr_top = lr_top * self.scale
        
        # Crop images
        lr_img = TF.crop(lr_img, lr_top, lr_left, lr_patch_size, lr_patch_size)
        hr_img = TF.crop(hr_img, hr_top, hr_left, self.patch_size, self.patch_size)
        
        return hr_img, lr_img
    
    def _augment(self, hr_img, lr_img):
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            hr_img = TF.hflip(hr_img)
            lr_img = TF.hflip(lr_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            hr_img = TF.vflip(hr_img)
            lr_img = TF.vflip(lr_img)
            
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_img = TF.rotate(hr_img, angle)
            lr_img = TF.rotate(lr_img, angle)
            
        return hr_img, lr_img


def get_div2k_loaders(data_path, scale=2, patch_size=96, batch_size=16, num_workers=0):
    """
    Get DIV2K data loaders
    
    Args:
        data_path (str): Path to DIV2K dataset
        scale (int): Scale factor
        patch_size (int): HR patch size for training
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Dataset paths
    train_hr_dir = os.path.join(data_path, 'DIV2K_train_HR')
    train_lr_dir = os.path.join(data_path, f'DIV2K_train_LR_bicubic/X{scale}')
    val_hr_dir = os.path.join(data_path, 'DIV2K_valid_HR')
    val_lr_dir = os.path.join(data_path, f'DIV2K_valid_LR_bicubic/X{scale}')
    
    # Check if directories exist
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} does not exist")
    
    # Create datasets
    train_dataset = DIV2KDataset(
        hr_dir=train_hr_dir,
        lr_dir=train_lr_dir if os.path.exists(train_lr_dir) else None,
        scale=scale,
        patch_size=patch_size,
        train=True
    )
    
    val_dataset = DIV2KDataset(
        hr_dir=val_hr_dir,
        lr_dir=val_lr_dir if os.path.exists(val_lr_dir) else None,
        scale=scale,
        patch_size=patch_size,  # Use same patch size for validation (center crop)
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    data_path = "./data/DV2K"
    train_loader, val_loader = get_div2k_loaders(data_path, scale=2, patch_size=96, batch_size=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for lr_batch, hr_batch in train_loader:
        print(f"LR batch shape: {lr_batch.shape}")
        print(f"HR batch shape: {hr_batch.shape}")
        print(f"LR range: [{lr_batch.min():.3f}, {lr_batch.max():.3f}]")
        print(f"HR range: [{hr_batch.min():.3f}, {hr_batch.max():.3f}]")
        break 