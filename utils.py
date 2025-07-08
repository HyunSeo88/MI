import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from torch.utils.tensorboard import SummaryWriter

import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def seed_everything(seed: int=42):
    # 재현성을 위해 시드 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_dataloaders(cfg):
    """
    cfg: 각종 변수가 저장되어 있는 dict
    설명: pytorch에서 훈련과 검증에 필요한 데이터 파이프라인을 설정한다. cfg를 받아 훈련용과 검증용 데이터로더를 dict 형태로 반환한다.
    return: {'train': train_loader, 'val':val_loader}
    
    """
    
    # 1) transform 정의
    train_transform = transforms.Compose([      #  transforms.Compose([])는 내부 리스트 안의 여러 변환을 순서대로 실행하는 파이프라인을 생성한다. 
        transforms.ToTensor(),
        # 필요시 추가 정규화 추가?
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 2) Dataset 인스턴스 생성
        # - 여기서 SRDataset은 내부적으로 Dataset 클래스를 상속받아서 __len__, __getitem__ 메서드를 구현한다.
        # - 즉, __getitem__ 메서드가 호출될 때마다 하나의 데이터에 transform하여 텐서 형태로 반환한다.
    from SR.data.preprocess import SRDataset 
    train_ds = SRDataset(cfg['data_path'], split='train', transform=train_transform)
    val_ds = SRDataset(cfg['data_path'], split='val', transform=val_transform)
    
    # 3) Dataloader 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers',4),
        pin_memory=True
    )
    return {'train':train_loader, 'val': val_loader}

def get_writer(log_dir: str):
    """
    TensorBoard SummaryWriter 반환.
    log_dir: 로그를 저장할 디렉토리
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def log_images(writer, tag: str, images: torch.Tensor, step: int, nrow: int = 4):
    """
    이미지 그리드를 TensorBoard에 기록.
    images: (B, C, H, W) 텐서
    """
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    writer.add_image(tag, grid, global_step=step)

def log_scalars(writer, metrics: dict, step: int):
    """
    여러 스칼라 값을 한 번에 기록.
    metrics: {'loss': 0.123, 'psnr': 27.5, ...}
    """
    for key, value in metrics.items():
        writer.add_scalar(key, value, global_step=step)
        
        
def compute_psnr(pred, target):
    # pred, target: numpy or torch.Tensor (0–1 범위)
    p = compare_psnr(target.cpu().numpy(), pred.cpu().numpy(), data_range=1.0)
    return p

def compute_ssim(pred, target):
    s = 0
    for c in range(pred.shape[1]):  # 채널별
        s += compare_ssim(
            target[0,c].cpu().numpy(),
            pred[0,c].cpu().numpy(),
            data_range=1.0
        )
    return s / pred.shape[1]