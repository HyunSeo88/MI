"""
Utility Functions for Super Resolution Training
SR 학습에 필요한 다양한 유틸리티 함수들
"""

import os
import json
import logging
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def save_model(path, model, optimizer, scheduler, epoch, loss=None, metrics=None):
    """
    모델 체크포인트 저장 (기존 utils.ipynb의 함수 확장)
    
    Args:
        path (str): 저장 경로
        model: 모델
        optimizer: 옵티마이저
        scheduler: 스케줄러
        epoch (int): 에포크
        loss (float): 손실값
        metrics (dict): 추가 메트릭들
    """
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat(),
    }
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(state_dict, path)
    print(f"Model saved to: {path}")


def load_model(path, model, optimizer=None, scheduler=None):
    """
    모델 체크포인트 로드
    
    Args:
        path (str): 모델 경로
        model: 모델
        optimizer: 옵티마이저 (선택사항)
        scheduler: 스케줄러 (선택사항)
        
    Returns:
        dict: 로드된 정보 (epoch, loss, metrics)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model'])
    
    # 옵티마이저 상태 로드 (있는 경우)
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 스케줄러 상태 로드 (있는 경우)
    if scheduler and checkpoint.get('scheduler'):
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', None),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', 'Unknown')
    }
    
    print(f"Model loaded from: {path}")
    print(f"Epoch: {info['epoch']}, Loss: {info['loss']}")
    
    return info


def setup_experiment_dir(experiment_name, base_dir="./results"):
    """
    실험 디렉토리 설정
    
    Args:
        experiment_name (str): 실험 이름
        base_dir (str): 기본 디렉토리
        
    Returns:
        str: 실험 디렉토리 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # 디렉토리 생성
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir


def setup_logging(exp_dir, log_level=logging.INFO):
    """
    로깅 설정
    
    Args:
        exp_dir (str): 실험 디렉토리
        log_level: 로그 레벨
        
    Returns:
        logger: 로거
    """
    log_file = os.path.join(exp_dir, "logs", "training.log")
    
    # 로거 설정
    logger = logging.getLogger('SR_Training')
    logger.setLevel(log_level)
    
    # 핸들러가 이미 있는 경우 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_config(config, exp_dir):
    """
    설정 저장
    
    Args:
        config (dict): 설정 딕셔너리
        exp_dir (str): 실험 디렉토리
    """
    config_path = os.path.join(exp_dir, "config.json")
    
    # datetime 객체를 문자열로 변환
    config_copy = {}
    for key, value in config.items():
        if isinstance(value, datetime):
            config_copy[key] = value.isoformat()
        else:
            config_copy[key] = value
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, indent=4, ensure_ascii=False)
    
    print(f"Config saved to: {config_path}")


def save_metrics_csv(metrics_history, exp_dir):
    """
    메트릭 히스토리를 CSV로 저장
    
    Args:
        metrics_history (list): 메트릭 히스토리
        exp_dir (str): 실험 디렉토리
    """
    if not metrics_history:
        return
    
    csv_path = os.path.join(exp_dir, "metrics.csv")
    
    # CSV 헤더 생성
    headers = ['epoch'] + list(metrics_history[0].keys())
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 헤더 쓰기
        f.write(','.join(headers) + '\n')
        
        # 데이터 쓰기
        for epoch, metrics in enumerate(metrics_history):
            row = [str(epoch)] + [str(metrics.get(key, '')) for key in headers[1:]]
            f.write(','.join(row) + '\n')
    
    print(f"Metrics saved to: {csv_path}")


def plot_training_curves(metrics_history, exp_dir, save_plot=True):
    """
    학습 곡선 플롯
    
    Args:
        metrics_history (list): 메트릭 히스토리
        exp_dir (str): 실험 디렉토리
        save_plot (bool): 플롯 저장 여부
    """
    if not metrics_history:
        return
    
    epochs = list(range(len(metrics_history)))
    
    # 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Loss 플롯
    if 'train_loss' in metrics_history[0] and 'val_loss' in metrics_history[0]:
        train_losses = [m['train_loss'] for m in metrics_history]
        val_losses = [m['val_loss'] for m in metrics_history]
        
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # PSNR 플롯
    if 'train_psnr' in metrics_history[0] and 'val_psnr' in metrics_history[0]:
        train_psnr = [m['train_psnr'] for m in metrics_history]
        val_psnr = [m['val_psnr'] for m in metrics_history]
        
        axes[0, 1].plot(epochs, train_psnr, label='Train PSNR', color='blue')
        axes[0, 1].plot(epochs, val_psnr, label='Val PSNR', color='red')
        axes[0, 1].set_title('PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # SSIM 플롯
    if 'train_ssim' in metrics_history[0] and 'val_ssim' in metrics_history[0]:
        train_ssim = [m['train_ssim'] for m in metrics_history]
        val_ssim = [m['val_ssim'] for m in metrics_history]
        
        axes[1, 0].plot(epochs, train_ssim, label='Train SSIM', color='blue')
        axes[1, 0].plot(epochs, val_ssim, label='Val SSIM', color='red')
        axes[1, 0].set_title('SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate 플롯
    if 'learning_rate' in metrics_history[0]:
        lrs = [m['learning_rate'] for m in metrics_history]
        
        axes[1, 1].plot(epochs, lrs, color='green')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = os.path.join(exp_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
    
    plt.show()


def tensor_to_pil(tensor):
    """
    텐서를 PIL 이미지로 변환
    
    Args:
        tensor (torch.Tensor): 이미지 텐서 [C, H, W] 또는 [B, C, H, W]
        
    Returns:
        PIL.Image: PIL 이미지
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 첫 번째 배치만 사용
    
    # [0, 1] 범위로 클램핑
    tensor = torch.clamp(tensor, 0, 1)
    
    # CPU로 이동 및 numpy 변환
    np_img = tensor.cpu().numpy()
    
    # 채널 순서 변경 [C, H, W] -> [H, W, C]
    np_img = np.transpose(np_img, (1, 2, 0))
    
    # [0, 255] 범위로 변환
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


def save_sample_images(lr_imgs, sr_imgs, hr_imgs, exp_dir, epoch, num_samples=4):
    """
    샘플 이미지 저장
    
    Args:
        lr_imgs (torch.Tensor): LR 이미지들
        sr_imgs (torch.Tensor): SR 이미지들  
        hr_imgs (torch.Tensor): HR 이미지들
        exp_dir (str): 실험 디렉토리
        epoch (int): 에포크
        num_samples (int): 저장할 샘플 개수
    """
    num_samples = min(num_samples, lr_imgs.size(0))
    
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    for i in range(num_samples):
        # LR 이미지
        lr_pil = tensor_to_pil(lr_imgs[i])
        axes[0, i].imshow(lr_pil)
        axes[0, i].set_title(f'LR {i+1}')
        axes[0, i].axis('off')
        
        # SR 이미지
        sr_pil = tensor_to_pil(sr_imgs[i])
        axes[1, i].imshow(sr_pil)
        axes[1, i].set_title(f'SR {i+1}')
        axes[1, i].axis('off')
        
        # HR 이미지
        hr_pil = tensor_to_pil(hr_imgs[i])
        axes[2, i].imshow(hr_pil)
        axes[2, i].set_title(f'HR {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # 이미지 저장
    img_path = os.path.join(exp_dir, "images", f"samples_epoch_{epoch:03d}.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample images saved to: {img_path}")


def get_device():
    """
    사용 가능한 디바이스 반환
    
    Returns:
        torch.device: 디바이스
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    모델 파라미터 개수 계산
    
    Args:
        model: 모델
        
    Returns:
        dict: 파라미터 정보
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def print_model_summary(model, model_name="Model"):
    """
    모델 요약 정보 출력
    
    Args:
        model: 모델
        model_name (str): 모델 이름
    """
    params = count_parameters(model)
    
    print(f"\n{model_name} Summary:")
    print("=" * 50)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Model size: {params['total'] * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)


def format_time(seconds):
    """
    초를 시:분:초 형식으로 변환
    
    Args:
        seconds (float): 초
        
    Returns:
        str: 포맷된 시간 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    # 테스트 코드
    print("Super Resolution Training Utilities")
    print("Available functions:")
    print("- save_model, load_model")
    print("- setup_experiment_dir, setup_logging")
    print("- save_config, save_metrics_csv")
    print("- plot_training_curves, save_sample_images")
    print("- tensor_to_pil, get_device")
    print("- count_parameters, print_model_summary")
    print("- format_time")
    
    # 디바이스 테스트
    device = get_device()
    
    # 시간 포맷 테스트
    print(f"Time format test: {format_time(3661)}")  # 1:01:01 