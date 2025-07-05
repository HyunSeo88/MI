"""
Super Resolution Training Engine
SR 모델 학습 및 평가를 위한 엔진
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
import torchmetrics.functional as F


def psnr_metric(outputs, targets):
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산
    
    Args:
        outputs (torch.Tensor): 예측된 SR 이미지 [B, C, H, W]
        targets (torch.Tensor): 실제 HR 이미지 [B, C, H, W]
        
    Returns:
        torch.Tensor: PSNR 값
    """
    return F.peak_signal_noise_ratio(outputs, targets, data_range=1.0)


def ssim_metric(outputs, targets):
    """
    SSIM (Structural Similarity Index) 계산
    
    Args:
        outputs (torch.Tensor): 예측된 SR 이미지 [B, C, H, W]
        targets (torch.Tensor): 실제 HR 이미지 [B, C, H, W]
        
    Returns:
        torch.Tensor: SSIM 값
    """
    return F.structural_similarity_index_measure(outputs, targets, data_range=1.0)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, 
                   use_psnr=True, use_ssim=False, clip_grad=None):
    """
    SR 모델의 한 에포크 학습
    
    Args:
        model: SR 모델
        loader: 학습 데이터 로더
        loss_fn: 손실 함수 (MSE, L1 등)
        optimizer: 옵티마이저
        scheduler: 스케줄러 (사용하지 않음, 에포크 단위로 외부에서 호출됨)
        device: 디바이스
        use_psnr (bool): PSNR 계산 여부
        use_ssim (bool): SSIM 계산 여부
        clip_grad (float): Gradient clipping 값
        
    Returns:
        dict: 학습 결과 (loss, psnr, ssim)
    """
    # 모델을 학습 모드로 설정
    model.train()

    # 평균 계산을 위한 메트릭들
    loss_meter = MeanMetric()
    psnr_meter = MeanMetric() if use_psnr else None
    ssim_meter = MeanMetric() if use_ssim else None
    
    # 학습 루프
    pbar = tqdm(loader, desc="Training")
    for lr_imgs, hr_imgs in pbar:
        # 데이터를 디바이스로 이동
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # Forward pass
        sr_imgs = model(lr_imgs)
        loss = loss_fn(sr_imgs, hr_imgs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # 메트릭 계산
        with torch.no_grad():
            # 값을 [0, 1] 범위로 클램핑
            sr_imgs_clamped = torch.clamp(sr_imgs, 0, 1)
            hr_imgs_clamped = torch.clamp(hr_imgs, 0, 1)
            
            if use_psnr:
                psnr = psnr_metric(sr_imgs_clamped, hr_imgs_clamped)
                psnr_meter.update(psnr.to('cpu'))
            
            if use_ssim:
                ssim = ssim_metric(sr_imgs_clamped, hr_imgs_clamped)
                ssim_meter.update(ssim.to('cpu'))
        
        # 통계 업데이트
        loss_meter.update(loss.to('cpu'))
        
        # Progress bar 업데이트
        postfix = {'Loss': f'{loss.item():.6f}'}
        if use_psnr:
            postfix['PSNR'] = f'{psnr.item():.2f}'
        if use_ssim:
            postfix['SSIM'] = f'{ssim.item():.4f}'
        pbar.set_postfix(postfix)
    
    # 결과 정리
    summary = {
        'loss': loss_meter.compute().item(),
    }
    
    if use_psnr:
        summary['psnr'] = psnr_meter.compute().item()
    
    if use_ssim:
        summary['ssim'] = ssim_meter.compute().item()

    return summary


def eval_one_epoch(model, loader, loss_fn, device, use_psnr=True, use_ssim=False):
    """
    SR 모델의 한 에포크 평가
    
    Args:
        model: SR 모델
        loader: 평가 데이터 로더  
        loss_fn: 손실 함수
        device: 디바이스
        use_psnr (bool): PSNR 계산 여부
        use_ssim (bool): SSIM 계산 여부
        
    Returns:
        dict: 평가 결과 (loss, psnr, ssim)
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 평균 계산을 위한 메트릭들
    loss_meter = MeanMetric()
    psnr_meter = MeanMetric() if use_psnr else None
    ssim_meter = MeanMetric() if use_ssim else None
    
    # 평가 루프
    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for lr_imgs, hr_imgs in pbar:
            # 데이터를 디바이스로 이동
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            loss = loss_fn(sr_imgs, hr_imgs)
            
            # 값을 [0, 1] 범위로 클램핑
            sr_imgs_clamped = torch.clamp(sr_imgs, 0, 1)
            hr_imgs_clamped = torch.clamp(hr_imgs, 0, 1)
            
            # 메트릭 계산
            if use_psnr:
                psnr = psnr_metric(sr_imgs_clamped, hr_imgs_clamped)
                psnr_meter.update(psnr.to('cpu'))
            
            if use_ssim:
                ssim = ssim_metric(sr_imgs_clamped, hr_imgs_clamped)
                ssim_meter.update(ssim.to('cpu'))
            
            # 통계 업데이트
            loss_meter.update(loss.to('cpu'))
            
            # Progress bar 업데이트
            postfix = {'Loss': f'{loss.item():.6f}'}
            if use_psnr:
                postfix['PSNR'] = f'{psnr.item():.2f}'
            if use_ssim:
                postfix['SSIM'] = f'{ssim.item():.4f}'
            pbar.set_postfix(postfix)
    
    # 결과 정리
    summary = {
        'loss': loss_meter.compute().item(),
    }
    
    if use_psnr:
        summary['psnr'] = psnr_meter.compute().item()
    
    if use_ssim:
        summary['ssim'] = ssim_meter.compute().item()

    return summary


def get_loss_function(loss_type='L1'):
    """
    손실 함수 가져오기
    
    Args:
        loss_type (str): 손실 함수 타입 ('L1', 'L2', 'MSE', 'Huber')
        
    Returns:
        loss_fn: 손실 함수
    """
    if loss_type.upper() == 'L1':
        return nn.L1Loss()
    elif loss_type.upper() in ['L2', 'MSE']:
        return nn.MSELoss()
    elif loss_type.upper() == 'HUBER':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_optimizer(model, optimizer_type='Adam', lr=1e-4, weight_decay=0):
    """
    옵티마이저 가져오기
    
    Args:
        model: 모델
        optimizer_type (str): 옵티마이저 타입
        lr (float): 학습률
        weight_decay (float): 가중치 감쇠
        
    Returns:
        optimizer: 옵티마이저
    """
    if optimizer_type.upper() == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.upper() == 'ADAMW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.upper() == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='CosineAnnealingLR', total_epochs=100, **kwargs):
    """
    스케줄러 가져오기
    
    Args:
        optimizer: 옵티마이저
        scheduler_type (str): 스케줄러 타입
        total_epochs (int): 총 에포크 수
        **kwargs: 추가 파라미터
        
    Returns:
        scheduler: 스케줄러
    """
    if scheduler_type.upper() == 'COSINEANNEALINGLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, **kwargs
        )
    elif scheduler_type.upper() == 'STEPLR':
        step_size = kwargs.get('step_size', total_epochs // 3)
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type.upper() == 'MULTISTEPLR':
        milestones = kwargs.get('milestones', [total_epochs // 2, total_epochs * 3 // 4])
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif scheduler_type.upper() == 'EXPONENTIALLR':
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        return None  # 스케줄러 사용하지 않음


if __name__ == "__main__":
    # 테스트 코드
    print("Super Resolution Training Engine")
    print("Available loss functions:", ['L1', 'L2', 'MSE', 'Huber'])
    print("Available optimizers:", ['Adam', 'AdamW', 'SGD'])
    print("Available schedulers:", ['CosineAnnealingLR', 'StepLR', 'MultiStepLR', 'ExponentialLR'])
    
    # 더미 데이터로 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 더미 텐서 생성
    lr_img = torch.randn(2, 3, 32, 32)
    hr_img = torch.randn(2, 3, 64, 64)
    
    # 메트릭 테스트
    try:
        psnr = psnr_metric(hr_img, hr_img)  # 동일한 이미지로 테스트
        ssim = ssim_metric(hr_img, hr_img)
        print(f"PSNR test: {psnr:.2f}")
        print(f"SSIM test: {ssim:.4f}")
    except Exception as e:
        print(f"Metric test failed: {e}") 