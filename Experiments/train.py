#!/usr/bin/env python3
"""
Super Resolution Training Script
SR 모델 학습을 위한 메인 스크립트

Usage:
    python train.py --model EDSR --scale 2 --epochs 100
    python train.py --model RCAN --scale 4 --batch-size 8 --lr 1e-5
    python train.py --resume-from ./results/EDSR_x2_20231201_120000/checkpoints/best.pth
"""

import os
import sys
import time
import torch
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import create_argument_parser, parse_config_from_args, print_config, validate_config, save_config
from dataset import get_div2k_loaders
from model_loader import create_model, print_model_info
from engine import (train_one_epoch, eval_one_epoch, get_loss_function, 
                   get_optimizer, get_scheduler)
from utils import (setup_experiment_dir, setup_logging, save_model, load_model,
                   save_metrics_csv, plot_training_curves, save_sample_images,
                   get_device, format_time)


class SuperResolutionTrainer:
    """Super Resolution 모델 훈련 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config (TrainingConfig): 훈련 설정
        """
        self.config = config
        
        # 실험 디렉토리 설정
        self.exp_dir = setup_experiment_dir(config.experiment_name)
        
        # 로깅 설정
        self.logger = setup_logging(self.exp_dir)
        
        # 설정 저장
        save_config(config, os.path.join(self.exp_dir, "config.json"))
        
        # 디바이스 설정
        if config.device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # 메트릭 히스토리
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        
        # 초기화할 것들
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.start_epoch = 0
    
    def setup_model(self):
        """모델 설정"""
        self.logger.info("Setting up model...")
        
        # 모델 파라미터에 scale 업데이트 (중복 방지)
        model_params = self.config.model_params.copy()
        model_params['scale'] = self.config.scale
        
        # 모델 생성
        self.model = create_model(
            self.config.model_name,
            **model_params
        )
        
        self.model = self.model.to(self.device)
        
        # 모델 정보 출력
        print_model_info(self.model, self.config.model_name)
        
        self.logger.info(f"Model {self.config.model_name} created successfully")
    
    def setup_data(self):
        """데이터 로더 설정"""
        self.logger.info("Setting up data loaders...")
        
        try:
            self.train_loader, self.val_loader = get_div2k_loaders(
                data_path=self.config.data_path,
                scale=self.config.scale,
                patch_size=self.config.patch_size,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers
            )
            
            self.logger.info(f"Train batches: {len(self.train_loader)}")
            self.logger.info(f"Val batches: {len(self.val_loader)}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup data loaders: {e}")
            raise
    
    def setup_training_components(self):
        """훈련 컴포넌트 설정 (손실함수, 옵티마이저, 스케줄러)"""
        self.logger.info("Setting up training components...")
        
        # 손실 함수
        self.loss_fn = get_loss_function(self.config.loss_type)
        self.logger.info(f"Loss function: {self.config.loss_type}")
        
        # 옵티마이저
        self.optimizer = get_optimizer(
            self.model,
            optimizer_type=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.logger.info(f"Optimizer: {self.config.optimizer}")
        
        # 스케줄러
        if self.config.scheduler:
            self.scheduler = get_scheduler(
                self.optimizer,
                scheduler_type=self.config.scheduler,
                total_epochs=self.config.epochs
            )
            self.logger.info(f"Scheduler: {self.config.scheduler}")
        else:
            self.scheduler = None
            self.logger.info("No scheduler used")
    
    def resume_training(self):
        """훈련 재개"""
        if self.config.resume_from:
            self.logger.info(f"Resuming training from: {self.config.resume_from}")
            
            try:
                info = load_model(
                    self.config.resume_from,
                    self.model,
                    self.optimizer,
                    self.scheduler
                )
                
                self.start_epoch = info['epoch'] + 1
                self.best_val_loss = info.get('loss', float('inf'))
                
                self.logger.info(f"Resumed from epoch {self.start_epoch}")
                
            except Exception as e:
                self.logger.error(f"Failed to resume training: {e}")
                raise
    
    def train_epoch(self, epoch):
        """한 에포크 훈련"""
        self.logger.info(f"Training epoch {epoch+1}/{self.config.epochs}")
        
        # 훈련
        train_metrics = train_one_epoch(
            model=self.model,
            loader=self.train_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            use_psnr=self.config.use_psnr,
            use_ssim=self.config.use_ssim
        )
        
        return train_metrics
    
    def validate_epoch(self, epoch):
        """한 에포크 검증"""
        if epoch % self.config.eval_every == 0:
            self.logger.info(f"Validating epoch {epoch+1}")
            
            val_metrics = eval_one_epoch(
                model=self.model,
                loader=self.val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                use_psnr=self.config.use_psnr,
                use_ssim=self.config.use_ssim
            )
            
            return val_metrics
        else:
            return None
    
    def save_checkpoint(self, epoch, train_metrics, val_metrics=None, is_best=False):
        """체크포인트 저장"""
        metrics = {
            'train': train_metrics,
            'val': val_metrics if val_metrics else {}
        }
        
        # 정기 체크포인트 저장
        if epoch % self.config.save_every == 0:
            checkpoint_path = os.path.join(
                self.exp_dir, "checkpoints", f"epoch_{epoch:03d}.pth"
            )
            save_model(
                checkpoint_path, self.model, self.optimizer, 
                self.scheduler, epoch, metrics=metrics
            )
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoints", "best.pth")
            save_model(
                best_path, self.model, self.optimizer,
                self.scheduler, epoch, metrics=metrics
            )
            self.logger.info(f"New best model saved! (epoch {epoch+1})")
        
        # 마지막 모델 저장
        last_path = os.path.join(self.exp_dir, "checkpoints", "last.pth")
        save_model(
            last_path, self.model, self.optimizer,
            self.scheduler, epoch, metrics=metrics
        )
    
    def save_sample_images_epoch(self, epoch):
        """샘플 이미지 저장"""
        if self.config.save_images and epoch % 10 == 0:
            try:
                # 검증 데이터에서 첫 번째 배치 가져오기
                data_iter = iter(self.val_loader)
                lr_imgs, hr_imgs = next(data_iter)
                
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                with torch.no_grad():
                    self.model.eval()
                    sr_imgs = self.model(lr_imgs)
                    sr_imgs = torch.clamp(sr_imgs, 0, 1)
                
                save_sample_images(lr_imgs, sr_imgs, hr_imgs, self.exp_dir, epoch)
                
            except Exception as e:
                self.logger.warning(f"Failed to save sample images: {e}")
    
    def train(self):
        """메인 훈련 루프"""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_start_time = time.time()
                
                # 훈련
                train_metrics = self.train_epoch(epoch)
                
                # 검증
                val_metrics = self.validate_epoch(epoch)
                
                # 메트릭 기록
                epoch_metrics = {
                    'train_loss': train_metrics['loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                if self.config.use_psnr:
                    epoch_metrics['train_psnr'] = train_metrics.get('psnr', 0)
                
                if self.config.use_ssim:
                    epoch_metrics['train_ssim'] = train_metrics.get('ssim', 0)
                
                if val_metrics:
                    epoch_metrics['val_loss'] = val_metrics['loss']
                    if self.config.use_psnr:
                        epoch_metrics['val_psnr'] = val_metrics.get('psnr', 0)
                    if self.config.use_ssim:
                        epoch_metrics['val_ssim'] = val_metrics.get('ssim', 0)
                
                self.metrics_history.append(epoch_metrics)
                
                # 최고 성능 체크
                is_best = False
                if val_metrics:
                    if self.config.use_psnr and 'psnr' in val_metrics:
                        if val_metrics['psnr'] > self.best_val_psnr:
                            self.best_val_psnr = val_metrics['psnr']
                            is_best = True
                    elif val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        is_best = True
                
                # 체크포인트 저장
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best)
                
                # 샘플 이미지 저장
                self.save_sample_images_epoch(epoch)
                
                # 로깅
                epoch_time = time.time() - epoch_start_time
                log_msg = (
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.6f}"
                )
                
                if self.config.use_psnr:
                    log_msg += f" | Train PSNR: {train_metrics.get('psnr', 0):.2f}dB"
                
                if val_metrics:
                    log_msg += f" | Val Loss: {val_metrics['loss']:.6f}"
                    if self.config.use_psnr:
                        log_msg += f" | Val PSNR: {val_metrics.get('psnr', 0):.2f}dB"
                
                log_msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                log_msg += f" | Time: {format_time(epoch_time)}"
                
                if is_best:
                    log_msg += " | 🌟 BEST!"
                
                self.logger.info(log_msg)
                
                # 에포크 단위 스케줄러 업데이트
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # 훈련 완료
            total_time = time.time() - start_time
            self.logger.info(f"Training completed! Total time: {format_time(total_time)}")
            
            # 메트릭 저장
            save_metrics_csv(self.metrics_history, self.exp_dir)
            
            # 훈련 곡선 저장
            if self.config.save_curves:
                plot_training_curves(self.metrics_history, self.exp_dir)
            
            self.logger.info(f"Results saved to: {self.exp_dir}")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # 현재 상태 저장
            self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise


def main():
    """메인 함수"""
    # 명령행 인수 파싱
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 설정 생성
    config = parse_config_from_args(args)
    
    # 설정 출력
    print_config(config)
    
    # 설정 검증
    errors = validate_config(config)
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # 훈련 시작
    try:
        trainer = SuperResolutionTrainer(config)
        
        # 모델 설정
        trainer.setup_model()
        
        # 데이터 설정
        trainer.setup_data()
        
        # 훈련 컴포넌트 설정
        trainer.setup_training_components()
        
        # 훈련 재개 (필요한 경우)
        trainer.resume_training()
        
        # 훈련 시작
        trainer.train()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 