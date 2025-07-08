# engine.py
# - 모델이나 데이터에 상관없이, 훈련과 validation의 전체 과정을 관장하는 역할을 한다.
# - 훈련의 반복적인 로직을 한 곳에 모아두어, 다른 파일들이 각자의 역할에 집중하게 한다.

import torch
import os
from utils import compute_psnr, compute_ssim, get_writer, log_scalars
class Trainer:
    def __init__(self, model, dataloaders, cfg):
        """
        model   : nn.Module 상속 받는 모델 -> eval(), train() 등의 메서드를 제공한다. 각 메서드는 학습 모드 <-> 추론 모드를 변경하여 드롭아웃, batchnorm등의 작동 방식을 변경한다.
        dataloaders: trainer 클래스에 데이터를 전달하는 역할 (mini-batch, shuffling, num_workers, iteration등을 적절하게 처리하여 데이터를 효율적으로 전달하는 역할)
        cfg: 설정 객체 (하이퍼파라미터, device 정보 등을 전달한다.)
        
        """
        # 디바이스 자동 감지
        if hasattr(cfg, 'device'):
            device = cfg.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # CUDA 사용 가능성 검증
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA가 사용 불가능합니다. CPU로 전환합니다.")
            device = 'cpu'
            
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.cfg = cfg
        # 옵티마이저와 스케줄러는 모델 정의 코드에서 구현할 거임.
        self.optimizers, self.schedulers = None, None
        
        # 로깅 설정
        self.writer = None
        if hasattr(cfg, 'log_dir'):
            self.writer = get_writer(cfg.log_dir)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss=0.0
        
        for batch_idx, (x,y) in enumerate(self.train_loader):           # x,y는 데이터로더가 만들어준 입력, 정답 mini batch이다.
            x,y  = x.to(self.device), y.to(self.device)
            
            # 1) forward 계산
            preds = self.model(x)
            
            # 2) Loss 계산
            loss = self.model.compute_loss(preds, y)
            total_loss += loss.item()
            
            # 3) backward + step
            for opt in self.optimizers: # self.optimizers가 두개 이상의 optimizer를 담고 있는 리스트일 경우를 처리한다. 예를 들어, GAN의 경우 두개의 optimizer 존재한다.
                opt.zero_grad()         #gradient 초기화
            loss.backward()
            for opt in self.optimizers:
                opt.step()
            
            # SRGAN 특별 처리: Discriminator step
            if hasattr(self.model, 'step_discriminator'):
                self.model.step_discriminator(self.optimizers)
            
            # 4) schedular step
            for sch in self.schedulers or []:
                sch.step()
        avg_loss = total_loss/len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        count =0
        
        with torch.no_grad():       # 이 블록 안에서는 gradient 계산을 수행하지 않도록 설정한다.
            for x,y in self.val_loader:
                x,y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss=self.model.compute_loss(preds,y)
                total_loss+=loss.item()
                
                psnr_total += compute_psnr(preds,y)
                ssim_total += compute_ssim(preds,y)
                count +=1
                
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = psnr_total  / count
        avg_ssim = ssim_total  / count
        
        return avg_loss, avg_psnr, avg_ssim
    
    def train(self, num_epochs, start_epoch=1, ckpt_dir="checkpoints"):
        # model코드에서 optimizer, scheduler 함수를 받아온다.
        self.optimizers, self.schedulers = self.model.configure_optimizers(self.cfg)
        
        os.makedirs(ckpt_dir, exist_ok=True)
        for epoch in range(start_epoch, num_epochs+1):
            train_loss         = self.train_epoch(epoch)
            val_loss, psnr, ssim = self.validate_epoch(epoch)
            print(f"[Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {psnr:.2f} dB | "
                f"SSIM: {ssim:.4f}")
                
            # 로깅
            if self.writer:
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'psnr': psnr,
                    'ssim': ssim
                }
                log_scalars(self.writer, metrics, epoch)
                
            # 에폭 종료 후 체크포인트 저장
            model_name = getattr(self.cfg, 'model_name', 'model')
            ckpt_path = f"{ckpt_dir}/{model_name}_epoch{epoch}.pth"
            self.save_checkpoint(epoch, ckpt_path)
            
    def save_checkpoint(self, epoch, path):
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sch.state_dict() for sch in self.schedulers],
        }
        torch.save(ckpt, path)
        
    def load_checkpoint(self, path):
        """
        체크포인트를 로드하여 모델, 옵티마이저, 스케줄러 상태를 복원
        """
        if not os.path.exists(path):
            print(f"체크포인트 파일을 찾을 수 없습니다: {path}")
            return 0
            
        ckpt = torch.load(path, map_location=self.device)
        
        # 모델 상태 로드
        self.model.load_state_dict(ckpt['model_state'])
        
        # 옵티마이저 상태 로드 (옵티마이저가 초기화된 후에만)
        if self.optimizers and 'optimizers' in ckpt:
            for opt, opt_state in zip(self.optimizers, ckpt['optimizers']):
                opt.load_state_dict(opt_state)
        
        # 스케줄러 상태 로드 (스케줄러가 초기화된 후에만)
        if self.schedulers and 'schedulers' in ckpt:
            for sch, sch_state in zip(self.schedulers, ckpt['schedulers']):
                sch.load_state_dict(sch_state)
        
        epoch = ckpt.get('epoch', 0)
        print(f"체크포인트 로드 완료: epoch {epoch}")
        return epoch
