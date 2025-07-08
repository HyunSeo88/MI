import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class block(nn.Module):
    def __init__(self, dim,kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim,dim,kernel_size,stride,padding=kernel_size//2)
        self.bn1=nn.BatchNorm2d(dim)
        self.prelu = nn.PReLU()
        self.conv2=nn.Conv2d(dim,dim,kernel_size,stride,padding=kernel_size//2)
        self.bn2=nn.BatchNorm2d(dim)
        
    def forward(self,x):
        h1 = self.conv1(x)
        bn1 = self.bn1(h1)
        prelu1 = self.prelu(bn1)
        h2 = self.conv2(prelu1)
        bn2 = self.bn2(h2)
        out = x+bn2
        return out

class PixelShuffle(nn.Module):
    def __init__(self, dim, scale, kernel_size):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(dim, dim * (self.scale**2), kernel_size, padding=kernel_size//2)
        self.shuffle = nn.PixelShuffle(self.scale)
    def forward(self, x):
        h = self.shuffle(self.conv(x))
        return h

class Generator(nn.Module):
    def __init__(self, dim, kernel_size=9, residual_kernel_size=3, stride=1, n_blocks=16):
        super().__init__()
        self.conv1 = nn.Conv2d(dim[0], dim[1], kernel_size, stride=stride, padding=kernel_size//2)
        self.prelu1 = nn.PReLU()
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.residual_blocks.append(block(dim[1], residual_kernel_size, stride))
        
        self.conv2 = nn.Conv2d(dim[1], dim[1], residual_kernel_size, stride, padding=residual_kernel_size//2)
        self.bn1 = nn.BatchNorm2d(dim[1])
        
        
        self.upsample = nn.Sequential(
           PixelShuffle(dim[1],2, kernel_size=3),
           nn.PReLU(),
           PixelShuffle(dim[1],2, kernel_size=3),
           nn.PReLU()
       )
     
        self.head = nn.Conv2d(dim[1], dim[0],kernel_size, stride, padding=kernel_size//2)
        
    def forward(self, x):
        shallow_feature = self.conv1(x)
        shallow_prelu = self.prelu1(shallow_feature)
        
        residual_out = shallow_prelu
        for res_block in self.residual_blocks:
            residual_out = res_block(residual_out)
        
        conv2_out = self.conv2(residual_out)
        bn_out = self.bn1(conv2_out)
        skip_out = shallow_prelu + bn_out
        
        upsample_out = self.upsample(skip_out)
        
        out = self.head(upsample_out)
        return out
        
        
        
class Discriminator(nn.Module):
    def __init__(self, dim=[3,64,128,256,512,1024], kernel_size=3, n_blocks=7):
        super().__init__()
        self.layers = nn.ModuleList()

        # 첫 두 레이어 (stride=1 → stride=2)
        self.layers.append(nn.Sequential(
            nn.Conv2d(dim[0], dim[1], kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(dim[1], dim[1], kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm2d(dim[1]),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        # 반복 블록
        for i in range(n_blocks-2):
            in_c, out_c = dim[i+1], dim[i+2]
            stride = 1 if i % 2 == 0 else 2
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # 전역 풀링 후 FC로 차원 축소
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim[-2], 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.global_pool(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)        # (B, C)
        x = self.fc(x)                   # (B, 1)
        return x


class SRGAN(nn.Module):
    def __init__(self, cfg, 
                 gen_dim=[3,64],    # Generator에 넘길 dim 파라미터
                 disc_dim=[3,64,128,256,512,1024],
                 gen_kwargs=None,   # kernel_size, n_blocks 등
                 disc_kwargs=None):
        super().__init__()
        
        # 1) Generator / Discriminator 인스턴스화
        gen_kwargs    = gen_kwargs    or {}
        disc_kwargs   = disc_kwargs   or {}
        self.gen  = Generator(gen_dim, **gen_kwargs)
        self.disc = Discriminator(disc_dim, **disc_kwargs)

        # 2) 손실 가중치
        self.lambda_adv = cfg.get('lambda_adv', 1e-3)

    def forward(self, x):
        return self.gen(x)

    def compute_loss(self, output, target):
        """
        SRGAN의 총 손실 (Generator + Discriminator)
        1) Generator Loss: Pixel-wise + Adversarial 
        2) Discriminator Loss: Real vs Fake
        """
        # 1) Generator Loss
        # Pixel loss
        loss_pixel = F.l1_loss(output, target)
        
        # Adversarial loss (Generator이 Discriminator를 속이는 정도)
        pred_fake = self.disc(output)
        real_labels = torch.ones_like(pred_fake, device=pred_fake.device)
        loss_gen_adv = F.binary_cross_entropy(pred_fake, real_labels)
        
        # Generator 총 손실
        loss_gen = loss_pixel + self.lambda_adv * loss_gen_adv
        
        # 2) Discriminator Loss
        # Real images
        pred_real = self.disc(target)
        real_labels = torch.ones_like(pred_real, device=pred_real.device)
        loss_disc_real = F.binary_cross_entropy(pred_real, real_labels)
        
        # Fake images (detach to stop gradient flow to generator)
        pred_fake_detached = self.disc(output.detach())
        fake_labels = torch.zeros_like(pred_fake_detached, device=pred_fake_detached.device)
        loss_disc_fake = F.binary_cross_entropy(pred_fake_detached, fake_labels)
        
        # Discriminator 총 손실
        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
        
        # 총 손실 반환 (Generator loss를 주로 사용)
        # Discriminator loss는 별도로 backward할 예정
        self.last_disc_loss = loss_disc  # Discriminator loss 저장
        return loss_gen
    
    def step_discriminator(self, optimizers):
        """
        Discriminator 전용 step 함수
        기존 engine.py와 호환성을 위해 추가
        """
        if hasattr(self, 'last_disc_loss') and len(optimizers) > 1:
            # Discriminator optimizer만 zero_grad 및 step
            optimizers[1].zero_grad()
            self.last_disc_loss.backward(retain_graph=True)
            optimizers[1].step()

    def configure_optimizers(self, cfg):
        """
        Generator와 Discriminator용 옵티마이저를 반환.
        Trainer는 zero_grad→backward→step 순으로 자동 처리.
        """
        lrG = cfg.get('lrG', cfg.get('lr', 1e-4))
        lrD = cfg.get('lrD', lrG)
        optG = optim.Adam(self.gen.parameters(),  lr=lrG, betas=(0.9, 0.999))
        optD = optim.Adam(self.disc.parameters(), lr=lrD, betas=(0.9, 0.999))

        # 스케줄러가 필요하면 리스트에 추가
        schedG = optim.lr_scheduler.StepLR(optG,
                   step_size=cfg.get('lr_stepG', 50),
                   gamma=cfg.get('lr_gamma', 0.5))
        schedD = optim.lr_scheduler.StepLR(optD,
                   step_size=cfg.get('lr_stepD', 50),
                   gamma=cfg.get('lr_gamma', 0.5))

        return [optG, optD], [schedG, schedD]