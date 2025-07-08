import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class CA(nn.Module):
    def __init__(self, dim, kernel_size, reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim//reduction, kernel_size = kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(dim//reduction, dim, kernel_size = kernel_size, stride =1, padding=kernel_size//2)
    def forward(self, x):
        g = x.mean([-1,-2], keepdim=True)  # 차원 유지
        scaled_g = self.conv1(g)
        h = F.relu(scaled_g)
        out = self.conv2(h)
        return torch.sigmoid(out)
    
class RCAB(nn.Module):
    def __init__(self, dim,kernel_size, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(dim,dim,kernel_size = kernel_size, stride = stride, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(dim,dim,kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.ca = CA(dim, kernel_size, reduction)
    def forward(self, x):
        res = self.conv2(F.relu(self.conv1(x)))
    
        s   = self.ca(res)
        res = res * s
        
        return x + res
    
class ResidualGroup(nn.Module):
    def __init__(self, dim, kernel_size, num_rcab):
        super().__init__()
        pad = kernel_size // 2
        # RCAB 여러 개 + 그룹 내 skip용 conv
        modules = [RCAB(dim, kernel_size) for _ in range(num_rcab)]
        modules.append(nn.Conv2d(dim, dim, kernel_size, padding=pad))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        # short skip connection
        return x + self.body(x)

class RCAN(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size, num_rg, num_rcab, scale):
        super().__init__()
        pad = kernel_size // 2
        # 1) Shallow Feature
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size, padding=pad)

        # 2) RIR: Residual-in-Residual
        self.RIR = nn.ModuleList([
            ResidualGroup(n_feats, kernel_size, num_rcab)
            for _ in range(num_rg)
        ])
        # 그룹 전체 skip을 위한 conv
        self.conv_after_RIR = nn.Conv2d(n_feats, n_feats, kernel_size, padding=pad)

        # 3) Upsample (ESPCN)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=pad),
            nn.PixelShuffle(scale)
        )

        # 4) Reconstruction
        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size, padding=pad)

    def forward(self, x):
        # shallow feature
        f1 = self.head(x)

        # deep feature via RIR
        res = f1
        for rg in self.RIR:
            res = rg(res)
        res = self.conv_after_RIR(res)

        # long skip + upsample + reconstruct
        f2  = self.upsample(f1 + res)
        out = self.tail(f2)
        return out
    
    def compute_loss(self, output, target):
        """
        RCAN 학습용 손실 함수.
        여기서는 픽셀 단위 재현 손실(L1)을 사용.
        """
        return F.l1_loss(output, target)

    def configure_optimizers(self, cfg):
        """
        옵티마이저(+스케줄러)를 생성하여 반환.
        Trainer가 zero_grad→backward→step, scheduler.step()을 자동 처리한다.
        """
        # 기본 Adam
        lr = cfg.get('lr', 1e-4)
        opt = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

        # 선택적 스케줄러
        schedulers = []
        if 'lr_step' in cfg:
            schedulers.append(
                optim.lr_scheduler.StepLR(opt,
                                          step_size=cfg['lr_step'],
                                          gamma=cfg.get('lr_gamma', 0.5))
            )
        return [opt], schedulers