import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size, scale_factor=1.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=kernel_size//2)
        self.scale = scale_factor
        
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        h = h*self.scale
        return x + h
    
    
class PixelShuffle(nn.Module):
    def __init__(self, dim, scale, kernel_size):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(dim, dim * (self.scale**2), kernel_size, padding=kernel_size//2)
        self.shuffle = nn.PixelShuffle(self.scale)
    def forward(self, x):
        h = self.shuffle(self.conv(x))
        return h
                
                
class EDSR(nn.Module):
    def __init__(self, n_colors, n_feats, kernel_size, n_resblocks, scale, res_scale=1.0):
        super().__init__()
        
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)
        self.body = nn.ModuleList()
        for _ in range(n_resblocks):
            self.body.append(
                ResBlock(dim=n_feats, kernel_size=kernel_size, scale_factor=res_scale)
            )
    
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)

        self.upsample = PixelShuffle(dim=n_feats, scale=scale, kernel_size=kernel_size)
        self.tail = nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        #head
        h = self.head(x)
        
        # Body
        res = h
        for block in self.body:
            res = block(res)
        res = self.body_conv(res)
        res += h

        # Tail
        out = self.upsample(res)
        out = self.tail(out)

        return out
    
    def compute_loss(self, output, target):
        """
        L1 loss 반환
        """
        return F.l1_loss(output, target)
    
    def configure_optimizers(self, cfg):
        # self.parameters()는 nn.Module()에서  기본으로 제공하는 메서드로, 모델 클래서에서 __init__에 등록된 모든 nn.Parameter(nn.Linear, nn.Conv2d 등 정의된 레이어의 가중치와 바이어스)를 자동으로 반환한다.
        opt = optim.Adam(self.parameters(), lr=cfg['lr'])
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=cfg.get('lr_step', 50), gamma=0.5)
        return [opt], [scheduler]   # 생성된 optimizer, scheduler를 리스트로 반환한다.