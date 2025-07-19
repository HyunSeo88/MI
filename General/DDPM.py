import torch.functional as F
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self):
        super().__init__()