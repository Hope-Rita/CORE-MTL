# models/heads/normal_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class NormalHead(nn.Module):
    def __init__(self, in_ch, hidden=128, output="normal"):
        super().__init__()
        self.output = output

        def _up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.body = nn.Sequential(
            _up_block(in_ch, hidden),
            _up_block(hidden, hidden),
            _up_block(hidden, 64),
            _up_block(64, 32),
        )

        # 输出层：Normal用3通道，Depth用1通道
        out_dim = 3 if output == "normal" else 1
        self.head = nn.Conv2d(32, out_dim, kernel_size=3, padding=1)

    def forward(self, z_s_map):
        h = self.body(z_s_map)
        y = self.head(h)
        if self.output == "normal":
            # 法线标准化
            n = F.normalize(y, dim=1, eps=1e-6)
            return n
        else:
            return y
