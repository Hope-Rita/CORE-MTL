# models/heads/albedo_head.py
import torch.nn as nn
class AlbedoHead(nn.Module):
    def __init__(self, in_ch, hidden=128):
        super().__init__()

        # [新增] 1x1 -> 14x14 (假设输入图片是 448x448, latent 是 14x14)
        # 也可以投射到更小的尺寸比如 8x8，然后慢慢上采样
        self.initial_size = 16  # 可以根据你的 img_size 调整，比如 img_size/32

        self.fc = nn.Linear(in_ch, hidden * self.initial_size * self.initial_size)
        self.relu = nn.ReLU(True)

        def _up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.body = nn.Sequential(
            _up_block(hidden, hidden),  # 16->32
            _up_block(hidden, 64),  # 32->64
            _up_block(64, 32),  # 64->128
            # 根据需要可能还要加层，取决于你的目标分辨率
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_p_vec):
        # z_p_vec: [B, C] 或 [B, C, 1, 1]
        x = z_p_vec.flatten(1)
        x = self.fc(x)
        x = self.relu(x)

        # Reshape back to map: [B, hidden, 16, 16]
        x = x.view(x.size(0), -1, self.initial_size, self.initial_size)

        x = self.body(x)
        return self.final_conv(x)

#
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x): return self.net(x)
#
#
# class AlbedoHead(nn.Module):
#     def __init__(self, in_ch, hidden=128):
#         super().__init__()
#
#         # 定义一个简单的上采样块: Upsample -> Conv -> BN -> ReLU
#         def _up_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )
#
#         self.net = nn.Sequential(
#             # 输入: [B, in_ch, 24, 24] (如果是384输入)
#             _up_block(in_ch, hidden),  # -> 48x48
#             _up_block(hidden, hidden),  # -> 96x96
#             _up_block(hidden, 64),  # -> 192x192
#             _up_block(64, 32),  # -> 384x384
#
#             # 输出层
#             nn.Conv2d(32, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()  # [0,1]
#         )
#
#     def forward(self, z_p_map):
#         return self.net(z_p_map)
