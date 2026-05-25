# models/heads/light_head.py
import torch
import torch.nn as nn

class LightHead(nn.Module):
    def __init__(self, in_ch, sh_deg=2):
        super().__init__()
        assert sh_deg == 2, "建议先用二阶SH稳定训练"
        self.out_dim = 3 * (sh_deg + 1) ** 2  # 3*9=27
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.LayerNorm(128),
            nn.Linear(128, self.out_dim)
        )
    def forward(self, g_vec):  # g_vec = GAP(feature_map) or z_s/z_p拼
        return self.mlp(g_vec)  # Bx27
