import torch.nn as nn
import torch
class LinearCKA(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, X, Y):
        # X,Y: [B, D]
        X = (X - X.mean(0)) #/ (X.std(0) + self.eps)
        Y = (Y - Y.mean(0)) #/ (Y.std(0) + self.eps)
        K = X @ X.t()
        L = Y @ Y.t()
        # 中心化
        n = K.size(0)
        H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0/n
        Kc, Lc = H @ K @ H, H @ L @ H
        num = (Kc * Lc).sum()
        den = (Kc.square().sum().sqrt() * Lc.square().sum().sqrt() + self.eps)
        return num / den  # 越大越相关；最小化它以去相关
