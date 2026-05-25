import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_cka import LinearCKA
from metrics.lpips import LPIPSMetric


# ==============================================================================
# 基础工具函数 (仅保留必须的)
# ==============================================================================

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4).clamp(0, 1)


def total_variation_l1(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def cross_covariance_abs(A: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """A: Bx3xHxW, N: Bx3xHxW"""
    B = A.size(0)
    a = A.view(B, 3, -1);
    a = a - a.mean(dim=2, keepdim=True)
    n = N.view(B, 3, -1);
    n = n - n.mean(dim=2, keepdim=True)
    cov = torch.bmm(a, n.transpose(1, 2)) / (a.size(-1) - 1)  # Bx3x3
    return cov.abs().mean()


def rgb_to_lab_safe(x: torch.Tensor) -> torch.Tensor:
    try:
        import kornia as K
        return K.color.rgb_to_lab(x)
    except Exception:
        return x


class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred: [B, 3, H, W]
        pred = F.normalize(pred, p=2, dim=1)
        # gt: [B, 3, H, W]
        # Mask out zero regions in GT
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1)
        dot_prod = (pred * gt).sum(dim=1, keepdim=True)
        num_valid = torch.sum(binary_mask)
        if num_valid > 0:
            loss = 1 - torch.sum(dot_prod * binary_mask) / num_valid
        else:
            loss = torch.tensor(0.0, device=pred.device)
        return loss


class EdgeConsistencyLoss(nn.Module):
    def __init__(self, levels: int = 3, eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.eps = eps
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3).contiguous()
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _grad(self, x: torch.Tensor):
        gx = F.conv2d(x, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.dtype), padding=1)
        return gx, gy

    def _charbonnier(self, x: torch.Tensor):
        return torch.sqrt(x * x + self.eps * self.eps)

    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        if pred_depth.dim() == 3: pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim() == 3:  gt_depth = gt_depth.unsqueeze(1)
        if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
            gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode="nearest")

        pd = torch.log1p(torch.clamp(pred_depth, min=0))
        gd = torch.log1p(torch.clamp(gt_depth, min=0))

        loss = 0.0
        pd_ms, gd_ms = pd, gd
        for _ in range(self.levels):
            gx_p, gy_p = self._grad(pd_ms)
            gx_g, gy_g = self._grad(gd_ms)
            loss = loss + self._charbonnier(gx_p - gx_g).mean() + self._charbonnier(gy_p - gy_g).mean()
            if _ < self.levels - 1:
                pd_ms = F.avg_pool2d(pd_ms, 2, 2)
                gd_ms = F.avg_pool2d(gd_ms, 2, 2)
        return loss


# ==============================================================================
# Composite Loss (瘦身版)
# ==============================================================================

class CompositeLoss(nn.Module):
    def __init__(self, loss_weights, dataset):
        super().__init__()
        self.weights = loss_weights.copy()
        self.dataset = dataset

        # 任务 Loss
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.depth_loss = nn.L1Loss()
        self.normal_loss = NormalLoss()

        # 独立性 Loss
        self.independence_loss = LinearCKA(eps=1e-6)

        # 重构 Loss
        self.recon_geom_loss = nn.L1Loss()
        self.recon_app_loss_lpips = LPIPSMetric(net='vgg')
        self.recon_app_loss_l1 = nn.L1Loss()
        self.edge_consistency_loss = EdgeConsistencyLoss()

    def update_weight(self, key, value):
        if key in self.weights:
            self.weights[key] = value

    def forward(self, outputs, targets):
        loss_dict = {}
        stage = int(outputs.get('stage', 2))
        _dev = outputs['pred_seg'].device

        # ===== 1. 下游任务 Loss (Task Loss) =====
        if stage == 0:
            l_task = torch.tensor(0.0, device=_dev)
            l_seg = l_depth = l_normal = torch.tensor(0.0, device=_dev)
        else:
            # Seg
            l_seg = self.seg_loss(outputs['pred_seg'], targets['segmentation'])

            # Depth (with mask)
            pred_d = outputs['pred_depth']
            gt_d = targets['depth']
            mask = (gt_d > 0).float().detach()
            num_valid = mask.sum()
            if num_valid > 0:
                l_depth = (torch.abs(pred_d - gt_d) * mask).sum() / num_valid
            else:
                l_depth = torch.tensor(0.0, device=_dev)

            # Normal [FIX: 使用 pred_normal]
            if 'pred_normal' in outputs and 'normal' in targets:
                l_normal = self.normal_loss(outputs['pred_normal'], targets['normal'])
            else:
                l_normal = torch.tensor(0.0, device=_dev)

            l_task = (self.weights['lambda_seg'] * l_seg +
                      self.weights['lambda_depth'] * l_depth +
                      self.weights['lambda_normal'] * l_normal)

        loss_dict.update({'seg_loss': l_seg, 'depth_loss': l_depth, 'normal_loss': l_normal, 'task_loss': l_task})

        # ===== 2. 独立性 Loss (CKA: Z_s vs Z_p) =====
        z_s = outputs['z_s']
        z_p = outputs['z_p']

        z_s_c = z_s - z_s.mean(dim=0, keepdim=True)
        z_p_c = z_p - z_p.mean(dim=0, keepdim=True)

        l_ind = self.independence_loss(z_s_c, z_p_c)
        loss_dict['independence_loss'] = l_ind

        if stage <= 1:
            loss_ind_weighted = torch.tensor(0.0, device=_dev)
        else:
            loss_ind_weighted = self.weights['lambda_independence'] * l_ind

        # ===== 3. 重构 Loss (Reconstruction) =====

        # 3.1 几何重构 (Z_s -> Depth)
        if self.weights.get('alpha_recon_geom') > 0:
            l_recon_g = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        else:
            l_recon_g = torch.tensor(0.0, device=_dev)

        # 3.2 外观重构 (Z_s + Z_p -> Image)
        l_recon_a = (self.recon_app_loss_lpips(outputs['recon_app'], targets['appearance_target']) +
                     self.weights['lambda_l1_recon'] * self.recon_app_loss_l1(outputs['recon_app'],
                                                                              targets['appearance_target']))

        # 3.3 物理分解重构 (Albedo * Shading -> Image)
        l_img_decomp = torch.tensor(0.0, device=_dev)
        if outputs.get('recon_decomp') is not None:
            target_img = targets.get('appearance_target', targets.get('image', None))
            l_img_decomp = F.l1_loss(srgb_to_linear(outputs['recon_decomp']), srgb_to_linear(target_img))

        # 辅助正则项 (TV Loss 等)
        A = outputs.get('albedo')
        S = outputs.get('shading')
        Nn = outputs.get('decomposition_normal')

        l_alb_tv = total_variation_l1(A) if A is not None else 0.0
        l_sh_gray = 0.0
        if S is not None:
            l_sh_gray = ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2).mean()

        l_xcov = 0.0
        if A is not None and Nn is not None:
            l_xcov = cross_covariance_abs(A, Nn)

        # 辅助输出 Loss
        l_recon_g_aux = self.recon_geom_loss(outputs['recon_geom_aux'],
                                             F.interpolate(targets['depth'], size=outputs['recon_geom_aux'].shape[2:],
                                                           mode='nearest'))
        l_recon_a_aux = self.recon_app_loss_l1(outputs['recon_app_aux'],
                                               F.interpolate(targets['appearance_target'],
                                                             size=outputs['recon_app_aux'].shape[2:], mode='bilinear'))

        l_edge = torch.tensor(0.0, device=_dev)
        if self.weights.get('lambda_edge_consistency') > 0:
            if 'pred_depth' in outputs and 'depth' in targets:
                l_edge = self.edge_consistency_loss(outputs['pred_depth'], targets['depth'])

        loss_dict.update({
            'recon_geom_loss': l_recon_g,
            'recon_app_loss': l_recon_a,
            'decomp_img': l_img_decomp,
            'edge_loss': l_edge,
        })

        # ===== 4. 总 Loss =====
        total_loss = (
                l_task +
                loss_ind_weighted +
                self.weights['alpha_recon_geom'] * l_recon_g +
                self.weights['beta_recon_app'] * l_recon_a +
                self.weights['lambda_img'] * l_img_decomp +
                self.weights.get('lambda_edge_consistency') * l_edge +
                # 辅助项
                0.5 * l_recon_g_aux +
                0.5 * l_recon_a_aux +
                self.weights.get('lambda_alb_tv') * l_alb_tv +
                self.weights.get('lambda_sh_gray') * l_sh_gray +
                self.weights.get('lambda_xcov') * l_xcov
        )

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict