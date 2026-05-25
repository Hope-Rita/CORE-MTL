import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用现有工程的组件
from losses.linear_cka import LinearCKA
from metrics.lpips import LPIPSMetric


class CelebALoss(nn.Module):
    def __init__(self, loss_config, device='cuda'):
        super().__init__()
        self.weights = loss_config
        self.device = device

        # 1. 任务 Loss: 40个属性的多标签分类
        # BCEWithLogitsLoss 内置了 Sigmoid，数值更稳定
        self.attr_loss_fn = nn.BCEWithLogitsLoss()

        # 2. 重构 Loss: L1 + LPIPS (可选)
        self.recon_l1 = nn.L1Loss()

        # 检查配置是否启用 LPIPS (感知损失对人脸重构很重要)
        self.use_lpips = loss_config.get('enable_lpips', True)
        if self.use_lpips:
            # LPIPSMetric 默认使用 vgg，需要确保 metrics/lpips.py 可用
            self.recon_lpips = LPIPSMetric(net='vgg').to(device)
            # 冻结 LPIPS 参数，防止训练时更新它
            for param in self.recon_lpips.parameters():
                param.requires_grad = False

        # 3. 独立性 Loss: CKA
        self.cka_loss_fn = LinearCKA()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: 字典，包含 pred_attr, recon_img, zs_vec, zp_vec
            targets: 字典，包含 attributes (GT), image (GT)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # ==================================================
        # 1. Attribute Classification Loss
        # ==================================================
        if 'pred_attr' in outputs and 'attributes' in targets:
            pred = outputs['pred_attr']
            gt = targets['attributes'].float()  # BCE 需要 float 目标

            l_attr = self.attr_loss_fn(pred, gt)

            w_attr = self.weights.get('lambda_attr', 1.0)
            total_loss += w_attr * l_attr
            loss_dict['attr_loss'] = l_attr.item()

        # ==================================================
        # 2. Reconstruction Loss
        # ==================================================
        if 'recon_img' in outputs and 'image' in targets:
            recon = outputs['recon_img']
            target_img = targets['image']

            # L1 Loss
            l_rec_l1 = self.recon_l1(recon, target_img)

            # LPIPS Loss
            l_rec_lpips = torch.tensor(0.0, device=self.device)
            if self.use_lpips:
                l_rec_lpips = self.recon_lpips(recon, target_img)

            # 组合重构 Loss
            # 这里的 beta_recon_app 控制总重构权重
            w_rec = self.weights.get('beta_recon_app', 1.0)
            # 内部 L1 和 LPIPS 的比例通常 LPIPS 占主导
            w_l1 = self.weights.get('lambda_l1_recon', 1.0)

            combined_rec = w_l1 * l_rec_l1 + l_rec_lpips
            total_loss += w_rec * combined_rec

            loss_dict['recon_l1'] = l_rec_l1.item()
            loss_dict['recon_lpips'] = l_rec_lpips.item()
            loss_dict['recon_total'] = combined_rec.item()

        # ==================================================
        # 3. Independence Loss (CKA)
        # ==================================================
        # 确保 Zs 和 Zp 在统计上不相关
        if 'zs_vec' in outputs and 'zp_vec' in outputs:
            zs = outputs['zs_vec']
            zp = outputs['zp_vec']

            l_cka = self.cka_loss_fn(zs, zp)

            w_ind = self.weights.get('lambda_independence', 0.0)
            total_loss += w_ind * l_cka
            loss_dict['cka_loss'] = l_cka.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict