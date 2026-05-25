import torch
import torch.nn as nn
import lpips


class LPIPSMetric(nn.Module):
    """
    A wrapper for the LPIPS metric to be used within our evaluation pipeline.
    """

    def __init__(self, net='alex'):
        super().__init__()
        # Use net='vgg' for VGG backbone, 'alex' for AlexNet
        self.loss_fn = lpips.LPIPS(net=net)

    def forward(self, pred_img, gt_img):
        """
        Calculates LPIPS distance.
        Args:
            pred_img (torch.Tensor): Predicted image, shape (B, C, H, W), range [0, 1]
            gt_img (torch.Tensor): Ground truth image, shape (B, C, H, W), range [0, 1]

        Note: LPIPS expects inputs to be in range [-1, 1].
        """
        # Rescale from [0, 1] to [-1, 1]
        pred_img_rescaled = pred_img * 2 - 1
        gt_img_rescaled = gt_img * 2 - 1

        distance = self.loss_fn(pred_img_rescaled, gt_img_rescaled)
        return distance.mean()