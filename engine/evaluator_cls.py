import torch
from tqdm import tqdm
import numpy as np
import logging


@torch.no_grad()
def evaluate_cls(model, val_loader, criterion, device):
    """
    CelebA 分类任务评估函数
    返回: Loss, Accuracy (per attribute & mean)
    """
    model.eval()

    total_loss = 0.0
    total_attr_loss = 0.0
    total_recon_loss = 0.0

    # 存储所有预测和真值，用于计算准确率
    all_preds = []
    all_targets = []

    # 使用 Tqdm 进度条
    pbar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in pbar:
        # 数据搬运
        imgs = batch['image'].to(device)
        attrs = batch['attributes'].to(device)

        # 构造输入输出字典
        targets = {
            'image': imgs,
            'attributes': attrs
        }

        # 前向传播
        outputs = model(imgs)

        # 计算 Loss
        loss_val, loss_dict = criterion(outputs, targets)

        total_loss += loss_val.item()
        total_attr_loss += loss_dict.get('attr_loss', 0.0)
        total_recon_loss += loss_dict.get('recon_total', 0.0)

        # 收集预测结果 (Logits -> Prob -> Binary)
        pred_logits = outputs['pred_attr']
        pred_binary = (torch.sigmoid(pred_logits) > 0.5).int().cpu()
        target_binary = attrs.int().cpu()

        all_preds.append(pred_binary)
        all_targets.append(target_binary)

    # 聚合结果
    all_preds = torch.cat(all_preds, dim=0)  # [N, 40]
    all_targets = torch.cat(all_targets, dim=0)  # [N, 40]

    # 计算每个属性的准确率
    # Correct: (Pred == Target)
    correct_mask = (all_preds == all_targets).float()
    acc_per_attr = correct_mask.mean(dim=0)  # [40]
    mean_acc = acc_per_attr.mean().item()  # scalar

    # 计算平均 Loss
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_attr_loss = total_attr_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches

    # 打印日志
    logging.info(f"\n--- Validation Results (CelebA) ---")
    logging.info(f"Total Loss: {avg_loss:.4f} | Attr Loss: {avg_attr_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")
    logging.info(f"Mean Accuracy (mA): {mean_acc * 100:.2f}%")

    # 可选：打印前5个和后5个属性的准确率，看看哪些难学
    # 这里假设我们不知道属性名，只打印索引
    logging.info(f"Top-5 Best Attrs Idx: {torch.topk(acc_per_attr, 5).indices.tolist()}")
    logging.info(f"Top-5 Worst Attrs Idx: {torch.topk(acc_per_attr, 5, largest=False).indices.tolist()}")

    return {
        'val_loss': avg_loss,
        'mean_acc': mean_acc,
        'attr_loss': avg_attr_loss,
        'recon_loss': avg_recon_loss
    }