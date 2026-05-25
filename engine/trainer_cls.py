import torch
import os
import logging
from tqdm import tqdm
from .evaluator_cls import evaluate_cls
from utils.general_utils import save_checkpoint


def train_cls(model, train_loader, val_loader, optimizer, criterion, config, device, checkpoint_dir, vis_dir=None):
    """
    CelebA 专用训练循环
    """
    train_cfg = config['training']
    epochs = int(train_cfg.get('epochs', 50))

    best_acc = 0.0
    start_epoch = 0

    # 学习率调度器 (简单的 Cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    for epoch in range(start_epoch, epochs):
        model.train()

        running_loss = 0.0
        running_attr_loss = 0.0
        running_cka_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch in pbar:
            # 1. 数据准备
            imgs = batch['image'].to(device)
            attrs = batch['attributes'].to(device)

            targets = {
                'image': imgs,
                'attributes': attrs
            }

            # 2. 前向传播
            outputs = model(imgs)

            # 3. 计算 Loss
            loss_val, loss_dict = criterion(outputs, targets)

            # 4. 反向传播
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # 5. 记录
            running_loss += loss_val.item()
            running_attr_loss += loss_dict.get('attr_loss', 0.0)
            running_cka_loss += loss_dict.get('cka_loss', 0.0)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_val.item():.3f}",
                'Attr': f"{loss_dict.get('attr_loss', 0):.3f}",
                'CKA': f"{loss_dict.get('cka_loss', 0):.3f}"
            })

        # 6. Epoch 结束，更新 LR
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        # 计算平均 Loss
        epoch_loss = running_loss / len(train_loader)
        epoch_attr_loss = running_attr_loss / len(train_loader)

        logging.info(
            f"\nEpoch {epoch + 1} | LR: {cur_lr:.6f} | Train Loss: {epoch_loss:.4f} (Attr: {epoch_attr_loss:.4f})")

        # 7. 验证 (Validation)
        # 每隔 1 个 epoch 验证一次，或者你可以设置为每 5 个
        if (epoch + 1) % 1 == 0:
            val_metrics = evaluate_cls(model, val_loader, criterion, device)

            # 8. 保存模型
            curr_acc = val_metrics['mean_acc']
            is_best = curr_acc > best_acc

            if is_best:
                best_acc = curr_acc
                logging.info(f"🏆 New Best Accuracy: {best_acc * 100:.2f}%")

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, is_best, checkpoint_dir)

    logging.info(f"Training Finished. Best Validation Accuracy: {best_acc * 100:.2f}%")