import yaml, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os, argparse
import logging
from datetime import datetime

# --- 基础工具 (复用现有) ---
from utils.general_utils import set_seed, setup_logging

# --- [新] CelebA 专用模块导入 (计划在后续步骤生成) ---
# 注意：这些文件目前还不存在，是我们在"平行宇宙"中即将创建的
try:
    from data_utils.celeba_dataset import CelebADataset
    from models.causal_celeba_model import CausalCelebAModel
    from losses.celeba_loss import CelebALoss
    from engine.trainer_cls import train_cls
except ImportError:
    print("⚠️ 警告: CelebA 专用模块尚未完全生成。请按照计划生成后续文件。")


def main(config_path):
    """
    CelebA 因果解耦实验入口 (40 Attributes + Reconstruction)
    """
    # 1. 加载配置并设置随机种子
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        set_seed(config['training']['seed'])
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return

    # 设置日志目录
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = os.path.join('runs_celeba', timestamp)  # 区分于原来的 runs
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    vis_dir = os.path.join(run_dir, 'visualizations')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    setup_logging(run_dir)
    logging.info(f"🚀 CelebA Causal Experiment Started: {timestamp}")
    logging.info(f"📂 Output Directory: {run_dir}")
    logging.info("=" * 60)
    logging.info(json.dumps(config, indent=4, default=str))
    logging.info("=" * 60)

    # 2. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"⚙️ Using device: {device}")

    # 3. 初始化数据集 (CelebA)
    logging.info("\n📚 Initializing CelebA Dataset...")
    data_cfg = config['data']
    target_num_attr = data_cfg.get('num_attributes')

    # 训练集
    train_dataset = CelebADataset(
        root_dir=data_cfg['dataset_path'],
        split='train',
        img_size=data_cfg.get('img_size', [128, 128]),
        num_attributes=target_num_attr,
        augmentation=True
    )

    # 验证集
    val_dataset = CelebADataset(
        root_dir=data_cfg['dataset_path'],
        split='val',  # 或 'test'
        img_size=data_cfg.get('img_size', [128, 128]),
        num_attributes=target_num_attr,
        augmentation=False
    )

    logging.info(f"   Train samples: {len(train_dataset)}")
    logging.info(f"   Val   samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True
    )

    # 4. 初始化模型 (CausalCelebAModel)
    logging.info("\n🧠 Initializing Model (ResNet18-Based Causal)...")
    model = CausalCelebAModel(config['model']).to(device)

    # 5. 优化器 & Loss
    train_cfg = config['training']
    base_lr = float(train_cfg['learning_rate'])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get('weight_decay', 1e-4))
    )

    # 专用 Loss 模块 (BCE + Recon + CKA)
    criterion = CelebALoss(config['losses'], device=device)

    # 6. 启动训练 (调用新的 trainer_cls)
    logging.info("\n🔥 Starting Training Loop...")
    if train_cfg.get('enable_training', True):
        train_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            vis_dir=vis_dir  # 传入可视化目录，方便训练中途看图
        )
    else:
        logging.info("🛑 Training disabled in config.")

    logging.info("\n✅ All Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认指向新的配置文件
    parser.add_argument('--config', type=str, default='configs/celeba/resnet18_40attr.yaml')
    args = parser.parse_args()
    main(args.config)