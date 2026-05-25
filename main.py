import yaml, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os, argparse
import logging
from datetime import datetime

# --- 数据集导入 ---
from data_utils.nyuv2_dataset import NYUv2Dataset
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset
from data_utils.cityscapes_c_dataset import CityscapesCDataset

# --- 模型与Loss导入 (仅保留 Causal 核心) ---
from models.causal_model import CausalMTLModel
from losses.composite_loss import CompositeLoss


# --- 引擎工具导入 ---
from engine.trainer import train
from engine.visualizer import generate_visual_reports
from engine.evaluator import evaluate
from engine.experiments import run_all_experiments
from utils.general_utils import set_seed, setup_logging


def main(config_path):
    """
    项目主函数（支持双验证集 Dual Validation）。
    """
    # 1. 加载配置并设置随机种子
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        set_seed(config['training']['seed'])
    except Exception as e:
        logging.info(f"❌ Error loading config file: {e}")
        return

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = os.path.join('runs', timestamp)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    vis_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    setup_logging(run_dir)
    logging.info("✅ Configuration loaded successfully.")
    logging.info(f"📂 All outputs for this run will be saved in: {run_dir}")
    logging.info("=" * 60)
    logging.info("🔧 Final Execution Configuration:")
    logging.info(json.dumps(config, indent=4, default=str))
    logging.info("=" * 60)

    # 2. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Using device: {device}")

    # 3. 初始化数据集和数据加载器
    logging.info("\nInitializing dataset...")
    try:
        data_cfg = config['data']
        dataset_type = data_cfg.get('type', 'nyuv2').lower()
        img_size = tuple(data_cfg['img_size'])
        dataset_path = data_cfg.get('dataset_path')

        logging.info(f"📋 Dataset Type: {dataset_type}")


        val_dataset_src = None  # 源域验证集 (Source) - 可选

        # === 数据集加载逻辑 ===
        if dataset_type == 'gta5_to_cityscapes':
            logging.info("🌍 Mode: GTA5 -> Cityscapes (Dual Validation)")

            # 1. 训练集 (GTA5 Train) - 开启增强
            train_dataset = GTA5Dataset(
                root_dir=data_cfg['train_dataset_path'],
                img_size=img_size,
                augmentation=True  # <--- 训练开启增强
            )

            # 2. 目标域验证集 (Cityscapes Val)
            val_dataset_tgt = CityscapesDataset(
                root_dir=data_cfg['val_dataset_path'],
                split='val'
            )

            # 3. 源域验证集 (GTA5 Val) - 关闭增强
            # 只有在 config 中提供了 source_val_path 才加载
            if 'source_val_path' in data_cfg:
                val_dataset_src = GTA5Dataset(
                    root_dir=data_cfg['source_val_path'],
                    img_size=img_size,
                    augmentation=False  # <--- 验证必须关闭增强
                )

        elif dataset_type == 'cityscapes':
            logging.info("🌍 Mode: Cityscapes")
            train_dataset = CityscapesDataset(root_dir=dataset_path, split='train')
            val_dataset_tgt = CityscapesDataset(root_dir=dataset_path, split='val')

        elif dataset_type == 'nyuv2':
            logging.info("🏠 Mode: NYUv2")
            train_dataset = NYUv2Dataset(root_dir=dataset_path, mode='train',
                                         augmentation=data_cfg.get('augmentation', False))
            val_dataset_tgt = NYUv2Dataset(root_dir=dataset_path, mode='val')

        else:
            raise ValueError(f"❌ Unsupported dataset type: '{dataset_type}'")

        # DataLoader 设置
        pin_memory = data_cfg.get('pin_memory', torch.cuda.is_available())

        # 训练 Loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory,
            drop_last=True
        )

        # 目标域验证 Loader (默认)
        val_loader_tgt = DataLoader(
            val_dataset_tgt,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=pin_memory
        )

        # 源域验证 Loader (可选)
        val_loader_src = None
        if val_dataset_src is not None:
            val_loader_src = DataLoader(
                val_dataset_src,
                batch_size=data_cfg['batch_size'],
                shuffle=False,
                num_workers=data_cfg['num_workers'],
                pin_memory=pin_memory
            )
            logging.info(f"📚 Dual Validation Enabled: Source (GTA5) & Target (Cityscapes)")

        logging.info(f"📚 Dataset loaded: {len(train_dataset)} training, {len(val_dataset_tgt)} target val samples.")

    except Exception as e:
        logging.info(f"❌ Error creating dataset/loaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 初始化模型 (CausalMTLModel Only)
    logging.info("\nInitializing CausalMTLModel...")
    base_lr = float(config['training']['learning_rate'])

    # 直接实例化 CausalMTLModel
    model = CausalMTLModel(config['model'], config['data']).to(device)

    # 参数分组：Backbone vs Heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # 优化器配置
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': base_lr},
        {'params': head_params, 'lr': base_lr}
    ], lr=base_lr, weight_decay=config['training']['weight_decay'])

    # Loss 配置 (仅使用 CompositeLoss)
    criterion = CompositeLoss(config['losses'].copy(), dataset_type).to(device)

    logging.info(f"🔧 Optimizer: {config['training']['optimizer']}, LR: {base_lr}")

    # 5. 学习率调度器 (Trainer 内部构建，此处传 None)
    scheduler = None

    # 6. 启动训练
    logging.info("\n----- Starting Training -----")
    if config['training'].get('enable_training', True):
        # 注意：这里传递了两个验证 Loader
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader_tgt,  # 目标域 (Cityscapes)
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            val_loader_source=val_loader_src  # 源域 (GTA5) - 可选
        )
    else:
        logging.info("🏃 Training is disabled in config.")

    # 7. 实验性分析
    exp_cfg = config.get('experiments', {})
    if exp_cfg.get('enable', False):
        logging.info("\n===== Running experiments =====")
        model.eval()
        run_all_experiments(model, val_loader_tgt, device)

    # 8. 可视化 (使用目标域数据)
    logging.info("\n----- Running Final Visualizations -----")
    best_ckpt = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_ckpt):
        try:
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            generate_visual_reports(model, val_loader_tgt, device, save_dir=vis_dir, num_reports=3)
        except Exception as e:
            logging.info(f"⚠️ Visualization failed: {e}")

    if hasattr(train_dataset, "close"): train_dataset.close()
    if hasattr(val_dataset_tgt, "close"): val_dataset_tgt.close()

    logging.info("\n🎉 Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    main(args.config)