import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import os, logging
import numpy as np
from contextlib import contextmanager

from .evaluator import evaluate
from utils.general_utils import save_checkpoint


# ----------------------------
# Utils & Helpers
# ----------------------------

@contextmanager
def bn_protection_mode(model):
    """
    CFA 上下文管理器：
    在处理反事实样本时，临时将模型切换到 eval 模式（针对 BN）。
    防止分布偏移的 'Frankenstein' 特征污染全局 BN 统计量。
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _set_requires_grad(module, requires_grad: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def _switch_stage_freeze(model, stage: int):
    """
    根据 Stage 切换冻结状态
    """
    if not hasattr(model, 'projector_p'): return

    # Stage 0: Decomposition Warmup (冻结任务头)
    if stage == 0:
        _set_requires_grad(model.head_seg, False)
        _set_requires_grad(model.head_depth, False)
        _set_requires_grad(model.head_normal, False)

        # 确保分解部分解冻
        _set_requires_grad(model.projector_s, True)
        _set_requires_grad(model.projector_p, True)
        _set_requires_grad(model.albedo_head, True)
        _set_requires_grad(model.normal_head, True)
        _set_requires_grad(model.light_head, True)
        _set_requires_grad(model.decoder_app, True)
        # logging.info("❄️ Stage-0: Task Heads FROZEN.")

    # Stage 1 & 2: Full Joint Training
    else:
        _set_requires_grad(model.head_seg, True)
        _set_requires_grad(model.head_depth, True)
        _set_requires_grad(model.head_normal, True)
        _set_requires_grad(model.projector_s, True)
        _set_requires_grad(model.projector_p, True)
        # logging.info(f"🔥 Stage-{stage}: All components UNFROZEN.")


def _get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_scheduler(optimizer, train_cfg):
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    sched_cfg = train_cfg.get("lr_scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "cosine")).lower()

    if sched_type == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))
        min_lr_factor = float(sched_cfg.get("min_lr_factor", 0.1))
        total_epochs = int(train_cfg.get("epochs", 30))
        t_max = int(sched_cfg.get("T_max", max(1, total_epochs - warmup_epochs)))
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=base_lr * min_lr_factor)
        return {"type": "cosine", "warmup_epochs": warmup_epochs, "base_lr": base_lr, "cosine": cosine}

    step_size = int(sched_cfg.get("step_size", 100))
    gamma = float(sched_cfg.get("gamma", 0.5))
    step = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return {"type": "step", "step": step}


def calculate_improvement(base_metrics, current_metrics, data_type='nyuv2'):
    improvement = 0
    count = 0
    # 1=越大越好, 0=越小越好
    metric_meta = {
        'seg_miou': 1, 'seg_pixel_acc': 1,
        'depth_abs_err': 0, 'depth_rel_err': 0,
        'normal_mean_angle': 0, 'normal_acc_30': 1,
        'normal_median_angle': 0, 'normal_acc_11': 1, 'normal_acc_22': 1
    }

    # 根据数据集类型过滤有效指标
    valid_keys = set(metric_meta.keys())
    if 'gta5' in data_type:
        valid_keys = {'seg_miou', 'seg_pixel_acc', 'depth_abs_err', 'depth_rel_err'}
    elif data_type == 'cityscapes':
        valid_keys = {'seg_miou', 'seg_pixel_acc', 'depth_abs_err', 'depth_rel_err'}

    for k, direction in metric_meta.items():
        if k not in valid_keys: continue
        if k in base_metrics and k in current_metrics:
            base = base_metrics[k]
            curr = current_metrics[k]
            if base == 0: continue
            if direction == 1:
                imp = (curr - base) / base
            else:
                imp = (base - curr) / base
            improvement += imp
            count += 1
    return improvement / max(1, count)


# ----------------------------
# Core Training Functions
# ----------------------------

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage: int, config):
    model.train()

    # ==========================================================================
    # 1. 鲁棒的 CFA 配置读取 (Robust Configuration Parsing)
    #    原则：显式优于隐式，拒绝默认值。
    # ==========================================================================
    train_cfg = config.get('training', {})
    cfa_cfg = train_cfg.get('cfa', None)  # 获取 CFA 配置块，如果没写则为 None

    cfa_enabled = False  # 默认为 False，除非配置文件显式开启

    # 只有当配置文件里写了 'cfa' 块时，才进行检查
    if cfa_cfg is not None:
        # [Rule 1] 强制要求 'enabled' 字段
        if 'enabled' not in cfa_cfg:
            raise ValueError(
                "❌ Config Error: Found 'cfa' block in config, but 'enabled' key is missing.\n"
                "   You MUST explicitly set 'enabled: true' or 'enabled: false'."
            )

        cfa_enabled = cfa_cfg['enabled']

        # [Rule 2] 只有开启时才检查参数，且不允许缺省
        if cfa_enabled:
            required_params = ['start_epoch', 'prob', 'lambda_cfa', 'mix_strategy']
            missing_params = [k for k in required_params if k not in cfa_cfg]

            if missing_params:
                raise ValueError(
                    f"❌ Config Error: CFA is enabled, but the following required parameters are missing: {missing_params}.\n"
                    "   We do not use default values. Please specify them in your yaml file."
                )

            # 安全读取 (既然通过了上面的检查，这里一定有值)
            cfa_start_epoch = cfa_cfg['start_epoch']
            cfa_prob = cfa_cfg['prob']
            lambda_cfa = cfa_cfg['lambda_cfa']
            cfa_mix_strategy = cfa_cfg['mix_strategy']

    # ==========================================================================

    # --- 指标追踪器 ---
    metrics_tracker = {
        "main_loss": [],
        "cfa_loss": [],
        "cfa_ratio": [],
        "z_drift": []
    }

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
    accumulation_steps = 1
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(pbar):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rgb = batch['rgb']

        # ==========================
        # Step 1: Main Forward (Pass 1)
        # ==========================
        outputs = model(rgb, stage=stage)
        crit_out = criterion(outputs, batch)

        if isinstance(crit_out, (tuple, list)):
            loss_main, loss_dict = crit_out
        else:
            loss_dict = crit_out
            loss_main = loss_dict['total_loss']

        # 记录基础指标
        current_cka = loss_dict.get('independence_loss', torch.tensor(1.0)).item()
        metrics_tracker["main_loss"].append(loss_main.item())

        # [OOM Fix] 立即反向传播释放显存
        loss_main.backward()

        # ==========================
        # Step 2: CFA Forward (Pass 2)
        # ==========================
        loss_cfa_val = 0.0
        cfa_active = False
        cfa_diagnostics = {}

        # 准入条件判断
        cond_stage = stage >= 2

        # 只有当 cfa_enabled 为 True 时，这些变量才会被定义，所以这里是安全的
        if cfa_enabled:
            cond_epoch = epoch >= cfa_start_epoch
            cond_batch = rgb.size(0) > 1

            should_run_cfa = cond_stage and cond_epoch and cond_batch

            if should_run_cfa and (torch.rand(1).item() < cfa_prob):
                cfa_active = True

                # A. 生成反事实样本 (detach!)
                z_s_map = outputs['z_s_map'].detach()
                z_p_map = outputs['z_p_map'].detach()

                I_cfa, diag_stats = model.generate_counterfactual_image(
                    z_s_map, z_p_map, strategy=cfa_mix_strategy
                )
                I_cfa = I_cfa.detach()
                metrics_tracker["z_drift"].append(diag_stats.get('mix_std', 0.0))

                # B. 反事实前向传播 (BN Protected!)
                with bn_protection_mode(model):
                    out_cfa = model(I_cfa, stage=stage)

                    # C. 计算 CFA Loss
                    _, cfa_loss_dict = criterion(out_cfa, batch)
                    l_cfa_task = cfa_loss_dict.get('task_loss', torch.tensor(0.0))
                    l_cfa_ind = cfa_loss_dict.get('independence_loss', torch.tensor(0.0))

                    loss_cfa = lambda_cfa * (l_cfa_task + 0.1 * l_cfa_ind)

                # [OOM Fix] 第二次反向传播
                loss_cfa.backward()

                # Probe
                main_task_loss = loss_dict.get('task_loss', torch.tensor(1.0)).item()
                cfa_task_loss_val = l_cfa_task.item()
                ratio = cfa_task_loss_val / (main_task_loss + 1e-6)
                metrics_tracker["cfa_ratio"].append(ratio)

                loss_cfa_val = loss_cfa.item()
                cfa_diagnostics = {
                    "Ratio": f"{ratio:.1f}x",
                    "MixStd": f"{diag_stats.get('mix_std', 0.0):.2f}"
                }

        # ==========================
        # Step 3: Optimize & Log
        # ==========================

        if cfa_active and (i % 50 == 0 or metrics_tracker["cfa_ratio"][-1] > 3.0):
            msg = f"[Iter {i}] MainL:{loss_main.item():.3f} CKA:{current_cka:.3f} | [CFA ON] CFAL:{loss_cfa_val:.3f} Ratio:{cfa_diagnostics['Ratio']} Z_Std:{cfa_diagnostics['MixStd']}"
            if float(cfa_diagnostics['Ratio'][:-1]) > 3.0:
                logging.warning(f"⚠️ [Probe Alert] High CFA Confusion! {msg}")
            else:
                logging.info(msg)

        if i % 50 == 0:
            logging.info(f"\n[Epoch {epoch + 1}][Iter {i}] 🔍 Loss Breakdown:")
            logging.info(f"  > Task Seg   : {loss_dict.get('seg_loss', 0):.4f}")
            logging.info(f"  > Task Depth : {loss_dict.get('depth_loss', 0):.4f}")
            logging.info(f"  > Indep (CKA): {loss_dict.get('independence_loss', 0):.4f} (Raw Value)")
            logging.info(f"  > Recon Geom : {loss_dict.get('recon_geom_loss', 0):.4f}")
            logging.info(f"  > Recon App  : {loss_dict.get('recon_app_loss', 0):.4f}")
            logging.info(f"  > Decomp L1  : {loss_dict.get('decomp_img', 0):.4f}")
            logging.info(f"  > Total Loss : {loss_main.item():.4f}")

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Update Pbar
        pf = {
            'L': f"{loss_main.item():.2f}",
            'Seg': f"{loss_dict.get('seg_loss', 0):.4f}",  # 新增
            'Dep': f"{loss_dict.get('depth_loss', 0):.4f}",  # 新增
            'CKA': f"{current_cka:.4f}"
        }
        if cfa_active:
            pf['CFA'] = f"{loss_cfa_val:.2f}"
        pbar.set_postfix(pf)

    # --- Epoch Summary ---
    avg_main = np.mean(metrics_tracker["main_loss"])
    avg_cfa = np.mean(metrics_tracker["cfa_loss"]) if metrics_tracker["cfa_loss"] else 0.0
    avg_ratio = np.mean(metrics_tracker["cfa_ratio"]) if metrics_tracker["cfa_ratio"] else 0.0

    logging.info(f"Epoch {epoch + 1} Report:")
    logging.info(f"  > Avg Main Loss : {avg_main:.4f}")
    if cfa_enabled and len(metrics_tracker["cfa_loss"]) > 0:
        logging.info(f"  > Avg CFA Loss  : {avg_cfa:.4f}")
        logging.info(f"  > Avg Confusion : {avg_ratio:.2f}x")

    return avg_main


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device,
          checkpoint_dir='checkpoints', val_loader_source=None):
    """
    Train function supporting dual validation (Target & Source).
    """
    data_type = config['data'].get('type', 'nyuv2').lower()
    train_cfg = config['training']

    stage0_epochs = int(train_cfg.get('stage0_epochs', 0))
    stage1_epochs = int(train_cfg.get('stage1_epochs', 0))
    total_epochs = int(train_cfg.get('epochs', 30))
    base_lr = float(train_cfg.get("learning_rate", 1e-4))

    ind_warmup_epochs = int(train_cfg.get('ind_warmup_epochs', 0))
    target_ind_lambda = float(config['losses'].get('lambda_independence', 0.0))

    best_relative_score = -float('inf')

    # [NEW] 记录源域最佳分数和指标
    best_score_src = -float('inf')
    best_metrics_src = {}

    baseline_metrics = None
    best_epoch = 0
    baseline_metrics_src = None

    # [NEW] 计算 Stage 2 正式开始的 Epoch 索引
    stage2_start_epoch = stage0_epochs + stage1_epochs

    sched = _build_scheduler(optimizer, train_cfg)
    logging.info(f"[LR Scheduler] {sched['type']}; base_lr={base_lr}")

    for epoch in range(total_epochs):
        # --- Stage Logic ---
        if epoch < stage0_epochs:
            stage = 0
        elif epoch < stage1_epochs + stage0_epochs:
            stage = 1
        else:
            stage = 2

        # 每个 Epoch 检查是否需要切换冻结状态
        _switch_stage_freeze(model, stage)

        if epoch == 0:
            logging.info(f"🔍 [Training Strategy] Total Epochs: {total_epochs}")
            logging.info(f"   Stage 0 (Decomp Only): 0 -> {stage0_epochs}")
            logging.info(f"   Stage 2 (Joint Train): {stage0_epochs} -> {total_epochs}")
            logging.info(f"   Baseline will be FIXED at Epoch {stage2_start_epoch + 1} (Start of Stage 2)")

        # --- Lambda Decay/Warmup Strategy ---
        current_ind_lambda = target_ind_lambda

        # 在 Stage 0 强制 lambda 为 0
        if stage == 0:
            current_ind_lambda = 0.0
        # 如果还在 Warmup 期间
        elif ind_warmup_epochs > 0 and (epoch - stage0_epochs) < ind_warmup_epochs:
            progress = epoch - stage0_epochs
            ratio = min(1.0, max(0.0, progress / float(ind_warmup_epochs)))
            current_ind_lambda = target_ind_lambda * ratio
            logging.info(f"   > Ind Loss Warmup: {progress}/{ind_warmup_epochs} = {ratio:.2f}")

        # 更新 Criterion 中的权重
        real_criterion = criterion.module if hasattr(criterion, 'module') else criterion
        if hasattr(real_criterion, 'weights'):
            real_criterion.weights['lambda_independence'] = torch.tensor(current_ind_lambda, device=device)

        # ---- LR Warmup (Cosine only) ----
        if sched["type"] == "cosine":
            warmup_epochs = sched["warmup_epochs"]
            if epoch < warmup_epochs:
                warmup_start = 0.1 * base_lr
                ratio = float(epoch + 1) / float(max(1, warmup_epochs))
                lr_now = warmup_start + (base_lr - warmup_start) * ratio
                _set_lr(optimizer, lr_now)

        cur_lr = _get_lr(optimizer)
        logging.info(f"\n----- Epoch {epoch + 1}/{total_epochs} (Stage {stage}) | lr={cur_lr:.6f} -----")

        # --- Train & Validate ---
        # 目标域
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage=stage,
                                     config=config)
        if data_type=="gta5_to_cityscapes":
            val_metrics = evaluate(model, val_loader, criterion, device,
                                   stage=stage, data_type="cityscapes")
        else:
            val_metrics = evaluate(model, val_loader, criterion, device,
                                   stage=stage,data_type=data_type)

        # --- Scheduler Step ---
        if sched["type"] == "cosine":
            if epoch >= sched["warmup_epochs"]:
                sched["cosine"].step()
        else:
            sched["step"].step()

        # --- Best Model Selection (Fixed Baseline Logic for Target Domain) ---
        is_best = False
        score = 0.0

        # Stage 0 & 1: 不计算 Score，跳过
        if stage < 2:
            logging.info(f"  -> Stage {stage} (Pre-training) - Skipping improvement calculation.")
            baseline_metrics = None  # 确保不使用预训练阶段作为基准

            # 依然保存 checkpoint，方便调试
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_relative_score,'config': config,
            }, False, checkpoint_dir=checkpoint_dir)

        else:
            # Stage 2: 只有进入联合训练阶段才开始评测
            if epoch == stage2_start_epoch:
                # 刚进入 Stage 2 的第一轮 -> 强制锁定为 Baseline
                baseline_metrics = val_metrics
                logging.info(f"  -> 🏁 Stage 2 Started. Setting FIXED BASELINE from current epoch.")

                # 保存一份作为 Stage 2 起点的存档
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': 0.0,
                    'baseline_metrics': baseline_metrics,'config': config,
                }, False, checkpoint_dir=checkpoint_dir, filename='checkpoint_stage2_start.pth.tar')

            elif baseline_metrics is not None:
                # Stage 2 后续轮次 -> 与锁定的 Baseline 比较
                score = calculate_improvement(baseline_metrics, val_metrics, data_type=data_type)
                is_best = (score > best_relative_score)

                if is_best:
                    best_relative_score = score
                    best_epoch = epoch + 1
                    best_metrics_details = val_metrics.copy()
                    logging.info(f"  -> 🏆 Best Model (vs Stage2-Start)! Score: {score:.2%}")

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': best_relative_score,
                    'baseline_metrics': baseline_metrics,'config': config,
                }, is_best, checkpoint_dir=checkpoint_dir)
            else:
                # 防御性代码：如果是断点续训且没加载到 baseline，以当前为准
                logging.info("  -> Warning: No baseline found (resumed?), setting current as baseline.")
                baseline_metrics = val_metrics

        # [NEW] Source Domain Validation Support (Dual Validation)
        if val_loader_source is not None:
            if data_type=="gta5_to_cityscapes":
                logging.info(f"🔍 [Val - Source] Evaluating on Source (GTA5)...")

            # 1. 评估 (Evaluate)
            # 强制传入 'gta5'，evaluator 会输出 seg_miou, seg_pixel_acc, depth_abs_err, depth_rel_err
                val_metrics_src = evaluate(model, val_loader_source, criterion, device,
                                           stage=stage, data_type='gta5')
            if stage >= 2:
                # A. 锁定 Baseline (如果是 Stage 2 第一轮，或者断点续训刚开始)
                if epoch == stage2_start_epoch or baseline_metrics_src is None:
                    baseline_metrics_src = val_metrics_src
                    logging.info(f"  -> 🏁 [Source] Stage 2 Started. Setting FIXED BASELINE from current epoch.")

                # B. 计算综合得分 (Score)
                # calculate_improvement 会自动识别 dict 里的 key:
                # + seg_miou (越大越好)
                # - depth_abs_err (越小越好)
                # Score > 0 表示整体比 Baseline 好
                score_src = calculate_improvement(baseline_metrics_src, val_metrics_src, data_type='gta5')

                # C. 记录最佳模型 (根据综合 Score)
                if score_src > best_score_src:
                    best_score_src = score_src
                    best_metrics_src = val_metrics_src.copy()

                    cur_miou = val_metrics_src.get('seg_miou', 0.0)
                    cur_acc = val_metrics_src.get('seg_pixel_acc', 0.0)  # 新增
                    cur_depth_abs = val_metrics_src.get('depth_abs_err', 0.0)
                    cur_depth_rel = val_metrics_src.get('depth_rel_err', 0.0)  # 新增

                    logging.info(
                        f"  ★ [Source Best] New Best (Score: {best_score_src:.2%}) | "
                        f"mIoU: {cur_miou:.4f} | Pixel Acc: {cur_acc:.4f} | "  # 打印 Seg 指标
                        f"Depth Abs: {cur_depth_abs:.4f} | Depth Rel: {cur_depth_rel:.4f}"  # 打印 Depth 指标
                    )
                    # 保存为独立文件
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_score': best_score_src,
                        'metrics': val_metrics_src,
                        'baseline_metrics': baseline_metrics_src,'config': config,  # 把 Baseline 也存进去，方便续训
                    }, False, checkpoint_dir=checkpoint_dir, filename='model_best_gta5.pth.tar')

    logging.info(f"\n✅ Training Finished. Best Epoch: {best_epoch}, Score: {best_relative_score:.2%}")

    if val_loader_source is not None:
        # [Modified] 打印最优 GTA5 的完整指标：mIoU, Pixel Acc, Abs Err, Rel Err
        final_src_miou = best_metrics_src.get('seg_miou', 0.0)
        final_src_acc = best_metrics_src.get('seg_pixel_acc', 0.0)
        final_src_depth_abs = best_metrics_src.get('depth_abs_err', 0.0)
        final_src_depth_rel = best_metrics_src.get('depth_rel_err', 0.0)

        logging.info(
            f"   Best Source Result -> mIoU: {final_src_miou:.4f} | Pixel Acc: {final_src_acc:.4f} | Depth Abs: {final_src_depth_abs:.4f} | Depth Rel: {final_src_depth_rel:.4f}")