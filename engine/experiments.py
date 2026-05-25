# engine/experiments.py
import torch, logging, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

@torch.no_grad()
def _rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt)).item()

@torch.no_grad()
def _color_augment_batch(x):
    # x: [B,3,H,W] tensor; 对每张图做强外观扰动（不改变几何）
    out = []
    for i in range(x.size(0)):
        img = x[i]
        # brightness/contrast/saturation/hue
        img = TF.adjust_brightness(img,  random.uniform(0.6, 1.4))
        img = TF.adjust_contrast(img,    random.uniform(0.6, 1.4))
        img = TF.adjust_saturation(img,  random.uniform(0.6, 1.4))
        img = TF.adjust_hue(img,         random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        out.append(torch.clamp(img, -1.0, 1.0))  # 依据你数据预处理范围适当调整
    return torch.stack(out, 0)

@torch.no_grad()
def swap_test(model, val_loader: DataLoader, device, max_batches: int = 8):
    """
    交换 z_p_depth：对 batch 做一个循环位移配对 (A,B)，
    比较 Full(A) / z_s-only(A) / Swap(A<-z_p^B) 相对 GT_A 的 RMSE。
    """
    model.eval()
    n, d_full, d_zs, d_swap = 0, 0.0, 0.0, 0.0
    hf_improve_cnt = 0

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches: break
        rgb = batch['rgb'].to(device)
        gt  = batch['depth'].to(device)

        # A 与 B（循环移位）
        rgbA, rgbB = rgb, torch.roll(rgb, shifts=1, dims=0)

        outA_full = model(rgbA, stage=2)
        outB_full = model(rgbB, stage=2)

        # A 的 z_s-only
        outA_zs = model(rgbA, stage=1)  # 关闭 FiLM 与残差
        predA_full = outA_full['pred_depth']
        predA_zs   = outA_zs['pred_depth']

        # A 用 B 的 z_p_depth_map
        outA_swap = model(
            rgbA, stage=2,
            override_zp_depth_map=outB_full['z_p_depth_map'].detach()
        )
        predA_swap = outA_swap['pred_depth']

        d_full += _rmse(predA_full, gt)
        d_zs   += _rmse(predA_zs,   gt)
        d_swap += _rmse(predA_swap, gt)
        n += 1

    d_full /= max(n,1); d_zs /= max(n,1); d_swap /= max(n,1)
    logging.info("\n[Swap Test] (A uses z_p from B)")
    logging.info(f"  RMSE Full(A):   {d_full:.4f}")
    logging.info(f"  RMSE z_s-only:  {d_zs:.4f}  (Δ={d_zs - d_full:+.4f} vs Full)")
    logging.info(f"  RMSE Swap(A<-B):{d_swap:.4f} (Δ={d_swap - d_full:+.4f} vs Full)")
    logging.info("  判据: 若 Δ(Swap) ≪ Δ(z_s-only)，更符合可解耦；若相近或更差，说明 z_p 携带低频几何或被忽略。")
    return {"rmse_full": d_full, "rmse_zs": d_zs, "rmse_swap": d_swap}

@torch.no_grad()
def appearance_invariance_test(model, val_loader: DataLoader, device, max_batches: int = 8):
    """
    外观强扰动前后：z_s / z_p 的稳定性 + RMSE 退化。
    """
    model.eval()
    n = 0
    zs_dist, zp_dist = 0.0, 0.0
    rmse_full_orig, rmse_full_aug = 0.0, 0.0
    rmse_zs_orig,   rmse_zs_aug   = 0.0, 0.0

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches: break
        rgb = batch['rgb'].to(device)
        gt  = batch['depth'].to(device)
        rgb_aug = _color_augment_batch(rgb)

        # Full & z_s-only
        out_o_full = model(rgb,      stage=2)
        out_a_full = model(rgb_aug,  stage=2)
        out_o_zs   = model(rgb,      stage=1)
        out_a_zs   = model(rgb_aug,  stage=1)

        # latent 距离（用 pooled 向量，稳定些）
        zs_dist += F.mse_loss(out_o_full['z_s'],      out_a_full['z_s']).item()
        zp_dist += F.mse_loss(out_o_full['z_p_depth'],out_a_full['z_p_depth']).item()

        rmse_full_orig += _rmse(out_o_full['pred_depth'], gt)
        rmse_full_aug  += _rmse(out_a_full['pred_depth'], gt)
        rmse_zs_orig   += _rmse(out_o_zs['pred_depth'],   gt)
        rmse_zs_aug    += _rmse(out_a_zs['pred_depth'],   gt)
        n += 1

    zs_dist /= max(n,1); zp_dist /= max(n,1)
    rmse_full_orig /= max(n,1); rmse_full_aug /= max(n,1)
    rmse_zs_orig   /= max(n,1); rmse_zs_aug   /= max(n,1)

    logging.info("\n[Appearance Invariance Test]")
    logging.info(f"  ||z_s(x)-z_s(x')||^2: {zs_dist:.6f}   (越小越稳定)")
    logging.info(f"  ||z_p(x)-z_p(x')||^2: {zp_dist:.6f}   (应显著大于 z_s 的变化)")
    logging.info(f"  RMSE Full(orig→aug):  {rmse_full_orig:.4f} → {rmse_full_aug:.4f} (Δ={rmse_full_aug-rmse_full_orig:+.4f})")
    logging.info(f"  RMSE z_s-only(orig→aug): {rmse_zs_orig:.4f} → {rmse_zs_aug:.4f} (Δ={rmse_zs_aug-rmse_zs_orig:+.4f})")
    logging.info("  判据: z_s 对外观更稳定、Full 在域转移下优于或接近 z_s-only，则更符合可解耦。")
    return {
        "zs_l2": zs_dist, "zp_l2": zp_dist,
        "rmse_full_orig": rmse_full_orig, "rmse_full_aug": rmse_full_aug,
        "rmse_zs_orig": rmse_zs_orig, "rmse_zs_aug": rmse_zs_aug
    }

@torch.no_grad()
def cross_env_generalization_test(model, val_loader: DataLoader, device, max_batches: int = 20):
    """
    在“外观域”上评估 Full 与 z_s-only 的退化。
    """
    model.eval()
    n = 0
    full_rmse_src, full_rmse_aug = 0.0, 0.0
    zs_rmse_src,   zs_rmse_aug   = 0.0, 0.0

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches: break
        rgb = batch['rgb'].to(device)
        gt  = batch['depth'].to(device)
        rgb_aug = _color_augment_batch(rgb)

        full_rmse_src += _rmse(model(rgb,     stage=2)['pred_depth'], gt)
        full_rmse_aug += _rmse(model(rgb_aug, stage=2)['pred_depth'], gt)
        zs_rmse_src   += _rmse(model(rgb,     stage=1)['pred_depth'], gt)
        zs_rmse_aug   += _rmse(model(rgb_aug, stage=1)['pred_depth'], gt)
        n += 1

    full_rmse_src /= max(n,1); full_rmse_aug /= max(n,1)
    zs_rmse_src   /= max(n,1); zs_rmse_aug   /= max(n,1)
    logging.info("\n[Cross-Env Generalization]")
    logging.info(f"  Full  RMSE: {full_rmse_src:.4f} → {full_rmse_aug:.4f} (Δ={full_rmse_aug-full_rmse_src:+.4f})")
    logging.info(f"  z_s   RMSE: {zs_rmse_src:.4f} → {zs_rmse_aug:.4f}   (Δ={zs_rmse_aug-zs_rmse_src:+.4f})")
    logging.info("  判据: Full 在增强域的退化不应显著劣于 z_s-only；若反而更差，z_p 可能带来了不稳定外观因素。")
    return {
        "full_src": full_rmse_src, "full_aug": full_rmse_aug,
        "zs_src": zs_rmse_src, "zs_aug": zs_rmse_aug
    }

@torch.no_grad()
def run_all_experiments(model, val_loader, device, max_batches_swap=8, max_batches_inv=8, max_batches_cross=20):
    res1 = swap_test(model, val_loader, device, max_batches=max_batches_swap)
    res2 = appearance_invariance_test(model, val_loader, device, max_batches=max_batches_inv)
    res3 = cross_env_generalization_test(model, val_loader, device, max_batches=max_batches_cross)
    return {"swap": res1, "invariance": res2, "cross_env": res3}
