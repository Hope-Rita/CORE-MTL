import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# 请确保 building_blocks.py 在同级目录下
from .building_blocks import ViTEncoder, ResNetEncoder, ResNetDecoderWithDeepSupervision
from .heads.albedo_head import AlbedoHead
from .heads.normal_head import NormalHead
from .heads.light_head import LightHead
from .layers.shading import shading_from_normals


# ==============================================================================
# 1. 基础组件 (ASPP)
# ==============================================================================

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# ==============================================================================
# 2. 通用任务预测头 (TaskHead)
# 严格限制：只接收 main_feat (Z_s) 和 low_level_feat，绝不接收 Z_p
# ==============================================================================

class TaskHead(nn.Module):
    def __init__(self, main_in_channels: int, out_channels: int,
                 low_level_in_channels: int = 256,
                 use_sigmoid=False):
        super().__init__()

        # 1. 上下文提取
        aspp_out = 256
        self.aspp = ASPP(in_channels=main_in_channels, out_channels=aspp_out)

        # 2. Skip Connection 投影
        self.low_level_proj_channels = 48
        self.project_low_level = nn.Sequential(
            nn.Conv2d(low_level_in_channels, self.low_level_proj_channels, 1, bias=False),
            nn.GroupNorm(4, self.low_level_proj_channels),
            nn.ReLU(inplace=True)
        )

        # 3. 最终预测卷积
        head_in_channels = aspp_out + self.low_level_proj_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, 1)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, z_s_feat, low_level_feat=None):
        # 1. 处理 Z_s
        x = self.aspp(z_s_feat)

        # 2. 融合 Low Level 细节
        if low_level_feat is not None:
            low = self.project_low_level(low_level_feat)
            x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, low], dim=1)

        # 3. 输出
        out = self.head(x)

        scale_factor = 4 if (low_level_feat is not None) else 8
        out = F.interpolate(out, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out


# ==============================================================================
# 3. 主模型 (CausalMTLModel)
# ==============================================================================
class CausalMTLModel(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config
        self.data_config = data_config

        # --- Encoder ---
        encoder_name = model_config['encoder_name']
        if 'resnet' in encoder_name:
            self.encoder = ResNetEncoder(name=encoder_name, pretrained=model_config['pretrained'], dilated=True)
            target_dim = 1024
            self.resnet_adapters = nn.ModuleList([
                nn.Conv2d(in_c, target_dim, kernel_size=1) for in_c in self.encoder.feature_dims
            ])
            encoder_feature_dim = target_dim
        else:
            self.encoder = ViTEncoder(name=encoder_name, pretrained=model_config['pretrained'],
                                      img_size=data_config['img_size'][0])
            encoder_feature_dim = self.encoder.feature_dim
            self.resnet_adapters = None

        self.latent_dim_s = model_config['latent_dim_s']
        self.latent_dim_p = model_config['latent_dim_p']

        combined_feature_dim = encoder_feature_dim * 4

        # --- Disentanglement Projectors ---
        # Z_s: 结构/语义
        self.projector_s = nn.Conv2d(combined_feature_dim, self.latent_dim_s, kernel_size=1)
        # Z_p: 风格/光照/纹理
        self.projector_p = nn.Conv2d(combined_feature_dim, self.latent_dim_p, kernel_size=1)

        # --- 统一通道数到 256 ---
        PROJ_CHANNELS = 256
        self.proj_f = nn.Conv2d(combined_feature_dim, PROJ_CHANNELS, kernel_size=1)
        self.proj_z_s = nn.Conv2d(self.latent_dim_s, PROJ_CHANNELS, kernel_size=1)
        # 移除了 self.proj_z_p，因为任务头不再需要它

        # --- Task Heads (Inputs: Z_s Only) ---
        num_seg_classes = model_config.get('num_seg_classes', 40)
        # 输入 = Backbone特征 + Z_s特征
        task_in_dim = PROJ_CHANNELS * 2

        self.head_seg = TaskHead(task_in_dim, num_seg_classes)
        self.head_depth = TaskHead(task_in_dim, 1, use_sigmoid=False)
        self.head_normal = TaskHead(task_in_dim, 3)

        # --- Reconstruction Decoders ---

        # 1. 几何重构 (Auxiliary): Z_s -> Depth
        self.decoder_geom = ResNetDecoderWithDeepSupervision(self.latent_dim_s, 1, tuple(data_config['img_size']))

        # 2. 图像重构: Z_s + Z_p -> Image
        self.decoder_app = ResNetDecoderWithDeepSupervision(
            self.latent_dim_s,  # Content Input Channel
            3,
            tuple(data_config['img_size']),
            style_dim=self.latent_dim_p  # Style Input Channel
        )

        # self.decoder_app = ResNetDecoderWithDeepSupervision(
        #     self.latent_dim_s + self.latent_dim_p,  # 拼接输入
        #     3, tuple(data_config['img_size'])
        # )
        self.final_app_activation = nn.Sigmoid()

        # 3. 物理分解 (Explicit Decomposition)
        self.albedo_head = AlbedoHead(self.latent_dim_p, hidden=128)
        self.normal_head = NormalHead(self.latent_dim_s, hidden=128)
        self.light_head = LightHead(in_ch=encoder_feature_dim)
        self._target_size = tuple(data_config['img_size'])

    def extract_features(self, x):
        raw_features = self.encoder(x)
        if self.resnet_adapters is not None:
            features = []
            target_h, target_w = raw_features[2].shape[-2:]
            for i, feat in enumerate(raw_features):
                feat = self.resnet_adapters[i](feat)
                if feat.shape[-2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                features.append(feat)
        else:
            features = raw_features
        combined_feat = torch.cat(features, dim=1)
        last_feat = features[-1]
        h = last_feat.mean(dim=[2, 3])
        return combined_feat, h, features, raw_features

    # ======================================================
    # [NEW] CFA 核心组件：生成反事实图像 + 诊断信息
    # ======================================================
    # [修复 models/causal_model.py 中的 generate_counterfactual_image]
    def generate_counterfactual_image(self, z_s_map, z_p_map, strategy='global'):
        """
        生成反事实图像 (适配 AdaIN 版本)
        Args:
            z_s_map: [B, C_s, H, W] 当前 batch 的结构特征
            z_p_map: [B, C_p, H, W] 当前 batch 的风格特征 (或者已经扩展过的 map)
        """
        B = z_s_map.size(0)

        # 1. 随机打乱索引 (Shuffle) 以获取“别人的风格”
        perm_idx = torch.randperm(B, device=z_s_map.device)
        z_p_shuffled = z_p_map[perm_idx]

        # 2. 准备 Style Vector (AdaIN 需要向量)
        # 无论输入是 Map 还是 Vector，我们都将其处理为 [B, C] 向量
        # 因为 AdaIN 解码器需要 style 参数为向量
        z_p_vec_shuffled = F.adaptive_avg_pool2d(z_p_shuffled, (1, 1)).flatten(1)

        # 3. 解码生成图像 (关键修改：不再拼接，而是传参)
        # 旧代码: z_mix = torch.cat([...]); self.decoder_app(z_mix)  <-- 报错根源
        # 新代码: 分别传入 Content 和 Style
        recon_app_logits, _ = self.decoder_app(z_s_map, z_p_vec_shuffled)

        I_cfa = self.final_app_activation(recon_app_logits)

        # =================================================
        # [诊断探针数据]
        # =================================================
        with torch.no_grad():
            norm_zs = z_s_map.norm(p=2, dim=1).mean()
            norm_zp = z_p_vec_shuffled.norm(p=2, dim=1).mean()
            # 这里的 mix_std 仅作参考，记录一下 Zs 的稳定性
            # (AdaIN 模式下没有显式的 z_mix 拼接特征了)
            mix_std = z_s_map.std()

        diagnostics = {
            "norm_zs": norm_zs.item(),
            "norm_zp": norm_zp.item(),
            "mix_mean": 0.0,
            "mix_std": mix_std.item()
        }

        return I_cfa, diagnostics



    def forward(self, x, stage: int = 2):
        # 1. 特征提取
        combined_feat, h, features, raw_features = self.extract_features(x)
        low_level_feat = raw_features[0]

        # 2. 潜变量生成
        z_s_map = self.projector_s(combined_feat)
        z_p_map = self.projector_p(combined_feat)

        # Global Vectors (for CKA)
        z_s = z_s_map.mean(dim=[2, 3])
        z_p_vec = F.adaptive_avg_pool2d(z_p_map, (1, 1)).flatten(1)
        B, _, H, W = z_s_map.shape
        # view 变回 [B, 64, 1, 1] -> expand 变成 [B, 64, H, W]
        # -1 表示保持该维度不变（即通道数保持 64）
        z_p_map = z_p_vec.view(B, -1, 1, 1).expand(B, -1, H, W)

        # 注意：下面的代码中，z_p = z_p_vec (如果你之前写了 z_p = z_p_map.mean... 请确认 z_p 变量指向的是向量)
        z_p = z_p_vec
        #z_p = z_p_map.mean(dim=[2, 3])

        # 3. 准备任务输入 (Z_s based)
        f_proj = self.proj_f(combined_feat)
        zs_proj = self.proj_z_s(z_s_map)
        task_input = torch.cat([f_proj, zs_proj], dim=1)

        # ======================================================================
        # [下游任务预测]
        # ======================================================================

        def run_head(head_module, inp, low_level):
            return head_module(inp, low_level)

        if self.training and task_input.requires_grad:
            pred_seg = checkpoint(run_head, self.head_seg, task_input, low_level_feat, use_reentrant=False)
            pred_depth = checkpoint(run_head, self.head_depth, task_input, low_level_feat, use_reentrant=False)
            pred_normal = checkpoint(run_head, self.head_normal, task_input, low_level_feat, use_reentrant=False)
        else:
            pred_seg = self.head_seg(task_input, low_level_feat)
            pred_depth = self.head_depth(task_input, low_level_feat)
            pred_normal = self.head_normal(task_input, low_level_feat)

        pred_normal = F.normalize(pred_normal, p=2, dim=1)

        # ======================================================================
        # [重构与辅助]
        # ======================================================================

        # 1. 几何一致性重构 (Z_s -> Depth)
        if self.training and z_s_map.requires_grad:
            recon_geom_final, recon_geom_aux = checkpoint(self.decoder_geom, z_s_map, use_reentrant=False)
        else:
            recon_geom_final, recon_geom_aux = self.decoder_geom(z_s_map)

        # 2. 图像重构 (Z_s + Z_p -> Image)
        #z_combined = torch.cat([z_s_map, z_p_map], dim=1)
        # if self.training and z_combined.requires_grad:
        #     recon_app_final_logits, recon_app_aux_logits = checkpoint(self.decoder_app, z_combined, use_reentrant=False)
        # else:
        #     recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_combined)

        if self.training and z_s_map.requires_grad:  # Checkpoint 需要特殊处理参数传递
            # 注意：checkpoint 只能传 Tensor，需要把 z_p_vec 传进去
            recon_app_final_logits, recon_app_aux_logits = checkpoint(
                self.decoder_app, z_s_map, z_p_vec, use_reentrant=False
            )
        else:
            recon_app_final_logits, recon_app_aux_logits = self.decoder_app(z_s_map, z_p_vec)

        recon_app_final = self.final_app_activation(recon_app_final_logits)
        recon_app_aux = self.final_app_activation(recon_app_aux_logits)

        # 3. 物理层分解
        #A = self.albedo_head(z_p_map)
        A = self.albedo_head(z_p_vec)
        N = self.normal_head(z_s_map)
        L = self.light_head(h)
        S = shading_from_normals(N, L)

        target_size = self._target_size
        if A.shape[-2:] != target_size:
            A = F.interpolate(A, size=target_size, mode='bilinear', align_corners=False)
            N = F.interpolate(N, size=target_size, mode='bilinear', align_corners=False)
            S = F.interpolate(S, size=target_size, mode='bilinear', align_corners=False)
        I_hat = torch.clamp(A * S, 0.0, 1.0)

        outputs = {
            'z_s': z_s, 'z_p': z_p,
            'z_s_map': z_s_map,
            'z_p_map': z_p_map,
            'pred_seg': pred_seg, 'pred_depth': pred_depth, 'pred_normal': pred_normal,

            # 物理分解输出
            'decomposition_normal': N, 'albedo': A, 'shading': S, 'sh_coeff': L, 'recon_decomp': I_hat,

            # 纯像素重构输出
            'recon_geom': recon_geom_final, 'recon_app': recon_app_final,
            'recon_geom_aux': recon_geom_aux, 'recon_app_aux': recon_app_aux,
            'stage': stage,
        }
        return outputs