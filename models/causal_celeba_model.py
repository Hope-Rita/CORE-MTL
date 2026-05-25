import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==============================================================================
# 1. 专用组件 (Local Components)
# ==============================================================================

class ResNet18Encoder(nn.Module):
    """
    专为 CelebA 任务设计的 ResNet18 编码器。
    不使用空洞卷积 (Dilation)，保持标准的 1/32 下采样率，以获得更大的感受野和更小的特征图。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 移除 FC 层，保留特征提取部分
        # ResNet18 结构: Stem -> Layer1(64) -> Layer2(128) -> Layer3(256) -> Layer4(512)
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        self.out_channels = 512

    def forward(self, x):
        # 返回最后一层的特征图
        return self.features(x)


class AttributeHead(nn.Module):
    """
    多标签分类头: Z_s (Vector) -> 40 Attributes
    """

    def __init__(self, in_features, num_attributes=40):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_attributes)
            # 注意：不加 Sigmoid，因为 Loss 中使用 BCEWithLogitsLoss
        )

    def forward(self, z_s_vec):
        return self.fc(z_s_vec)


class SimpleDecoder(nn.Module):
    """
    图像重构解码器: (Z_s Map + Z_p Map) -> RGB Image
    假设输入特征图是原图的 1/32 (ResNet18默认)，需要上采样 5 次 (2^5=32)。
    """

    def __init__(self, in_channels, out_channels=3):
        super().__init__()

        # Helper for ConvBlock
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # 假设输入 512ch (例如 4x4 for 128px img)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = conv_block(in_channels, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = conv_block(256, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = conv_block(128, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = conv_block(64, 32)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)  # 最后一层 1x1 卷积输出 RGB
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        x = self.conv5(self.up5(x))
        return self.sigmoid(x)


# ==============================================================================
# 2. 核心模型 (CausalCelebAModel)
# ==============================================================================

class CausalCelebAModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        # 1. Backbone
        # 默认使用 ResNet18，更适合 CelebA
        self.encoder = ResNet18Encoder(pretrained=model_config.get('pretrained', True))
        enc_dim = self.encoder.out_channels  # 512

        # 2. Projectors (Latent S & P)
        # 将 Backbone 特征映射到 Z_s 和 Z_p 空间
        # 即使 latent_dim 较小，这里也先保持 spatial map (Conv2d 1x1)
        self.latent_dim_s = model_config.get('latent_dim_s', 256)
        self.latent_dim_p = model_config.get('latent_dim_p', 256)

        self.proj_zs = nn.Sequential(
            nn.Conv2d(enc_dim, self.latent_dim_s, 1, bias=False),
            nn.BatchNorm2d(self.latent_dim_s),
            nn.ReLU(inplace=True)
        )

        self.proj_zp = nn.Sequential(
            nn.Conv2d(enc_dim, self.latent_dim_p, 1, bias=False),
            nn.BatchNorm2d(self.latent_dim_p),
            nn.ReLU(inplace=True)
        )

        # 3. Task Head (Attributes)
        # 负责预测 40 个属性
        self.num_attributes = model_config.get('num_attributes', 40)
        self.attr_head = AttributeHead(self.latent_dim_s, self.num_attributes)

        # 4. Reconstruction Decoder
        # 负责从 Z_s + Z_p 还原图像
        # 输入通道 = dim_s + dim_p
        self.decoder = SimpleDecoder(self.latent_dim_s + self.latent_dim_p)

    def forward(self, x):
        # 1. Encode
        feat_map = self.encoder(x)  # [B, 512, H/32, W/32]

        # 2. Decouple -> Spatial Maps
        zs_map = self.proj_zs(feat_map)  # [B, dim_s, H', W']
        zp_map = self.proj_zp(feat_map)  # [B, dim_p, H', W']

        # 3. Branch 1: Attributes (Zs -> GAP -> Vector -> FC)
        zs_vec = F.adaptive_avg_pool2d(zs_map, (1, 1)).flatten(1)
        pred_attr = self.attr_head(zs_vec)

        # 4. Branch 2: Reconstruction (Cat(Zs, Zp) -> Decoder)
        # 拼接两个 map
        z_combined = torch.cat([zs_map, zp_map], dim=1)
        recon_img = self.decoder(z_combined)

        # 5. Extract Zp Vector (for CKA Loss)
        # 为了计算 CKA (Independence)，我们需要 Zp 也是向量形式
        zp_vec = F.adaptive_avg_pool2d(zp_map, (1, 1)).flatten(1)

        return {
            'pred_attr': pred_attr,  # [B, 40] logits
            'recon_img': recon_img,  # [B, 3, H, W]
            'zs_vec': zs_vec,  # [B, dim_s]
            'zp_vec': zp_vec,  # [B, dim_p]
            'zs_map': zs_map,  # [B, dim_s, H', W']
            'zp_map': zp_map  # [B, dim_p, H', W']
        }