# ====================================================================================
# models/building_blocks.py -- FINAL REVISED VERSION
# ====================================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models import ResNet50_Weights


# [在文件顶部 imports 下方添加]
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # 预测仿射参数 (Scale, Shift)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        # style: [B, style_dim] -> [B, 2*C]
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = h.chunk(2, 1)

        # Instance Norm
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8

        # Modulate: (x - mean)/std * (1+gamma) + beta
        # 注意：这里用 (1+gamma) 是为了让初始状态接近 Identity，训练更稳定
        return (x - mean) / std * (1 + gamma) + beta


# [新增一个支持 Style 的 Block，放在 ResidualBlock 附近]
class AdaINResidualBlock(nn.Module):
    """
    支持 AdaIN 的残差块：
    Content (x) 来自上一层, Style (z_p) 来自全局向量
    """

    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.adain1 = AdaIN(style_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.adain2 = AdaIN(style_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
            self.upsample = True
        else:
            self.shortcut = nn.Identity()
            self.upsample = False

        # 如果需要上采样主路径
        if self.upsample:
            self.main_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, style):
        # Main Path
        # 1. AdaIN -> ReLU -> Upsample(if needed) -> Conv
        out = self.adain1(x, style)
        out = self.relu(out)
        if self.upsample:
            out = self.main_upsample(out)
        out = self.conv1(out)

        # 2. AdaIN -> ReLU -> Conv
        out = self.adain2(out, style)
        out = self.relu(out)
        out = self.conv2(out)

        # Shortcut Path
        res = self.shortcut(x)

        return out + res


# [修改 ResNetDecoderWithDeepSupervision]
class ResNetDecoderWithDeepSupervision(nn.Module):
    def __init__(self, input_channels, output_channels, target_size=(224, 224), style_dim=0):
        super().__init__()
        self.target_size = target_size
        self.style_dim = style_dim  # 新增参数

        # 初始层只处理 Z_s
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        self.attention = SelfAttention(512)

        # 如果 style_dim > 0，说明是 Appearance Decoder，使用 AdaIN
        # 如果 style_dim == 0，说明是 Geometry Decoder，使用标准 ResidualBlock

        if style_dim > 0:
            self.use_adain = True
            self.res_block1 = AdaINResidualBlock(512, 256, style_dim)
            self.res_block2 = AdaINResidualBlock(256, 128, style_dim)
            self.res_block3 = AdaINResidualBlock(128, 64, style_dim)
        else:
            self.use_adain = False
            self.res_block1 = ResidualBlock(512, 256)
            self.res_block2 = ResidualBlock(256, 128)
            self.res_block3 = ResidualBlock(128, 64)

        self.aux_head = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.final_upsample = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x, style=None):
        # x: Z_s [B, C, H, W]
        x = self.initial_conv(x)
        x = self.attention(x)

        if self.use_adain:
            assert style is not None, "Appearance Decoder requires style vector!"
            x = self.res_block1(x, style)
            x_56 = self.res_block2(x, style)
            out_aux = self.aux_head(x_56)
            x_final = self.res_block3(x_56, style)
        else:
            x = self.res_block1(x)
            x_56 = self.res_block2(x)
            out_aux = self.aux_head(x_56)
            x_final = self.res_block3(x_56)

        out_final = self.final_upsample(x_final)
        return out_final, out_aux
class ViTEncoder(nn.Module):
    """
    A wrapper for the Vision Transformer to extract multi-scale patch token features.

    功能升级：
    1. [工程] 严格根据 img_size (224/384) 自动匹配最佳预训练权重 (DEFAULT/SWAG)。
    2. [架构] 返回多层特征列表 (indices: 2, 5, 8, 11) 以支持高分辨率解码。
    """

    def __init__(self, name="vit_b_16", pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        if name == "vit_b_16":
            weights = None
            if pretrained:
                if img_size == 384:
                    # 384x384 输入，使用 SWAG 高分权重 (最佳推荐)
                    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                    print(f"✅ [ViTEncoder] Using 384x384 pretrained weights (SWAG_E2E_V1).")
                elif img_size == 224:
                    # 兼容默认情况
                    weights = models.ViT_B_16_Weights.DEFAULT
                    print(f"✅ [ViTEncoder] Using 224x224 pretrained weights (DEFAULT).")
                else:
                    raise ValueError(f"For pretrained ViT, img_size must be 224 or 384. Got {img_size}.")

            # 初始化模型，严格绑定 image_size
            self.vit: VisionTransformer = models.vit_b_16(weights=weights, image_size=img_size)
            self.feature_dim = 768
        else:
            raise ValueError(f"Encoder '{name}' not supported.")

        # Remove the final classification head
        self.vit.heads = nn.Identity()

    def forward(self, x):
        # 1. 预处理 (torchvision 内部会检查输入尺寸是否匹配 img_size)
        x = self.vit._process_input(x)

        # 2. 拼接 Class Token
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # 3. 手动执行 Encoder Layers 以提取中间层特征
        # 注意：torchvision 的 encoder 包含 pos_embedding + dropout + layers + ln

        # 3.1 加位置编码 + Dropout
        x = x + self.vit.encoder.pos_embedding
        if hasattr(self.vit.encoder, 'dropout'):
            x = self.vit.encoder.dropout(x)

        features = []
        # 定义要提取的层索引 (0-11)，这里取 [2, 5, 8, 11] 即第 3, 6, 9, 12 层
        # 这种取法覆盖了低、中、高层语义
        out_indices = [2, 5, 8, 11]

        # 3.2 逐层前向传播
        for i, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)

            if i in out_indices:
                # 提取 Patch Tokens (去掉 Class Token)
                patch_tokens = x[:, 1:, :]
                b, _, c = patch_tokens.shape

                # 整理为 2D 特征图: [B, 768, H/16, W/16]
                feat = patch_tokens.permute(0, 2, 1).view(b, c, self.grid_size, self.grid_size)
                features.append(feat)

        # 返回特征列表，最后一层 features[-1] 即为最高层特征
        return features


class ResNetEncoder(nn.Module):
    """
    ResNet50 Encoder wrapper adapted for CausalMTL.
    Supports 'resnet_dilated' mode to align with LibMTL.
    """

    def __init__(self, name="resnet50", pretrained=True, dilated=True):
        super().__init__()

        # 1. 加载标准 ResNet50
        # LibMTL 使用 resnet_dilated，通常意味着最后一块使用空洞卷积
        # replace_stride_with_dilation=[False, False, True] 会让 Layer4 的 Stride=1, Dilation=2
        # 这样输出的特征图尺寸是 1/16 (OS=16)，比标准的 1/32 更适合密集预测
        replace_stride = [False, True, True] if dilated else [False, False, False]

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride
        )

        # 记录各层通道数，用于后续 Adapter
        self.feature_dims = [256, 512, 1024, 2048]

        # 移除 FC 层和 AvgPool，只保留特征提取部分
        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1  # 1/4
        self.layer2 = self.backbone.layer2  # 1/8
        self.layer3 = self.backbone.layer3  # 1/16
        self.layer4 = self.backbone.layer4  # 1/16 (if dilated) else 1/32

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # 返回多尺度特征列表
        return [c1, c2, c3, c4]

class MLP(nn.Module):
    # ... (Code from above, no changes needed here)
    """
    A simple Multi-Layer Perceptron for projection heads.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.out_features = output_dim
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    """
    一个标准的卷积解码器，加入了BatchNorm2d来稳定梯度流。
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.output_channels = output_channels

        self.decoder_net = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False),  # 使用BN时，卷积层可以不用偏置
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Block 2: 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Block 3: 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Block 4: 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Output Layer: 112x112 -> 224x224
            # 最后一层通常不加BN和ReLU，直接输出logits
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *feature_maps):
        combined_features = torch.cat(feature_maps, dim=1)
        return self.decoder_net(combined_features)


class ConvDecoder(nn.Module):
    """
    一个增强版的卷积解码器，使用InstanceNorm2d来保证在小批次下的训练稳定性。
    """
    def __init__(self, latent_dim, output_channels, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        start_size = target_size[0] // 16

        self.upsample_in = nn.Sequential(
            nn.Linear(latent_dim, 512 * start_size * start_size),
            nn.ReLU(True)
        )
        self.start_size = start_size

        self.net = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256), # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128), # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),  # <-- 使用 InstanceNorm2d 替换 BatchNorm2d
            nn.ReLU(True),
            # 112x112 -> 224x224 (最后一层不加归一化和ReLU)
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.upsample_in(x)
        x = x.view(-1, 512, self.start_size, self.start_size)
        x = self.net(x)
        return x


class SelfAttention(nn.Module):
    """
    一个高效的、用于卷积特征图的自注意力模块。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 定义Q, K, V的卷积投影层
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 伽马参数，用于残差连接的加权，初始化为0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # 1. 计算 Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x (W*H)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x (W*H)

        # 2. 计算注意力图 (Attention Map)
        energy = torch.bmm(proj_query, proj_key)  # B x (W*H) x (W*H)
        attention = self.softmax(energy)  # B x (W*H) x (W*H)

        # 3. 将注意力应用于 Value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # 4. 残差连接
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    """
    一个标准的残差上采样块，包含快捷连接。
    它首先上采样，然后通过两个卷积层，最后将输入添加到输出上。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 快捷连接路径：需要上采样并调整通道数以匹配主路径输出
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        # 主路径
        self.main_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        main = self.main_path(x)
        return self.relu(shortcut + main)

#
# class ResNetDecoderWithDeepSupervision(nn.Module):
#     """
#     使用残差块构建的解码器，并集成了深度监督功能。
#     """
#     def __init__(self, input_channels, output_channels, target_size=(224, 224)):
#         super().__init__()
#         self.target_size = target_size
#         start_size = target_size[0] // 16  # 224 / 16 = 14
#
#         # 初始层：将扁平的latent vector转换为14x14的特征图
#         self.initial_conv = nn.Sequential(
#             nn.Conv2d(input_channels, 512, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(True)
#         )
#         self.start_size = start_size
#         self.attention = SelfAttention(512)
#         # 上采样模块 (ResNet blocks)
#         self.res_block1 = ResidualBlock(512, 256)  # 14x14 -> 28x28
#         self.res_block2 = ResidualBlock(256, 128)  # 28x28 -> 56x56
#
#
#
#         self.res_block3 = ResidualBlock(128, 64)   # 56x56 -> 112x112
#
#         # --- 深度监督分支 ---
#         # 从 56x56 的特征图 (self.res_block2的输出) 创建一个辅助预测
#         self.aux_head = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
#
#         # 主输出路径
#         self.final_upsample = nn.Sequential(
#             #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 112x112 -> 224x224
#             nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
#         )
#
#     def forward(self, x):
#         # 1. 初始上采样 (输入 x 是 48x48 的 z_s_map)
#         x = self.initial_conv(x)  # -> [B, 512, 48, 48]
#
#         # 【关键】在这里做 Attention，显存开销极小
#         x = self.attention(x)  # -> [B, 512, 48, 48]
#
#         # 2. 通过残差块
#         x = self.res_block1(x)  # -> [B, 256, 96, 96]
#         x_56 = self.res_block2(x)  # -> [B, 128, 192, 192] (这里就是你的 x_56)
#
#         # 3. 计算辅助输出 (旁路，不影响主路)
#         out_aux = self.aux_head(x_56)
#
#         # 4. 继续主路径
#         x_final = self.res_block3(x_56)  # -> [B, 64, 384, 384]
#         out_final = self.final_upsample(x_final)
#
#         # 5. 返回主输出和辅助输出
#         return out_final, out_aux