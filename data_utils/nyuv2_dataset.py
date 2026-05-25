import os
import torch
import torch.nn.functional as F
import numpy as np
import fnmatch
import random
from torch.utils.data import Dataset
from torchvision import transforms


class RandomScaleCrop(object):
    """
    随机缩放并裁剪，最后插值回原尺寸。
    保证输出分辨率与输入分辨率一致 (Height, Width)。
    """

    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        # img: [C, H, W], label: [H, W], depth: [1, H, W], normal: [3, H, W]
        height, width = img.shape[-2:]

        # 随机选择一个缩放比例
        sc = self.scale[random.randint(0, len(self.scale) - 1)]

        # 裁剪窗口的大小 (缩放后的视窗)
        h, w = int(height / sc), int(width / sc)

        # 随机选择裁剪起点
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        # 1. Crop: 裁剪出 ROI
        # 2. Interpolate: 强制缩放回原图尺寸 (height, width) -> 保证分辨率不变
        # Bilinear for continuous (img, normal), Nearest for discrete/depth (label, depth)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                             align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w].float(), size=(height, width),
                               mode='nearest').squeeze(0).squeeze(0).long()
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear',
                                align_corners=True).squeeze(0)


        return img_, label_, depth_ / sc, normal_


class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, mode='train', augmentation=False):
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.mode = mode
        self.augmentation = augmentation

        # LibMTL 文件夹结构: root/train/image/*.npy
        sub_dir = 'train' if mode == 'train' else 'val'
        self.data_path = os.path.join(self.root, sub_dir)

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")

        image_dir = os.path.join(self.data_path, 'image')
        self.index_list = fnmatch.filter(os.listdir(image_dir), '*.npy')
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()

        self.num_samples = len(self.index_list)
        print(f"[{mode.upper()}] Found {self.num_samples} samples in {self.data_path}")

        # ==================================================
        # 定义增强变换 (只针对 RGB)
        # ==================================================
        # 1. ColorJitter: 亮度/对比度/饱和度 0.4, 色相 0.1 (比较通用的强增强)
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)



    def __getitem__(self, i):
        index = self.index_list[i]

        # 1. 读取 .npy
        img_path = os.path.join(self.data_path, 'image', f'{index}.npy')
        label_path = os.path.join(self.data_path, 'label', f'{index}.npy')
        depth_path = os.path.join(self.data_path, 'depth', f'{index}.npy')
        normal_path = os.path.join(self.data_path, 'normal', f'{index}.npy')

        img_np = np.load(img_path)  # [H, W, 3] float64 [0,1]
        label_np = np.load(label_path)  # [H, W]
        depth_np = np.load(depth_path)  # [H, W]
        normal_np = np.load(normal_path)  # [H, W, 3]

        # 2. 转换 Tensor & 维度 (HWC -> CHW)
        # 强制转为 float32，PyTorch 默认 float32，float64 会浪费显存且可能报错
        image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()
        semantic = torch.from_numpy(label_np).long()
        normal = torch.from_numpy(np.moveaxis(normal_np, -1, 0)).float()

        if depth_np.ndim == 2:
            depth = torch.from_numpy(depth_np).float().unsqueeze(0)
        else:
            depth = torch.from_numpy(np.moveaxis(depth_np, -1, 0)).float()

        # ==================================================
        # 3. Data Augmentation
        # ==================================================
        if self.augmentation:
            # A. 几何变换 (Geometry) - 必须所有模态同步变换
            # Random Scale Crop (缩放并裁剪，保持分辨率不变)
            image, semantic, depth, normal = RandomScaleCrop(scale=[1.0, 1.2, 1.5])(image, semantic, depth, normal)

            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = - normal[0, :, :]  # 翻转 normal 的 x 分量

            # B. 像素变换 (Photometric) - 只变换 RGB
            # 必须放在几何变换之后，Normalize 之前
            image = self.color_jitter(image)

        # 4. Normalize (建议对 Train 和 Val 都做，但一定要在增强后做)
        # 这是一个非常好的习惯，能让 Loss 下降更平滑
        #image = self.normalize(image)

        # 构造返回字典
        return {
            'rgb': image,
            'depth': depth,
            'segmentation': semantic,
            'normal': normal,
            'scene_type': torch.tensor(0, dtype=torch.long),


            'appearance_target': image,
            'depth_meters': depth
        }

    def __len__(self):
        return self.num_samples