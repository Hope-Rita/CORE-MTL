import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CityscapesCDataset(Dataset):
    """
    Cityscapes -> Cityscapes-C 泛化测试数据集
    支持遍历: root_dir / corruption / severity / city / image.png
    """

    def __init__(self, root_dir, gt_dir, corruption, severity, img_size=(128, 256)):
        """
        Args:
            root_dir: Cityscapes-C 根目录
            gt_dir: Clean Cityscapes 预处理目录 (包含 val/label/*.npy)
            corruption: 腐蚀类型字符串 (如 'fog')
            severity: 强度 (1-5)
        """
        super().__init__()
        self.gt_dir = gt_dir
        self.target_size = (img_size[1], img_size[0])  # PIL uses (W, H)

        # 构造当前测试的具体路径
        self.images_dir = os.path.join(root_dir, corruption, str(severity))
        self.img_paths = []

        if os.path.exists(self.images_dir):
            # 遍历城市子文件夹
            subfolders = sorted([d for d in os.listdir(self.images_dir)
                                 if os.path.isdir(os.path.join(self.images_dir, d))])

            if len(subfolders) > 0:
                for city in subfolders:
                    city_path = os.path.join(self.images_dir, city)
                    files = sorted([f for f in os.listdir(city_path) if f.endswith('.png')])
                    for f in files:
                        self.img_paths.append(os.path.join(city_path, f))
            else:
                # 兼容扁平结构
                files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
                for f in files:
                    self.img_paths.append(os.path.join(self.images_dir, f))

        self.num_samples = len(self.img_paths)
        # 静默模式，避免打印太多 log
        # print(f"[Cityscapes-C] {corruption} (Sev {severity}): Found {self.num_samples} images.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Image
        img_path = self.img_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')
        img_resized = img_pil.resize(self.target_size, resample=Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # 2. GT (通过 Index 读取 Clean Label)
        label_path = os.path.join(self.gt_dir, 'val', 'label', f'{idx}.npy')
        depth_path = os.path.join(self.gt_dir, 'val', 'depth', f'{idx}.npy')

        H, W = self.target_size[1], self.target_size[0]

        try:
            if os.path.exists(label_path):
                label = torch.from_numpy(np.load(label_path)).long()
            else:
                label = torch.zeros((H, W), dtype=torch.long)

            if os.path.exists(depth_path):
                d_np = np.load(depth_path)
                if d_np.ndim == 3:
                    d_np = np.moveaxis(d_np, -1, 0)
                elif d_np.ndim == 2:
                    d_np = d_np[np.newaxis, ...]
                depth = torch.from_numpy(d_np).float()
            else:
                depth = torch.zeros((1, H, W), dtype=torch.float)
        except Exception:
            label = torch.zeros((H, W), dtype=torch.long)
            depth = torch.zeros((1, H, W), dtype=torch.float)

        return {
            'rgb': img_tensor,
            'segmentation': label,
            'depth': depth,
            'appearance_target': img_tensor,
            'scene_type': torch.tensor(0, dtype=torch.long)
        }