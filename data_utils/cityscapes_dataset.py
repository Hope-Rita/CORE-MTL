import os
import torch
import fnmatch
import numpy as np
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        严格遵循 LibMTL 预处理格式的 Dataset 读取类。
        假设输入 .npy 文件已经是对齐的、尺寸一致的，不做任何 Resize 或 Normalize。
        """
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.split = split

        # LibMTL 文件夹结构: root/train/image/*.npy
        folder_split = 'train' if split == 'train' else 'val'
        self.data_path = os.path.join(self.root, folder_split)

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path not found: {self.data_path}")

        # 获取文件列表
        # 假设文件名是 0.npy, 1.npy ... 需要按数字排序
        image_dir = os.path.join(self.data_path, 'image')
        self.index_list = fnmatch.filter(os.listdir(image_dir), '*.npy')
        # 排序以确保对应关系
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()

        self.num_samples = len(self.index_list)
        print(f"[{split.upper()}] Found {self.num_samples} samples in {self.data_path}")

    def __getitem__(self, i):
        index = self.index_list[i]

        # ==================================================
        # 1. 读取 .npy 文件
        # ==================================================
        # 格式通常为:
        # Image: [H, W, 3]
        # Label: [H, W]
        # Depth: [H, W, 1] (或 [H, W])
        img_np = np.load(os.path.join(self.data_path, 'image', f'{index}.npy'))
        label_np = np.load(os.path.join(self.data_path, 'label', f'{index}.npy'))
        depth_np = np.load(os.path.join(self.data_path, 'depth', f'{index}.npy'))

        # ==================================================
        # 2. 维度变换 (HWC -> CHW) & 转 Tensor
        # ==================================================
        # Numpy 默认是 HWC，PyTorch 需要 CHW

        # Image: [H, W, 3] -> [3, H, W]
        if img_np.ndim == 3:
            image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()
        else:
            # 异常处理：如果是 HW
            image = torch.from_numpy(img_np).float().unsqueeze(0)

        # Label: [H, W] -> [H, W] (Label 不需要 channel 维，或者根据 loss 需要)
        # 通常 CrossEntropy 需要 [H, W] 的 Long Tensor
        semantic = torch.from_numpy(label_np).long()

        # Depth: [H, W, 1] -> [1, H, W]
        if depth_np.ndim == 3:
            depth = torch.from_numpy(np.moveaxis(depth_np, -1, 0)).float()
        elif depth_np.ndim == 2:
            depth = torch.from_numpy(depth_np).float().unsqueeze(0)
        else:
            depth = torch.from_numpy(depth_np).float()

        # ==================================================
        # 3. 构造返回字典 (不做任何 Resize/Normalize)
        # ==================================================
        return {
            'rgb': image,  # 原始 float 数据
            'depth': depth,  # 原始深度/视差
            'segmentation': semantic,  # 原始标签
            'scene_type': torch.tensor(0, dtype=torch.long),  # 占位符

            # 兼容性字段：
            # appearance_target 用于重构 Loss，这里直接用原始 RGB
            'appearance_target': image,

            # depth_meters 用于特定 metric，这里直接用原始 depth
            'depth_meters': depth
        }

    def __len__(self):
        return self.num_samples