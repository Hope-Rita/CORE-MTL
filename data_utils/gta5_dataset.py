import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

GTA5_TO_7_CLASSES = {
    0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1,
    7: 0, 8: 0, 9: -1, 10: -1, 11: 1, 12: 1, 13: 1,
    14: -1, 15: -1, 16: -1, 17: 2, 18: -1, 19: 2, 20: 2,
    21: 3, 22: 3, 23: 4, 24: 5, 25: 5, 26: 6, 27: 6,
    28: 6, 29: -1, 30: -1, 31: 6, 32: 6, 33: 6
}


class GTA5Dataset(Dataset):
    def __init__(self, root_dir, img_size, augmentation=False):
        """
        新增 augmentation 参数控制是否翻转
        支持读取同级目录下的 depth (.npy) 信息
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.augmentation = augmentation

        self.images_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(self.images_dir):
            self.images_dir = os.path.join(root_dir, 'Images')

        self.labels_dir = os.path.join(root_dir, 'labels')
        if not os.path.exists(self.labels_dir):
            self.labels_dir = os.path.join(root_dir, 'Labels')

        # 1. 自动定位 depth 文件夹
        self.depth_dir = os.path.join(root_dir, 'depth')

        self.images = []
        self.targets = []
        self.depths = []  # 存储深度路径

        search_pattern = os.path.join(self.images_dir, "*.png")
        files = glob.glob(search_pattern)
        files.sort()  # 排序，确保图像顺序一致

        for img_path in files:
            file_name = os.path.basename(img_path)
            label_path = os.path.join(self.labels_dir, file_name)

            # 2. 构造深度文件路径 (假设 0001.png -> 0001.npy)
            depth_file_name = os.path.splitext(file_name)[0] + '.npy'
            depth_path = os.path.join(self.depth_dir, depth_file_name)

            if os.path.exists(label_path):
                self.images.append(img_path)
                self.targets.append(label_path)

                # 如果对应的深度文件存在，则记录；否则给None
                if os.path.exists(depth_path):
                    self.depths.append(depth_path)
                else:
                    self.depths.append(None)

        self.mapping = np.zeros(256, dtype=np.int64) - 1
        for k, v in GTA5_TO_7_CLASSES.items():
            if k >= 0:
                self.mapping[k] = v

        print(f"[GTA5] Found {len(self.images)} samples in {root_dir} (Aug={self.augmentation})")
        print(f"[GTA5] Depth samples found: {len([d for d in self.depths if d is not None])}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.targets[idx]
        depth_path = self.depths[idx]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # 3. 读取深度信息并归一化
        if depth_path is not None:
            # 读取 .npy (float32, 0~255)
            depth_np = np.load(depth_path).astype(np.float32)
            # 转为 PIL 'F' 模式
            depth_np = depth_np / 255.0
            depth_img = Image.fromarray(depth_np, mode='F')
        else:
            # 如果缺失深度，生成全0图
            depth_img = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.float32), mode='F')

        # 4. Resize (保持一致)
        # resize 接受 (width, height)
        target_size = (self.img_size[1], self.img_size[0])

        img = img.resize(target_size, Image.BILINEAR)
        label = label.resize(target_size, Image.NEAREST)
        depth_img = depth_img.resize(target_size, Image.BILINEAR)  # 深度图用双线性插值

        # 5. Augmentation (同步翻转)
        if self.augmentation and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
            depth_img = transforms.functional.hflip(depth_img)

        # 6. 转 Tensor
        to_tensor = transforms.ToTensor()
        rgb_tensor_unnormalized = to_tensor(img).float()

        label_np = np.array(label, dtype=np.int64)
        label_np[label_np > 255] = 255
        label_mapped = self.mapping[label_np]
        seg_tensor = torch.from_numpy(label_mapped).long()

        # Depth Tensor [1, H, W]
        # 因为已经是 float32 且归一化到了 0-1，ToTensor 直接转换即可
        depth_tensor = to_tensor(depth_img).float()

        return {
            'rgb': rgb_tensor_unnormalized,
            'depth': depth_tensor,
            'segmentation': seg_tensor,
            'scene_type': torch.tensor(0),
            'appearance_target': rgb_tensor_unnormalized
        }