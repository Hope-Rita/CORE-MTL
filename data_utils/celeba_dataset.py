import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CelebADataset(Dataset):
    """
    CelebA 数据集加载器 (Scalability 实验适配版)
    - 自动读取 list_attr_celeba.txt 获取属性
    - 支持 num_attributes 参数，自动截取前 T 个属性用于 Scalability 实验
    """

    def __init__(self, root_dir, split='train', img_size=[128, 128], num_attributes=40, augmentation=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.img_size = tuple(img_size)  # (H, W)
        self.num_attributes = num_attributes  # [关键修改] 任务数量 T

        # 1. 检查文件路径
        self.attr_path = os.path.join(root_dir, 'list_attr_celeba.txt')
        self.partition_path = os.path.join(root_dir, 'list_eval_partition.txt')
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')

        # 兼容两种常见的解压路径结构
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba')
            if not os.path.exists(self.img_dir):
                raise ValueError(f"❌ Image directory not found in {root_dir}")

        # 2. 读取属性标签
        # header=1 表示第二行是属性名，第一行是数量
        try:
            self.attr_df = pd.read_csv(self.attr_path, delim_whitespace=True, header=1)
        except Exception as e:
            raise ValueError(f"❌ Failed to read attribute file: {self.attr_path}. Error: {e}")

        # [关键逻辑] 截取前 T 个属性
        # CelebA 总共有 40 个属性。为了 Scalability 实验，我们取前 num_attributes 个
        if self.num_attributes < 40:
            print(f"⚡ [Scalability Experiment] Selecting top {self.num_attributes} attributes.")
            self.attr_df = self.attr_df.iloc[:, :self.num_attributes]

        # 3. 读取划分文件 (0:Train, 1:Val, 2:Test)
        try:
            partition_df = pd.read_csv(self.partition_path, delim_whitespace=True, header=None,
                                       names=['image_id', 'partition'])
        except Exception as e:
            raise ValueError(f"❌ Failed to read partition file: {self.partition_path}. Error: {e}")

        # 4. 根据 split 筛选数据
        split_map = {'train': 0, 'val': 1, 'test': 2, 'valid': 1}
        target_partition = split_map.get(self.split, 0)

        # 获取该 split 下的所有图片文件名
        self.split_images = partition_df[partition_df['partition'] == target_partition]['image_id'].tolist()

        # 过滤掉 attr_df 中不存在的图片 (取交集)
        self.attr_df = self.attr_df.loc[self.attr_df.index.intersection(self.split_images)]

        # 更新 image list
        self.image_ids = self.attr_df.index.tolist()

        print(
            f"[CelebA] Split: {self.split.upper()} | Samples: {len(self.image_ids)} | Attributes: {self.attr_df.shape[1]}")

        # 5. 定义变换
        if augmentation:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 1. 读取图片
        img = Image.open(img_path).convert('RGB')

        # 2. 获取属性标签 (已截取到 T 个)
        # CelebA 原始标签是 1 和 -1，映射到 1 和 0
        attr_raw = self.attr_df.loc[img_name].values.astype(int)
        attr_label = (attr_raw + 1) // 2
        attr_tensor = torch.from_numpy(attr_label).float()

        # 3. 图像变换
        img_tensor = self.transform(img)

        return {
            'image': img_tensor,  # [3, H, W]
            'attributes': attr_tensor,  # [T]
            'appearance_target': img_tensor  # [3, H, W]
        }