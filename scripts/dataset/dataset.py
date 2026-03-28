from torch.utils.data import Dataset, DataLoader
import json
import nibabel as nib
import torch
import torchvision.transforms as transforms
import time
import numpy as np
import os


class DoctorDataset(Dataset):
    """
    医学影像数据集类，用于加载和处理医学影像数据
    
    该类继承自 PyTorch 的 Dataset 类，支持数据的预加载和动态加载
    """
    
    def __init__(self, data_path, root="./", preload=False):
        """
        初始化数据集
        
        Args:
            data_path: JSON 数据路径，包含图像、标签和消息的路径信息
            root: 根目录，用于拼接完整的文件路径
            preload: 是否预加载数据到内存，预加载可以提高训练速度但会占用更多内存
        """
        super().__init__()
        self.root = root
        # 读取 JSON 数据文件
        self.data = self.read_json(data_path)
        self.preload = preload
        # 定义数据转换，将 numpy 数组转换为 PyTorch 张量
        self.transform = transforms.ToTensor()
        
        # 预加载数据到内存
        if preload:
            self._preload_data()

    def read_json(self, data_path):
        """
        读取 JSON 文件
        
        Args:
            data_path: JSON 数据路径
            
        Returns:
            解析后的数据列表，每个元素包含图像路径、标签路径和消息
        """
        try:
            # 拼接完整的文件路径并打开 JSON 文件
            with open(os.path.join(self.root, data_path), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return []

    def _preload_data(self):
        """
        预加载数据到内存
        
        将数据集中的所有图像和标签加载到内存中，减少训练时的 I/O 操作
        """
        print("Preloading data...")
        self.preloaded_data = []
        # 遍历数据集中的每个项目
        for item in self.data:
            try:
                # 构建图像和标签的完整路径
                image_path = os.path.join(self.root, item['image'])
                label_path = os.path.join(self.root, item['label'])
                
                # 检查文件是否存在
                if not os.path.exists(image_path) or not os.path.exists(label_path):
                    print(f"Warning: Missing files for item: {item}")
                    continue
                
                # 加载 NIfTI 格式的图像和标签
                nii_image = nib.load(image_path)
                nii_label = nib.load(label_path)
                
                # 转换为 numpy 数组并保存
                image = nii_image.get_fdata()
                label = nii_label.get_fdata()
                message = item['message']
                
                # 将数据添加到预加载列表
                self.preloaded_data.append((image, label, message))
            except Exception as e:
                print(f"Error preloading item: {e}")
        print(f"Preloaded {len(self.preloaded_data)} items")

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            数据集的样本数量
        """
        if self.preload:
            return len(self.preloaded_data)
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx: 索引
            
        Returns:
            image: 形状为 (204, 1, 340, 340) 的图像张量
            label: 形状为 (204, 1, 340, 340) 的标签张量
            message: 与样本相关的文本消息
        """
        try:
            if self.preload:
                # 使用预加载的数据
                image, label, message = self.preloaded_data[idx]
            else:
                # 动态加载数据
                item = self.data[idx]
                image_path = os.path.join(self.root, item['image'])
                label_path = os.path.join(self.root, item['label'])
                
                # 加载 NIfTI 格式的图像和标签
                nii_image = nib.load(image_path)
                nii_label = nib.load(label_path)
                image = nii_image.get_fdata()
                label = nii_label.get_fdata()
                message = item['message']
            
            # 转换为张量并调整形状为 (204, 1, 340, 340)
            image = self.transform(image).reshape(204, 1, 340, 340)
            label = self.transform(label).reshape(204, 1, 340, 340)
            
            return image, label, message
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # 返回默认值，避免训练过程因数据加载错误而中断
            return torch.zeros(204, 1, 340, 340), torch.zeros(204, 1, 340, 340), ""


if __name__ == "__main__":
    t1 = time.time()
    # 测试预加载模式
    dataset = DoctorDataset(root="./", data_path="dataset/llm/qa_pairs.json", preload=True)
    # 创建数据加载器
    b = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # 遍历数据加载器
    for idx, data in enumerate(b):
        print(idx, data[0].shape, data[1].shape)
    print(f"程序已结束，耗时{round(time.time() - t1, 2)}s")
