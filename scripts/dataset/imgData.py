import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import random
import torchvision.transforms as transforms

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from scripts.img.imgProcess import process_nii_gz, mask_process

class ImgDataset(Dataset):
    def __init__(self, nii_list, mask_list, use_augmentation=True, rotation_prob=0.5):
        """
        初始化 ImgDataset
        
        Args:
            nii_list: nii.gz 图像文件路径列表
            mask_list: nii.gz 掩码文件路径列表
            use_augmentation: 是否使用数据增强
            rotation_prob: 随机旋转的概率
        """
        self.nii_list = nii_list
        self.mask_list = mask_list
        self.use_augmentation = use_augmentation
        self.rotation_prob = rotation_prob
        self.data = None
        self.masks = None
        self._load_data()

    def _load_data(self):
        """
        加载数据，将所有医学图像拼接成4维张量
        """
        data_list = []
        masks_list = []
        
        for nii_path, mask_path in zip(self.nii_list, self.mask_list):
            # 加载图像数据
            img_data = process_nii_gz(nii_path)
            data_list.append(img_data)
            # 加载掩码数据
            mask_data = process_nii_gz(mask_path)
            mask_data = mask_process(mask_data)
            masks_list.append(mask_data)
        
        # 将所有数据拼接成4维张量 [图像总数, 通道数, H, W]
        self.data = torch.cat(data_list, dim=0)
        self.masks = torch.cat(masks_list, dim=0)

    def __len__(self):
        """
        返回数据集长度
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        获取指定索引的数据
        
        Args:
            index: 数据索引
            
        Returns:
            图像数据和对应的掩码
        """
        image = self.data[index]
        mask = self.masks[index]
        
        # 应用数据增强
        if self.use_augmentation and random.random() < self.rotation_prob:
            # 随机选择旋转角度
            angle = random.choice([90, 180, 270])
            
            # 对图像应用旋转
            image = transforms.functional.rotate(image, angle)
            # 对掩码应用相同的旋转
            mask = transforms.functional.rotate(mask, angle)
        
        return image, mask
    
if __name__ == "__main__":
    dataList = ['./dataset/image/train/50/ADC.nii.gz']
    maskList = ['./dataset/image/train/50/tumor.nii.gz']
    
    # 测试数据增强功能
    print("Testing with data augmentation...")
    dataset = ImgDataset(dataList, maskList, use_augmentation=True, rotation_prob=1.0)  # 100%概率旋转
    
    # 获取原始数据
    original_img, original_mask = dataset[50]
    
    # 获取增强后的数据
    augmented_img, augmented_mask = dataset[50]
    
    print(f"Original image shape: {original_img.shape}")
    print(f"Original mask shape: {original_mask.shape}")
    print(f"Augmented image shape: {augmented_img.shape}")
    print(f"Augmented mask shape: {augmented_mask.shape}")
    
    # 可视化结果
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_img[0].numpy(), cmap="gray")
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(original_mask[0].numpy(), cmap="gray")
    plt.title("Original Mask")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(augmented_img[0].numpy(), cmap="gray")
    plt.title("Augmented Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(augmented_mask[0].numpy(), cmap="gray")
    plt.title("Augmented Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
       
