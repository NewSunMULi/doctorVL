import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from scripts.img.imgProcess import process_nii_gz, mask_process

class ImgDataset(Dataset):
    def __init__(self, nii_list, mask_list):
        """
        初始化 ImgDataset
        
        Args:
            nii_list: nii.gz 图像文件路径列表
            mask_list: nii.gz 掩码文件路径列表
        """
        self.nii_list = nii_list
        self.mask_list = mask_list
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
        return self.data[index], self.masks[index]
    
if __name__ == "__main__":
    dataList = ['./dataset/image/train/50/ADC.nii.gz']
    maskList = ['./dataset/image/train/50/tumor.nii.gz']
    dataset = ImgDataset(dataList, maskList)
    img = dataset[50]
    img1 = img[0].numpy()
    img2 = img[1].numpy()
    print(img1.shape)
    print(img2.shape)
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1[0], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(img2[0], cmap="gray")
    plt.show()
       
