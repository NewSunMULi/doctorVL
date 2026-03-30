import torch
from ..img.imgProcess import process_nii_gz, mask_process
from torch.utils.data import Dataset


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
