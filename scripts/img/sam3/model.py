import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Sam3Processor, Sam3Model
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import numpy as np
import os
from typing import Optional, List, Dict, Union


class ImageDataset(Dataset):
    """
    图像数据集类，用于加载和处理图像及对应的掩码
    
    该类继承自 PyTorch 的 Dataset 类，支持将 numpy 数组转换为 PIL 图像并进行处理
    """
    
    def __init__(self, images: np.ndarray, masks: np.ndarray, 
                 image_size: tuple = (256, 256), transform=None):
        """
        初始化 ImageDataset
        
        Args:
            images: 图像数据数组，形状为 (num_samples, height, width, channels)
            masks: 掩码数据数组，形状为 (num_samples, height, width)
            image_size: 图像调整后的大小，默认 (256, 256)
            transform: 图像变换函数，默认 None
        """
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transform
        
        # 确保图像和掩码数量匹配
        assert images.shape[0] == masks.shape[0], "Number of images and masks must match"
        
    def _numpy_to_image(self, array: np.ndarray) -> Image.Image:
        """
        将 numpy 数组转换为 PIL Image
        
        Args:
            array: numpy 数组，形状为 (height, width) 或 (height, width, channels)
            
        Returns:
            PIL Image 对象
        """
        # 统一数据类型转换
        if array.dtype not in (np.uint8, np.float32, np.float64):
            array = array.astype(np.uint8)
        elif array.dtype in (np.float32, np.float64):
            array = (array * 255).astype(np.uint8)
        
        # 根据维度创建图像
        if array.ndim == 2:
            image = Image.fromarray(array, mode='L').convert('RGB')
        elif array.ndim == 3:
            if array.shape[2] == 3:
                image = Image.fromarray(array, mode='RGB')
            elif array.shape[2] == 4:
                image = Image.fromarray(array, mode='RGBA').convert('RGB')
            else:
                raise ValueError(f"Unsupported array shape: {array.shape}")
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        
        return image
    
    def _process_mask(self, mask_array: np.ndarray) -> torch.Tensor:
        """
        处理掩码数据
        
        Args:
            mask_array: 掩码 numpy 数组
            
        Returns:
            处理后的掩码张量
        """
        # 统一数据类型转换
        if mask_array.dtype == np.uint8:
            mask_array = mask_array.astype(np.float32) / 255.0
        elif mask_array.dtype not in (np.float32, np.float64):
            mask_array = mask_array.astype(np.float32)
        
        # 调整维度
        if mask_array.ndim == 2:
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        elif mask_array.ndim == 3:
            mask_tensor = torch.from_numpy(mask_array.squeeze(0))
        else:
            raise ValueError(f"Unsupported mask array shape: {mask_array.shape}")
        
        return mask_tensor
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集的样本数量
        """
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Image.Image]]:
        """
        获取单个数据样本
        
        Args:
            idx: 索引
            
        Returns:
            包含图像、掩码、原始大小和图像路径的字典
        """
        image_array = self.images[idx]
        mask_array = self.masks[idx]
        
        # 处理图像
        image = self._numpy_to_image(image_array)
        original_size = image.size
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # 处理掩码
        mask = self._process_mask(mask_array)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 调整掩码大小
        mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
        mask_resized = Image.fromarray(mask_np, mode='L')
        mask_resized = mask_resized.resize(self.image_size, Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': mask_tensor,
            'original_size': original_size,
            'image_path': f"numpy_array_{idx}"
        }


class Sam3Doctor:
    """
    Sam3 模型包装类，用于医学影像分割
    
    该类封装了 Sam3 模型的加载、推理和训练相关功能
    """
    
    def __init__(self, model_path):
        """
        初始化 Sam3Doctor
        
        Args:
            model_path: 模型路径
        """
        # 使用半精度加载模型，减少内存消耗
        self.model = Sam3Model.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lora_config = None
        self.peft_model = None

    def __call__(self, img, text=""):
        """
        推理函数
        
        Args:
            img: 输入图像
            text: 文本提示，用于引导分割
            
        Returns:
            预测的掩码
        """
        # 预处理输入
        ip = self.processor(images=img, text=text, return_tensors="pt").to(self.device, non_blocking=True)
        
        # 使用混合精度推理
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(**ip)
        
        # 后处理输出
        output = self.processor.post_process_instance_segmentation(
            outputs, 
            threshold=0.5,  # 置信度阈值
            mask_threshold=0.5,  # 掩码阈值
            target_sizes=ip.get("original_sizes").tolist()
        )[0]

        return output['masks']

    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, 
                   target_modules: Optional[List[str]] = None):
        """
        设置 LoRA 配置
        
        Args:
            r: LoRA 秩，控制 LoRA 的容量
            lora_alpha: LoRA 缩放因子
            lora_dropout: LoRA dropout，防止过拟合
            target_modules: 目标模块列表，默认包含注意力和前馈网络模块
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            inference_mode=False
        )
        
        self.peft_model = get_peft_model(self.model, self.lora_config)
        self.peft_model.print_trainable_parameters()

    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        if self.peft_model is not None:
            # 保存 LoRA 模型
            self.peft_model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        else:
            # 保存基础模型
            self.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"Base model saved to {save_path}")

    def load_lora_model(self, lora_path: str):
        """
        加载 LoRA 模型
        
        Args:
            lora_path: LoRA 模型路径
        """
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(self.model, lora_path, torch_dtype=torch.bfloat16)
        self.peft_model.to(self.device)
        print(f"LoRA model loaded from {lora_path}")


if __name__ == "__main__":
    # 测试 Sam3Doctor
    model = Sam3Doctor("./model/sam3/sam3-8b5")
    img1 = Image.open("./img/img2.jpeg").convert("L")
    print(img1.size)
    img_t = torch.from_numpy(np.array(img1)).unsqueeze(0)
    print(img_t.shape)
