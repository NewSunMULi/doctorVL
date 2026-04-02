import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from transformers import Sam3Processor, Sam3Model, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.dataset.imgData import ImgDataset


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_path, device_map="auto")
        self.processor = Sam3Processor.from_pretrained(model_path)

    def __call__(self, img: Image.Image, text=""):
        """
        推理函数
        
        Args:
            img: 输入图像
            text: 文本提示，用于引导分割
            
        Returns:
            预测的掩码
        """
        # 预处理输入
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pred_masks[:, 0:1, :, :][0]
        # 后处理输出
        output = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # 置信度阈值
            mask_threshold=0.5,  # 掩码阈值
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        return output['masks']

    def load_lora(self, lora_path):
        """
        加载训练好的LoRA适配器
        
        Args:
            lora_path: LoRA适配器的路径
        """
        # 加载LoRA适配器
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.print_trainable_parameters()
        self.processor = Sam3Processor.from_pretrained(lora_path)
        print(f"LoRA适配器已从 {lora_path} 加载")

    def train(self, nii_list, mask_list, output_dir, batch_size=4, epochs=10, learning_rate=1e-4, lora_r=8, lora_alpha=16, text_prompt=""):
        """
        训练Sam3模型
        
        Args:
            nii_list: nii.gz 图像文件路径列表
            mask_list: nii.gz 掩码文件路径列表
            output_dir: 模型保存目录
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            lora_r: LoRA的秩
            lora_alpha: LoRA的alpha参数
            text_prompt: 文本提示，用于引导分割
        """        
        # 冻结模型原始参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # 包装模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 创建数据集和数据加载器
        dataset = ImgDataset(nii_list, mask_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 计算总步数并设置学习率调度器
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # 开始训练
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                # 将数据移至设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 为批次中的每个元素创建文本提示
                text_prompts = [text_prompt] * images.shape[0]
                # 预处理输入
                inputs = self.processor(images=images, text=text_prompts, return_tensors="pt").to(self.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 处理输出形状：选择第一个实例并调整大小以匹配标签
                # 输出形状: [batch_size, num_instances, height, width]
                # 我们只关注第一个实例（肿瘤）
                pred_masks = outputs.pred_masks[:, 0:1, :, :]  # 选择第一个实例，保持通道维度
                
                # 调整大小以匹配标签形状
                import torch.nn.functional as F
                pred_masks = F.interpolate(pred_masks, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # 应用sigmoid激活函数，确保输出在[0,1]范围内
                pred_masks = torch.sigmoid(pred_masks)
                
                # 计算损失
                loss = criterion(pred_masks, masks)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        
        # 保存模型
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"模型已保存到 {output_dir}")


if __name__ == "__main__":
    # 测试 Sam3Doctor
    model = Sam3Doctor("./model/sam3/sam3-8b5")
    
    # 测试训练
    test_nii_list = ["./dataset/image/train/50/P1.nii.gz", "./dataset/image/train/50/P2.nii.gz", "./dataset/image/train/50/P3.nii.gz"]
    test_mask_list = ["./dataset/image/train/50/tumor.nii.gz"] * 3
        
    dataset1 = ImgDataset(test_nii_list, test_mask_list)
    
    # 测试加载LoRA适配器
    lora_path = './model/sam3_lora'
    try:
        model.load_lora(lora_path)
        print("LoRA加载测试成功")
    except Exception as e:
        print(f"LoRA加载测试失败: {e}")
        print("注意：这是正常的，因为可能还没有训练过LoRA模型")

    img = dataset1[50]
    op = model(img[0], text="tumor")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('QtAgg')
    plt.subplot(1, 2, 1)
    plt.imshow(op[0].numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img[1][0].numpy(), cmap='gray')
    plt.show()